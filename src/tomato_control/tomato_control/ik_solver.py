import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class IKResult:
    success: bool
    joint_angles: Dict[str, float]
    reason: str = ""
    wrist_target: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict[str, float]] = None


class TomatoArmIK:
    """
    AIK for the tomato arm.

    IK target frame:
        origin = joint_2 rotation origin
        axes are parallel to base_link
        +X = forward
        +Y = left
        +Z = up

    Target points must already be transformed into this fixed,
    joint_2-origin frame. The physical base_link-to-joint_2 translation from
    the URDF is retained for diagnostics, but it is not subtracted during IK.

    joint_1 = base yaw about Z
    joint_2 = shoulder pitch
    joint_3 = elbow pitch
    joint_4 = wrist pitch

    The URDF supplies the effective joint_1 zero direction and the complete
    three-dimensional wrist-to-tool-tip offset. Motor direction inversion is
    intentionally outside this geometric solver.

    """

    def __init__(
        self,
        upper_arm_length: float,
        forearm_length: float,
        tool_length: float = 0.0,
        tool_offset: Optional[Tuple[float, float, float]] = None,
        shoulder_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        base_yaw_zero_offset: float = 0.0,
        base_yaw_direction: float = 1.0,
        tool_lateral_axis_sign: float = 1.0,
        tool_z_vertical_sign: float = 1.0,
        joint_names: Tuple[str, str, str, str] = (
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
        ),
        joint_limits: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    ):
        self.L1 = float(upper_arm_length)
        self.L2 = float(forearm_length)

        # Backward compatibility: callers that provide only tool_length still
        # get the original straight, collinear tool model. URDF-based
        # construction supplies the full wrist-to-tip translation instead.
        if tool_offset is None:
            tool_offset = (float(tool_length), 0.0, 0.0)

        if len(tool_offset) != 3:
            raise ValueError("tool_offset must contain exactly three values")

        self.tool_offset = tuple(float(v) for v in tool_offset)
        self.tool_offset_x = self.tool_offset[0]
        self.tool_offset_y = self.tool_offset[1]
        self.tool_offset_z = self.tool_offset[2]

        self.base_yaw_zero_offset = self._normalize_angle(
            float(base_yaw_zero_offset)
        )

        if abs(abs(float(base_yaw_direction)) - 1.0) > 1e-9:
            raise ValueError(
                "base_yaw_direction must be either +1 or -1"
            )
        self.base_yaw_direction = 1.0 if base_yaw_direction > 0.0 else -1.0

        if abs(abs(float(tool_lateral_axis_sign)) - 1.0) > 1e-9:
            raise ValueError(
                "tool_lateral_axis_sign must be either +1 or -1"
            )
        self.tool_lateral_axis_sign = (
            1.0 if tool_lateral_axis_sign > 0.0 else -1.0
        )

        if abs(abs(float(tool_z_vertical_sign)) - 1.0) > 1e-9:
            raise ValueError(
                "tool_z_vertical_sign must be either +1 or -1"
            )
        self.tool_z_vertical_sign = (
            1.0 if tool_z_vertical_sign > 0.0 else -1.0
        )

        # Retained for existing diagnostics. Reachability now uses the full
        # vector components rather than this scalar magnitude.
        self.L_tool = self._norm(self.tool_offset)

        # Controller targets are joint_2-origin-relative, so this is zero for
        # the normal URDF construction path. The attribute is retained because
        # controller diagnostics report the mathematical shoulder offset.
        self.shoulder_offset = tuple(float(v) for v in shoulder_offset)
        self.joint_2_origin_in_base = (0.0, 0.0, 0.0)

        self.joint_1_name = joint_names[0]
        self.joint_2_name = joint_names[1]
        self.joint_3_name = joint_names[2]
        self.joint_4_name = joint_names[3]

        self.joint_limits = joint_limits or {}

    @classmethod
    def from_robot_description(
        cls,
        robot_description: str,
        joint_2_name: str = "joint_2",
        joint_3_name: str = "joint_3",
        joint_4_name: str = "joint_4",
        joint_5_name: str = "joint_5",
        tool_tip_joint_name: Optional[str] = "tool_tip_joint",
    ):
        """
        Build IK solver from URDF robot_description.

        Expected URDF structure:
            joint_1 origin/axis = base-yaw zero orientation and direction
            joint_2 origin = physical base_link to shoulder-pitch transform
            joint_3 origin = shoulder to elbow offset
            joint_4 origin = elbow to wrist offset
            joint_5 origin = wrist to end effector/tool offset
            tool_tip_joint origin = end effector to physical tool tip offset

        The URDF joint_2 translation is not used as an IK subtraction because
        all controller targets are already expressed relative to joint_2.
        """

        root = ET.fromstring(robot_description)

        _, joint_1_rpy = cls._get_joint_origin(root, "joint_1")
        joint_1_axis = cls._get_joint_axis(root, "joint_1")
        joint_2_origin_in_base, joint_2_rpy = cls._get_joint_origin(
            root,
            joint_2_name,
        )
        joint_2_axis = cls._get_joint_axis(root, joint_2_name)
        joint_3_xyz, joint_3_rpy = cls._get_joint_origin(
            root,
            joint_3_name,
        )
        joint_4_xyz, joint_4_rpy = cls._get_joint_origin(
            root,
            joint_4_name,
        )
        joint_5_xyz, joint_5_rpy = cls._get_joint_origin(
            root,
            joint_5_name,
        )

        upper_arm_length = cls._norm(joint_3_xyz)
        forearm_length = cls._norm(joint_4_xyz)

        if upper_arm_length <= 0.0 or forearm_length <= 0.0:
            raise ValueError("URDF arm link lengths must be greater than zero")

        # Derive the logical zero direction of joint_1 from the actual URDF
        # transforms. The joint_1 origin has a 180-degree yaw, but joint_2's
        # fixed rotation and -Y axis make positive shoulder pitch move toward
        # base +X. Therefore the effective arm-forward yaw at q1=0 is 0, not
        # pi. Computing the direction avoids applying that 180-degree change
        # twice in IK.
        joint_1_rotation = cls._rpy_to_rotation(joint_1_rpy)
        joint_2_rotation = cls._rpy_to_rotation(joint_2_rpy)
        base_to_joint_2_rotation = cls._matmul_rotation(
            joint_1_rotation,
            joint_2_rotation,
        )

        upper_arm_unit = tuple(
            component / upper_arm_length for component in joint_3_xyz
        )
        positive_shoulder_motion = cls._rotate_vector(
            base_to_joint_2_rotation,
            cls._cross(joint_2_axis, upper_arm_unit),
        )
        forward_xy_norm = math.hypot(
            positive_shoulder_motion[0],
            positive_shoulder_motion[1],
        )

        if forward_xy_norm < 1e-9:
            raise ValueError(
                "Could not derive the arm-forward direction from joint_2"
            )

        base_yaw_zero_offset = math.atan2(
            positive_shoulder_motion[1],
            positive_shoulder_motion[0],
        )

        joint_1_axis_in_base = cls._rotate_vector(
            joint_1_rotation,
            joint_1_axis,
        )
        if (
            math.hypot(joint_1_axis_in_base[0], joint_1_axis_in_base[1])
            > 1e-6
            or abs(abs(joint_1_axis_in_base[2]) - 1.0) > 1e-6
        ):
            raise ValueError(
                "The analytical IK requires joint_1 to rotate about base Z"
            )
        base_yaw_direction = (
            1.0 if joint_1_axis_in_base[2] > 0.0 else -1.0
        )
        # Compose the fixed transforms instead of adding their magnitudes.
        # joint_5_xyz is expressed in wrist_link. tool_tip_joint's translation
        # is expressed in end_effector_link, so rotate it by joint_5's fixed
        # orientation before adding it.
        tool_offset = joint_5_xyz

        # Include the additional fixed offset from end_effector_link to tool_tip_link
        if tool_tip_joint_name is not None:
            tool_tip_joint = root.find(f".//joint[@name='{tool_tip_joint_name}']")

            if tool_tip_joint is not None:
                tool_tip_joint_xyz, _ = cls._get_joint_origin(
                    root,
                    tool_tip_joint_name,
                )
                joint_5_rotation = cls._rpy_to_rotation(joint_5_rpy)
                rotated_tool_tip_xyz = cls._rotate_vector(
                    joint_5_rotation,
                    tool_tip_joint_xyz,
                )
                tool_offset = tuple(
                    joint_5_xyz[index] + rotated_tool_tip_xyz[index]
                    for index in range(3)
                )

        # Determine how the wrist-frame Y and Z components map into the
        # solver's arm plane. This URDF's 180-degree base rotation makes
        # wrist +Y point toward the arm plane's right, so the negative tool Y
        # offset moves the tip toward robot-left.
        joint_3_rotation = cls._rpy_to_rotation(joint_3_rpy)
        joint_4_rotation = cls._rpy_to_rotation(joint_4_rpy)
        wrist_zero_rotation = cls._matmul_rotation(
            cls._matmul_rotation(
                base_to_joint_2_rotation,
                joint_3_rotation,
            ),
            joint_4_rotation,
        )

        forward_unit = (
            math.cos(base_yaw_zero_offset),
            math.sin(base_yaw_zero_offset),
            0.0,
        )
        left_unit = (-forward_unit[1], forward_unit[0], 0.0)
        wrist_y_at_zero = cls._rotate_vector(
            wrist_zero_rotation,
            (0.0, 1.0, 0.0),
        )
        lateral_alignment = sum(
            wrist_y_at_zero[index] * left_unit[index]
            for index in range(3)
        )

        if abs(abs(lateral_alignment) - 1.0) > 1e-6:
            raise ValueError(
                "The analytical IK requires wrist Y to remain perpendicular "
                "to the vertical arm plane"
            )
        tool_lateral_axis_sign = 1.0 if lateral_alignment > 0.0 else -1.0

        horizontal_joint_2_rotation = cls._axis_angle_to_rotation(
            joint_2_axis,
            math.pi / 2.0,
        )
        wrist_horizontal_rotation = cls._matmul_rotation(
            cls._matmul_rotation(
                cls._matmul_rotation(
                    base_to_joint_2_rotation,
                    horizontal_joint_2_rotation,
                ),
                joint_3_rotation,
            ),
            joint_4_rotation,
        )
        wrist_x_horizontal = cls._rotate_vector(
            wrist_horizontal_rotation,
            (1.0, 0.0, 0.0),
        )
        horizontal_alignment = sum(
            wrist_x_horizontal[index] * forward_unit[index]
            for index in range(3)
        )
        if (
            horizontal_alignment < 1.0 - 1e-5
            or abs(wrist_x_horizontal[2]) > 1e-5
        ):
            raise ValueError(
                "The analytical IK requires q2=pi/2 to point wrist +X "
                "forward and horizontally"
            )

        wrist_z_horizontal = cls._rotate_vector(
            wrist_horizontal_rotation,
            (0.0, 0.0, 1.0),
        )
        if abs(abs(wrist_z_horizontal[2]) - 1.0) > 1e-5:
            raise ValueError(
                "The analytical IK requires wrist Z to be vertical when "
                "the tool is horizontal"
            )
        tool_z_vertical_sign = 1.0 if wrist_z_horizontal[2] > 0.0 else -1.0

        joint_limits = cls._get_joint_limits(root)

        solver = cls(
            upper_arm_length=upper_arm_length,
            forearm_length=forearm_length,
            tool_offset=tool_offset,
            # The mathematical origin is the joint_2 origin itself.
            shoulder_offset=(0.0, 0.0, 0.0),
            base_yaw_zero_offset=base_yaw_zero_offset,
            base_yaw_direction=base_yaw_direction,
            tool_lateral_axis_sign=tool_lateral_axis_sign,
            tool_z_vertical_sign=tool_z_vertical_sign,
            joint_names=("joint_1", joint_2_name, joint_3_name, joint_4_name),
            joint_limits=joint_limits,
        )
        solver.joint_2_origin_in_base = tuple(
            float(value) for value in joint_2_origin_in_base
        )
        return solver

    @staticmethod
    def _norm(xyz):
        return math.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2)

    @staticmethod
    def _get_joint_xyz(root, joint_name):
        xyz, _ = TomatoArmIK._get_joint_origin(root, joint_name)
        return xyz

    @staticmethod
    def _get_joint_axis(root, joint_name):
        joint = root.find(f".//joint[@name='{joint_name}']")

        if joint is None:
            raise ValueError(f"Joint '{joint_name}' not found in URDF")

        axis = joint.find("axis")
        axis_str = "1 0 0" if axis is None else axis.attrib.get(
            "xyz",
            "1 0 0",
        )

        try:
            axis_xyz = tuple(float(value) for value in axis_str.split())
        except ValueError as error:
            raise ValueError(
                f"Joint '{joint_name}' has invalid axis xyz='{axis_str}'"
            ) from error

        if len(axis_xyz) != 3:
            raise ValueError(
                f"Joint '{joint_name}' has invalid axis xyz='{axis_str}'"
            )

        axis_norm = TomatoArmIK._norm(axis_xyz)
        if axis_norm < 1e-12:
            raise ValueError(f"Joint '{joint_name}' has a zero-length axis")

        return tuple(component / axis_norm for component in axis_xyz)

    @staticmethod
    def _get_joint_origin(root, joint_name):
        joint = root.find(f".//joint[@name='{joint_name}']")

        if joint is None:
            raise ValueError(f"Joint '{joint_name}' not found in URDF")

        origin = joint.find("origin")

        if origin is None:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

        xyz_str = origin.attrib.get("xyz", "0 0 0")
        xyz = [float(v) for v in xyz_str.split()]

        if len(xyz) != 3:
            raise ValueError(f"Joint '{joint_name}' has invalid xyz='{xyz_str}'")

        rpy_str = origin.attrib.get("rpy", "0 0 0")
        rpy = [float(v) for v in rpy_str.split()]

        if len(rpy) != 3:
            raise ValueError(f"Joint '{joint_name}' has invalid rpy='{rpy_str}'")

        return tuple(xyz), tuple(rpy)

    @staticmethod
    def _rpy_to_rotation(rpy):
        """Return the URDF fixed-axis Rz(yaw) Ry(pitch) Rx(roll) matrix."""

        roll, pitch, yaw = rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        return (
            (
                cy * cp,
                cy * sp * sr - sy * cr,
                cy * sp * cr + sy * sr,
            ),
            (
                sy * cp,
                sy * sp * sr + cy * cr,
                sy * sp * cr - cy * sr,
            ),
            (-sp, cp * sr, cp * cr),
        )

    @staticmethod
    def _axis_angle_to_rotation(axis, angle):
        """Return a Rodrigues rotation matrix for a normalized axis."""

        x, y, z = axis
        cosine = math.cos(angle)
        sine = math.sin(angle)
        one_minus_cosine = 1.0 - cosine

        return (
            (
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ),
            (
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ),
            (
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ),
        )

    @staticmethod
    def _matmul_rotation(left, right):
        return tuple(
            tuple(
                sum(left[row][index] * right[index][column] for index in range(3))
                for column in range(3)
            )
            for row in range(3)
        )

    @staticmethod
    def _cross(left, right):
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    @staticmethod
    def _rotate_vector(rotation, vector):
        return tuple(
            sum(rotation[row][column] * vector[column] for column in range(3))
            for row in range(3)
        )

    @staticmethod
    def _get_joint_limits(root):
        limits = {}

        for joint in root.findall(".//joint"):
            name = joint.attrib.get("name")
            joint_type = joint.attrib.get("type")

            if name is None:
                continue

            if joint_type == "continuous":
                limits[name] = (None, None)
                continue

            limit = joint.find("limit")

            if limit is None:
                limits[name] = (None, None)
                continue

            lower = limit.attrib.get("lower")
            upper = limit.attrib.get("upper")

            lower = float(lower) if lower is not None else None
            upper = float(upper) if upper is not None else None

            limits[name] = (lower, upper)

        return limits

    @staticmethod
    def _clamp(value, low=-1.0, high=1.0):
        return max(low, min(high, value))

    @staticmethod
    def _normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _check_joint_limits(self, joint_angles):
        for joint_name, angle in joint_angles.items():
            if joint_name not in self.joint_limits:
                continue

            lower, upper = self.joint_limits[joint_name]

            if lower is not None and angle < lower:
                return False, f"{joint_name}={angle:.3f} below lower limit {lower:.3f}"

            if upper is not None and angle > upper:
                return False, f"{joint_name}={angle:.3f} above upper limit {upper:.3f}"

        return True, ""

    def solve(
        self,
        x: float,
        y: float,
        z: float,
        tool_angle_from_horizontal: float = 0.0,
        elbow_solution: str = "up",
        target_is_tool_tip: bool = True,
        check_limits: bool = True,
    ) -> IKResult:
        """
        Solve IK for a target point in the fixed joint_2-origin frame.

        Args:
            x, y, z:
                Target position relative to the joint_2 rotation origin,
                expressed along base_link-parallel axes, in meters.

            tool_angle_from_horizontal:
                Desired tool/end-effector angle in the vertical arm plane.
                0 rad = tool points forward/horizontal.
                pi/2 rad = tool points straight up.
                -pi/2 rad = tool points straight down.

            elbow_solution:
                "down" or "up".
                Over or under hand

            target_is_tool_tip:
                If True, account for the complete three-dimensional
                wrist-to-tip offset before solving base/shoulder/elbow.
                If False, treat the target as the wrist point.

            check_limits:
                If True, reject solutions outside the URDF joint limits.

        Returns:
            IKResult with joint angles in radians.
        """

        x = float(x)
        y = float(y)
        z = float(z)

        # Collapse the target into the vertical arm plane. A lateral tool-tip
        # offset means the wrist plane does not point directly at the target:
        # the target, radial arm component, and fixed lateral component form a
        # right triangle in base XY.
        r_base = math.sqrt(x ** 2 + y ** 2)

        if target_is_tool_tip:
            tool_lateral = (
                self.tool_lateral_axis_sign * self.tool_offset_y
            )
        else:
            tool_lateral = 0.0

        if r_base + 1e-12 < abs(tool_lateral):
            return IKResult(
                success=False,
                joint_angles={},
                reason=(
                    "Target is inside the tool's fixed lateral offset: "
                    f"horizontal_radius={r_base:.3f} < "
                    f"lateral_offset={abs(tool_lateral):.3f}"
                ),
            )

        radial_to_tip = math.sqrt(
            max(0.0, r_base ** 2 - tool_lateral ** 2)
        )

        target_bearing = math.atan2(y, x)
        lateral_bearing = math.atan2(tool_lateral, radial_to_tip)
        arm_plane_bearing = target_bearing - lateral_bearing

        # JOINT 1
        # The effective q1=0 arm direction is derived from joint_1's fixed
        # origin and the direction in which positive joint_2 moves. In the
        # current URDF those transforms cancel to make q1=0 face base +X.
        q1 = self.base_yaw_direction * (
            arm_plane_bearing - self.base_yaw_zero_offset
        )
        q1 = self._normalize_angle(q1)

        # Targets are already relative to joint_2. For the normal construction
        # path shoulder_offset is therefore exactly zero. Keeping these lines
        # preserves compatibility with direct constructor users while making
        # the intended frame explicit in from_robot_description().
        shoulder_x = self.shoulder_offset[0]
        shoulder_z = self.shoulder_offset[2]

        # Collapse to 2d arm plane
        r = radial_to_tip - shoulder_x
        z_rel = z - shoulder_z

        # Account for the planar components of the complete wrist-to-tool-tip
        # vector. Local +X follows the requested tool angle. The direction of
        # local +Z relative to the plane is derived from the URDF.
        if target_is_tool_tip:
            tool_radial = (
                self.tool_offset_x
                * math.cos(tool_angle_from_horizontal)
                - self.tool_z_vertical_sign
                * self.tool_offset_z
                * math.sin(tool_angle_from_horizontal)
            )
            tool_vertical = (
                self.tool_offset_x
                * math.sin(tool_angle_from_horizontal)
                + self.tool_z_vertical_sign
                * self.tool_offset_z
                * math.cos(tool_angle_from_horizontal)
            )
            wrist_r = r - tool_radial
            wrist_z = z_rel - tool_vertical
        else:
            tool_radial = 0.0
            tool_vertical = 0.0
            wrist_r = r
            wrist_z = z_rel

        # Distance from shoulder to wrist target
        h_sq = wrist_r ** 2 + wrist_z ** 2
        h = math.sqrt(h_sq)

        # Reject if too close to shoulder
        if h < 1e-9:
            return IKResult(
                success=False,
                joint_angles={},
                reason="Target is too close to shoulder origin",
            )

        max_reach = self.L1 + self.L2
        min_reach = abs(self.L1 - self.L2)

        # Reject if too far
        if h > max_reach:
            return IKResult(
                success=False,
                joint_angles={},
                reason=f"Target unreachable: h={h:.3f} > max_reach={max_reach:.3f}",
                wrist_target=(wrist_r, wrist_z),
            )

        # Reject if too close
        if h < min_reach:
            return IKResult(
                success=False,
                joint_angles={},
                reason=f"Target unreachable: h={h:.3f} < min_reach={min_reach:.3f}",
                wrist_target=(wrist_r, wrist_z),
            )

        
        # Standard planar 2-link IK.
        #
        # q3_geom is the forearm angle relative to the upper arm in geometric coords.
        # q3_geom = 0 means arm is straight.
        cos_q3_geom = (h_sq - self.L1 ** 2 - self.L2 ** 2) / (2.0 * self.L1 * self.L2)
        cos_q3_geom = self._clamp(cos_q3_geom)

        q3_abs = math.acos(cos_q3_geom)

        # Positive q3_geom places the elbow below the shoulder-to-wrist line.
        if elbow_solution == "down":
            q3_geom = q3_abs
        elif elbow_solution == "up":
            q3_geom = -q3_abs
        else:
            return IKResult(
                success=False,
                joint_angles={},
                reason=f"Invalid elbow_solution='{elbow_solution}', use 'down' or 'up'",
            )

        target_angle = math.atan2(wrist_z, wrist_r)

        shoulder_geom = target_angle - math.atan2(
            self.L2 * math.sin(q3_geom),
            self.L1 + self.L2 * math.cos(q3_geom),
        )

        forearm_geom = shoulder_geom + q3_geom

        # Convert geometric planar angles into your robot joint convention.
        #
        # Your zero pose has links vertical:
        #   q2 = 0 -> upper arm points up
        #   q2 = pi/2 -> upper arm points forward/horizontal
        q2 = (math.pi / 2.0) - shoulder_geom

        # Positive q3 bends the forearm forward/down from the upper arm.
        q3 = -q3_geom

        # q4 sets tool orientation.
        # If tool_angle_from_horizontal = 0, the tool points forward/horizontal.
        q4 = forearm_geom - tool_angle_from_horizontal

        joint_angles = {
            self.joint_1_name: q1,
            self.joint_2_name: q2,
            self.joint_3_name: q3,
            self.joint_4_name: q4,
        }

        if check_limits:
            ok, reason = self._check_joint_limits(joint_angles)
            if not ok:
                return IKResult(
                    success=False,
                    joint_angles=joint_angles,
                    reason=reason,
                    wrist_target=(wrist_r, wrist_z),
                    metadata={
                        "r_base": r_base,
                        "radial_to_tip": radial_to_tip,
                        "tool_lateral": tool_lateral,
                        "target_bearing": target_bearing,
                        "arm_plane_bearing": arm_plane_bearing,
                        "base_yaw_zero_offset": self.base_yaw_zero_offset,
                        "r": r,
                        "z_rel": z_rel,
                        "h": h,
                        "shoulder_geom": shoulder_geom,
                        "forearm_geom": forearm_geom,
                    },
                )

        return IKResult(
            success=True,
            joint_angles=joint_angles,
            reason="success",
            wrist_target=(wrist_r, wrist_z),
            metadata={
                "r_base": r_base,
                "radial_to_tip": radial_to_tip,
                "tool_lateral": tool_lateral,
                "target_bearing": target_bearing,
                "arm_plane_bearing": arm_plane_bearing,
                "base_yaw_zero_offset": self.base_yaw_zero_offset,
                "r": r,
                "z_rel": z_rel,
                "h": h,
                "shoulder_geom": shoulder_geom,
                "forearm_geom": forearm_geom,
                "tool_angle_from_horizontal": tool_angle_from_horizontal,
                "tool_offset_x": self.tool_offset_x,
                "tool_offset_y": self.tool_offset_y,
                "tool_offset_z": self.tool_offset_z,
                "tool_radial": tool_radial,
                "tool_vertical": tool_vertical,
            },
        )

    def solve_all(
        self,
        x: float,
        y: float,
        z: float,
        tool_angle_from_horizontal: float = 0.0,
        target_is_tool_tip: bool = True,
        check_limits: bool = True,
    ):
        """
        Return both elbow-down and elbow-up IK attempts
        """

        return {
            "down": self.solve(
                x,
                y,
                z,
                tool_angle_from_horizontal=tool_angle_from_horizontal,
                elbow_solution="down",
                target_is_tool_tip=target_is_tool_tip,
                check_limits=check_limits,
            ),
            "up": self.solve(
                x,
                y,
                z,
                tool_angle_from_horizontal=tool_angle_from_horizontal,
                elbow_solution="up",
                target_is_tool_tip=target_is_tool_tip,
                check_limits=check_limits,
            ),
        }