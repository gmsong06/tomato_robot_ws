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

    Robot base frame:
        +X = forward
        +Y = left
        +Z = up

    Assumes target point is transformed into robot base frame

    joint_1 = base yaw about Z
    joint_2 = shoulder pitch
    joint_3 = elbow pitch
    joint_4 = wrist pitch

    """

    def __init__(
        self,
        upper_arm_length: float,
        forearm_length: float,
        tool_length: float = 0.0,
        shoulder_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
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
        self.L_tool = float(tool_length)

        self.shoulder_offset = tuple(float(v) for v in shoulder_offset)

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
            joint_2 origin = base/yaw frame to shoulder pitch offset
            joint_3 origin = shoulder to elbow offset
            joint_4 origin = elbow to wrist offset
            joint_5 origin = wrist to end effector/tool offset
            tool_tip_joint origin = end effector to physical tool tip offset
        """

        root = ET.fromstring(robot_description)

        shoulder_offset = cls._get_joint_xyz(root, joint_2_name)
        joint_3_xyz = cls._get_joint_xyz(root, joint_3_name)
        joint_4_xyz = cls._get_joint_xyz(root, joint_4_name)
        joint_5_xyz = cls._get_joint_xyz(root, joint_5_name)

        upper_arm_length = cls._norm(joint_3_xyz)
        forearm_length = cls._norm(joint_4_xyz)
        tool_length = cls._norm(joint_5_xyz)

        # Include the additional fixed offset from end_effector_link to tool_tip_link
        if tool_tip_joint_name is not None:
            tool_tip_joint = root.find(f".//joint[@name='{tool_tip_joint_name}']")

            if tool_tip_joint is not None:
                tool_tip_joint_xyz = cls._get_joint_xyz(root, tool_tip_joint_name)
                tool_length += cls._norm(tool_tip_joint_xyz)

        joint_limits = cls._get_joint_limits(root)

        return cls(
            upper_arm_length=upper_arm_length,
            forearm_length=forearm_length,
            tool_length=tool_length,
            shoulder_offset=shoulder_offset,
            joint_names=("joint_1", joint_2_name, joint_3_name, joint_4_name),
            joint_limits=joint_limits,
        )

    @staticmethod
    def _norm(xyz):
        return math.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2)

    @staticmethod
    def _get_joint_xyz(root, joint_name):
        joint = root.find(f".//joint[@name='{joint_name}']")

        if joint is None:
            raise ValueError(f"Joint '{joint_name}' not found in URDF")

        origin = joint.find("origin")

        if origin is None:
            return (0.0, 0.0, 0.0)

        xyz_str = origin.attrib.get("xyz", "0 0 0")
        xyz = [float(v) for v in xyz_str.split()]

        if len(xyz) != 3:
            raise ValueError(f"Joint '{joint_name}' has invalid xyz='{xyz_str}'")

        return tuple(xyz)

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
        Solve IK for a target point in robot base frame.

        Args:
            x, y, z:
                Target position in robot base frame, meters.

            tool_angle_from_horizontal:
                Desired tool/end-effector angle in the vertical arm plane.
                0 rad = tool points forward/horizontal.
                pi/2 rad = tool points straight up.
                -pi/2 rad = tool points straight down.

            elbow_solution:
                "down" or "up".
                Over or under hand

            target_is_tool_tip:
                If True, subtract tool length before solving shoulder/elbow.
                If False, treat the target as the wrist point.

            check_limits:
                If True, reject solutions outside the URDF joint limits.

        Returns:
            IKResult with joint angles in radians.
        """

        x = float(x)
        y = float(y)
        z = float(z)

        # JOINT 1
        # Yaw
        q1 = math.atan2(y, x)
        q1 = self._normalize_angle(q1)

        # Collapse 3D target into 2D arm plane after base yaw
        r_base = math.sqrt(x ** 2 + y ** 2)

        # Account for shoulder pitch joint offset from robot base origin
        # (x is 0, z is very thin, so might be trivial but just to be safe)
        shoulder_x = self.shoulder_offset[0]
        shoulder_z = self.shoulder_offset[2]

        # Collapse to 2d arm plane
        r = r_base - shoulder_x
        z_rel = z - shoulder_z

        # Account for end-effector length
        if target_is_tool_tip:
            wrist_r = r - self.L_tool * math.cos(tool_angle_from_horizontal)
            wrist_z = z_rel - self.L_tool * math.sin(tool_angle_from_horizontal)
        else:
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
                "r": r,
                "z_rel": z_rel,
                "h": h,
                "shoulder_geom": shoulder_geom,
                "forearm_geom": forearm_geom,
                "tool_angle_from_horizontal": tool_angle_from_horizontal,
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