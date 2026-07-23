from __future__ import annotations

from dataclasses import dataclass
import math

from rclpy.node import Node


@dataclass(frozen=True)
class ControllerConfig:
    """Validated configuration for the tomato controller."""

    minimum_valid_disparity_px: float
    maximum_valid_disparity_px: float
    minimum_valid_disparity_ratio: float
    roi_total_shrink_fraction: float
    surface_disparity_percentile: float

    camera_joint_2_x_m: float
    camera_joint_2_y_m: float
    camera_joint_2_z_m: float
    camera_pitch_down_degrees: float

    pregrasp_distance_m: float
    retreat_distance_m: float
    tool_angle_from_horizontal_rad: float
    elbow_configuration: str

    contact_standoff_m: float
    contact_lateral_offset_m: float
    contact_vertical_offset_m: float

    motor_commands_enabled: bool
    joint_command_topic: str
    command_interval_seconds: float
    invert_base_yaw_motor_command: bool

    manual_approval_required: bool
    approval_service_name: str
    selection_service_name: str
    clear_selection_service_name: str
    retract_service_name: str

    # Fixed ROS/URDF joint-space pose used after the tomato-relative retreat.
    home_joint_positions_rad: tuple[float, float, float, float]

    @staticmethod
    def declare_parameters(node: Node) -> None:
        node.declare_parameter("robot_description", "")

        # Disparity filtering.
        node.declare_parameter("min_valid_disparity", 1.0)
        node.declare_parameter("max_valid_disparity", 400.0)
        node.declare_parameter("min_valid_ratio", 0.10)

        # Total fraction removed from the box. A value of 0.40 removes 20%
        # from each side and keeps the center 60%.
        node.declare_parameter("roi_shrink", 0.20)
        node.declare_parameter("surface_disparity_percentile", 75.0)

        # Left rectified camera pose relative to the joint_2 rotation origin.
        # The axes remain parallel to base_link: +X forward, +Y left, +Z up.
        node.declare_parameter("camera_x_m", -0.20)
        node.declare_parameter("camera_y_m", 0.051555)
        node.declare_parameter("camera_z_m", 0.647)
        node.declare_parameter("camera_pitch_down_deg", 35.0)

        # Tomato-relative three-point trajectory.
        node.declare_parameter("pregrasp_offset_m", 0.05)
        node.declare_parameter("retreat_offset_m", 0.05)
        node.declare_parameter("tool_angle_from_horizontal", 0.0)
        node.declare_parameter("elbow_solution", "up")

        # Contact corrections in the fixed joint_2-origin frame.
        node.declare_parameter("contact_surface_offset_m", 0.03)
        node.declare_parameter("contact_y_offset_m", 0.0)
        node.declare_parameter("contact_z_offset_m", 0.0)

        # Motor output.
        node.declare_parameter("enable_motor_commands", False)
        node.declare_parameter(
            "joint_command_topic",
            "/joint_target_positions",
        )
        node.declare_parameter("command_interval_sec", 2.0)
        node.declare_parameter("invert_joint_1_command", True)

        # Manual tomato selection.
        node.declare_parameter(
            "selection_service_name",
            "/controller/select_tomato",
        )
        node.declare_parameter(
            "clear_selection_service_name",
            "/controller/clear_selection",
        )
        node.declare_parameter(
            "retract_service_name",
            "/controller/retract",
        )

        # Fixed home pose in ROS/URDF radians.
        node.declare_parameter(
            "home_joint_positions",
            [
                -0.0928058376670813,
                0.10471975511965978,
                1.53588974175501,
                0.32903887900147005,
            ],
        )

        # Manual approval.
        node.declare_parameter("require_manual_approval", True)
        node.declare_parameter(
            "approval_service_name",
            "/controller/set_motion_approval",
        )

    @classmethod
    def from_node(cls, node: Node) -> "ControllerConfig":
        config = cls(
            minimum_valid_disparity_px=float(
                node.get_parameter("min_valid_disparity").value
            ),
            maximum_valid_disparity_px=float(
                node.get_parameter("max_valid_disparity").value
            ),
            minimum_valid_disparity_ratio=float(
                node.get_parameter("min_valid_ratio").value
            ),
            roi_total_shrink_fraction=float(
                node.get_parameter("roi_shrink").value
            ),
            surface_disparity_percentile=float(
                node.get_parameter("surface_disparity_percentile").value
            ),
            camera_joint_2_x_m=float(
                node.get_parameter("camera_x_m").value
            ),
            camera_joint_2_y_m=float(
                node.get_parameter("camera_y_m").value
            ),
            camera_joint_2_z_m=float(
                node.get_parameter("camera_z_m").value
            ),
            camera_pitch_down_degrees=float(
                node.get_parameter("camera_pitch_down_deg").value
            ),
            pregrasp_distance_m=float(
                node.get_parameter("pregrasp_offset_m").value
            ),
            retreat_distance_m=float(
                node.get_parameter("retreat_offset_m").value
            ),
            tool_angle_from_horizontal_rad=float(
                node.get_parameter("tool_angle_from_horizontal").value
            ),
            elbow_configuration=str(
                node.get_parameter("elbow_solution").value
            ),
            contact_standoff_m=float(
                node.get_parameter("contact_surface_offset_m").value
            ),
            contact_lateral_offset_m=float(
                node.get_parameter("contact_y_offset_m").value
            ),
            contact_vertical_offset_m=float(
                node.get_parameter("contact_z_offset_m").value
            ),
            motor_commands_enabled=bool(
                node.get_parameter("enable_motor_commands").value
            ),
            joint_command_topic=str(
                node.get_parameter("joint_command_topic").value
            ),
            command_interval_seconds=float(
                node.get_parameter("command_interval_sec").value
            ),
            invert_base_yaw_motor_command=bool(
                node.get_parameter("invert_joint_1_command").value
            ),
            manual_approval_required=bool(
                node.get_parameter("require_manual_approval").value
            ),
            approval_service_name=str(
                node.get_parameter("approval_service_name").value
            ),
            selection_service_name=str(
                node.get_parameter("selection_service_name").value
            ),
            clear_selection_service_name=str(
                node.get_parameter(
                    "clear_selection_service_name"
                ).value
            ),
            retract_service_name=str(
                node.get_parameter("retract_service_name").value
            ),
            home_joint_positions_rad=tuple(
                float(value)
                for value in node.get_parameter(
                    "home_joint_positions"
                ).value
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.minimum_valid_disparity_px < 0.0:
            raise ValueError("min_valid_disparity must be nonnegative")

        if self.maximum_valid_disparity_px <= self.minimum_valid_disparity_px:
            raise ValueError(
                "max_valid_disparity must be greater than min_valid_disparity"
            )

        if not 0.0 <= self.minimum_valid_disparity_ratio <= 1.0:
            raise ValueError("min_valid_ratio must be between 0 and 1")

        if not 0.0 <= self.roi_total_shrink_fraction < 1.0:
            raise ValueError("roi_shrink must be in the range [0, 1)")

        if not 0.0 <= self.surface_disparity_percentile <= 100.0:
            raise ValueError(
                "surface_disparity_percentile must be between 0 and 100"
            )

        if self.pregrasp_distance_m < 0.0:
            raise ValueError("pregrasp_offset_m must be nonnegative")

        if self.retreat_distance_m < 0.0:
            raise ValueError("retreat_offset_m must be nonnegative")

        if self.contact_standoff_m < 0.0:
            raise ValueError("contact_surface_offset_m must be nonnegative")

        if self.command_interval_seconds <= 0.0:
            raise ValueError("command_interval_sec must be greater than 0")

        if self.elbow_configuration not in {"up", "down"}:
            raise ValueError("elbow_solution must be 'up' or 'down'")

        if len(self.home_joint_positions_rad) != 4:
            raise ValueError(
                "home_joint_positions must contain exactly four angles"
            )

        if not all(
            math.isfinite(angle)
            for angle in self.home_joint_positions_rad
        ):
            raise ValueError(
                "home_joint_positions must contain only finite values"
            )

    # Temporary compatibility aliases for code written before the controller
    # target frame was made explicit. These values are not base_link-relative.
    @property
    def camera_base_x_m(self) -> float:
        return self.camera_joint_2_x_m

    @property
    def camera_base_y_m(self) -> float:
        return self.camera_joint_2_y_m

    @property
    def camera_base_z_m(self) -> float:
        return self.camera_joint_2_z_m