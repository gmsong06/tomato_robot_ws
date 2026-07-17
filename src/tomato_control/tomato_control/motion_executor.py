from __future__ import annotations

from collections import deque

from std_msgs.msg import Float64MultiArray, MultiArrayDimension

from tomato_control.controller_config import ControllerConfig
from tomato_control.controller_models import TomatoCandidate, WaypointCommand


class MotionExecutor:
    """Translate ROS joint solutions into physical motor commands and queue them."""

    JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4")

    def __init__(self, node, config: ControllerConfig):
        self.node = node
        self.config = config
        self.logger = node.get_logger()

        self.publisher = node.create_publisher(
            Float64MultiArray,
            config.joint_command_topic,
            10,
        )

        self.queued_commands: deque[WaypointCommand] = deque()
        self.is_in_progress = False
        self.active_candidate: TomatoCandidate | None = None

    def start(self, candidate: TomatoCandidate) -> bool:
        if self.is_in_progress:
            self.logger.info(
                "Motion already in progress; not starting another approach"
            )
            return False

        self.active_candidate = candidate

        if not self.config.motor_commands_enabled:
            self.logger.warn(
                "Motor commands are disabled. Set "
                "enable_motor_commands:=true to publish to the motor node."
            )
            self._log_dry_run(candidate)
            self.reset()
            self.logger.info(
                "Dry run complete. Controller is ready for another tomato."
            )
            return True

        self.queued_commands = deque(candidate.waypoint_commands)
        self.is_in_progress = True

        detection = candidate.detection
        self.logger.info(
            "Starting horizontal approach for "
            f"id={detection.detection_id}, "
            f"ripeness={detection.final_ripeness}"
        )

        self.publish_next_command()
        return True

    def publish_next_command(self) -> None:
        if not self.is_in_progress:
            return

        if not self.queued_commands:
            self.finish()
            return

        self.publish_command(self.queued_commands.popleft())

    def publish_command(self, command: WaypointCommand) -> None:
        motor_joint_angles = self.convert_ros_angles_to_motor_angles(
            command.joint_angles
        )

        message = Float64MultiArray()
        joint_dimension = MultiArrayDimension()
        joint_dimension.label = "joints"
        joint_dimension.size = len(self.JOINT_NAMES)
        joint_dimension.stride = len(self.JOINT_NAMES)
        message.layout.dim.append(joint_dimension)
        message.data = [
            motor_joint_angles[joint_name]
            for joint_name in self.JOINT_NAMES
        ]

        self.publisher.publish(message)
        self.logger.info(
            f"Published {command.name} motor command: "
            f"{motor_joint_angles}"
        )

    def convert_ros_angles_to_motor_angles(
        self,
        ros_joint_angles: dict[str, float],
    ) -> dict[str, float]:
        motor_joint_angles: dict[str, float] = {}

        for joint_name in self.JOINT_NAMES:
            motor_angle_rad = float(ros_joint_angles[joint_name])

            if (
                joint_name == "joint_1"
                and self.config.invert_base_yaw_motor_command
            ):
                motor_angle_rad = -motor_angle_rad

            motor_joint_angles[joint_name] = motor_angle_rad

        return motor_joint_angles

    def finish(self) -> None:
        completed_detection_id = None

        if self.active_candidate is not None:
            completed_detection_id = (
                self.active_candidate.detection.detection_id
            )

        self.reset()

        if completed_detection_id is None:
            self.logger.info("Motion sequence complete")
        else:
            self.logger.info(
                "Motion sequence complete for "
                f"detection id={completed_detection_id}"
            )

        self.logger.info(
            "Controller is ready for another tomato approach and approval."
        )

    def reset(self) -> None:
        self.queued_commands.clear()
        self.is_in_progress = False
        self.active_candidate = None

    def _log_dry_run(self, candidate: TomatoCandidate) -> None:
        for command in candidate.waypoint_commands:
            position = command.waypoint.position_base
            motor_angles = self.convert_ros_angles_to_motor_angles(
                command.joint_angles
            )

            self.logger.info(
                f"DRY RUN {command.name} target_base=("
                f"x={position.x_m:.3f}, "
                f"y={position.y_m:.3f}, "
                f"z={position.z_m:.3f}), "
                f"ros_joints={command.joint_angles}, "
                f"motor_joints={motor_angles}"
            )
