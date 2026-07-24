from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence

from std_msgs.msg import Float64MultiArray, MultiArrayDimension

from tomato_control.controller_config import ControllerConfig
from tomato_control.controller_models import TomatoCandidate, WaypointCommand


class MotionExecutor:
    """Publish queued joint commands at a fixed interval."""

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
        self.active_sequence_name: str | None = None
        self._on_sequence_complete: Callable[[], None] | None = None

    def start(self, candidate: TomatoCandidate) -> bool:
        """Backward-compatible helper that executes every candidate command."""

        return self.start_commands(
            candidate,
            candidate.waypoint_commands,
            sequence_name="full tomato trajectory",
        )

    def start_commands(
        self,
        candidate: TomatoCandidate,
        commands: Sequence[WaypointCommand],
        *,
        sequence_name: str,
        on_complete: Callable[[], None] | None = None,
    ) -> bool:
        """Start a specific subset of a candidate's solved commands."""

        if self.is_in_progress:
            self.logger.info(
                "Motion already in progress; not starting another sequence"
            )
            return False

        command_list = list(commands)
        if not command_list:
            self.logger.warn(
                f"Cannot start {sequence_name}: command list is empty"
            )
            return False

        self.active_candidate = candidate
        self.active_sequence_name = sequence_name
        self._on_sequence_complete = on_complete

        if not self.config.motor_commands_enabled:
            self.logger.warn(
                "Motor commands are disabled. Running this sequence as a "
                "dry run only."
            )
            self._log_dry_run_commands(command_list)
            callback = self._on_sequence_complete
            self.reset()
            if callback is not None:
                callback()
            return True

        self.queued_commands = deque(command_list)
        self.is_in_progress = True

        detection_id = candidate.detection.detection_id
        self.logger.info(
            f"Starting {sequence_name} for detection id={detection_id}"
        )

        # Send the first command immediately. The ROS timer sends each
        # remaining command after command_interval_sec.
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

        completed_sequence_name = self.active_sequence_name
        completion_callback = self._on_sequence_complete
        self.reset()

        self.logger.info(
            f"Completed {completed_sequence_name or 'motion sequence'} "
            f"for detection id={completed_detection_id}"
        )

        if completion_callback is not None:
            completion_callback()

    def reset(self) -> None:
        self.queued_commands.clear()
        self.is_in_progress = False
        self.active_candidate = None
        self.active_sequence_name = None
        self._on_sequence_complete = None

    def _log_dry_run_commands(
        self,
        commands: Sequence[WaypointCommand],
    ) -> None:
        for command in commands:
            motor_angles = self.convert_ros_angles_to_motor_angles(
                command.joint_angles
            )

            if command.waypoint is None:
                target_description = "joint-space target"
            else:
                position = command.waypoint.position_base
                target_description = (
                    "target_base=("
                    f"x={position.x_m:.3f}, "
                    f"y={position.y_m:.3f}, "
                    f"z={position.z_m:.3f})"
                )

            self.logger.info(
                f"DRY RUN {command.name} {target_description}, "
                f"ros_joints={command.joint_angles}, "
                f"motor_joints={motor_angles}"
            )