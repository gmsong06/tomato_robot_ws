import math
from typing import Optional

import rclpy
import yaml
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from tomato_interfaces.msg import TorqueState
from tomato_interfaces.srv import SetTorque

from lerobot.motors import (
    Motor,
    MotorCalibration,
    MotorNormMode,
)
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from tomato_motor_control import constants


TICKS_PER_REVOLUTION = 4096
ZERO_POSITION_TICKS = TICKS_PER_REVOLUTION // 2
ENCODER_MIN_TICKS = 0
ENCODER_MAX_TICKS = TICKS_PER_REVOLUTION - 1


class FeetechMotorNode(Node):
    def __init__(self):
        super().__init__("feetech_motor_node")

        # ---------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------
        self.declare_parameter("port", constants.DEFAULT_PORT)
        self.declare_parameter("goal_time", 100)
        self.declare_parameter(
            "motor_config_path",
            "/home/ann/tomato_robot_ws/src/"
            "tomato_motor_control/config/motors.yaml",
        )

        # How often to resend the latest target until the encoders
        # confirm that all joints reached it.
        self.declare_parameter("goal_retry_period_sec", 0.25)

        # Maximum allowed joint error before a target is considered reached.
        self.declare_parameter("goal_tolerance_rad", 0.03)

        # Stop retrying after this amount of time.
        self.declare_parameter("goal_retry_timeout_sec", 10.0)

        # Reject only exceptionally large one-command movements. The former
        # 0.80 rad limit blocked valid shoulder/elbow moves between home and a
        # solved waypoint; 1.75 rad permits up to about 100 degrees while the
        # calibrated per-joint command ranges remain authoritative.
        self.declare_parameter("max_goal_delta_rad", 1.75)

        # Keep commanded positions away from recorded calibration endpoints.
        self.declare_parameter("command_limit_margin_ticks", 10)

        # A valid single-turn calibration must leave the encoder seam outside
        # the recorded range. The old joint_1 range [1, 4095] fails this check.
        self.declare_parameter("encoder_seam_margin_ticks", 64)

        # Joint 1 is intentionally restricted to +/-85 degrees. This fits
        # inside the new calibrated range while leaving a small safety margin.
        self.declare_parameter(
            "joint_1_soft_limit_rad",
            math.radians(85.0),
        )

        self.port_name = str(
            self.get_parameter("port").value
        )
        self.goal_time = int(
            self.get_parameter("goal_time").value
        )
        self.motor_config_path = str(
            self.get_parameter("motor_config_path").value
        )

        self.goal_retry_period_sec = float(
            self.get_parameter("goal_retry_period_sec").value
        )
        self.goal_tolerance_rad = float(
            self.get_parameter("goal_tolerance_rad").value
        )
        self.goal_retry_timeout_sec = float(
            self.get_parameter("goal_retry_timeout_sec").value
        )
        self.max_goal_delta_rad = float(
            self.get_parameter("max_goal_delta_rad").value
        )
        self.command_limit_margin_ticks = int(
            self.get_parameter("command_limit_margin_ticks").value
        )
        self.encoder_seam_margin_ticks = int(
            self.get_parameter("encoder_seam_margin_ticks").value
        )
        self.joint_1_soft_limit_rad = float(
            self.get_parameter("joint_1_soft_limit_rad").value
        )

        if self.goal_retry_period_sec <= 0.0:
            raise ValueError(
                "goal_retry_period_sec must be greater than zero"
            )

        if self.goal_tolerance_rad <= 0.0:
            raise ValueError(
                "goal_tolerance_rad must be greater than zero"
            )

        if self.goal_retry_timeout_sec <= 0.0:
            raise ValueError(
                "goal_retry_timeout_sec must be greater than zero"
            )

        if self.max_goal_delta_rad <= 0.0:
            raise ValueError(
                "max_goal_delta_rad must be greater than zero"
            )

        if self.command_limit_margin_ticks < 0:
            raise ValueError(
                "command_limit_margin_ticks cannot be negative"
            )

        if not 0 <= self.encoder_seam_margin_ticks < ZERO_POSITION_TICKS:
            raise ValueError(
                "encoder_seam_margin_ticks must be in [0, 2048)"
            )

        if not 0.0 < self.joint_1_soft_limit_rad <= math.pi:
            raise ValueError(
                "joint_1_soft_limit_rad must be in (0, pi]"
            )

        # ---------------------------------------------------------
        # Load motor configuration
        # ---------------------------------------------------------
        with open(self.motor_config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict) or "motors" not in config:
            raise RuntimeError(
                f"No 'motors' section found in {self.motor_config_path}"
            )

        self.motor_config = config["motors"]

        if not isinstance(self.motor_config, dict) or not self.motor_config:
            raise RuntimeError(
                f"The 'motors' section in {self.motor_config_path} "
                "must be a non-empty mapping"
            )

        self.joint_names = list(self.motor_config.keys())
        self.validate_motor_config()

        self.calibration = {
            joint_name: MotorCalibration(
                id=int(motor_info["id"]),
                drive_mode=0,
                homing_offset=int(motor_info["homing_offset"]),
                range_min=int(motor_info["range_min"]),
                range_max=int(motor_info["range_max"]),
            )
            for joint_name, motor_info in self.motor_config.items()
        }

        self.get_logger().info(
            f"Configured joint order: {self.joint_names}"
        )

        # ---------------------------------------------------------
        # Create motor objects and connect to the bus
        # ---------------------------------------------------------
        self.motors = {
            joint_name: Motor(
                int(motor_info["id"]),
                motor_info.get("model", "sts3215"),
                MotorNormMode.RANGE_0_100,
            )
            for joint_name, motor_info in self.motor_config.items()
        }

        self.bus = FeetechMotorsBus(
            port=self.port_name,
            motors=self.motors,
            calibration=self.calibration,
        )

        self.bus.connect(handshake=False)

        try:
            self.bus.disable_torque()

            if not self.bus.is_calibrated:
                self.get_logger().warning(
                    "Motor calibration does not match YAML; "
                    "writing calibration with torque disabled"
                )
                self.bus.write_calibration(self.calibration)

            self.configure_motors()

        except Exception:
            try:
                self.bus.disable_torque()
                self.bus.disconnect(disable_torque=False)
            except Exception as cleanup_error:
                self.get_logger().error(
                    f"Motor setup cleanup failed: {cleanup_error}"
                )
            raise

        # ---------------------------------------------------------
        # Active target state
        # ---------------------------------------------------------
        self.active_goal_ticks: Optional[dict[str, int]] = None
        self.active_goal_started_at: Optional[Time] = None

        # Updated whenever Present_Position is read.
        self.latest_position_ticks: dict[str, int] = {}

        # ---------------------------------------------------------
        # ROS publishers
        # ---------------------------------------------------------
        self.joint_state_publisher = self.create_publisher(
            JointState,
            "/joint_states",
            10,
        )

        self.torque_state_publisher = self.create_publisher(
            TorqueState,
            "/torque_states",
            10,
        )

        # ---------------------------------------------------------
        # ROS subscriptions
        # ---------------------------------------------------------
        self.target_subscription = self.create_subscription(
            Float64MultiArray,
            "/joint_target_positions",
            self.target_callback,
            10,
        )

        # ---------------------------------------------------------
        # ROS services
        # ---------------------------------------------------------
        self.torque_service = self.create_service(
            SetTorque,
            "/set_torque",
            self.set_torque_callback,
        )

        # ---------------------------------------------------------
        # Timers
        # ---------------------------------------------------------

        # Publish encoder positions at 10 Hz.
        self.joint_state_timer = self.create_timer(
            0.1,
            self.publish_joint_state,
        )

        # Retry the active target until it has been reached.
        self.goal_retry_timer = self.create_timer(
            self.goal_retry_period_sec,
            self.retry_active_goal,
        )

        # Torque state does not need to be read at 10 Hz.
        self.torque_state_timer = self.create_timer(
            1.0,
            self.publish_torque_state,
        )

        self.get_logger().info(
            f"Position motor node connected to "
            f"{len(self.motors)} motors on {self.port_name}"
        )

        self.get_logger().info(
            "Goal retry configuration: "
            f"period={self.goal_retry_period_sec:.2f}s, "
            f"tolerance={self.goal_tolerance_rad:.3f}rad, "
            f"timeout={self.goal_retry_timeout_sec:.1f}s, "
            f"maximum initial delta={self.max_goal_delta_rad:.3f}rad"
        )

        self.get_logger().warning(
            "Motor torque is OFF at startup. Use /set_torque only after "
            "checking /joint_states."
        )

    # =========================================================
    # Motor setup
    # =========================================================

    def validate_motor_config(self):
        required_fields = {
            "id",
            "homing_offset",
            "range_min",
            "range_max",
        }
        seen_motor_ids = set()

        for joint_name, motor_info in self.motor_config.items():
            if not isinstance(motor_info, dict):
                raise RuntimeError(
                    f"Configuration for {joint_name} must be a mapping"
                )

            missing_fields = required_fields.difference(motor_info)
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise RuntimeError(
                    f"Configuration for {joint_name} is missing: {missing}"
                )

            motor_id = int(motor_info["id"])
            homing_offset = int(motor_info["homing_offset"])
            minimum_ticks = int(motor_info["range_min"])
            maximum_ticks = int(motor_info["range_max"])

            if motor_id in seen_motor_ids:
                raise RuntimeError(
                    f"Motor ID {motor_id} is assigned more than once"
                )
            seen_motor_ids.add(motor_id)

            if not -2047 <= homing_offset <= 2047:
                raise RuntimeError(
                    f"{joint_name} homing_offset={homing_offset} is outside "
                    "the signed 12-bit Feetech range [-2047, 2047]"
                )

            if not (
                ENCODER_MIN_TICKS
                <= minimum_ticks
                < ZERO_POSITION_TICKS
                < maximum_ticks
                <= ENCODER_MAX_TICKS
            ):
                raise RuntimeError(
                    f"{joint_name} calibration [{minimum_ticks}, "
                    f"{maximum_ticks}] must contain zero tick "
                    f"{ZERO_POSITION_TICKS} and remain inside "
                    f"[{ENCODER_MIN_TICKS}, {ENCODER_MAX_TICKS}]"
                )

            if (
                minimum_ticks < self.encoder_seam_margin_ticks
                or maximum_ticks
                > ENCODER_MAX_TICKS - self.encoder_seam_margin_ticks
            ):
                raise RuntimeError(
                    f"{joint_name} calibration [{minimum_ticks}, "
                    f"{maximum_ticks}] is too close to the 0/4095 encoder "
                    "seam. Recalibrate before enabling torque."
                )

            if maximum_ticks - minimum_ticks <= (
                2 * self.command_limit_margin_ticks
            ):
                raise RuntimeError(
                    f"{joint_name} calibration range is too small for "
                    f"command_limit_margin_ticks="
                    f"{self.command_limit_margin_ticks}"
                )

            minimum_rad, maximum_rad = self.command_limits_radians(
                joint_name
            )
            self.get_logger().info(
                f"{joint_name}: calibration=[{minimum_ticks}, "
                f"{maximum_ticks}] ticks, command range="
                f"[{minimum_rad:.3f}, {maximum_rad:.3f}] rad "
                f"([{math.degrees(minimum_rad):.1f}, "
                f"{math.degrees(maximum_rad):.1f}] deg)"
            )

    def configure_motors(self):
        for joint_name in self.joint_names:
            self.bus.disable_torque(joint_name)

            self.bus.write(
                "Operating_Mode",
                joint_name,
                OperatingMode.POSITION.value,
                normalize=False,
            )

            self.bus.write(
                "Goal_Time",
                joint_name,
                self.goal_time,
                normalize=False,
            )

            self.get_logger().info(
                f"{joint_name}: POSITION mode, "
                f"Goal_Time={self.goal_time}, torque=OFF"
            )

    # =========================================================
    # Unit conversion
    # =========================================================

    def tick_midpoint(self, joint_name: str) -> float:
        # Feetech/LeRobot homing calibration maps the chosen zero pose
        # to the half-turn value. range_min/range_max are limits, not zero.
        del joint_name
        return float(ZERO_POSITION_TICKS)

    def command_tick_limits(
        self,
        joint_name: str,
    ) -> tuple[int, int]:
        motor_info = self.motor_config[joint_name]
        minimum_ticks = (
            int(motor_info["range_min"])
            + self.command_limit_margin_ticks
        )
        maximum_ticks = (
            int(motor_info["range_max"])
            - self.command_limit_margin_ticks
        )

        if joint_name == "joint_1":
            soft_limit_ticks = int(
                math.floor(
                    self.joint_1_soft_limit_rad
                    * TICKS_PER_REVOLUTION
                    / (2.0 * math.pi)
                )
            )
            minimum_ticks = max(
                minimum_ticks,
                ZERO_POSITION_TICKS - soft_limit_ticks,
            )
            maximum_ticks = min(
                maximum_ticks,
                ZERO_POSITION_TICKS + soft_limit_ticks,
            )

        if minimum_ticks >= maximum_ticks:
            raise RuntimeError(
                f"No usable command range remains for {joint_name}"
            )

        return minimum_ticks, maximum_ticks

    def command_limits_radians(
        self,
        joint_name: str,
    ) -> tuple[float, float]:
        minimum_ticks, maximum_ticks = self.command_tick_limits(joint_name)
        return (
            self.ticks_to_radians(joint_name, minimum_ticks),
            self.ticks_to_radians(joint_name, maximum_ticks),
        )

    def ticks_to_radians(
        self,
        joint_name: str,
        position_ticks: int,
    ) -> float:
        midpoint_ticks = self.tick_midpoint(joint_name)

        return (
            float(position_ticks) - midpoint_ticks
        ) * 2.0 * math.pi / TICKS_PER_REVOLUTION

    def radians_to_ticks(
        self,
        joint_name: str,
        position_radians: float,
    ) -> int:
        if not math.isfinite(position_radians):
            raise ValueError(
                f"{joint_name} target must be a finite number"
            )

        midpoint_ticks = self.tick_midpoint(joint_name)

        target_ticks = int(
            round(
                midpoint_ticks
                + position_radians
                * TICKS_PER_REVOLUTION
                / (2.0 * math.pi)
            )
        )

        minimum_ticks, maximum_ticks = self.command_tick_limits(joint_name)

        if not minimum_ticks <= target_ticks <= maximum_ticks:
            minimum_rad, maximum_rad = self.command_limits_radians(joint_name)
            raise ValueError(
                f"{joint_name} target {position_radians:.4f}rad is outside "
                f"the safe range [{minimum_rad:.4f}, "
                f"{maximum_rad:.4f}]rad"
            )

        return target_ticks

    def goal_tolerance_ticks(self) -> int:
        tolerance_ticks = int(
            round(
                self.goal_tolerance_rad
                * TICKS_PER_REVOLUTION
                / (2.0 * math.pi)
            )
        )

        return max(1, tolerance_ticks)

    def maximum_goal_delta_ticks(self) -> int:
        return max(
            1,
            int(
                math.floor(
                    self.max_goal_delta_rad
                    * TICKS_PER_REVOLUTION
                    / (2.0 * math.pi)
                )
            ),
        )

    # =========================================================
    # Goal handling
    # =========================================================

    def target_callback(
        self,
        message: Float64MultiArray,
    ):
        if len(message.data) != len(self.joint_names):
            self.get_logger().warning(
                f"Expected {len(self.joint_names)} joint targets, "
                f"received {len(message.data)}"
            )
            return

        torque_states = self.read_torque_states()
        if torque_states is None:
            self.get_logger().error(
                "Rejected joint target because torque state could not be read"
            )
            return

        disabled_joints = [
            joint_name
            for joint_name, enabled in torque_states.items()
            if not enabled
        ]
        if disabled_joints:
            self.get_logger().error(
                "Rejected joint target because torque is disabled for: "
                + ", ".join(disabled_joints)
            )
            return

        # Read fresh positions before validating the requested movement.
        self.read_joint_positions()
        missing_feedback = [
            joint_name
            for joint_name in self.joint_names
            if joint_name not in self.latest_position_ticks
        ]
        if missing_feedback:
            self.get_logger().error(
                "Rejected joint target because position feedback is missing "
                "for: " + ", ".join(missing_feedback)
            )
            return

        goal_ticks: dict[str, int] = {}

        try:
            for joint_name, target_radians in zip(
                self.joint_names,
                message.data,
            ):
                goal_ticks[joint_name] = self.radians_to_ticks(
                    joint_name,
                    float(target_radians),
                )

            self.validate_goal_delta(goal_ticks)

        except (TypeError, ValueError) as error:
            self.get_logger().error(
                f"Rejected unsafe joint target: {error}"
            )
            return

        # A new command replaces the previous active command.
        self.active_goal_ticks = goal_ticks
        self.active_goal_started_at = self.get_clock().now()

        target_summary = " | ".join(
            f"{joint_name}:"
            f"{float(target_radians):.4f}rad"
            f"->{goal_ticks[joint_name]}ticks"
            for joint_name, target_radians in zip(
                self.joint_names,
                message.data,
            )
        )

        self.get_logger().info(
            f"Received new joint target [{target_summary}]"
        )

        # Send immediately. The retry timer will send it again if the
        # encoders do not confirm that it was reached.
        if not self.write_goal_ticks(goal_ticks):
            self.clear_active_goal()

    def validate_goal_delta(
        self,
        goal_ticks: dict[str, int],
    ):
        maximum_delta_ticks = self.maximum_goal_delta_ticks()

        for joint_name, target_ticks in goal_ticks.items():
            current_ticks = self.latest_position_ticks.get(joint_name)
            if current_ticks is None:
                raise ValueError(
                    f"No position feedback for {joint_name}"
                )

            calibration_minimum, calibration_maximum = (
                self.calibration_tick_limits(joint_name)
            )
            if not (
                calibration_minimum
                <= current_ticks
                <= calibration_maximum
            ):
                raise ValueError(
                    f"{joint_name} present position {current_ticks} is "
                    f"outside calibration range [{calibration_minimum}, "
                    f"{calibration_maximum}]"
                )

            delta_ticks = abs(target_ticks - current_ticks)
            if delta_ticks > maximum_delta_ticks:
                delta_radians = (
                    delta_ticks
                    * 2.0
                    * math.pi
                    / TICKS_PER_REVOLUTION
                )
                raise ValueError(
                    f"{joint_name} requested movement "
                    f"{delta_radians:.4f}rad exceeds max_goal_delta_rad="
                    f"{self.max_goal_delta_rad:.4f}"
                )

    def validate_goal_ticks(
        self,
        goal_ticks: dict[str, int],
    ):
        if set(goal_ticks) != set(self.joint_names):
            raise ValueError(
                "Goal must contain exactly: "
                + ", ".join(self.joint_names)
            )

        for joint_name, target_ticks in goal_ticks.items():
            minimum_ticks, maximum_ticks = self.command_tick_limits(
                joint_name
            )
            if not minimum_ticks <= int(target_ticks) <= maximum_ticks:
                raise ValueError(
                    f"{joint_name} goal {target_ticks} is outside safe tick "
                    f"range [{minimum_ticks}, {maximum_ticks}]"
                )

    def write_goal_ticks(
        self,
        goal_ticks: dict[str, int],
    ) -> bool:
        try:
            self.validate_goal_ticks(goal_ticks)

            torque_states = self.read_torque_states()
            if torque_states is None or not all(torque_states.values()):
                self.get_logger().error(
                    "Refused to write Goal_Position while any motor torque "
                    "is disabled"
                )
                return False

            self.bus.sync_write(
                "Goal_Position",
                goal_ticks,
                normalize=False,
            )

            goal_summary = " | ".join(
                f"{joint_name}:{target_ticks}"
                for joint_name, target_ticks in goal_ticks.items()
            )

            self.get_logger().info(
                f"Wrote goal ticks [{goal_summary}]"
            )

            return True

        except (TypeError, ValueError) as error:
            self.get_logger().error(
                f"Refused unsafe motor target: {error}"
            )
            return False

        except Exception as error:
            self.get_logger().error(
                f"Failed to write motor targets: {error}"
            )
            return False

    def active_goal_has_been_reached(self) -> bool:
        if self.active_goal_ticks is None:
            return True

        tolerance_ticks = self.goal_tolerance_ticks()

        for joint_name, target_ticks in self.active_goal_ticks.items():
            current_ticks = self.latest_position_ticks.get(joint_name)

            if current_ticks is None:
                return False

            joint_error_ticks = abs(
                current_ticks - target_ticks
            )

            if joint_error_ticks > tolerance_ticks:
                return False

        return True

    def retry_active_goal(self):
        if self.active_goal_ticks is None:
            return

        if self.active_goal_has_been_reached():
            final_summary = " | ".join(
                (
                    f"{joint_name}:"
                    f"{self.latest_position_ticks[joint_name]}"
                    f"/{target_ticks}"
                )
                for joint_name, target_ticks
                in self.active_goal_ticks.items()
            )

            self.get_logger().info(
                f"All motors reached active target "
                f"[{final_summary}]"
            )

            self.clear_active_goal()
            return

        if self.active_goal_started_at is not None:
            elapsed_seconds = (
                self.get_clock().now()
                - self.active_goal_started_at
            ).nanoseconds / 1e9

            if elapsed_seconds >= self.goal_retry_timeout_sec:
                error_summary = self.active_goal_error_summary()

                self.get_logger().error(
                    f"Motor target timed out after "
                    f"{elapsed_seconds:.1f} seconds. "
                    f"Errors: [{error_summary}]"
                )

                self.clear_active_goal()
                return

        # The first command may occasionally not be acted on by every
        # motor. Resend the complete target until feedback confirms it.
        if not self.write_goal_ticks(self.active_goal_ticks):
            self.get_logger().error(
                "Stopped retrying because the active goal is no longer safe"
            )
            self.clear_active_goal()

    def active_goal_error_summary(self) -> str:
        if self.active_goal_ticks is None:
            return "no active goal"

        summaries = []

        for joint_name, target_ticks in self.active_goal_ticks.items():
            current_ticks = self.latest_position_ticks.get(joint_name)

            if current_ticks is None:
                summaries.append(
                    f"{joint_name}:no feedback"
                )
                continue

            error_ticks = target_ticks - current_ticks
            error_radians = (
                error_ticks
                * 2.0
                * math.pi
                / TICKS_PER_REVOLUTION
            )

            summaries.append(
                f"{joint_name}:"
                f"current={current_ticks},"
                f"target={target_ticks},"
                f"error={error_radians:.3f}rad"
            )

        return " | ".join(summaries)

    def clear_active_goal(self):
        self.active_goal_ticks = None
        self.active_goal_started_at = None

    # =========================================================
    # Torque service
    # =========================================================

    def read_torque_states(self) -> Optional[dict[str, bool]]:
        states: dict[str, bool] = {}

        for joint_name in self.joint_names:
            try:
                torque_value = self.bus.read(
                    "Torque_Enable",
                    joint_name,
                    normalize=False,
                )
                states[joint_name] = bool(torque_value)

            except Exception as error:
                self.get_logger().error(
                    f"Failed to read torque state for "
                    f"{joint_name}: {error}"
                )
                return None

        return states

    def read_position_tick(
        self,
        joint_name: str,
    ) -> int:
        return int(
            self.bus.read(
                "Present_Position",
                joint_name,
                normalize=False,
            )
        )

    def calibration_tick_limits(
        self,
        joint_name: str,
    ) -> tuple[int, int]:
        motor_info = self.motor_config[joint_name]
        return (
            int(motor_info["range_min"]),
            int(motor_info["range_max"]),
        )

    def prepare_motors_for_torque_enable(
        self,
        joints_to_enable: list[str],
    ):
        if not joints_to_enable:
            return

        hold_positions: dict[str, int] = {}

        for joint_name in joints_to_enable:
            current_ticks = self.read_position_tick(joint_name)
            minimum_ticks, maximum_ticks = self.calibration_tick_limits(
                joint_name
            )

            if not minimum_ticks <= current_ticks <= maximum_ticks:
                raise RuntimeError(
                    f"Refusing to enable {joint_name}: present position "
                    f"{current_ticks} is outside calibration range "
                    f"[{minimum_ticks}, {maximum_ticks}]"
                )

            self.latest_position_ticks[joint_name] = current_ticks
            hold_positions[joint_name] = current_ticks

        # Replace any stale Goal_Position with the measured position before
        # torque is enabled. Every value was checked against calibration.
        for joint_name, current_ticks in hold_positions.items():
            self.bus.write(
                "Goal_Position",
                joint_name,
                current_ticks,
                normalize=False,
            )

        for joint_name in joints_to_enable:
            self.bus.enable_torque(joint_name)

    def set_torque_callback(
        self,
        request,
        response,
    ):
        enabled_values = list(request.enabled)

        try:
            if len(enabled_values) == 0:
                response.success = False
                response.message = (
                    "Request must contain at least one torque value"
                )
                return response

            if len(enabled_values) == 1:
                enabled_values = (
                    enabled_values * len(self.joint_names)
                )

            elif len(enabled_values) != len(self.joint_names):
                response.success = False
                response.message = (
                    f"Expected either 1 value or "
                    f"{len(self.joint_names)} values, "
                    f"received {len(enabled_values)}"
                )
                return response

            # Any torque change invalidates an old motion target.
            self.clear_active_goal()

            joints_to_disable = [
                joint_name
                for joint_name, should_enable in zip(
                    self.joint_names,
                    enabled_values,
                )
                if not should_enable
            ]
            joints_to_enable = [
                joint_name
                for joint_name, should_enable in zip(
                    self.joint_names,
                    enabled_values,
                )
                if should_enable
            ]

            # Disable first, then replace stale goals before enabling.
            for joint_name in joints_to_disable:
                self.bus.disable_torque(joint_name)

            self.prepare_motors_for_torque_enable(joints_to_enable)

            torque_summary = " | ".join(
                (
                    f"{joint_name}:"
                    f"{'ON' if should_enable else 'OFF'}"
                )
                for joint_name, should_enable in zip(
                    self.joint_names,
                    enabled_values,
                )
            )

            response.success = True
            response.message = (
                f"Torque set [{torque_summary}]"
            )

            self.get_logger().info(response.message)
            self.publish_torque_state()

        except Exception as error:
            self.clear_active_goal()
            try:
                self.bus.disable_torque()
            except Exception as disable_error:
                self.get_logger().error(
                    f"Emergency torque disable also failed: {disable_error}"
                )

            response.success = False
            response.message = (
                f"Failed to set torque; emergency torque disable was "
                f"requested: {error}"
            )
            self.get_logger().error(response.message)

        return response

    # =========================================================
    # Joint position feedback
    # =========================================================

    def read_joint_positions(
        self,
    ) -> tuple[list[str], list[float]]:
        received_joint_names = []
        joint_positions_radians = []

        for joint_name in self.joint_names:
            try:
                position_ticks = self.read_position_tick(joint_name)

            except Exception as error:
                self.get_logger().warning(
                    f"Failed to read {joint_name}: {error}"
                )
                continue

            self.latest_position_ticks[joint_name] = (
                position_ticks
            )

            received_joint_names.append(joint_name)
            joint_positions_radians.append(
                self.ticks_to_radians(
                    joint_name,
                    position_ticks,
                )
            )

        return (
            received_joint_names,
            joint_positions_radians,
        )

    def publish_joint_state(self):
        joint_names, joint_positions = (
            self.read_joint_positions()
        )

        if not joint_names:
            return

        message = JointState()
        message.header.stamp = (
            self.get_clock().now().to_msg()
        )
        message.name = joint_names
        message.position = joint_positions

        self.joint_state_publisher.publish(message)

    # =========================================================
    # Torque feedback
    # =========================================================

    def publish_torque_state(self):
        message = TorqueState()
        message.header.stamp = (
            self.get_clock().now().to_msg()
        )

        torque_states = self.read_torque_states()
        if torque_states is None:
            return

        for joint_name, enabled in torque_states.items():
            message.name.append(joint_name)
            message.enabled.append(enabled)

        self.torque_state_publisher.publish(message)

    # =========================================================
    # Shutdown
    # =========================================================

    def destroy_node(self):
        try:
            self.clear_active_goal()

            self.bus.disable_torque()
            self.publish_torque_state()

            self.bus.disconnect(
                disable_torque=False,
            )

        except Exception as error:
            self.get_logger().warning(
                f"Motor shutdown encountered an error: {error}"
            )

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node: Optional[FeetechMotorNode] = None

    try:
        node = FeetechMotorNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()