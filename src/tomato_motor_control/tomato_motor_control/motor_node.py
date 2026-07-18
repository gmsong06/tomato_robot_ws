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

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from tomato_motor_control import constants


TICKS_PER_REVOLUTION = 4096


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

        # ---------------------------------------------------------
        # Load motor configuration
        # ---------------------------------------------------------
        with open(self.motor_config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if "motors" not in config:
            raise RuntimeError(
                f"No 'motors' section found in {self.motor_config_path}"
            )

        self.motor_config = config["motors"]
        self.joint_names = list(self.motor_config.keys())

        self.get_logger().info(
            f"Configured joint order: {self.joint_names}"
        )

        # ---------------------------------------------------------
        # Create motor objects and connect to the bus
        # ---------------------------------------------------------
        self.motors = {
            joint_name: Motor(
                motor_info["id"],
                motor_info.get("model", "sts3215"),
                MotorNormMode.RANGE_0_100,
            )
            for joint_name, motor_info in self.motor_config.items()
        }

        self.bus = FeetechMotorsBus(
            port=self.port_name,
            motors=self.motors,
        )

        self.bus.connect(handshake=False)

        self.configure_motors()

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
            f"timeout={self.goal_retry_timeout_sec:.1f}s"
        )

    # =========================================================
    # Motor setup
    # =========================================================

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

            self.bus.enable_torque(joint_name)

            self.get_logger().info(
                f"{joint_name}: POSITION mode, "
                f"Goal_Time={self.goal_time}"
            )

    # =========================================================
    # Unit conversion
    # =========================================================

    def tick_midpoint(self, joint_name: str) -> float:
        motor_info = self.motor_config[joint_name]

        return (
            float(motor_info["range_min"])
            + float(motor_info["range_max"])
        ) / 2.0

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
        motor_info = self.motor_config[joint_name]
        midpoint_ticks = self.tick_midpoint(joint_name)

        target_ticks = int(
            round(
                midpoint_ticks
                + position_radians
                * TICKS_PER_REVOLUTION
                / (2.0 * math.pi)
            )
        )

        minimum_ticks = int(motor_info["range_min"])
        maximum_ticks = int(motor_info["range_max"])

        return max(
            minimum_ticks,
            min(maximum_ticks, target_ticks),
        )

    def goal_tolerance_ticks(self) -> int:
        tolerance_ticks = int(
            round(
                self.goal_tolerance_rad
                * TICKS_PER_REVOLUTION
                / (2.0 * math.pi)
            )
        )

        return max(1, tolerance_ticks)

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

        goal_ticks = {}

        for joint_name, target_radians in zip(
            self.joint_names,
            message.data,
        ):
            goal_ticks[joint_name] = self.radians_to_ticks(
                joint_name,
                float(target_radians),
            )

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
        self.write_goal_ticks(goal_ticks)

    def write_goal_ticks(
        self,
        goal_ticks: dict[str, int],
    ) -> bool:
        try:
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
        self.write_goal_ticks(self.active_goal_ticks)

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

            # Stop retrying an old target whenever torque is disabled
            # for any motor.
            if not all(enabled_values):
                self.clear_active_goal()

            for joint_name, should_enable in zip(
                self.joint_names,
                enabled_values,
            ):
                if should_enable:
                    self.bus.enable_torque(joint_name)
                else:
                    self.bus.disable_torque(joint_name)

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
            response.success = False
            response.message = (
                f"Failed to set torque: {error}"
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
                position_ticks = int(
                    self.bus.read(
                        "Present_Position",
                        joint_name,
                        normalize=False,
                    )
                )

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

        for joint_name in self.joint_names:
            try:
                torque_value = self.bus.read(
                    "Torque_Enable",
                    joint_name,
                    normalize=False,
                )

                message.name.append(joint_name)
                message.enabled.append(
                    bool(torque_value)
                )

            except Exception as error:
                self.get_logger().warning(
                    f"Failed to read torque state for "
                    f"{joint_name}: {error}"
                )

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
    node = FeetechMotorNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()