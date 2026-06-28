
import math
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from tomato_interfaces.srv import SetTorque
from tomato_interfaces.msg import TorqueState

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from tomato_motor_control import constants


TICKS_PER_REV = 4096


class FeetechMotorNode(Node):
    def __init__(self):
        super().__init__("feetech_motor_node")

        self.declare_parameter("port", constants.DEFAULT_PORT)
        self.declare_parameter("goal_time", 100)
        self.declare_parameter(
            "motor_config_path",
            "/home/ann/tomato_robot_ws/src/tomato_motor_control/config/motors.yaml",
        )

        self.port_name = self.get_parameter("port").value
        self.goal_time = int(self.get_parameter("goal_time").value)
        config_path = self.get_parameter("motor_config_path").value

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.motor_config = self.config["motors"]

        self.motors = {
            joint_name: Motor(
                info["id"],
                info.get("model", "sts3215"),
                MotorNormMode.RANGE_0_100,
            )
            for joint_name, info in self.motor_config.items()
        }

        self.bus = FeetechMotorsBus(
            port=self.port_name,
            motors=self.motors,
        )

        self.bus.connect(handshake=False)

        for name in self.motors.keys():
            self.bus.disable_torque(name)

            self.bus.write(
                "Operating_Mode",
                name,
                OperatingMode.POSITION.value,
                normalize=False,
            )

            self.bus.write(
                "Goal_Time",
                name,
                self.goal_time,
                normalize=False,
            )

            self.bus.enable_torque(name)

            self.get_logger().info(
                f"{name}: POSITION mode, Goal_Time={self.goal_time}"
            )

        self.joint_pub = self.create_publisher(
            JointState,
            "/joint_states",
            10,
        )

        self.torque_state_pub = self.create_publisher(
            TorqueState,
            "/torque_states",
            10,
        )

        self.target_sub = self.create_subscription(
            Float64MultiArray,
            "/joint_target_positions",
            self.target_callback,
            10,
        )

        self.torque_srv = self.create_service(
            SetTorque,
            "/set_torque",
            self.set_torque_callback,
        )

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(
            f"Position motor node connected to {len(self.motors)} motors on {self.port_name}"
        )

    def tick_midpoint(self, joint_name):
        info = self.motor_config[joint_name]
        return (info["range_min"] + info["range_max"]) / 2.0

    def ticks_to_rad(self, joint_name, ticks):
        mid = self.tick_midpoint(joint_name)
        return (ticks - mid) * 2.0 * math.pi / TICKS_PER_REV

    def rad_to_ticks(self, joint_name, rad):
        info = self.motor_config[joint_name]
        mid = self.tick_midpoint(joint_name)
        ticks = int(mid + rad * TICKS_PER_REV / (2.0 * math.pi))
        return max(info["range_min"], min(info["range_max"], ticks))

    def target_callback(self, msg: Float64MultiArray):
        names = list(self.motors.keys())

        if len(msg.data) != len(names):
            self.get_logger().warn(
                f"Expected {len(names)} joint targets, got {len(msg.data)}"
            )
            return

        goals = {}

        for name, target_rad in zip(names, msg.data):
            goal_tick = self.rad_to_ticks(name, float(target_rad))
            goals[name] = goal_tick

        summary = " | ".join(
            f"{name}:{goal_tick}" for name, goal_tick in goals.items()
        )

        self.get_logger().info(
            f"Goal ticks [{summary}]",
            throttle_duration_sec=0.25,
        )

        self.bus.sync_write(
            "Goal_Position",
            goals,
            normalize=False,
        )

    def set_torque_callback(self, request, response):
        names = list(self.motors.keys())
        enabled_values = list(request.enabled)

        try:
            if len(enabled_values) == 0:
                response.success = False
                response.message = "Request must contain at least one torque value"
                return response

            if len(enabled_values) == 1:
                enabled_values = enabled_values * len(names)

            elif len(enabled_values) != len(names):
                response.success = False
                response.message = (
                    f"Expected either 1 value or {len(names)} values, "
                    f"got {len(enabled_values)}"
                )
                return response

            for name, enable in zip(names, enabled_values):
                if enable:
                    self.bus.enable_torque(name)
                else:
                    self.bus.disable_torque(name)

            summary = " | ".join(
                f"{name}:{'ON' if enable else 'OFF'}"
                for name, enable in zip(names, enabled_values)
            )

            response.success = True
            response.message = f"Torque set [{summary}]"
            self.get_logger().info(response.message)

            self.publish_torque_state()

        except Exception as e:
            response.success = False
            response.message = f"Failed to set torque: {e}"
            self.get_logger().error(response.message)

        return response

    def read_joint_positions(self):
        names = []
        positions = []

        for name in self.motors.keys():
            try:
                pos_tick = self.bus.read(
                    "Present_Position",
                    name,
                    normalize=False,
                )
            except Exception as e:
                self.get_logger().warn(f"Failed to read {name}: {e}")
                continue

            names.append(name)
            positions.append(self.ticks_to_rad(name, int(pos_tick)))

        return names, positions

    def publish_joint_state(self):
        names, positions = self.read_joint_positions()

        if not names:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = positions

        self.joint_pub.publish(msg)

    def publish_torque_state(self):
        msg = TorqueState()
        msg.header.stamp = self.get_clock().now().to_msg()

        for name in self.motors.keys():
            try:
                torque_value = self.bus.read(
                    "Torque_Enable",
                    name,
                    normalize=False,
                )

                msg.name.append(name)
                msg.enabled.append(bool(torque_value))

            except Exception as e:
                self.get_logger().warn(
                    f"Failed to read torque state for {name}: {e}"
                )

        self.torque_state_pub.publish(msg)

    def timer_callback(self):
        self.publish_joint_state()
        self.publish_torque_state()

    def destroy_node(self):
        try:
            self.bus.disable_torque()
            self.publish_torque_state()
            self.bus.disconnect(disable_torque=False)
        except Exception:
            pass

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
        rclpy.shutdown()


if __name__ == "__main__":
    main()