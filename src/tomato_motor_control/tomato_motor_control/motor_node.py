
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

from tomato_motor_control import constants


class FeetechMotorNode(Node):
    def __init__(self):
        super().__init__("feetech_motor_node")

        self.declare_parameter("port", constants.DEFAULT_PORT)
        self.declare_parameter(
            "motor_config_path",
            "/home/ann/tomato_robot_ws/src/tomato_motor_control/config/motors.yaml",
        )

        config_path = self.get_parameter("motor_config_path").value

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.motors = {
            joint_name: Motor(
                info["id"],
                info.get("model", "sts3215"),
                MotorNormMode.RANGE_0_100,
            )
            for joint_name, info in config["motors"].items()
        }

        self.port_name = self.get_parameter("port").value

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
                OperatingMode.VELOCITY.value,
                normalize=False,
            )
            mode = self.bus.read("Operating_Mode", name, normalize=False)
            self.get_logger().info(f"{name} operating mode: {mode}")
            self.bus.enable_torque(name)

        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)

        self.target_sub = self.create_subscription(
            Float64MultiArray,
            "/motor_target_velocities",
            self.target_callback,
            10,
        )

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(
            f"Connected to {len(self.motors)} motor(s) on {self.port_name}"
        )

    def target_callback(self, msg: Float64MultiArray):
        names = list(self.motors.keys())

        if len(msg.data) != len(names):
            self.get_logger().warn(
                f"Expected {len(names)} velocities, got {len(msg.data)}: {list(msg.data)}"
            )
            return

        velocities = [int(v) for v in msg.data]

        command_summary = " | ".join(
            f"{name}:{vel:+d}" for name, vel in zip(names, velocities)
        )
        self.get_logger().info(f"Commanding [{command_summary}]")

        for name, velocity in zip(names, velocities):
            self.bus.write(
                "Goal_Velocity",
                name,
                velocity,
                normalize=False,
            )

    def timer_callback(self):
        names = []
        positions = []

        for name in self.motors.keys():
            try:
                pos = self.bus.read(
                    "Present_Position",
                    name,
                    normalize=False,
                )
            except Exception as e:
                self.get_logger().warn(f"Failed to read {name}: {e}")
                continue

            names.append(name)
            positions.append(constants.ticks_to_rad(int(pos)))

        if not names:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = positions

        self.joint_pub.publish(msg)

    def destroy_node(self):
        try:
            for name in self.motors.keys():
                self.bus.write(
                    "Goal_Velocity",
                    name,
                    0,
                    normalize=False,
                )
                self.bus.disable_torque(name)

            self.bus.disconnect()
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