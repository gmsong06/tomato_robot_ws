#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor


TICKS_PER_REV = 4095


class FeetechMotorNode(Node):
    def __init__(self):
        super().__init__("feetech_motor_node")

        self.declare_parameter("port", "/dev/tty.usbmodem5B3D0464591")
        self.declare_parameter("motor_id", 1)
        self.declare_parameter("motor_model", "sts3215")

        self.port_name = self.get_parameter("port").value
        self.motor_id = int(self.get_parameter("motor_id").value)
        self.motor_model = self.get_parameter("motor_model").value

        self.motor_name = f"joint_{self.motor_id}"

        self.bus = FeetechMotorsBus(
            port=self.port_name,
            motors={
                self.motor_name: Motor(
                    id=self.motor_id,
                    model=self.motor_model,
                    norm_mode=None,
                )
            },
        )

        self.bus.connect()
        self.bus.enable_torque(self.motor_name)

        self.get_logger().info(
            f"Connected to {self.motor_model} motor {self.motor_id} on {self.port_name}"
        )

        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)

        self.target_sub = self.create_subscription(
            Float64,
            "/motor_target_ticks",
            self.target_callback,
            10,
        )

        self.timer = self.create_timer(0.1, self.timer_callback)

    def ticks_to_rad(self, ticks: int) -> float:
        return (float(ticks) / TICKS_PER_REV) * 2.0 * math.pi

    def read_position(self):
        try:
            pos = self.bus.read("Present_Position", self.motor_name)
            return int(pos)
        except Exception as e:
            self.get_logger().warn(f"Failed to read position: {e}")
            return None

    def target_callback(self, msg: Float64):
        target_ticks = int(msg.data)

        self.get_logger().info(
            f"Commanding {self.motor_name} to {target_ticks} ticks"
        )

        try:
            self.bus.write("Goal_Position", self.motor_name, target_ticks)
        except Exception as e:
            self.get_logger().warn(f"Failed to command motor: {e}")

    def timer_callback(self):
        pos = self.read_position()
        if pos is None:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [self.motor_name]
        msg.position = [self.ticks_to_rad(pos)]

        self.joint_pub.publish(msg)
        self.get_logger().info(f"{self.motor_name}: {pos} ticks")

    def destroy_node(self):
        try:
            self.bus.disable_torque(self.motor_name)
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