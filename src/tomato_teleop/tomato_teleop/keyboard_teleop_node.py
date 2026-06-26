#!/usr/bin/env python3

import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class KeyboardTeleopNode(Node):
    def __init__(self):
        super().__init__("keyboard_teleop_node")

        self.declare_parameter("velocity", 1700.0)
        self.declare_parameter("num_motors", 6)
        self.declare_parameter("motor_index", 0)
        self.declare_parameter("publish_hz", 50.0)

        self.velocity = float(self.get_parameter("velocity").value)
        self.num_motors = int(self.get_parameter("num_motors").value)
        self.motor_index = int(self.get_parameter("motor_index").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)

        self.motor_pub = self.create_publisher(
            Float64MultiArray,
            "/motor_target_velocities",
            10,
        )

        self.timer = self.create_timer(1.0 / self.publish_hz, self.timer_callback)

        self.get_logger().info(
            f"Keyboard teleop started: motor_index={self.motor_index}, "
            f"num_motors={self.num_motors}, velocity={self.velocity}"
        )

        print(
            """
Keyboard teleop:
  w      : selected motor forward
  s      : selected motor backward
  space  : stop
  q      : quit
"""
        )

    def publish_selected_velocity(self, velocity):
        msg = Float64MultiArray()
        msg.data = [0.0] * self.num_motors
        msg.data[self.motor_index] = velocity
        self.motor_pub.publish(msg)

    def read_key_nonblocking(self):
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if readable:
            return sys.stdin.read(1)
        return None

    def timer_callback(self):
        key = self.read_key_nonblocking()

        velocity = 0.0

        if key == "w":
            velocity = self.velocity
        elif key == "s":
            velocity = -self.velocity
        elif key == " ":
            velocity = 0.0
        elif key == "q":
            self.publish_selected_velocity(0.0)
            self.get_logger().info("Quit requested")
            rclpy.shutdown()
            return

        self.publish_selected_velocity(velocity)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleopNode()

    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_selected_velocity(0.0)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()