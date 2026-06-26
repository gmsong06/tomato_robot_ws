#!/usr/bin/env python3

import math
import select
import sys
import termios
import tty

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


TICKS_PER_REV = 4096


class KeyboardTeleopNode(Node):
    def __init__(self):
        super().__init__("keyboard_teleop_node")

        self.declare_parameter("step_rad", 0.05)
        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter(
            "motor_config_path",
            "/home/ann/tomato_robot_ws/src/tomato_motor_control/config/motors.yaml",
        )

        self.step_rad = float(self.get_parameter("step_rad").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.motor_config_path = self.get_parameter("motor_config_path").value

        with open(self.motor_config_path, "r") as f:
            config = yaml.safe_load(f)

        self.motor_config = config["motors"]
        self.joint_names = list(self.motor_config.keys())
        self.num_motors = len(self.joint_names)

        self.targets = [0.0] * self.num_motors
        self.got_joint_state = False
        self.joint_limits = self.load_joint_limits()

        self.target_pub = self.create_publisher(
            Float64MultiArray,
            "/joint_target_positions",
            10,
        )

        self.joint_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )

        self.keymap = {
            "q": (0, +1),
            "a": (0, -1),
            "w": (1, +1),
            "s": (1, -1),
            "e": (2, +1),
            "d": (2, -1),
            "r": (3, +1),
            "f": (3, -1),
            "t": (4, +1),
            "g": (4, -1),
            "y": (5, +1),
            "h": (5, -1),
        }

        self.get_logger().info(
            f"Loaded {self.num_motors} joints from {self.motor_config_path}"
        )

        print(
            """
Position teleop:

  q/a : joint 1 +/-
  w/s : joint 2 +/-
  e/d : joint 3 +/-
  r/f : joint 4 +/-
  t/g : joint 5 +/-
  y/h : joint 6 +/-

  space : republish current target
  x     : quit

Waiting for first /joint_states before accepting movement keys...
"""
        )

    def load_joint_limits(self):
        limits = []

        for joint_name in self.joint_names:
            info = self.motor_config[joint_name]

            range_min = float(info["range_min"])
            range_max = float(info["range_max"])
            mid = (range_min + range_max) / 2.0

            min_rad = (range_min - mid) * 2.0 * math.pi / TICKS_PER_REV
            max_rad = (range_max - mid) * 2.0 * math.pi / TICKS_PER_REV

            limits.append((min_rad, max_rad))

            self.get_logger().info(
                f"{joint_name}: limits [{min_rad:+.2f}, {max_rad:+.2f}] rad"
            )

        return limits

    def clamp_target(self, joint_idx, value):
        lo, hi = self.joint_limits[joint_idx]
        return max(lo, min(hi, value))

    def clamp_all_targets(self):
        for i in range(self.num_motors):
            self.targets[i] = self.clamp_target(i, self.targets[i])

    def joint_state_callback(self, msg: JointState):
        if self.got_joint_state:
            return

        if len(msg.position) < self.num_motors:
            return

        self.targets = list(msg.position[: self.num_motors])
        self.clamp_all_targets()
        self.got_joint_state = True

        self.get_logger().info("Teleop ready. Initialized targets from /joint_states.")

    def publish_targets(self):
        self.clamp_all_targets()

        msg = Float64MultiArray()
        msg.data = self.targets
        self.target_pub.publish(msg)

        pretty = " | ".join(
            f"{self.joint_names[i]}:{self.targets[i]:+.2f}"
            for i in range(self.num_motors)
        )
        self.get_logger().info(f"Targets [{pretty}]")

    def read_key_nonblocking(self):
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if readable:
            return sys.stdin.read(1)
        return None

    def handle_key(self, key):
        if key is None:
            return True

        if key == "x":
            return False

        if not self.got_joint_state:
            if key in self.keymap or key == " ":
                self.get_logger().warn("Still waiting for first /joint_states...")
            return True

        if key == " ":
            self.publish_targets()

        elif key in self.keymap:
            joint_idx, direction = self.keymap[key]

            if joint_idx >= self.num_motors:
                return True

            new_target = self.targets[joint_idx] + direction * self.step_rad
            self.targets[joint_idx] = self.clamp_target(joint_idx, new_target)
            self.publish_targets()

        return True

    def run(self):
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setcbreak(sys.stdin.fileno())

            dt = 1.0 / self.publish_hz

            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=dt)

                key = self.read_key_nonblocking()
                keep_running = self.handle_key(key)

                if not keep_running:
                    break

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleopNode()

    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()