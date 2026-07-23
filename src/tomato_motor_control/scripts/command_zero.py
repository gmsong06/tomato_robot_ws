#!/usr/bin/env python3

import argparse
import math
import sys
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from tomato_interfaces.msg import TorqueState
from tomato_interfaces.srv import SetTorque


JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4"]
JOINT_TARGET_TOPIC = "/joint_target_positions"
JOINT_STATE_TOPIC = "/joint_states"
TORQUE_STATE_TOPIC = "/torque_states"
TORQUE_SERVICE = "/set_torque"


class ZeroCommandNode(Node):
    def __init__(self):
        super().__init__("zero_command_test")

        self.latest_positions: Optional[dict[str, float]] = None
        self.latest_torque: Optional[dict[str, bool]] = None

        self.create_subscription(
            JointState,
            JOINT_STATE_TOPIC,
            self.joint_state_callback,
            10,
        )
        self.create_subscription(
            TorqueState,
            TORQUE_STATE_TOPIC,
            self.torque_state_callback,
            10,
        )
        self.target_publisher = self.create_publisher(
            Float64MultiArray,
            JOINT_TARGET_TOPIC,
            10,
        )
        self.torque_client = self.create_client(
            SetTorque,
            TORQUE_SERVICE,
        )

    def joint_state_callback(self, message: JointState):
        if len(message.name) != len(message.position):
            return

        positions = dict(zip(message.name, message.position))
        if not all(name in positions for name in JOINT_NAMES):
            return

        self.latest_positions = {
            name: float(positions[name])
            for name in JOINT_NAMES
        }

    def torque_state_callback(self, message: TorqueState):
        if len(message.name) != len(message.enabled):
            return

        states = dict(zip(message.name, message.enabled))
        if not all(name in states for name in JOINT_NAMES):
            return

        self.latest_torque = {
            name: bool(states[name])
            for name in JOINT_NAMES
        }

    def publish_zero_target(self):
        message = Float64MultiArray()
        message.data = [0.0, 0.0, 0.0, 0.0]
        self.target_publisher.publish(message)

    def emergency_disable_torque(self):
        print("\nRequesting emergency torque disable...", flush=True)

        if not self.torque_client.wait_for_service(timeout_sec=1.0):
            print(
                "ERROR: /set_torque is unavailable. Stop the motor node "
                "or remove motor power immediately.",
                file=sys.stderr,
            )
            return

        request = SetTorque.Request()
        request.enabled = [False]
        future = self.torque_client.call_async(request)

        deadline = time.monotonic() + 2.0
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() >= deadline:
                print(
                    "ERROR: torque-disable request timed out. Stop the "
                    "motor node or remove motor power immediately.",
                    file=sys.stderr,
                )
                return

        try:
            response = future.result()
            print(f"Torque service response: {response.message}")
        except Exception as error:
            print(
                f"ERROR: torque-disable request failed: {error}",
                file=sys.stderr,
            )


def wait_for_feedback(
    node: ZeroCommandNode,
    timeout_sec: float,
) -> bool:
    deadline = time.monotonic() + timeout_sec

    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if (
            node.latest_positions is not None
            and node.latest_torque is not None
        ):
            return True

    return False


def print_positions(label: str, positions: dict[str, float]):
    print(label)
    for name in JOINT_NAMES:
        radians = positions[name]
        print(
            f"  {name}: {radians:+.6f} rad "
            f"({math.degrees(radians):+.2f} deg)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Safely test a [0, 0, 0, 0] ROS joint target."
    )
    parser.add_argument(
        "--max-start-error-rad",
        type=float,
        default=0.20,
        help=(
            "Refuse the test if any joint begins farther than this from "
            "zero (default: 0.20 rad / 11.5 deg)."
        ),
    )
    parser.add_argument(
        "--runaway-margin-rad",
        type=float,
        default=0.10,
        help=(
            "Emergency-stop if a joint moves this much farther from zero "
            "than where it started (default: 0.10 rad / 5.7 deg)."
        ),
    )
    parser.add_argument(
        "--tolerance-rad",
        type=float,
        default=0.03,
        help="Zero-position success tolerance (default: 0.03 rad).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=8.0,
        help="Motion-monitoring timeout (default: 8 seconds).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the typed ZERO confirmation.",
    )
    args = parser.parse_args()

    if args.max_start_error_rad <= 0.0:
        parser.error("--max-start-error-rad must be positive")
    if args.runaway_margin_rad <= 0.0:
        parser.error("--runaway-margin-rad must be positive")
    if args.tolerance_rad <= 0.0:
        parser.error("--tolerance-rad must be positive")
    if args.timeout_sec <= 0.0:
        parser.error("--timeout-sec must be positive")

    rclpy.init()
    node = ZeroCommandNode()
    command_was_sent = False

    try:
        print("Waiting for joint and torque feedback...")
        if not wait_for_feedback(node, timeout_sec=5.0):
            raise RuntimeError(
                "Did not receive both /joint_states and /torque_states. "
                "Start only the updated motor node first."
            )

        assert node.latest_positions is not None
        assert node.latest_torque is not None

        start_positions = dict(node.latest_positions)
        print_positions("Current positions:", start_positions)

        nonfinite = [
            name
            for name, value in start_positions.items()
            if not math.isfinite(value)
        ]
        if nonfinite:
            raise RuntimeError(
                "Non-finite joint feedback for: " + ", ".join(nonfinite)
            )

        too_far = [
            name
            for name, value in start_positions.items()
            if abs(value) > args.max_start_error_rad
        ]
        if too_far:
            raise RuntimeError(
                "Refusing zero command because these joints are too far "
                "from zero: " + ", ".join(too_far)
            )

        torque_off = [
            name
            for name, enabled in node.latest_torque.items()
            if not enabled
        ]
        if torque_off:
            raise RuntimeError(
                "Torque must already be enabled for all joints. Disabled: "
                + ", ".join(torque_off)
            )

        discovery_deadline = time.monotonic() + 2.0
        while (
            node.target_publisher.get_subscription_count() == 0
            and time.monotonic() < discovery_deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.05)

        if node.target_publisher.get_subscription_count() == 0:
            raise RuntimeError(
                "No subscriber is connected to /joint_target_positions"
            )

        if not args.yes:
            confirmation = input(
                "\nSupport the arm and keep emergency power accessible.\n"
                "Type ZERO to command all four joints to 0 rad: "
            )
            if confirmation != "ZERO":
                print("Cancelled; no command was sent.")
                return

        node.publish_zero_target()
        command_was_sent = True
        print("Zero target published. Monitoring motion...")

        deadline = time.monotonic() + args.timeout_sec
        last_print = 0.0

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.latest_positions is None:
                continue

            positions = node.latest_positions

            runaway = [
                name
                for name in JOINT_NAMES
                if abs(positions[name])
                > abs(start_positions[name]) + args.runaway_margin_rad
            ]
            if runaway:
                print_positions("Unsafe motion detected:", positions)
                node.emergency_disable_torque()
                raise RuntimeError(
                    "Joint(s) moved away from zero: "
                    + ", ".join(runaway)
                )

            if all(
                abs(positions[name]) <= args.tolerance_rad
                for name in JOINT_NAMES
            ):
                print_positions("Zero target reached:", positions)
                print(
                    "PASS: all joints reached zero. Torque remains enabled "
                    "to hold the arm."
                )
                return

            now = time.monotonic()
            if now - last_print >= 0.5:
                print_positions("Monitoring:", positions)
                last_print = now

        if command_was_sent:
            node.emergency_disable_torque()
        raise RuntimeError(
            f"Zero target was not reached within {args.timeout_sec:.1f}s"
        )

    except KeyboardInterrupt:
        if command_was_sent:
            node.emergency_disable_torque()
        print("\nInterrupted.")

    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
