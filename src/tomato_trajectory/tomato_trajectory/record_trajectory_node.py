
from pathlib import Path
import select
import sys
import time
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from tomato_interfaces.srv import SetTorque
from tomato_trajectory.torque_utils import call_set_torque


class RecordTrajectoryNode(Node):
    def __init__(self):
        super().__init__("record_trajectory_node")

        self.declare_parameter("name", "test_motion")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter(
            "output_dir",
            "/home/ann/tomato_robot_ws/src/tomato_trajectory/trajectories",
        )

        self.name = self.get_parameter("name").value
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.output_dir = Path(self.get_parameter("output_dir").value)

        self.latest_joint_state = None
        self.recording = False
        self.points = []
        self.start_time = None
        self.joint_names = None

        self.torque_client = self.create_client(SetTorque, "/set_torque")

        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )

        self.timer = self.create_timer(
            1.0 / self.rate_hz,
            self.record_timer_callback,
        )

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg

    def wait_for_enter_while_spinning(self, prompt):
        print(prompt)
        print("Press ENTER to continue...")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            readable, _, _ = select.select([sys.stdin], [], [], 0.0)

            if readable:
                sys.stdin.readline()
                return

    def wait_for_joint_state(self):
        self.get_logger().info("Waiting for first /joint_states...")

        while rclpy.ok() and self.latest_joint_state is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.joint_names = list(self.latest_joint_state.name)

        self.get_logger().info(
            f"Got joint states for: {self.joint_names}"
        )

    def start_recording(self):
        self.points = []
        self.start_time = time.time()
        self.recording = True
        self.get_logger().info("Recording started.")

    def stop_recording(self):
        self.recording = False
        self.get_logger().info(
            f"Recording stopped. {len(self.points)} points captured."
        )

    def record_timer_callback(self):
        if not self.recording:
            return

        if self.latest_joint_state is None:
            return

        t = time.time() - self.start_time

        point = {
            "t": float(t),
            "positions": [float(x) for x in self.latest_joint_state.position],
        }

        self.points.append(point)

        if len(self.points) % int(self.rate_hz) == 0:
            self.get_logger().info(f"Recording... {len(self.points)} points")

    def save_trajectory(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.output_dir / f"{self.name}.yaml"

        data = {
            "name": self.name,
            "rate_hz": self.rate_hz,
            "joint_names": self.joint_names,
            "points": self.points,
        }

        with open(output_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        self.get_logger().info(f"Saved trajectory to {output_path}")

    def run(self):
        self.wait_for_joint_state()

        self.wait_for_enter_while_spinning(
            "\nReady to enter recording mode."
        )

        call_set_torque(self, self.torque_client, False)

        print("\nMove the arm by hand.")
        self.start_recording()

        self.wait_for_enter_while_spinning(
            "\nRecording now."
        )

        self.stop_recording()
        self.save_trajectory()

        self.wait_for_enter_while_spinning(
            "\nReady to re-enable torque."
        )

        call_set_torque(self, self.torque_client, True)


def main(args=None):
    rclpy.init(args=args)
    node = RecordTrajectoryNode()

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Recording interrupted.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()