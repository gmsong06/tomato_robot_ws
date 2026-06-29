
from pathlib import Path
import time
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from tomato_interfaces.srv import SetTorque
from tomato_trajectory.torque_utils import call_set_torque


class ReplayTrajectoryNode(Node):
    def __init__(self):
        super().__init__("replay_trajectory_node")

        self.declare_parameter("name", "test_motion")
        self.declare_parameter(
            "trajectory_dir",
            "/home/ann/tomato_robot_ws/src/tomato_trajectory/trajectories",
        )
        self.declare_parameter("speed_scale", 1.0)
        self.declare_parameter("move_to_start", True)

        self.name = self.get_parameter("name").value
        self.trajectory_dir = Path(self.get_parameter("trajectory_dir").value)
        self.speed_scale = float(self.get_parameter("speed_scale").value)
        self.move_to_start = bool(self.get_parameter("move_to_start").value)

        self.latest_joint_state = None

        self.torque_client = self.create_client(SetTorque, "/set_torque")

        self.target_pub = self.create_publisher(
            Float64MultiArray,
            "/joint_target_positions",
            10,
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg

    def wait_for_joint_state(self):
        self.get_logger().info("Waiting for first /joint_states...")

        while rclpy.ok() and self.latest_joint_state is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info(
            f"Got joint states for: {list(self.latest_joint_state.name)}"
        )

    def load_trajectory(self):
        path = self.trajectory_dir / f"{self.name}.yaml"

        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if "joint_names" not in data or "points" not in data:
            raise ValueError("Trajectory YAML must contain joint_names and points")

        if len(data["points"]) == 0:
            raise ValueError("Trajectory has no points")

        self.get_logger().info(
            f"Loaded trajectory {path} with {len(data['points'])} points"
        )

        return data

    def publish_positions(self, positions):
        msg = Float64MultiArray()
        msg.data = [float(x) for x in positions]
        self.target_pub.publish(msg)

    def move_slowly_to_start(self, start_positions, duration=2.0, rate_hz=20.0):
        if self.latest_joint_state is None:
            return

        current = list(self.latest_joint_state.position)

        if len(current) != len(start_positions):
            self.get_logger().warn(
                f"Current joint count {len(current)} != trajectory joint count {len(start_positions)}"
            )
            return

        steps = max(1, int(duration * rate_hz))

        self.get_logger().info("Moving slowly to trajectory start...")

        for i in range(steps + 1):
            alpha = i / steps

            interp = [
                (1.0 - alpha) * c + alpha * s
                for c, s in zip(current, start_positions)
            ]

            self.publish_positions(interp)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(1.0 / rate_hz)

        self.get_logger().info("Reached trajectory start.")

    def replay(self, trajectory):
        points = trajectory["points"]

        self.get_logger().info("Replaying trajectory...")

        start_wall_time = time.time()
        trajectory_start_t = float(points[0]["t"])

        for point in points:
            if not rclpy.ok():
                break

            target_t = (
                float(point["t"]) - trajectory_start_t
            ) / self.speed_scale

            while rclpy.ok():
                elapsed = time.time() - start_wall_time

                if elapsed >= target_t:
                    break

                rclpy.spin_once(self, timeout_sec=0.001)
                time.sleep(0.001)

            self.publish_positions(point["positions"])

        self.get_logger().info("Replay complete.")

    def run(self):
        trajectory = self.load_trajectory()
        self.wait_for_joint_state()

        call_set_torque(self, self.torque_client, True)

        first_positions = trajectory["points"][0]["positions"]

        if self.move_to_start:
            self.move_slowly_to_start(first_positions)

        input("\nPress ENTER to replay trajectory...")

        self.replay(trajectory)


def main(args=None):
    rclpy.init(args=args)
    node = ReplayTrajectoryNode()

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Replay interrupted.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()