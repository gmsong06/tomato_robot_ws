#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from tomato_interfaces.msg import TomatoDetectionArray


class TomatoReactiveControllerNode(Node):
    def __init__(self):
        super().__init__("tomato_reactive_controller_node")

        self.declare_parameter("velocity", 1700.0)
        self.declare_parameter("num_motors", 1)

        self.velocity = float(self.get_parameter("velocity").value)
        self.num_motors = int(self.get_parameter("num_motors").value)

        self.detections_sub = self.create_subscription(
            TomatoDetectionArray,
            "/tomato_detections",
            self.detections_callback,
            10,
        )

        self.motor_pub = self.create_publisher(
            Float64MultiArray,
            "/motor_target_velocities",
            10,
        )

        self.get_logger().info("Tomato reactive controller started")

    def detections_callback(self, msg: TomatoDetectionArray):
        cmd = Float64MultiArray()
        self.get_logger().info(f"Received {len(msg.detections)} detections")
        if len(msg.detections) > 0:
            cmd.data = [self.velocity] + [0.0] * (self.num_motors - 1)
            self.get_logger().info("Tomato detected → motor running")
        else:
            cmd.data = [0.0] * self.num_motors
            self.get_logger().info("No tomato → motor stopped")

        self.motor_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = TomatoReactiveControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Float64MultiArray()
        stop.data = [0.0] * node.num_motors
        node.motor_pub.publish(stop)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()