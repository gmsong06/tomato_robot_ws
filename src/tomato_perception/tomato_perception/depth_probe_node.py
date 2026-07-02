#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from stereo_msgs.msg import DisparityImage


class DepthProbeNode(Node):
    def __init__(self):
        super().__init__("depth_probe_node")

        self.declare_parameter("u", 320)
        self.declare_parameter("v", 240)
        self.declare_parameter("window", 30)

        # Expected disparity range for ~30.5 in / 0.775 m
        self.declare_parameter("min_valid_disparity", 80.0)
        self.declare_parameter("max_valid_disparity", 180.0)

        self.u = int(self.get_parameter("u").value)
        self.v = int(self.get_parameter("v").value)
        self.window = int(self.get_parameter("window").value)

        self.min_valid_disparity = float(
            self.get_parameter("min_valid_disparity").value
        )
        self.max_valid_disparity = float(
            self.get_parameter("max_valid_disparity").value
        )

        self.bridge = CvBridge()
        self.printed_camera_params = False

        self.sub = self.create_subscription(
            DisparityImage,
            "/stereo/disparity",
            self.callback,
            10,
        )

        self.get_logger().info(
            f"Depth probe started at ROI center=({self.u}, {self.v}), "
            f"window={self.window}px, "
            f"valid disparity range=[{self.min_valid_disparity}, "
            f"{self.max_valid_disparity}]"
        )

        self.declare_parameter("baseline_m", 0.115)
        self.baseline_m = float(self.get_parameter("baseline_m").value)

    def callback(self, msg: DisparityImage):
        disp = self.bridge.imgmsg_to_cv2(
            msg.image,
            desired_encoding="32FC1",
        )

        if not self.printed_camera_params:
            self.get_logger().info(
                f"Disparity camera params: f={msg.f:.3f}, T={msg.t:.6f}"
            )
            self.printed_camera_params = True

        h, w = disp.shape
        half = self.window // 2

        x1 = max(0, self.u - half)
        x2 = min(w, self.u + half)
        y1 = max(0, self.v - half)
        y2 = min(h, self.v + half)

        roi = disp[y1:y2, x1:x2]

        valid = (
            np.isfinite(roi)
            & (roi > self.min_valid_disparity)
            & (roi < self.max_valid_disparity)
        )

        valid_count = int(np.count_nonzero(valid))
        total_count = int(roi.size)

        if valid_count == 0:
            self.get_logger().warn(
                f"No valid disparity in ROI. "
                f"ROI=({x1}:{x2}, {y1}:{y2}), "
                f"valid range=[{self.min_valid_disparity}, "
                f"{self.max_valid_disparity}]"
            )
            return

        valid_disparities = roi[valid]

        median_disp = float(np.median(valid_disparities))
        mean_disp = float(np.mean(valid_disparities))

        depth_m = abs(msg.f * 0.10) / median_disp

        self.get_logger().info(
            f"ROI center=({self.u}, {self.v}), "
            f"valid={valid_count}/{total_count}, "
            f"median disparity={median_disp:.2f}px, "
            f"mean disparity={mean_disp:.2f}px, "
            f"depth={depth_m:.3f} m, "
            f"depth={depth_m * 100:.1f} cm"
        )


def main(args=None):
    rclpy.init(args=args)
    node = DepthProbeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()