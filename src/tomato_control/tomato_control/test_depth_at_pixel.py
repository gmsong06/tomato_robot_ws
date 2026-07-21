#!/usr/bin/env python3

import argparse
from types import SimpleNamespace

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from stereo_msgs.msg import DisparityImage

from tomato_control.tomato_depth_estimator import TomatoDepthEstimator


class DepthAtPixel(Node):
    def __init__(self, args):
        super().__init__("depth_at_pixel")

        config = SimpleNamespace(
            minimum_valid_disparity_px=1.0,
            maximum_valid_disparity_px=400.0,
            minimum_valid_disparity_ratio=0.10,
            roi_total_shrink_fraction=args.roi_shrink,
            surface_disparity_percentile=args.percentile,
        )

        self.estimator = TomatoDepthEstimator(config)
        self.bridge = CvBridge()
        self.done = False

        half = args.box_size // 2

        # Fake a tomato bounding box centered on the requested pixel.
        self.detection = SimpleNamespace(
            x1=args.u - half,
            y1=args.v - half,
            x2=args.u + half,
            y2=args.v + half,
        )

        self.subscription = self.create_subscription(
            DisparityImage,
            "/stereo/disparity",
            self.disparity_callback,
            qos_profile_sensor_data,
        )

        print(
            f"Waiting for disparity at pixel "
            f"(u={args.u}, v={args.v})..."
        )

    def disparity_callback(self, message):
        disparity_image = self.bridge.imgmsg_to_cv2(
            message.image,
            desired_encoding="32FC1",
        )

        result = self.estimator.estimate(
            disparity_image,
            message,
            self.detection,
        )

        print("\nDepth result")
        print(f"focal length: {message.f:.3f} px")
        print(f"baseline: {message.t:.6f} m")

        if result is None:
            print("FAILED: the requested region was invalid")
            self.done = True
            return

        print(
            f"sampled ROI: "
            f"x=[{result.roi.x_min}, {result.roi.x_max}), "
            f"y=[{result.roi.y_min}, {result.roi.y_max})"
        )
        print(
            f"valid pixels: {result.valid_pixel_count}/"
            f"{result.total_pixel_count}"
        )
        print(f"valid ratio: {result.valid_pixel_ratio:.3f}")

        if not result.is_valid:
            print("FAILED: insufficient valid disparity")
            self.done = True
            return

        print(f"median disparity: {result.median_disparity_px:.3f} px")
        print(
            f"selected disparity: "
            f"{result.surface_disparity_px:.3f} px"
        )
        print(f"estimated depth: {result.optical_depth_m:.4f} m")

        self.done = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("u", type=int, help="Horizontal pixel coordinate")
    parser.add_argument("v", type=int, help="Vertical pixel coordinate")
    parser.add_argument("--box-size", type=int, default=40)
    parser.add_argument("--roi-shrink", type=float, default=0.40)
    parser.add_argument("--percentile", type=float, default=75.0)
    args = parser.parse_args()

    rclpy.init(args=[])
    node = DepthAtPixel(args)

    while rclpy.ok() and not node.done:
        rclpy.spin_once(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()