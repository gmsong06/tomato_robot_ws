import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from stereo_msgs.msg import DisparityImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class DisparityImageExtractNode(Node):
    def __init__(self):
        super().__init__("disparity_image_extract_node")

        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            DisparityImage,
            "/stereo/disparity",
            self.callback,
            10,
        )

        self.pub = self.create_publisher(
            Image,
            "/stereo/disparity/color",
            10,
        )

        self.get_logger().info("Published disparity image")

    def callback(self, msg: DisparityImage):
        disparity_msg = msg.image

        disparity = self.bridge.imgmsg_to_cv2(
            disparity_msg,
            desired_encoding="32FC1",
        )

        valid = np.isfinite(disparity) & (disparity > msg.min_disparity)

        if not np.any(valid):
            return

        disp_min = np.min(disparity[valid])
        disp_max = np.max(disparity[valid])

        if disp_max <= disp_min:
            return

        normalized = np.zeros_like(disparity, dtype=np.uint8)
        normalized[valid] = (
            255.0 * (disparity[valid] - disp_min) / (disp_max - disp_min)
        ).astype(np.uint8)

        color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        # Invalid pixels black
        color[~valid] = (0, 0, 0)

        out_msg = self.bridge.cv2_to_imgmsg(color, encoding="bgr8")
        out_msg.header = msg.header

        self.pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DisparityImageExtractNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()