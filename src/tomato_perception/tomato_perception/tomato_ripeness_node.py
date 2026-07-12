
import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node

from tomato_interfaces.msg import (
    TomatoDetectionArray,
    TomatoRipeness,
    TomatoRipenessArray,
)

MIN_TOMATO_PIXEL_RATIO = 0.05
MIN_WARM_RATIO = 0.20


class TomatoRipenessNode(Node):
    def __init__(self):
        super().__init__("tomato_ripeness_node")

        self.bridge = CvBridge()

        self.detections_sub = self.create_subscription(
            TomatoDetectionArray,
            "/tomato_detections",
            self.tomato_detections_callback,
            10,
        )

        self.ripeness_pub = self.create_publisher(
            TomatoRipenessArray,
            "/tomato_ripeness",
            10,
        )

        self.get_logger().info("Tomato ripeness node started")

    def get_color_masks_bgr(self, crop):
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        valid = (s > 50) & (v > 50)

        red = valid & ((h <= 10) | (h >= 170))
        orange = valid & (h > 10) & (h <= 22)
        yellow = valid & (h > 22) & (h <= 28)
        green = valid & (h > 28) & (h <= 90)

        return red, orange, yellow, green

    def classify_crop(self, crop):
        red, orange, yellow, green = self.get_color_masks_bgr(crop)

        crop_area = crop.shape[0] * crop.shape[1]
        if crop_area == 0:
            return "unknown", 0.0

        red_count = int(red.sum())
        orange_count = int(orange.sum())
        yellow_count = int(yellow.sum())
        green_count = int(green.sum())

        warm_count = red_count + orange_count + yellow_count
        tomato_count = warm_count + green_count

        tomato_pixel_ratio = tomato_count / crop_area

        if tomato_pixel_ratio < MIN_TOMATO_PIXEL_RATIO:
            return "unknown", 0.0

        warm_ratio = warm_count / tomato_count
        green_ratio = green_count / tomato_count

        if warm_count == 0:
            warm_strength = 0.0
        else:
            warm_strength = (
                1.0 * red_count
                + 0.6 * orange_count
                + 0.05 * yellow_count
            ) / warm_count

        ripeness_score = float(warm_strength)

        if warm_ratio < MIN_WARM_RATIO:
            final_ripeness = "green" if green_ratio > 0.60 else "unknown"
        elif warm_strength >= 0.75:
            final_ripeness = "fully_ripened"
        elif warm_strength >= 0.45:
            final_ripeness = "half_ripened"
        else:
            final_ripeness = "green"

        return final_ripeness, ripeness_score

    def tomato_detections_callback(self, msg: TomatoDetectionArray):
        self.get_logger().info(f"Received {len(msg.detections)} tomato detection(s)")

        ripeness_array_msg = TomatoRipenessArray()
        ripeness_array_msg.header = msg.header

        for detection in msg.detections:
            crop = self.bridge.imgmsg_to_cv2(
                detection.image,
                desired_encoding="bgr8",
            )

            final_ripeness, ripeness_score = self.classify_crop(crop)

            tomato_ripeness = TomatoRipeness()
            tomato_ripeness.header = detection.header
            tomato_ripeness.detection_id = detection.detection_id

            tomato_ripeness.yolo_ripeness = detection.yolo_ripeness
            tomato_ripeness.final_ripeness = final_ripeness

            tomato_ripeness.yolo_confidence = detection.yolo_confidence
            tomato_ripeness.ripeness_score = float(ripeness_score)

            tomato_ripeness.x1 = detection.x1
            tomato_ripeness.y1 = detection.y1
            tomato_ripeness.x2 = detection.x2
            tomato_ripeness.y2 = detection.y2

            ripeness_array_msg.ripenesses.append(tomato_ripeness)

            self.get_logger().info(
                f"id={detection.detection_id}, "
                f"YOLO={detection.yolo_ripeness}, "
                f"final={final_ripeness}, "
                f"score={ripeness_score:.2f}"
            )

        self.ripeness_pub.publish(ripeness_array_msg)

        self.get_logger().info(
            f"Published {len(ripeness_array_msg.ripenesses)} tomato ripeness result(s)"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TomatoRipenessNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()