import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


class TomatoDetectionNode(Node):
    def __init__(self):
        super().__init__("tomato_detection_node")

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            10,
        )

        self.tomato_seen_pub = self.create_publisher(
            Bool,
            "/tomato_seen",
            10,
        )

        self.tomato_crop_pub = self.create_publisher(
            Image,
            "/tomato_crop/image_raw",
            10,
        )

        self.tomato_yolo_ripeness_pub = self.create_publisher(
            String,
            "/tomato_yolo_ripeness",
            10,
        )

        self.get_logger().info("Tomato detection node started")

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # TODO: replace this stub with actual YOLO tomato detector.
        tomato_seen = False
        tomato_crop = None
        yolo_ripeness = "unknown"

        # Example future output:
        # tomato_seen = True
        # tomato_crop = frame[y1:y2, x1:x2]
        # yolo_ripeness = "red"

        seen_msg = Bool()
        seen_msg.data = tomato_seen
        self.tomato_seen_pub.publish(seen_msg)

        ripeness_msg = String()
        ripeness_msg.data = yolo_ripeness
        self.tomato_yolo_ripeness_pub.publish(ripeness_msg)

        if tomato_crop is not None:
            crop_msg = self.bridge.cv2_to_imgmsg(tomato_crop, encoding="bgr8")
            crop_msg.header.stamp = msg.header.stamp
            crop_msg.header.frame_id = "tomato_crop_frame"
            self.tomato_crop_pub.publish(crop_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TomatoDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()