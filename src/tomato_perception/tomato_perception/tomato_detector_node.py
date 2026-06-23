import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from ultralytics import YOLO


class TomatoDetectionNode(Node):
    def __init__(self):
        super().__init__("tomato_detection_node")

        self.declare_parameter(
            "model_path",
            "/home/ann/tomato_robot_ws/src/tomato_perception/models/tomato_detector.pt",
        )

        self.declare_parameter(
            "yolo_conf",
            0.5
        )

        self.model = YOLO(
            self.get_parameter("model_path").value
        )

        self.class_names = self.model.names

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

    def compute_iou(self, box1, box2):
        x11, y11, x12, y12 = box1
        x21, y21, x22, y22 = box2
        xi1, yi1 = max(x11, x21), max(y11, y21)
        xi2, yi2 = min(x12, x22), min(y12, y22)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = max(0, x12 - x11) * max(0, y12 - y11)
        area2 = max(0, x22 - x21) * max(0, y22 - y21)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0


    def remove_duplicate_detections(self, boxes):
        detections = sorted(
            [
                {
                    "cls_id": int(b.cls.item()),
                    "conf":   float(b.conf.item()),
                    "coords": tuple(map(int, b.xyxy[0].tolist())),
                }
                for b in boxes
            ],
            key=lambda x: x["conf"],
            reverse=True,
        )
        kept = []
        for det in detections:
            if not any(self.compute_iou(det["coords"], k["coords"]) > DUPLICATE_IOU_THRESHOLD for k in kept):
                kept.append(det)
        return kept

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        results = self.model.predict(
            source=frame,
            conf=self.yolo_conf,
            verbose=False,
        )

        filtered_boxes = self.remove_duplicate_detections(results[0].boxes)
        
        self.get_logger().info(f"Detections: {len(results[0].boxes)} → {len(filtered_boxes)} after removing duplicates")

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