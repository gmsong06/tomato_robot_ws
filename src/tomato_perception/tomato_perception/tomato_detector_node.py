import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from ultralytics import YOLO

from tomato_interfaces.msg import TomatoDetection, TomatoDetectionArray


DUPLICATE_IOU_THRESHOLD = 0.5

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

        self.detections_pub = self.create_publisher(
            TomatoDetectionArray,
            "/tomato_detections",
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

        seen_msg = Bool()
        seen_msg.data = len(filtered_boxes) > 0
        self.tomato_seen_pub.publish(seen_msg)

        array_msg = TomatoDetectionArray()
        array_msg.header = msg.header

        h, w = frame.shape[:2]

        for det in filtered_boxes:
            x1, y1, x2, y2 = det["coords"]

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            
            tomato_msg = TomatoDetection()
            tomato_msg.header = msg.header
            tomato_msg.ripeness = str(self.class_names.get(det["cls_id"], det["cls_id"]))
            tomato_msg.confidence = float(det["conf"])
            tomato_msg.x1 = int(x1)
            tomato_msg.y1 = int(y1)
            tomato_msg.x2 = int(x2)
            tomato_msg.y2 = int(y2)

            tomato_msg.image = self.bridge.cv2_to_imgmsg(crop, encoding="bgr8")
            tomato_msg.image.header.stamp = msg.header.stamp
            tomato_msg.image.header.frame_id = "tomato_crop_frame"

            array_msg.detections.append(tomato_msg)
        
        self.detections_pub.publish(array_msg)

        self.get_logger().info(
            f"Published {len(array_msg.detections)} tomato detection(s)"
        )
        

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