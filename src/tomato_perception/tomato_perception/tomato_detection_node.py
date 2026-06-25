
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
import time

from tomato_interfaces.msg import TomatoDetection, TomatoDetectionArray


DUPLICATE_IOU_THRESHOLD = 0.5
MIN_BOX_AREA_RATIO = 0.001


class TomatoDetectionNode(Node):
    def __init__(self):
        super().__init__("tomato_detection_node")

        self.declare_parameter(
            "model_path",
            "/home/ann/tomato_robot_ws/src/tomato_perception/models/src/tomato_perception/models/yolo11s_6.pt",
        )
        self.declare_parameter("yolo_conf", 0.5)

        self.yolo_conf = self.get_parameter("yolo_conf").value
        self.model = YOLO(self.get_parameter("model_path").value)
        self.class_names = self.model.names

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, 10
        )

        self.detections_pub = self.create_publisher(
            TomatoDetectionArray, "/tomato_detections", 10
        )

        self.annotated_pub = self.create_publisher(
            Image, "/tomato_detections/annotated_image", 10
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
                    "conf": float(b.conf.item()),
                    "coords": tuple(map(int, b.xyxy[0].tolist())),
                }
                for b in boxes
            ],
            key=lambda x: x["conf"],
            reverse=True,
        )

        kept = []
        for det in detections:
            is_duplicate = any(
                self.compute_iou(det["coords"], kept_det["coords"])
                > DUPLICATE_IOU_THRESHOLD
                for kept_det in kept
            )
            if not is_duplicate:
                kept.append(det)

        return kept

    def box_area_ratio(self, x1, y1, x2, y2, frame):
        h, w = frame.shape[:2]
        image_area = h * w
        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        return box_area / image_area if image_area > 0 else 0.0

    def simplify_yolo_class(self, yolo_class):
        yolo_class = str(yolo_class).lower()

        if "fully" in yolo_class:
            return "fully_ripened"
        if "half" in yolo_class:
            return "half_ripened"
        if "green" in yolo_class:
            return "green"

        return "unknown"

    def image_callback(self, msg: Image):
        start = time.perf_counter()
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        results = self.model.predict(
            source=frame,
            conf=self.yolo_conf,
            imgsz=320,
            verbose=False,
        )

        annotated_frame = results[0].plot()
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        annotated_msg.header = msg.header
        # self.annotated_pub.publish(annotated_msg)

        filtered_boxes = self.remove_duplicate_detections(results[0].boxes)

        array_msg = TomatoDetectionArray()
        array_msg.header = msg.header

        h, w = frame.shape[:2]
        detection_id = 0

        for det in filtered_boxes:
            x1, y1, x2, y2 = det["coords"]

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            if self.box_area_ratio(x1, y1, x2, y2, frame) < MIN_BOX_AREA_RATIO:
                continue

            crop = frame[y1:y2, x1:x2]
            raw_class_name = self.class_names.get(det["cls_id"], str(det["cls_id"]))

            tomato_msg = TomatoDetection()
            tomato_msg.header = msg.header
            tomato_msg.detection_id = detection_id
            tomato_msg.yolo_ripeness = self.simplify_yolo_class(raw_class_name)
            tomato_msg.yolo_confidence = float(det["conf"])

            tomato_msg.x1 = int(x1)
            tomato_msg.y1 = int(y1)
            tomato_msg.x2 = int(x2)
            tomato_msg.y2 = int(y2)

            tomato_msg.image = self.bridge.cv2_to_imgmsg(crop, encoding="bgr8")
            tomato_msg.image.header = msg.header

            array_msg.detections.append(tomato_msg)
            detection_id += 1

        self.detections_pub.publish(array_msg)

        self.get_logger().info(
            f"YOLO: {(time.perf_counter()-start)*1000:.1f} ms"
        )
        self.get_logger().info(f"Published {len(array_msg.detections)} tomato detection(s)")


def main(args=None):
    rclpy.init(args=args)
    node = TomatoDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()