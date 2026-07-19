import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

from tomato_interfaces.msg import TomatoDetection, TomatoDetectionArray
from tomato_perception.tomato_tracker import TomatoTracker


DUPLICATE_IOU_THRESHOLD = 0.5
MIN_BOX_AREA_RATIO = 0.001


class TomatoDetectionNode(Node):
    def __init__(self):
        super().__init__("tomato_detection_node")

        # ---------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------
        self.declare_parameter(
            "model_path",
            "/home/ann/tomato_robot_ws/src/"
            "tomato_perception/models/yolo11s_6.pt",
        )
        self.declare_parameter("yolo_conf", 0.5)

        self.declare_parameter("tracker_iou_threshold", 0.25)
        self.declare_parameter("tracker_max_center_distance_px", 80.0)
        self.declare_parameter("tracker_max_missed_frames", 10)

        model_path = str(self.get_parameter("model_path").value)
        self.yolo_confidence_threshold = float(
            self.get_parameter("yolo_conf").value
        )

        tracker_iou_threshold = float(
            self.get_parameter("tracker_iou_threshold").value
        )
        tracker_max_center_distance_px = float(
            self.get_parameter("tracker_max_center_distance_px").value
        )
        tracker_max_missed_frames = int(
            self.get_parameter("tracker_max_missed_frames").value
        )

        # ---------------------------------------------------------
        # YOLO and persistent tracker
        # ---------------------------------------------------------
        self.get_logger().info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names

        self.tracker = TomatoTracker(
            iou_threshold=tracker_iou_threshold,
            max_center_distance_px=tracker_max_center_distance_px,
            max_missed_frames=tracker_max_missed_frames,
        )

        # ---------------------------------------------------------
        # ROS communication
        # ---------------------------------------------------------
        self.bridge = CvBridge()

        self.image_subscription = self.create_subscription(
            Image,
            "/stereo/left/image_rect",
            self.image_callback,
            10,
        )

        self.detections_publisher = self.create_publisher(
            TomatoDetectionArray,
            "/tomato_detections",
            10,
        )

        self.annotated_image_publisher = self.create_publisher(
            Image,
            "/tomato_detections/annotated_image",
            10,
        )

        self.get_logger().info(
            "Tomato detection node started with persistent tracking: "
            f"IoU threshold={tracker_iou_threshold:.2f}, "
            f"max center distance={tracker_max_center_distance_px:.1f}px, "
            f"max missed frames={tracker_max_missed_frames}"
        )

    # =========================================================
    # Detection filtering
    # =========================================================

    @staticmethod
    def compute_iou(box1, box2):
        x11, y11, x12, y12 = box1
        x21, y21, x22, y22 = box2

        intersection_x1 = max(x11, x21)
        intersection_y1 = max(y11, y21)
        intersection_x2 = min(x12, x22)
        intersection_y2 = min(y12, y22)

        intersection_area = (
            max(0, intersection_x2 - intersection_x1)
            * max(0, intersection_y2 - intersection_y1)
        )

        area1 = max(0, x12 - x11) * max(0, y12 - y11)
        area2 = max(0, x22 - x21) * max(0, y22 - y21)
        union_area = area1 + area2 - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def remove_duplicate_detections(self, boxes):
        detections = sorted(
            [
                {
                    "class_id": int(box.cls.item()),
                    "confidence": float(box.conf.item()),
                    "coords": tuple(map(int, box.xyxy[0].tolist())),
                }
                for box in boxes
            ],
            key=lambda detection: detection["confidence"],
            reverse=True,
        )

        kept_detections = []

        for detection in detections:
            is_duplicate = any(
                self.compute_iou(
                    detection["coords"],
                    kept_detection["coords"],
                )
                > DUPLICATE_IOU_THRESHOLD
                for kept_detection in kept_detections
            )

            if not is_duplicate:
                kept_detections.append(detection)

        return kept_detections

    @staticmethod
    def box_area_ratio(x1, y1, x2, y2, frame):
        image_height, image_width = frame.shape[:2]
        image_area = image_height * image_width
        box_area = max(0, x2 - x1) * max(0, y2 - y1)

        return box_area / image_area if image_area > 0 else 0.0

    @staticmethod
    def simplify_yolo_class(yolo_class):
        normalized_class = str(yolo_class).lower()

        if "fully" in normalized_class:
            return "fully_ripened"
        if "half" in normalized_class:
            return "half_ripened"
        if "green" in normalized_class:
            return "green"

        return "unknown"

    @staticmethod
    def draw_detection(
        frame,
        x1,
        y1,
        x2,
        y2,
        track_id,
        label,
        confidence,
    ):
        text = f"ID {track_id} | {label} | {confidence:.2f}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        text_y = max(y1 - 8, 15)

        cv2.putText(
            frame,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # =========================================================
    # Image processing
    # =========================================================

    def image_callback(self, message: Image):
        callback_start = time.perf_counter()

        frame = self.bridge.imgmsg_to_cv2(
            message,
            desired_encoding="bgr8",
        )
        annotated_frame = frame.copy()

        inference_start = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=self.yolo_confidence_threshold,
            imgsz=320,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        raw_detections = self.remove_duplicate_detections(
            results[0].boxes
        )

        image_height, image_width = frame.shape[:2]
        valid_detections = []

        for detection in raw_detections:
            x1, y1, x2, y2 = detection["coords"]

            x1 = max(0, min(x1, image_width - 1))
            x2 = max(0, min(x2, image_width))
            y1 = max(0, min(y1, image_height - 1))
            y2 = max(0, min(y2, image_height))

            if x2 <= x1 or y2 <= y1:
                continue

            if (
                self.box_area_ratio(x1, y1, x2, y2, frame)
                < MIN_BOX_AREA_RATIO
            ):
                continue

            detection["coords"] = (x1, y1, x2, y2)
            valid_detections.append(detection)

        tracked_detections = self.tracker.update(valid_detections)

        detection_array_message = TomatoDetectionArray()
        detection_array_message.header = message.header

        for detection in tracked_detections:
            track_id = int(detection["track_id"])
            x1, y1, x2, y2 = detection["coords"]

            crop = frame[y1:y2, x1:x2]

            raw_class_name = self.class_names.get(
                detection["class_id"],
                str(detection["class_id"]),
            )
            simplified_class = self.simplify_yolo_class(
                raw_class_name
            )

            tomato_message = TomatoDetection()
            tomato_message.header = message.header

            # This field now contains a persistent tracking ID rather than
            # a per-frame array index. No ROS interface change is required.
            tomato_message.detection_id = track_id

            tomato_message.yolo_ripeness = simplified_class
            tomato_message.yolo_confidence = float(
                detection["confidence"]
            )

            tomato_message.x1 = int(x1)
            tomato_message.y1 = int(y1)
            tomato_message.x2 = int(x2)
            tomato_message.y2 = int(y2)

            tomato_message.image = self.bridge.cv2_to_imgmsg(
                crop,
                encoding="bgr8",
            )
            tomato_message.image.header = message.header

            detection_array_message.detections.append(
                tomato_message
            )

            self.draw_detection(
                annotated_frame,
                x1,
                y1,
                x2,
                y2,
                track_id,
                simplified_class,
                float(detection["confidence"]),
            )

        self.detections_publisher.publish(
            detection_array_message
        )

        annotated_message = self.bridge.cv2_to_imgmsg(
            annotated_frame,
            encoding="bgr8",
        )
        annotated_message.header = message.header
        self.annotated_image_publisher.publish(annotated_message)

        callback_ms = (time.perf_counter() - callback_start) * 1000.0

        self.get_logger().info(
            f"YOLO={inference_ms:.1f}ms | "
            f"callback={callback_ms:.1f}ms | "
            f"detections={len(detection_array_message.detections)} | "
            f"active IDs={self.tracker.active_track_ids}"
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
