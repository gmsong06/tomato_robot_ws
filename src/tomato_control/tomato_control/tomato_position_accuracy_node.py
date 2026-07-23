from __future__ import annotations

import csv
import math
from pathlib import Path

import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from stereo_msgs.msg import DisparityImage

from tomato_control.camera_geometry import CameraGeometry
from tomato_control.controller_config import ControllerConfig
from tomato_control.controller_models import Point3D
from tomato_control.tomato_depth_estimator import TomatoDepthEstimator
from tomato_interfaces.msg import TomatoRipenessArray


class TomatoPositionAccuracyNode(Node):
    """Log tomato position estimates and optional ground-truth error."""

    CSV_FIELDS = (
        "stamp_sec",
        "detection_id",
        "ripeness",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "sample_index",
        "roi_center_u_px",
        "roi_center_v_px",
        "valid_pixel_count",
        "total_pixel_count",
        "valid_pixel_ratio",
        "median_disparity_px",
        "mean_disparity_px",
        "surface_disparity_px",
        "optical_depth_m",
        "camera_x_m",
        "camera_y_m",
        "camera_z_m",
        "origin_x_m",
        "origin_y_m",
        "origin_z_m",
        "truth_x_m",
        "truth_y_m",
        "truth_z_m",
        "error_x_m",
        "error_y_m",
        "error_z_m",
        "error_norm_m",
        "mean_error_x_m",
        "mean_error_y_m",
        "mean_error_z_m",
        "rmse_m",
    )

    def __init__(self):
        super().__init__("tomato_position_accuracy_node")

        ControllerConfig.declare_parameters(self)
        self.config = ControllerConfig.from_node(self)

        self.declare_parameter("camera_info_topic", "/stereo/left/camera_info")
        self.declare_parameter("ripeness_topic", "/tomato_ripeness")
        self.declare_parameter("disparity_topic", "/stereo/disparity")
        self.declare_parameter("sync_slop_sec", 0.15)
        self.declare_parameter("target_detection_id", -1)
        self.declare_parameter("use_ground_truth", False)
        self.declare_parameter("truth_x_m", 0.0)
        self.declare_parameter("truth_y_m", 0.0)
        self.declare_parameter("truth_z_m", 0.0)
        self.declare_parameter(
            "output_csv",
            "/tmp/tomato_position_accuracy.csv",
        )
        self.declare_parameter("append_csv", True)
        self.declare_parameter("log_every_n", 1)

        self.camera_info_topic = str(
            self.get_parameter("camera_info_topic").value
        )
        self.ripeness_topic = str(
            self.get_parameter("ripeness_topic").value
        )
        self.disparity_topic = str(
            self.get_parameter("disparity_topic").value
        )
        self.sync_slop_sec = float(
            self.get_parameter("sync_slop_sec").value
        )
        self.target_detection_id = int(
            self.get_parameter("target_detection_id").value
        )
        self.use_ground_truth = bool(
            self.get_parameter("use_ground_truth").value
        )
        self.truth_point = Point3D(
            x_m=float(self.get_parameter("truth_x_m").value),
            y_m=float(self.get_parameter("truth_y_m").value),
            z_m=float(self.get_parameter("truth_z_m").value),
        )
        self.output_csv = Path(
            str(self.get_parameter("output_csv").value)
        )
        self.append_csv = bool(self.get_parameter("append_csv").value)
        self.log_every_n = max(1, int(self.get_parameter("log_every_n").value))

        if self.sync_slop_sec < 0.0:
            raise ValueError("sync_slop_sec must be nonnegative")

        self.bridge = CvBridge()
        self.camera_geometry = CameraGeometry(self.config)
        self.depth_estimator = TomatoDepthEstimator(self.config)
        self.logged_estimate_count = 0
        self.ground_truth_sample_count = 0
        self.sum_error_x_m = 0.0
        self.sum_error_y_m = 0.0
        self.sum_error_z_m = 0.0
        self.sum_squared_error_norm_m = 0.0

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )

        self.ripeness_sub = Subscriber(
            self,
            TomatoRipenessArray,
            self.ripeness_topic,
        )
        self.disparity_sub = Subscriber(
            self,
            DisparityImage,
            self.disparity_topic,
        )
        self.synchronizer = ApproximateTimeSynchronizer(
            [self.ripeness_sub, self.disparity_sub],
            queue_size=10,
            slop=self.sync_slop_sec,
        )
        self.synchronizer.registerCallback(self.synced_callback)

        self.csv_file = None
        self.csv_writer = None
        if str(self.output_csv):
            self.open_csv()

        target_text = (
            "all detections"
            if self.target_detection_id < 0
            else f"detection id={self.target_detection_id}"
        )
        self.get_logger().info(
            "Tomato position accuracy node started for "
            f"{target_text}; writing CSV to {self.output_csv}"
        )
        self.get_logger().info(
            "Camera pose relative to robot origin: "
            f"x={self.config.camera_origin_x_m:.4f} m, "
            f"y={self.config.camera_origin_y_m:.4f} m, "
            f"z={self.config.camera_origin_z_m:.4f} m, "
            f"pitch_down={self.config.camera_pitch_down_degrees:.2f} deg"
        )
        if self.use_ground_truth and self.target_detection_id < 0:
            self.get_logger().warn(
                "Ground truth is enabled for all detections. Set "
                "target_detection_id when measuring one fixed tomato."
            )

    def open_csv(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.output_csv.exists()
        mode = "a" if self.append_csv else "w"
        self.csv_file = self.output_csv.open(mode, newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=self.CSV_FIELDS,
        )
        if not self.append_csv or not file_exists:
            self.csv_writer.writeheader()
            self.csv_file.flush()

    def camera_info_callback(self, message: CameraInfo) -> None:
        intrinsics = self.camera_geometry.update_intrinsics(message)
        self.get_logger().debug(
            "Updated camera intrinsics: "
            f"fx={intrinsics.focal_x_px:.2f}, "
            f"fy={intrinsics.focal_y_px:.2f}, "
            f"cx={intrinsics.principal_x_px:.2f}, "
            f"cy={intrinsics.principal_y_px:.2f}"
        )

    def synced_callback(
        self,
        ripeness_message: TomatoRipenessArray,
        disparity_message: DisparityImage,
    ) -> None:
        if self.camera_geometry.intrinsics is None:
            self.get_logger().warn(
                f"Waiting for camera intrinsics on {self.camera_info_topic}"
            )
            return

        disparity_image = self.bridge.imgmsg_to_cv2(
            disparity_message.image,
            desired_encoding="32FC1",
        )

        for detection in ripeness_message.ripenesses:
            detection_id = int(detection.detection_id)
            if (
                self.target_detection_id >= 0
                and detection_id != self.target_detection_id
            ):
                continue

            self.process_detection(
                ripeness_message,
                disparity_image,
                disparity_message,
                detection,
            )

    def process_detection(
        self,
        ripeness_message: TomatoRipenessArray,
        disparity_image,
        disparity_message: DisparityImage,
        detection,
    ) -> None:
        detection_id = int(detection.detection_id)
        depth = self.depth_estimator.estimate(
            disparity_image,
            disparity_message,
            detection,
        )

        if depth is None:
            self.get_logger().warn(
                f"id={detection_id}: invalid bbox/ROI; skipping"
            )
            return

        if not depth.is_valid or depth.optical_depth_m is None:
            self.get_logger().warn(
                f"id={detection_id}: unreliable disparity "
                f"valid={depth.valid_pixel_count}/"
                f"{depth.total_pixel_count} "
                f"ratio={depth.valid_pixel_ratio:.3f}; skipping"
            )
            return

        camera_point = self.camera_geometry.back_project_pixel(
            depth.roi.center_u_px,
            depth.roi.center_v_px,
            depth.optical_depth_m,
        )
        if camera_point is None:
            self.get_logger().warn(
                f"id={detection_id}: camera intrinsics unavailable; skipping"
            )
            return

        origin_point = self.camera_geometry.transform_camera_point_to_origin(
            camera_point
        )
        errors = self.compute_errors(origin_point)
        running_errors = self.update_running_errors(errors)
        stamp_sec = (
            float(ripeness_message.header.stamp.sec)
            + float(ripeness_message.header.stamp.nanosec) * 1e-9
        )

        row = {
            "stamp_sec": f"{stamp_sec:.9f}",
            "detection_id": detection_id,
            "ripeness": detection.final_ripeness,
            "bbox_x1": int(detection.x1),
            "bbox_y1": int(detection.y1),
            "bbox_x2": int(detection.x2),
            "bbox_y2": int(detection.y2),
            "sample_index": self.logged_estimate_count + 1,
            "roi_center_u_px": depth.roi.center_u_px,
            "roi_center_v_px": depth.roi.center_v_px,
            "valid_pixel_count": depth.valid_pixel_count,
            "total_pixel_count": depth.total_pixel_count,
            "valid_pixel_ratio": f"{depth.valid_pixel_ratio:.6f}",
            "median_disparity_px": self.format_optional_float(
                depth.median_disparity_px
            ),
            "mean_disparity_px": self.format_optional_float(
                depth.mean_disparity_px
            ),
            "surface_disparity_px": self.format_optional_float(
                depth.surface_disparity_px
            ),
            "optical_depth_m": f"{depth.optical_depth_m:.6f}",
            "camera_x_m": f"{camera_point.x_m:.6f}",
            "camera_y_m": f"{camera_point.y_m:.6f}",
            "camera_z_m": f"{camera_point.z_m:.6f}",
            "origin_x_m": f"{origin_point.x_m:.6f}",
            "origin_y_m": f"{origin_point.y_m:.6f}",
            "origin_z_m": f"{origin_point.z_m:.6f}",
            "truth_x_m": self.truth_value("x_m"),
            "truth_y_m": self.truth_value("y_m"),
            "truth_z_m": self.truth_value("z_m"),
            "error_x_m": self.format_optional_float(errors["x"]),
            "error_y_m": self.format_optional_float(errors["y"]),
            "error_z_m": self.format_optional_float(errors["z"]),
            "error_norm_m": self.format_optional_float(errors["norm"]),
            "mean_error_x_m": self.format_optional_float(running_errors["mean_x"]),
            "mean_error_y_m": self.format_optional_float(running_errors["mean_y"]),
            "mean_error_z_m": self.format_optional_float(running_errors["mean_z"]),
            "rmse_m": self.format_optional_float(running_errors["rmse"]),
        }

        if self.csv_writer is not None:
            self.csv_writer.writerow(row)
            self.csv_file.flush()

        self.logged_estimate_count += 1
        if self.logged_estimate_count % self.log_every_n == 0:
            self.log_estimate(
                detection_id,
                detection,
                depth,
                origin_point,
                errors,
                running_errors,
            )

    def compute_errors(self, origin_point: Point3D) -> dict[str, float | None]:
        if not self.use_ground_truth:
            return {
                "x": None,
                "y": None,
                "z": None,
                "norm": None,
            }

        error_x = origin_point.x_m - self.truth_point.x_m
        error_y = origin_point.y_m - self.truth_point.y_m
        error_z = origin_point.z_m - self.truth_point.z_m
        return {
            "x": error_x,
            "y": error_y,
            "z": error_z,
            "norm": math.sqrt(
                error_x * error_x
                + error_y * error_y
                + error_z * error_z
            ),
        }

    def update_running_errors(
        self,
        errors: dict[str, float | None],
    ) -> dict[str, float | None]:
        if not self.use_ground_truth:
            return {
                "mean_x": None,
                "mean_y": None,
                "mean_z": None,
                "rmse": None,
            }

        error_x = float(errors["x"])
        error_y = float(errors["y"])
        error_z = float(errors["z"])
        error_norm = float(errors["norm"])

        self.ground_truth_sample_count += 1
        self.sum_error_x_m += error_x
        self.sum_error_y_m += error_y
        self.sum_error_z_m += error_z
        self.sum_squared_error_norm_m += error_norm * error_norm

        sample_count = float(self.ground_truth_sample_count)
        return {
            "mean_x": self.sum_error_x_m / sample_count,
            "mean_y": self.sum_error_y_m / sample_count,
            "mean_z": self.sum_error_z_m / sample_count,
            "rmse": math.sqrt(
                self.sum_squared_error_norm_m / sample_count
            ),
        }

    def truth_value(self, axis_name: str) -> str:
        if not self.use_ground_truth:
            return ""
        return f"{getattr(self.truth_point, axis_name):.6f}"

    @staticmethod
    def format_optional_float(value: float | None) -> str:
        return "" if value is None else f"{float(value):.6f}"

    def log_estimate(
        self,
        detection_id: int,
        detection,
        depth,
        origin_point: Point3D,
        errors: dict[str, float | None],
        running_errors: dict[str, float | None],
    ) -> None:
        message = (
            f"id={detection_id}, ripeness={detection.final_ripeness}, "
            f"origin=("
            f"x={origin_point.x_m:.3f}, "
            f"y={origin_point.y_m:.3f}, "
            f"z={origin_point.z_m:.3f}) m, "
            f"depth={depth.optical_depth_m:.3f} m, "
            f"surface_disp={depth.surface_disparity_px:.2f} px, "
            f"median_disp={depth.median_disparity_px:.2f} px, "
            f"valid={depth.valid_pixel_count}/{depth.total_pixel_count}"
        )

        if self.use_ground_truth:
            message += (
                ", error=("
                f"dx={errors['x']:.3f}, "
                f"dy={errors['y']:.3f}, "
                f"dz={errors['z']:.3f}, "
                f"norm={errors['norm']:.3f}) m, "
                f"mean_dz={running_errors['mean_z']:.3f} m, "
                f"rmse={running_errors['rmse']:.3f} m"
            )

        self.get_logger().info(message)

    def destroy_node(self) -> None:
        if self.csv_file is not None:
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TomatoPositionAccuracyNode()

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
