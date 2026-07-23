from __future__ import annotations

from dataclasses import dataclass

import cv2
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from stereo_msgs.msg import DisparityImage

from tomato_control.camera_geometry import CameraGeometry
from tomato_control.controller_config import ControllerConfig
from tomato_control.controller_models import BoundingBox, Point3D
from tomato_control.tomato_depth_estimator import TomatoDepthEstimator


@dataclass(frozen=True)
class PixelProbeDetection:
    """Small fake detection box centered on a clicked pixel."""

    x1: int
    y1: int
    x2: int
    y2: int


class TomatoPixelPositionProbeNode(Node):
    """Click a rectified left-image pixel and log its 3D position."""

    def __init__(self):
        super().__init__("tomato_pixel_position_probe_node")

        ControllerConfig.declare_parameters(self)
        self.config = ControllerConfig.from_node(self)

        self.declare_parameter("left_image_topic", "/stereo/left/image_rect")
        self.declare_parameter("camera_info_topic", "/stereo/left/camera_info")
        self.declare_parameter("disparity_topic", "/stereo/disparity")
        self.declare_parameter("sync_slop_sec", 0.15)
        self.declare_parameter("probe_half_window_px", 6)
        self.declare_parameter("show_window", True)
        self.declare_parameter("window_name", "tomato pixel position probe")
        self.declare_parameter("display_scale", 1.0)
        self.declare_parameter("pixel_u", -1)
        self.declare_parameter("pixel_v", -1)
        self.declare_parameter("exit_after_first_probe", False)

        self.left_image_topic = str(
            self.get_parameter("left_image_topic").value
        )
        self.camera_info_topic = str(
            self.get_parameter("camera_info_topic").value
        )
        self.disparity_topic = str(
            self.get_parameter("disparity_topic").value
        )
        self.sync_slop_sec = float(
            self.get_parameter("sync_slop_sec").value
        )
        self.probe_half_window_px = max(
            0,
            int(self.get_parameter("probe_half_window_px").value),
        )
        self.show_window = bool(self.get_parameter("show_window").value)
        self.window_name = str(self.get_parameter("window_name").value)
        self.display_scale = float(self.get_parameter("display_scale").value)
        self.parameter_pixel_u = int(self.get_parameter("pixel_u").value)
        self.parameter_pixel_v = int(self.get_parameter("pixel_v").value)
        self.exit_after_first_probe = bool(
            self.get_parameter("exit_after_first_probe").value
        )

        if self.sync_slop_sec < 0.0:
            raise ValueError("sync_slop_sec must be nonnegative")

        if self.display_scale <= 0.0:
            raise ValueError("display_scale must be greater than 0")

        self.bridge = CvBridge()
        self.camera_geometry = CameraGeometry(self.config)
        self.depth_estimator = TomatoDepthEstimator(self.config)

        self.latest_disparity_image: np.ndarray | None = None
        self.latest_disparity_message: DisparityImage | None = None
        self.latest_image_shape: tuple[int, int] | None = None
        self.last_probe_pixel: tuple[int, int] | None = None
        self.last_probe_roi: BoundingBox | None = None
        self.completed_parameter_probe = False
        self.gui_available = self.show_window
        self.logged_waiting_for_intrinsics = False

        if self.show_window:
            self.create_window()

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.left_image_sub = Subscriber(
            self,
            Image,
            self.left_image_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.disparity_sub = Subscriber(
            self,
            DisparityImage,
            self.disparity_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.synchronizer = ApproximateTimeSynchronizer(
            [self.left_image_sub, self.disparity_sub],
            queue_size=10,
            slop=self.sync_slop_sec,
        )
        self.synchronizer.registerCallback(self.synced_callback)

        self.get_logger().info(
            "Pixel probe started. Click the left rectified image to print "
            "the 3D point in camera and robot-origin frames."
        )
        self.get_logger().info(
            "Depth sampling uses TomatoDepthEstimator on a "
            f"{2 * self.probe_half_window_px + 1}x"
            f"{2 * self.probe_half_window_px + 1} px box around the pixel."
        )
        self.get_logger().info(
            "Camera pose relative to robot origin: "
            f"x={self.config.camera_origin_x_m:.4f} m, "
            f"y={self.config.camera_origin_y_m:.4f} m, "
            f"z={self.config.camera_origin_z_m:.4f} m, "
            f"pitch_down={self.config.camera_pitch_down_degrees:.2f} deg"
        )

    def create_window(self) -> None:
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
        except cv2.error as error:
            self.gui_available = False
            self.get_logger().error(
                "Could not create OpenCV click window. Use pixel_u/pixel_v "
                f"with show_window:=false on a headless machine: {error}"
            )

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
        image_message: Image,
        disparity_message: DisparityImage,
    ) -> None:
        display_image = self.image_message_to_bgr(image_message)
        disparity_image = self.bridge.imgmsg_to_cv2(
            disparity_message.image,
            desired_encoding="32FC1",
        )

        self.latest_disparity_image = disparity_image
        self.latest_disparity_message = disparity_message
        self.latest_image_shape = disparity_image.shape[:2]

        if self.gui_available:
            self.show_display_image(display_image)

        if self.should_run_parameter_probe():
            self.completed_parameter_probe = True
            self.probe_pixel(self.parameter_pixel_u, self.parameter_pixel_v)

    def image_message_to_bgr(self, image_message: Image) -> np.ndarray:
        try:
            return self.bridge.imgmsg_to_cv2(
                image_message,
                desired_encoding="bgr8",
            )
        except Exception:
            image = self.bridge.imgmsg_to_cv2(
                image_message,
                desired_encoding="passthrough",
            )

        if image.ndim == 2:
            if image.dtype != np.uint8:
                image = cv2.normalize(
                    image,
                    None,
                    alpha=0,
                    beta=255,
                    norm_type=cv2.NORM_MINMAX,
                ).astype(np.uint8)
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        return image.copy()

    def show_display_image(self, display_image: np.ndarray) -> None:
        overlay = display_image.copy()

        if self.last_probe_roi is not None:
            cv2.rectangle(
                overlay,
                (self.last_probe_roi.x_min, self.last_probe_roi.y_min),
                (self.last_probe_roi.x_max, self.last_probe_roi.y_max),
                (0, 255, 255),
                1,
            )

        if self.last_probe_pixel is not None:
            cv2.drawMarker(
                overlay,
                self.last_probe_pixel,
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
            )

        if self.display_scale != 1.0:
            overlay = cv2.resize(
                overlay,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
                interpolation=cv2.INTER_NEAREST,
            )

        try:
            cv2.imshow(self.window_name, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                self.get_logger().info("Closing pixel probe")
                rclpy.shutdown()
        except cv2.error as error:
            self.gui_available = False
            self.get_logger().error(
                "OpenCV display failed. Use pixel_u/pixel_v with "
                f"show_window:=false on a headless machine: {error}"
            )

    def mouse_callback(self, event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        pixel_u = int(round(float(x) / self.display_scale))
        pixel_v = int(round(float(y) / self.display_scale))
        self.probe_pixel(pixel_u, pixel_v)

    def should_run_parameter_probe(self) -> bool:
        return (
            not self.completed_parameter_probe
            and self.parameter_pixel_u >= 0
            and self.parameter_pixel_v >= 0
        )

    def probe_pixel(self, pixel_u: int, pixel_v: int) -> None:
        if self.latest_disparity_image is None:
            self.get_logger().warn("No synchronized disparity image yet")
            return

        if self.latest_disparity_message is None:
            self.get_logger().warn("No synchronized disparity message yet")
            return

        if self.camera_geometry.intrinsics is None:
            if not self.logged_waiting_for_intrinsics:
                self.get_logger().warn(
                    f"Waiting for camera intrinsics on {self.camera_info_topic}"
                )
                self.logged_waiting_for_intrinsics = True
            return

        image_height, image_width = self.latest_disparity_image.shape[:2]
        if (
            pixel_u < 0
            or pixel_v < 0
            or pixel_u >= image_width
            or pixel_v >= image_height
        ):
            self.get_logger().warn(
                f"Pixel ({pixel_u}, {pixel_v}) is outside image bounds "
                f"{image_width}x{image_height}"
            )
            return

        detection = self.pixel_to_detection(pixel_u, pixel_v)
        depth = self.depth_estimator.estimate(
            self.latest_disparity_image,
            self.latest_disparity_message,
            detection,
        )

        if depth is None:
            self.get_logger().warn(
                f"Pixel ({pixel_u}, {pixel_v}) produced an empty ROI"
            )
            return

        self.last_probe_pixel = (pixel_u, pixel_v)
        self.last_probe_roi = depth.roi

        if not depth.is_valid or depth.optical_depth_m is None:
            self.get_logger().warn(
                f"Pixel ({pixel_u}, {pixel_v}) has unreliable disparity: "
                f"valid={depth.valid_pixel_count}/"
                f"{depth.total_pixel_count}, "
                f"ratio={depth.valid_pixel_ratio:.3f}"
            )
            return

        camera_point = self.camera_geometry.back_project_pixel(
            pixel_u,
            pixel_v,
            depth.optical_depth_m,
        )
        if camera_point is None:
            self.get_logger().warn("Camera intrinsics unavailable")
            return

        origin_point = self.camera_geometry.transform_camera_point_to_origin(
            camera_point
        )
        raw_disparity_px = self.raw_disparity_at(pixel_u, pixel_v)
        raw_depth_m = self.depth_from_disparity(raw_disparity_px)

        self.log_probe(
            pixel_u,
            pixel_v,
            raw_disparity_px,
            raw_depth_m,
            depth,
            camera_point,
            origin_point,
        )

        if self.exit_after_first_probe:
            rclpy.shutdown()

    def pixel_to_detection(
        self,
        pixel_u: int,
        pixel_v: int,
    ) -> PixelProbeDetection:
        radius = self.probe_half_window_px
        return PixelProbeDetection(
            x1=pixel_u - radius,
            y1=pixel_v - radius,
            x2=pixel_u + radius + 1,
            y2=pixel_v + radius + 1,
        )

    def raw_disparity_at(self, pixel_u: int, pixel_v: int) -> float | None:
        if self.latest_disparity_image is None:
            return None

        raw_disparity = float(self.latest_disparity_image[pixel_v, pixel_u])
        if not np.isfinite(raw_disparity):
            return None

        if raw_disparity <= self.config.minimum_valid_disparity_px:
            return None

        if raw_disparity >= self.config.maximum_valid_disparity_px:
            return None

        return raw_disparity

    def depth_from_disparity(
        self,
        disparity_px: float | None,
    ) -> float | None:
        if disparity_px is None or self.latest_disparity_message is None:
            return None

        return (
            abs(
                self.latest_disparity_message.f
                * self.latest_disparity_message.t
            )
            / disparity_px
        )

    def log_probe(
        self,
        pixel_u: int,
        pixel_v: int,
        raw_disparity_px: float | None,
        raw_depth_m: float | None,
        depth,
        camera_point: Point3D,
        origin_point: Point3D,
    ) -> None:
        self.get_logger().info(
            f"pixel=({pixel_u}, {pixel_v}), "
            f"origin_xyz_m=("
            f"{origin_point.x_m:.4f}, "
            f"{origin_point.y_m:.4f}, "
            f"{origin_point.z_m:.4f}), "
            f"camera_xyz_m=("
            f"{camera_point.x_m:.4f}, "
            f"{camera_point.y_m:.4f}, "
            f"{camera_point.z_m:.4f}), "
            f"depth_m={depth.optical_depth_m:.4f}, "
            f"raw_depth_m={self.format_float(raw_depth_m)}, "
            f"surface_disp_px={depth.surface_disparity_px:.2f}, "
            f"median_disp_px={depth.median_disparity_px:.2f}, "
            f"raw_disp_px={self.format_float(raw_disparity_px)}, "
            f"valid={depth.valid_pixel_count}/{depth.total_pixel_count}"
        )

    @staticmethod
    def format_float(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    def destroy_node(self) -> None:
        if self.show_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TomatoPixelPositionProbeNode()

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
