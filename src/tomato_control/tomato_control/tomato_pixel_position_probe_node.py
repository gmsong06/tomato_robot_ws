from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Lock, Thread
from urllib.parse import parse_qs, urlparse

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
        self.declare_parameter("web_server_enabled", True)
        self.declare_parameter("web_bind_host", "127.0.0.1")
        self.declare_parameter("web_port", 8765)
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
        self.web_server_enabled = bool(
            self.get_parameter("web_server_enabled").value
        )
        self.web_bind_host = str(self.get_parameter("web_bind_host").value)
        self.web_port = int(self.get_parameter("web_port").value)
        self.parameter_pixel_u = int(self.get_parameter("pixel_u").value)
        self.parameter_pixel_v = int(self.get_parameter("pixel_v").value)
        self.exit_after_first_probe = bool(
            self.get_parameter("exit_after_first_probe").value
        )

        if self.sync_slop_sec < 0.0:
            raise ValueError("sync_slop_sec must be nonnegative")

        if self.display_scale <= 0.0:
            raise ValueError("display_scale must be greater than 0")

        if not 0 < self.web_port <= 65535:
            raise ValueError("web_port must be in the range [1, 65535]")

        self.bridge = CvBridge()
        self.camera_geometry = CameraGeometry(self.config)
        self.depth_estimator = TomatoDepthEstimator(self.config)

        self.frame_lock = Lock()
        self.latest_display_image: np.ndarray | None = None
        self.latest_disparity_image: np.ndarray | None = None
        self.latest_disparity_message: DisparityImage | None = None
        self.latest_image_shape: tuple[int, int] | None = None
        self.last_probe_pixel: tuple[int, int] | None = None
        self.last_probe_roi: BoundingBox | None = None
        self.last_probe_result: dict | None = None
        self.completed_parameter_probe = False
        self.gui_available = self.show_window
        self.logged_waiting_for_intrinsics = False
        self.web_server: ThreadingHTTPServer | None = None
        self.web_server_thread: Thread | None = None

        if self.show_window:
            self.create_window()

        if self.web_server_enabled and not self.gui_available:
            self.start_web_server()

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
            f"{2 * self.probe_half_window_px + 1} px box around the pixel; "
            "click output includes median and P75 estimates."
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
                "Could not create OpenCV click window; falling back to "
                f"browser mode when enabled: {error}"
            )

    def start_web_server(self) -> None:
        handler_class = self.make_web_request_handler()

        try:
            self.web_server = ThreadingHTTPServer(
                (self.web_bind_host, self.web_port),
                handler_class,
            )
        except OSError as error:
            self.get_logger().error(
                f"Could not start browser pixel probe on "
                f"{self.web_bind_host}:{self.web_port}: {error}"
            )
            return

        self.web_server_thread = Thread(
            target=self.web_server.serve_forever,
            daemon=True,
        )
        self.web_server_thread.start()

        display_host = (
            "127.0.0.1"
            if self.web_bind_host in {"", "0.0.0.0"}
            else self.web_bind_host
        )
        self.get_logger().info(
            "Browser pixel probe available at "
            f"http://{display_host}:{self.web_port}"
        )

    def make_web_request_handler(self):
        node = self

        class PixelProbeRequestHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed_url = urlparse(self.path)

                if parsed_url.path == "/":
                    self.respond_html(node.browser_html())
                    return

                if parsed_url.path == "/frame.jpg":
                    frame_jpeg = node.render_frame_jpeg()
                    if frame_jpeg is None:
                        self.respond_text(
                            HTTPStatus.SERVICE_UNAVAILABLE,
                            "No image frame received yet",
                        )
                        return
                    self.respond_bytes("image/jpeg", frame_jpeg)
                    return

                if parsed_url.path == "/probe":
                    self.handle_probe(parsed_url.query)
                    return

                if parsed_url.path == "/latest":
                    self.respond_json(node.latest_probe_result())
                    return

                self.respond_text(HTTPStatus.NOT_FOUND, "Not found")

            def handle_probe(self, query: str) -> None:
                query_values = parse_qs(query)
                try:
                    pixel_u = int(query_values.get("u", [""])[0])
                    pixel_v = int(query_values.get("v", [""])[0])
                except ValueError:
                    self.respond_json(
                        {
                            "ok": False,
                            "message": "Expected integer u and v parameters",
                        },
                        HTTPStatus.BAD_REQUEST,
                    )
                    return

                result = node.probe_pixel(pixel_u, pixel_v)
                status = (
                    HTTPStatus.OK
                    if result.get("ok")
                    else HTTPStatus.BAD_REQUEST
                )
                self.respond_json(result, status)

            def respond_html(self, html: str) -> None:
                self.respond_bytes(
                    "text/html; charset=utf-8",
                    html.encode("utf-8"),
                )

            def respond_json(
                self,
                data: dict,
                status: HTTPStatus = HTTPStatus.OK,
            ) -> None:
                body = json.dumps(data).encode("utf-8")
                self.send_response(status)
                self.send_header(
                    "Content-Type",
                    "application/json; charset=utf-8",
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def respond_bytes(self, content_type: str, body: bytes) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def respond_text(
                self,
                status: HTTPStatus,
                message: str,
            ) -> None:
                body = message.encode("utf-8")
                self.send_response(status)
                self.send_header(
                    "Content-Type",
                    "text/plain; charset=utf-8",
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args) -> None:
                return

        return PixelProbeRequestHandler

    def browser_html(self) -> str:
        return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tomato Pixel Probe</title>
  <style>
    body {
      margin: 0;
      font-family: system-ui, sans-serif;
      background: #111;
      color: #eee;
    }
    main {
      display: grid;
      gap: 12px;
      padding: 12px;
    }
    #frame {
      max-width: 100%;
      height: auto;
      border: 1px solid #444;
      cursor: crosshair;
      image-rendering: auto;
    }
    pre {
      margin: 0;
      padding: 10px;
      background: #1e1e1e;
      border: 1px solid #444;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <main>
    <img id="frame" alt="left rectified camera frame" src="/frame.jpg">
    <pre id="result">Click a pixel in the image.</pre>
  </main>
  <script>
    const frame = document.getElementById("frame");
    const result = document.getElementById("result");

    function refreshFrame() {
      frame.src = "/frame.jpg?t=" + Date.now();
    }

    function formatPoint(name, point) {
      if (!point) {
        return name + ": n/a";
      }
      return name + ": (" +
        point.x_m.toFixed(4) + ", " +
        point.y_m.toFixed(4) + ", " +
        point.z_m.toFixed(4) + ") m";
    }

    function showResult(data) {
      if (!data.ok) {
        result.textContent = data.message;
        return;
      }

      result.textContent =
        "pixel: (" + data.pixel_u + ", " + data.pixel_v + ")\\n" +
        formatPoint("P75 origin", data.p75.origin) + "\\n" +
        formatPoint("median origin", data.median.origin) + "\\n" +
        formatPoint("P75 camera", data.p75.camera) + "\\n" +
        formatPoint("median camera", data.median.camera) + "\\n" +
        "P75 depth: " + formatNumber(data.p75.depth_m, 4) + " m\\n" +
        "median depth: " + formatNumber(data.median.depth_m, 4) + " m\\n" +
        "raw depth: " + data.raw_depth_m + "\\n" +
        "P75 disparity: " +
          formatNumber(data.p75.disparity_px, 2) + " px\\n" +
        "median disparity: " +
          formatNumber(data.median.disparity_px, 2) + " px\\n" +
        "configured estimator P" + data.surface_percentile + ": " +
          formatNumber(data.surface_disparity_px, 2) + " px\\n" +
        "raw disparity: " + data.raw_disparity_px + "\\n" +
        "valid pixels: " + data.valid_pixel_count +
          "/" + data.total_pixel_count +
          " (" + data.valid_pixel_ratio.toFixed(3) + ")";
    }

    function formatNumber(value, digits) {
      if (typeof value !== "number") {
        return value;
      }
      return value.toFixed(digits);
    }

    frame.addEventListener("click", async (event) => {
      const rect = frame.getBoundingClientRect();
      const u = Math.round(
        (event.clientX - rect.left) * frame.naturalWidth / rect.width
      );
      const v = Math.round(
        (event.clientY - rect.top) * frame.naturalHeight / rect.height
      );
      const response = await fetch("/probe?u=" + u + "&v=" + v);
      showResult(await response.json());
      refreshFrame();
    });

    setInterval(refreshFrame, 250);
  </script>
</body>
</html>
"""

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

        with self.frame_lock:
            self.latest_display_image = display_image.copy()
            self.latest_disparity_image = disparity_image.copy()
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

    def render_frame_jpeg(self) -> bytes | None:
        with self.frame_lock:
            if self.latest_display_image is None:
                return None

            overlay = self.latest_display_image.copy()
            last_probe_roi = self.last_probe_roi
            last_probe_pixel = self.last_probe_pixel

        self.draw_probe_overlay(overlay, last_probe_roi, last_probe_pixel)

        ok, encoded_image = cv2.imencode(
            ".jpg",
            overlay,
            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )
        if not ok:
            return None

        return encoded_image.tobytes()

    def latest_probe_result(self) -> dict:
        with self.frame_lock:
            if self.last_probe_result is None:
                return {
                    "ok": False,
                    "message": "No pixel has been probed yet",
                }
            return dict(self.last_probe_result)

    def show_display_image(self, display_image: np.ndarray) -> None:
        overlay = display_image.copy()

        with self.frame_lock:
            last_probe_roi = self.last_probe_roi
            last_probe_pixel = self.last_probe_pixel

        self.draw_probe_overlay(overlay, last_probe_roi, last_probe_pixel)

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

    @staticmethod
    def draw_probe_overlay(
        overlay: np.ndarray,
        last_probe_roi: BoundingBox | None,
        last_probe_pixel: tuple[int, int] | None,
    ) -> None:
        if last_probe_roi is not None:
            cv2.rectangle(
                overlay,
                (last_probe_roi.x_min, last_probe_roi.y_min),
                (last_probe_roi.x_max, last_probe_roi.y_max),
                (0, 255, 255),
                1,
            )

        if last_probe_pixel is not None:
            cv2.drawMarker(
                overlay,
                last_probe_pixel,
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
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

    def probe_pixel(self, pixel_u: int, pixel_v: int) -> dict:
        with self.frame_lock:
            disparity_image = (
                None
                if self.latest_disparity_image is None
                else self.latest_disparity_image.copy()
            )
            disparity_message = self.latest_disparity_message

        if disparity_image is None:
            return self.fail_probe("No synchronized disparity image yet")

        if disparity_message is None:
            return self.fail_probe("No synchronized disparity message yet")

        if self.camera_geometry.intrinsics is None:
            if not self.logged_waiting_for_intrinsics:
                self.get_logger().warn(
                    f"Waiting for camera intrinsics on {self.camera_info_topic}"
                )
                self.logged_waiting_for_intrinsics = True
            return self.fail_probe("No camera intrinsics received yet")

        image_height, image_width = disparity_image.shape[:2]
        if (
            pixel_u < 0
            or pixel_v < 0
            or pixel_u >= image_width
            or pixel_v >= image_height
        ):
            return self.fail_probe(
                f"Pixel ({pixel_u}, {pixel_v}) is outside image bounds "
                f"{image_width}x{image_height}"
            )

        detection = self.pixel_to_detection(pixel_u, pixel_v)
        depth = self.depth_estimator.estimate(
            disparity_image,
            disparity_message,
            detection,
        )

        if depth is None:
            return self.fail_probe(
                f"Pixel ({pixel_u}, {pixel_v}) produced an empty ROI"
            )

        with self.frame_lock:
            self.last_probe_pixel = (pixel_u, pixel_v)
            self.last_probe_roi = depth.roi

        if not depth.is_valid or depth.optical_depth_m is None:
            return self.fail_probe(
                f"Pixel ({pixel_u}, {pixel_v}) has unreliable disparity: "
                f"valid={depth.valid_pixel_count}/"
                f"{depth.total_pixel_count}, "
                f"ratio={depth.valid_pixel_ratio:.3f}"
            )

        camera_point = self.camera_geometry.back_project_pixel(
            pixel_u,
            pixel_v,
            depth.optical_depth_m,
        )
        if camera_point is None:
            return self.fail_probe("Camera intrinsics unavailable")

        origin_point = self.camera_geometry.transform_camera_point_to_origin(
            camera_point
        )
        median_depth_m = self.depth_from_disparity(
            disparity_message,
            depth.median_disparity_px,
        )
        median_camera_point, median_origin_point = self.points_from_depth(
            pixel_u,
            pixel_v,
            median_depth_m,
        )
        p75_disparity_px = self.disparity_percentile_in_roi(
            disparity_image,
            depth.roi,
            75.0,
        )
        p75_depth_m = self.depth_from_disparity(
            disparity_message,
            p75_disparity_px,
        )
        p75_camera_point, p75_origin_point = self.points_from_depth(
            pixel_u,
            pixel_v,
            p75_depth_m,
        )
        raw_disparity_px = self.raw_disparity_at(
            disparity_image,
            pixel_u,
            pixel_v,
        )
        raw_depth_m = self.depth_from_disparity(
            disparity_message,
            raw_disparity_px,
        )

        result = self.make_probe_result(
            pixel_u,
            pixel_v,
            raw_disparity_px,
            raw_depth_m,
            depth,
            camera_point,
            origin_point,
            median_depth_m,
            median_camera_point,
            median_origin_point,
            p75_disparity_px,
            p75_depth_m,
            p75_camera_point,
            p75_origin_point,
        )
        with self.frame_lock:
            self.last_probe_result = result

        self.log_probe(result)

        if self.exit_after_first_probe:
            rclpy.shutdown()

        return result

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

    def raw_disparity_at(
        self,
        disparity_image: np.ndarray,
        pixel_u: int,
        pixel_v: int,
    ) -> float | None:
        raw_disparity = float(disparity_image[pixel_v, pixel_u])
        if not np.isfinite(raw_disparity):
            return None

        if raw_disparity <= self.config.minimum_valid_disparity_px:
            return None

        if raw_disparity >= self.config.maximum_valid_disparity_px:
            return None

        return raw_disparity

    def depth_from_disparity(
        self,
        disparity_message: DisparityImage,
        disparity_px: float | None,
    ) -> float | None:
        if disparity_px is None:
            return None

        return abs(disparity_message.f * disparity_message.t) / disparity_px

    def disparity_percentile_in_roi(
        self,
        disparity_image: np.ndarray,
        roi: BoundingBox,
        percentile: float,
    ) -> float | None:
        disparity_roi = disparity_image[
            roi.y_min:roi.y_max,
            roi.x_min:roi.x_max,
        ]
        valid_mask = (
            np.isfinite(disparity_roi)
            & (disparity_roi > self.config.minimum_valid_disparity_px)
            & (disparity_roi < self.config.maximum_valid_disparity_px)
        )
        if not np.any(valid_mask):
            return None

        return float(np.percentile(disparity_roi[valid_mask], percentile))

    def points_from_depth(
        self,
        pixel_u: int,
        pixel_v: int,
        depth_m: float | None,
    ) -> tuple[Point3D | None, Point3D | None]:
        if depth_m is None:
            return None, None

        camera_point = self.camera_geometry.back_project_pixel(
            pixel_u,
            pixel_v,
            depth_m,
        )
        if camera_point is None:
            return None, None

        origin_point = self.camera_geometry.transform_camera_point_to_origin(
            camera_point
        )
        return camera_point, origin_point

    def fail_probe(self, message: str) -> dict:
        result = {
            "ok": False,
            "message": message,
        }
        with self.frame_lock:
            self.last_probe_result = result

        self.get_logger().warn(message)
        return result

    def make_probe_result(
        self,
        pixel_u: int,
        pixel_v: int,
        raw_disparity_px: float | None,
        raw_depth_m: float | None,
        depth,
        camera_point: Point3D,
        origin_point: Point3D,
        median_depth_m: float | None,
        median_camera_point: Point3D | None,
        median_origin_point: Point3D | None,
        p75_disparity_px: float | None,
        p75_depth_m: float | None,
        p75_camera_point: Point3D | None,
        p75_origin_point: Point3D | None,
    ) -> dict:
        return {
            "ok": True,
            "pixel_u": pixel_u,
            "pixel_v": pixel_v,
            "origin": self.point_to_dict(origin_point),
            "camera": self.point_to_dict(camera_point),
            "depth_m": depth.optical_depth_m,
            "raw_depth_m": self.optional_number(raw_depth_m),
            "surface_percentile": self.config.surface_disparity_percentile,
            "surface_disparity_px": depth.surface_disparity_px,
            "median_disparity_px": depth.median_disparity_px,
            "mean_disparity_px": depth.mean_disparity_px,
            "median": {
                "disparity_px": self.optional_number(
                    depth.median_disparity_px
                ),
                "depth_m": self.optional_number(median_depth_m),
                "camera": self.optional_point_to_dict(median_camera_point),
                "origin": self.optional_point_to_dict(median_origin_point),
            },
            "p75": {
                "disparity_px": self.optional_number(p75_disparity_px),
                "depth_m": self.optional_number(p75_depth_m),
                "camera": self.optional_point_to_dict(p75_camera_point),
                "origin": self.optional_point_to_dict(p75_origin_point),
            },
            "raw_disparity_px": self.optional_number(raw_disparity_px),
            "valid_pixel_count": depth.valid_pixel_count,
            "total_pixel_count": depth.total_pixel_count,
            "valid_pixel_ratio": depth.valid_pixel_ratio,
            "roi": {
                "x_min": depth.roi.x_min,
                "y_min": depth.roi.y_min,
                "x_max": depth.roi.x_max,
                "y_max": depth.roi.y_max,
            },
        }

    @staticmethod
    def point_to_dict(point: Point3D) -> dict[str, float]:
        return {
            "x_m": point.x_m,
            "y_m": point.y_m,
            "z_m": point.z_m,
        }

    @staticmethod
    def optional_point_to_dict(
        point: Point3D | None,
    ) -> dict[str, float] | None:
        if point is None:
            return None

        return TomatoPixelPositionProbeNode.point_to_dict(point)

    @staticmethod
    def optional_number(value: float | None) -> float | str:
        return "n/a" if value is None else value

    def log_probe(
        self,
        result: dict,
    ) -> None:
        p75 = result["p75"]
        median = result["median"]
        self.get_logger().info(
            f"pixel=({result['pixel_u']}, {result['pixel_v']}), "
            f"p75_origin_xyz_m={self.format_point(p75['origin'])}, "
            f"median_origin_xyz_m={self.format_point(median['origin'])}, "
            f"p75_camera_xyz_m={self.format_point(p75['camera'])}, "
            f"median_camera_xyz_m={self.format_point(median['camera'])}, "
            f"p75_depth_m={self.format_float(p75['depth_m'])}, "
            f"median_depth_m={self.format_float(median['depth_m'])}, "
            f"raw_depth_m={self.format_float(result['raw_depth_m'])}, "
            f"p75_disp_px={self.format_float(p75['disparity_px'])}, "
            f"median_disp_px={self.format_float(median['disparity_px'])}, "
            f"estimator_p{result['surface_percentile']:g}_disp_px="
            f"{result['surface_disparity_px']:.2f}, "
            f"raw_disp_px={self.format_float(result['raw_disparity_px'])}, "
            f"valid={result['valid_pixel_count']}/"
            f"{result['total_pixel_count']}"
        )

    @staticmethod
    def format_float(value: float | str) -> str:
        return value if isinstance(value, str) else f"{value:.4f}"

    @staticmethod
    def format_point(point: dict[str, float] | None) -> str:
        if point is None:
            return "n/a"

        return (
            f"({point['x_m']:.4f}, "
            f"{point['y_m']:.4f}, "
            f"{point['z_m']:.4f})"
        )

    def destroy_node(self) -> None:
        if self.web_server is not None:
            self.web_server.shutdown()
            self.web_server.server_close()

        if self.show_window and self.gui_available:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
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
