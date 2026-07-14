import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        self.declare_parameter("camera_id", 0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 10.0)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter(
            "camera_info_topic",
            "/camera/camera_info",
        )
        self.declare_parameter("frame_id", "camera_frame")
        self.declare_parameter("calibration_file", "")

        self.camera_id = int(
            self.get_parameter("camera_id").value
        )
        self.width = int(
            self.get_parameter("width").value
        )
        self.height = int(
            self.get_parameter("height").value
        )
        self.fps = float(
            self.get_parameter("fps").value
        )
        self.image_topic = str(
            self.get_parameter("image_topic").value
        )
        self.camera_info_topic = str(
            self.get_parameter("camera_info_topic").value
        )
        self.frame_id = str(
            self.get_parameter("frame_id").value
        )
        self.calibration_file = str(
            self.get_parameter("calibration_file").value
        )

        if self.width <= 0:
            raise ValueError("width must be greater than 0")

        if self.height <= 0:
            raise ValueError("height must be greater than 0")

        if self.fps <= 0.0:
            raise ValueError("fps must be greater than 0")

        self.bridge = CvBridge()
        self.camera_info_template = self.load_camera_info(
            self.calibration_file
        )

        # Sensor-data QoS is appropriate for camera streams. If the system
        # falls behind, dropping an old frame is better than building latency.
        self.image_pub = self.create_publisher(
            Image,
            self.image_topic,
            qos_profile_sensor_data,
        )
        self.camera_info_pub = self.create_publisher(
            CameraInfo,
            self.camera_info_topic,
            qos_profile_sensor_data,
        )

        cmd = [
            "rpicam-vid",
            "--camera",
            str(self.camera_id),
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--framerate",
            str(self.fps),
            "--codec",
            "mjpeg",
            "--nopreview",
            "--buffer-count",
            "4",
            "-t",
            "0",
            "-o",
            "-",
        ]

        self.get_logger().info(
            "Starting camera command: " + " ".join(cmd)
        )

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,

            # Do not leave stderr connected to an unread PIPE. A full stderr
            # pipe can block rpicam-vid and stop the image stream.
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

        if self.proc.stdout is None:
            raise RuntimeError(
                "Failed to open rpicam-vid stdout pipe"
            )

        self.buffer = bytearray()

        # The reader thread continuously drains the MJPEG pipe. The ROS timer
        # publishes only the newest complete frame so stale frames do not
        # accumulate when the rest of the pipeline is under load.
        self.frame_lock = threading.Lock()
        self.latest_jpg = None
        self.latest_frame_sequence = 0
        self.last_published_sequence = 0
        self.stop_event = threading.Event()

        self.reader_thread = threading.Thread(
            target=self.reader_loop,
            name=f"camera_{self.camera_id}_mjpeg_reader",
            daemon=True,
        )
        self.reader_thread.start()

        self.process_exit_logged = False

        self.timer = self.create_timer(
            1.0 / self.fps,
            self.timer_callback,
        )

        self.get_logger().info(
            f"Publishing {self.image_topic} and "
            f"{self.camera_info_topic} at up to "
            f"{self.fps:.1f} FPS"
        )

    def load_camera_info(self, path):
        msg = CameraInfo()
        msg.width = self.width
        msg.height = self.height
        msg.distortion_model = "plumb_bob"

        if not path:
            self.get_logger().warn(
                "No calibration file provided. "
                "Publishing placeholder CameraInfo."
            )
            return msg

        path = Path(path)

        if not path.exists():
            self.get_logger().warn(
                f"Calibration file not found: {path}"
            )
            return msg

        with open(path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        msg.width = int(data["image_width"])
        msg.height = int(data["image_height"])

        msg.distortion_model = data.get(
            "distortion_model",
            "plumb_bob",
        )

        msg.k = [
            float(x)
            for x in data["camera_matrix"]["data"]
        ]

        msg.d = [
            float(x)
            for x in data["distortion_coefficients"]["data"]
        ]

        msg.r = [
            float(x)
            for x in data["rectification_matrix"]["data"]
        ]

        msg.p = [
            float(x)
            for x in data["projection_matrix"]["data"]
        ]

        self.get_logger().info(
            f"Loaded calibration from {path}"
        )

        return msg

    def make_camera_info(self, stamp):
        msg = CameraInfo()

        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id

        msg.width = self.camera_info_template.width
        msg.height = self.camera_info_template.height
        msg.distortion_model = (
            self.camera_info_template.distortion_model
        )

        msg.k = list(self.camera_info_template.k)
        msg.d = list(self.camera_info_template.d)
        msg.r = list(self.camera_info_template.r)
        msg.p = list(self.camera_info_template.p)

        return msg

    def reader_loop(self):
        """
        Continuously read the MJPEG byte stream and keep the newest complete JPEG.

        Pipe reading is done outside the ROS executor so a blocking camera read
        cannot prevent subscriptions, services, or timers from running.
        """

        while not self.stop_event.is_set():
            try:
                chunk = self.proc.stdout.read(65536)
            except (OSError, ValueError):
                break

            if not chunk:
                if self.proc.poll() is not None:
                    break

                continue

            self.buffer.extend(chunk)

            newest_complete_jpg = None

            # Extract every complete JPEG currently in the buffer. Keep only
            # the newest one because older frames are already stale.
            while True:
                start = self.buffer.find(b"\xff\xd8")

                if start == -1:
                    # No JPEG start marker exists. Prevent unlimited growth if
                    # malformed or unrelated bytes enter the stream.
                    if len(self.buffer) > 1048576:
                        self.buffer.clear()

                    break

                end = self.buffer.find(
                    b"\xff\xd9",
                    start + 2,
                )

                if end == -1:
                    # Preserve the incomplete JPEG but discard bytes before it.
                    if start > 0:
                        del self.buffer[:start]

                    break

                newest_complete_jpg = bytes(
                    self.buffer[start:end + 2]
                )

                del self.buffer[:end + 2]

            if newest_complete_jpg is not None:
                with self.frame_lock:
                    self.latest_jpg = newest_complete_jpg
                    self.latest_frame_sequence += 1

    def timer_callback(self):
        if self.proc.poll() is not None:
            if not self.process_exit_logged:
                self.get_logger().error(
                    "rpicam-vid process exited with code "
                    f"{self.proc.returncode}"
                )

                self.process_exit_logged = True

            return

        with self.frame_lock:
            if (
                self.latest_jpg is None
                or self.latest_frame_sequence
                == self.last_published_sequence
            ):
                return

            # bytes is immutable, so it is safe to use after releasing
            # the lock.
            jpg = self.latest_jpg
            frame_sequence = self.latest_frame_sequence

        arr = np.frombuffer(
            jpg,
            dtype=np.uint8,
        )

        frame = cv2.imdecode(
            arr,
            cv2.IMREAD_COLOR,
        )

        if frame is None:
            self.get_logger().warn(
                "Failed to decode JPEG frame"
            )
            return

        stamp = self.get_clock().now().to_msg()

        image_msg = self.bridge.cv2_to_imgmsg(
            frame,
            encoding="bgr8",
        )

        image_msg.header.stamp = stamp
        image_msg.header.frame_id = self.frame_id

        # The matching Image and CameraInfo use the exact same timestamp.
        camera_info_msg = self.make_camera_info(stamp)

        self.image_pub.publish(image_msg)
        self.camera_info_pub.publish(camera_info_msg)

        self.last_published_sequence = frame_sequence

    def destroy_node(self):
        self.stop_event.set()

        if (
            hasattr(self, "proc")
            and self.proc.poll() is None
        ):
            self.proc.terminate()

            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2)

        if (
            hasattr(self, "reader_thread")
            and self.reader_thread.is_alive()
        ):
            self.reader_thread.join(timeout=1.0)

        if (
            hasattr(self, "proc")
            and self.proc.stdout is not None
        ):
            self.proc.stdout.close()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = CameraNode()

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