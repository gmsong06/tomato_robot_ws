import subprocess
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        self.declare_parameter("camera_id", 0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("frame_id", "camera_frame")
        self.declare_parameter("calibration_file", "")

        self.camera_id = int(self.get_parameter("camera_id").value)
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fps = int(self.get_parameter("fps").value)
        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.calibration_file = self.get_parameter("calibration_file").value

        self.bridge = CvBridge()
        self.camera_info_template = self.load_camera_info(self.calibration_file)

        self.image_pub = self.create_publisher(Image, self.image_topic, 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, self.camera_info_topic, 10)

        cmd = [
            "rpicam-vid",
            "--camera", str(self.camera_id),
            "--width", str(self.width),
            "--height", str(self.height),
            "--framerate", str(self.fps),
            "--codec", "mjpeg",
            "--nopreview",
            "--buffer-count", "4",
            "-t", "0",
            "-o", "-",
        ]

        self.get_logger().info("Starting camera command: " + " ".join(cmd))

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        self.buffer = bytearray()
        self.timer = self.create_timer(0.001, self.timer_callback)

    def load_camera_info(self, path):
        msg = CameraInfo()
        msg.width = self.width
        msg.height = self.height
        msg.distortion_model = "plumb_bob"

        if not path:
            self.get_logger().warn("No calibration file provided. Publishing placeholder CameraInfo.")
            return msg

        path = Path(path)
        if not path.exists():
            self.get_logger().warn(f"Calibration file not found: {path}")
            return msg

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        msg.width = int(data["image_width"])
        msg.height = int(data["image_height"])
        msg.distortion_model = data.get("distortion_model", "plumb_bob")
        msg.k = [float(x) for x in data["camera_matrix"]["data"]]
        msg.d = [float(x) for x in data["distortion_coefficients"]["data"]]
        msg.r = [float(x) for x in data["rectification_matrix"]["data"]]
        msg.p = [float(x) for x in data["projection_matrix"]["data"]]

        self.get_logger().info(f"Loaded calibration from {path}")
        return msg

    def make_camera_info(self, stamp):
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.width = self.camera_info_template.width
        msg.height = self.camera_info_template.height
        msg.distortion_model = self.camera_info_template.distortion_model
        msg.k = self.camera_info_template.k
        msg.d = self.camera_info_template.d
        msg.r = self.camera_info_template.r
        msg.p = self.camera_info_template.p
        return msg

    def timer_callback(self):
        if self.proc.poll() is not None:
            stderr = self.proc.stderr.read().decode(errors="ignore")
            self.get_logger().error("rpicam-vid process exited")
            if stderr:
                self.get_logger().error(stderr)
            return

        chunk = self.proc.stdout.read(4096)
        if not chunk:
            return

        self.buffer.extend(chunk)

        start = self.buffer.find(b"\xff\xd8")
        end = self.buffer.find(b"\xff\xd9")

        if start == -1 or end == -1 or end < start:
            return

        jpg = self.buffer[start:end + 2]
        del self.buffer[:end + 2]

        arr = np.frombuffer(jpg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn("Failed to decode JPEG frame")
            return

        stamp = self.get_clock().now().to_msg()

        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = self.frame_id

        camera_info_msg = self.make_camera_info(stamp)

        self.image_pub.publish(image_msg)
        self.camera_info_pub.publish(camera_info_msg)

    def destroy_node(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()

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
        rclpy.shutdown()


if __name__ == "__main__":
    main()