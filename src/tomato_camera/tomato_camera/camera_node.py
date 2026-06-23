import subprocess

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        self.declare_parameter("camera_id", 0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30)
        self.declare_parameter("topic_name", "/camera/image_raw")
        self.declare_parameter("frame_id", "camera_frame")

        self.camera_id = self.get_parameter("camera_id").value
        self.width = self.get_parameter("width").value
        self.height = self.get_parameter("height").value
        self.fps = self.get_parameter("fps").value
        self.topic_name = self.get_parameter("topic_name").value
        self.frame_id = self.get_parameter("frame_id").value

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, self.topic_name, 10)

        cmd = [
            "rpicam-vid",
            "--camera", str(self.camera_id),
            "--width", str(self.width),
            "--height", str(self.height),
            "--framerate", str(self.fps),
            "--codec", "mjpeg",
            "--inline",
            "--nopreview",
            "--buffer-count", "1",
            "-t", "0",
            "-o", "-",
        ]

        self.get_logger().info("Starting camera command: " + " ".join(cmd))

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

        self.buffer = bytearray()
        self.timer = self.create_timer(0.001, self.timer_callback)

        self.get_logger().info(
            f"Camera node started: camera_id={self.camera_id}, "
            f"{self.width}x{self.height}@{self.fps}, topic={self.topic_name}"
        )

    def timer_callback(self):
        if self.proc.poll() is not None:
            self.get_logger().error("rpicam-vid process exited")
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

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        self.image_pub.publish(msg)

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