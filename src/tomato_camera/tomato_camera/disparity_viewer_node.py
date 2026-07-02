import rclpy
from rclpy.node import Node
from stereo_msgs.msg import DisparityImage
from sensor_msgs.msg import Image


class DisparityImageExtractNode(Node):
    def __init__(self):
        super().__init__("disparity_image_extract_node")

        self.sub = self.create_subscription(
            DisparityImage,
            "/stereo/disparity",
            self.callback,
            10,
        )

        self.pub = self.create_publisher(
            Image,
            "/stereo/disparity/image",
            10,
        )

    def callback(self, msg):
        image = msg.image
        image.header = msg.header
        self.pub.publish(image)


def main(args=None):
    rclpy.init(args=args)
    node = DisparityImageExtractNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()