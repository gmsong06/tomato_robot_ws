from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    rpicam-still --camera 0 -o cam0.jpg

    return LaunchDescription([
        motor_node,
    ])