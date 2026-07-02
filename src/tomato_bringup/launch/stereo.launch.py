from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    left_camera_node = Node(
        package="tomato_camera",
        executable="camera_node",
        name="left_camera_node",
        parameters=[{
            "camera_id": 0,
            "width": 640,
            "height": 480,
            "fps": 30,
            "image_topic": "/stereo/left/image_raw",
            "camera_info_topic": "/stereo/left/camera_info",
            "frame_id": "stereo_left_camera_frame",
            "calibration_file": "/home/ann/tomato_robot_ws/src/tomato_camera/config/stereo/left_camera.yaml",
        }],
    )

    right_camera_node = Node(
        package="tomato_camera",
        executable="camera_node",
        name="right_camera_node",
        parameters=[{
            "camera_id": 1,
            "width": 640,
            "height": 480,
            "fps": 30,
            "image_topic": "/stereo/right/image_raw",
            "camera_info_topic": "/stereo/right/camera_info",
            "frame_id": "stereo_right_camera_frame",
            "calibration_file": "/home/ann/tomato_robot_ws/src/tomato_camera/config/stereo/right_camera.yaml",
        }],
    )

    return LaunchDescription([
        left_camera_node,
        right_camera_node
    ])
