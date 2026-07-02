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
            "fps": 15,
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
            "fps": 15,
            "image_topic": "/stereo/right/image_raw",
            "camera_info_topic": "/stereo/right/camera_info",
            "frame_id": "stereo_right_camera_frame",
            "calibration_file": "/home/ann/tomato_robot_ws/src/tomato_camera/config/stereo/right_camera.yaml",
        }],
    )

    left_rectify = Node(
        package="image_proc",
        executable="rectify_node",
        name="left_rectify",
        remappings=[
            ("image", "/stereo/left/image_raw"),
            ("camera_info", "/stereo/left/camera_info"),
            ("image_rect", "/stereo/left/image_rect"),
        ],
    )

    right_rectify = Node(
        package="image_proc",
        executable="rectify_node",
        name="right_rectify",
        remappings=[
            ("image", "/stereo/right/image_raw"),
            ("camera_info", "/stereo/right/camera_info"),
            ("image_rect", "/stereo/right/image_rect"),
        ],
    )

    disparity = Node(
        package="stereo_image_proc",
        executable="disparity_node",
        name="disparity_node",
        parameters=[
            {
                "approximate_sync": True,
                "queue_size": 10,
                "min_disparity": 0,
                "max_disparity": 64,
                "block_size": 15,
                "uniqueness_ratio": 10.0,
                "speckle_size": 100,
                "speckle_range": 4,
                "disp12_max_diff": 1,
            }
        ],
        remappings=[
            ("left/image_rect", "/stereo/left/image_rect"),
            ("left/camera_info", "/stereo/left/camera_info"),
            ("right/image_rect", "/stereo/right/image_rect"),
            ("right/camera_info", "/stereo/right/camera_info"),
            ("disparity", "/stereo/disparity"),
        ],
    )

    return LaunchDescription([
        left_camera_node,
        right_camera_node,
        left_rectify,
        right_rectify,
        disparity,
    ])

