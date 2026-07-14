from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    left_calibration_file = PathJoinSubstitution([
        FindPackageShare("tomato_camera"),
        "config",
        "stereo",
        "left_camera.yaml",
    ])

    right_calibration_file = PathJoinSubstitution([
        FindPackageShare("tomato_camera"),
        "config",
        "stereo",
        "right_camera.yaml",
    ])

    disparity_config_file = PathJoinSubstitution([
        FindPackageShare("tomato_camera"),
        "config",
        "disparity.yaml",
    ])

    left_camera_node = Node(
        package="tomato_camera",
        executable="camera_node",
        name="left_camera_node",
        parameters=[{
            "camera_id": 0,
            "width": 640,
            "height": 480,
            "fps": 15.0,
            "image_topic": "/stereo/left/image_raw",
            "camera_info_topic": "/stereo/left/camera_info",
            "frame_id": "stereo_left_camera_frame",
            "calibration_file": left_calibration_file,
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
            "fps": 15.0,
            "image_topic": "/stereo/right/image_raw",
            "camera_info_topic": "/stereo/right/camera_info",
            "frame_id": "stereo_right_camera_frame",
            "calibration_file": right_calibration_file,
        }],
    )

    left_rectify = Node(
        package="image_proc",
        executable="rectify_node",
        name="left_rectify",
        namespace="/stereo/left",
        output="screen",
        ros_arguments=[
            "--log-level",
            "error",
        ],
        remappings=[
            ("image", "image_raw"),
            ("camera_info", "camera_info"),
            ("image_rect", "image_rect"),
        ],
    )

    right_rectify = Node(
        package="image_proc",
        executable="rectify_node",
        name="right_rectify",
        namespace="/stereo/right",
        output="screen",
        ros_arguments=[
            "--log-level",
            "error",
        ],
        remappings=[
            ("image", "image_raw"),
            ("camera_info", "camera_info"),
            ("image_rect", "image_rect"),
        ],
    )

    disparity = Node(
        package="stereo_image_proc",
        executable="disparity_node",
        name="disparity_node",
        parameters=[
            disparity_config_file,
        ],
        remappings=[
            ("left/image_rect", "/stereo/left/image_rect"),
            ("left/camera_info", "/stereo/left/camera_info"),
            ("right/image_rect", "/stereo/right/image_rect"),
            ("right/camera_info", "/stereo/right/camera_info"),
            ("disparity", "/stereo/disparity"),
        ],
    )

    disparity_viewer = Node(
        package="tomato_camera",
        executable="disparity_viewer_node",
        name="disparity_viewer_node",
        parameters=[

        ],
    )

    return LaunchDescription([
        left_camera_node,
        right_camera_node,
        left_rectify,
        right_rectify,
        disparity,
        disparity_viewer
    ])