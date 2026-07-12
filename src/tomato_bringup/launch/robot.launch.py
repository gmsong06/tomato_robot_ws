from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():

    motor_node = Node(
        package="tomato_motor_control",
        executable="motor_node",
        name="feetech_motor_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "port": "/dev/ttyACM0",
                "motor_config_path": "/home/ann/tomato_robot_ws/src/tomato_motor_control/config/motors.yaml",
                "goal_time": 100,
            }
        ],
    )

    stereo_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("tomato_camera"),
                "launch",
                "stereo.launch.py",
            ])
        )
    )

    tomato_detection_node = Node(
        package="tomato_perception",
        executable="tomato_detection_node",
        name="tomato_detection_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "model_path": "/home/ann/tomato_robot_ws/src/tomato_perception/models/yolo11s_6.pt",
                "yolo_conf": 0.5,
            }
        ],
    )

    tomato_ripeness_node = Node(
        package="tomato_perception",
        executable="tomato_ripeness_node",
        name="tomato_ripeness_node",
        output="screen",
        emulate_tty=True,
    )

    tomato_reactive_controller_node = Node(
        package="tomato_perception",
        executable="tomato_reactive_controller_node",
        name="tomato_reactive_controller_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "min_valid_disparity": 80.0,
                "max_valid_disparity": 220.0,
                "min_valid_ratio": 0.10,
                "roi_shrink": 0.20,
            }
        ],
    )

    return LaunchDescription([
        # motor_node,
        stereo_camera,
        tomato_detection_node,
        tomato_ripeness_node,
        tomato_reactive_controller_node,
    ])