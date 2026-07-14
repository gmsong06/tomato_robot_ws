from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import Command, PathJoinSubstitution


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

    robot_description_content = Command([
        "xacro ",
        PathJoinSubstitution([
            FindPackageShare("tomato_description"),
            "urdf",
            "tomato_arm.urdf.xacro",
        ]),
    ])

    robot_description = {
        "robot_description": robot_description_content
    }

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    controller_node = Node(
        package="tomato_control",
        executable="controller_node",
        name="controller_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            robot_description,
            {
                # Keep the previous IK-only test disabled.
                "ik_test_mode": False,

                # Enable the camera optical-frame to base_link transform test.
                # This mode publishes RViz markers only and never sends motor commands.
                "transform_test_mode": True,

                # Test point in the left rectified camera optical frame.
                # Camera optical convention: +X right, +Y down, +Z forward.
                "transform_test_camera_x_m": 0.00,
                "transform_test_camera_y_m": 0.00,
                "transform_test_camera_z_m": 0.50,
                "transform_test_publish_rate_hz": 2.0,
                "transform_test_marker_topic": "/camera_transform_test_markers",

                # Manual eye-to-hand transform currently being tested.
                "camera_x_m": -0.20,
                "camera_y_m": 0.0524,
                "camera_z_m": 0.65,
                "camera_pitch_down_deg": 45.0,

                "enable_motor_commands": False,
            },
        ],
    )

    return LaunchDescription([
        # motor_node,
        # stereo_camera,
        # tomato_detection_node,
        # tomato_ripeness_node,
        robot_state_publisher_node,
        controller_node,
    ])


# For the default transform test point:
#
# point_camera = x=0.000, y=0.000, z=0.500 m
#
# Expected point in base_link with the current manual transform:
#
# point_base = x=0.1536, y=0.0524, z=0.2964 m
