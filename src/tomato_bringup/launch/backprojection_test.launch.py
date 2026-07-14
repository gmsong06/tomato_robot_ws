from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import Command, PathJoinSubstitution


def generate_launch_description():

    stereo_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("tomato_camera"),
                "launch",
                "stereo.launch.py",
            ])
        )
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

    # Publish a zero-position JointState so robot_state_publisher publishes
    # the complete robot TF tree while the controller is in test mode.
    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[
            robot_description,
            {
                "rate": 10.0,
            },
        ],
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
                # Keep the previous isolated tests disabled.
                "ik_test_mode": False,
                "transform_test_mode": False,

                # Use the real rectified CameraInfo with a manually specified
                # pixel and known depth. Disparity and motor commands stay off.
                "backprojection_test_mode": True,

                # A negative u or v means use cx or cy from CameraInfo.
                # The first test therefore uses the principal point and should
                # back-project to camera point (0, 0, depth).
                "backprojection_test_u_px": 908.0,
                "backprojection_test_v_px": -1.0,
                "backprojection_test_depth_m": 0.50,
                "backprojection_test_publish_rate_hz": 2.0,
                "backprojection_test_marker_topic": "/backprojection_test_markers",

                # Manual eye-to-hand transform already checked synthetically.
                "camera_x_m": -0.20,
                "camera_y_m": 0.0524,
                "camera_z_m": 0.65,
                "camera_pitch_down_deg": 45.0,

                "enable_motor_commands": False,
            },
        ],
    )

    return LaunchDescription([
        stereo_camera,
        robot_state_publisher_node,
        joint_state_publisher_node,
        controller_node,
    ])
