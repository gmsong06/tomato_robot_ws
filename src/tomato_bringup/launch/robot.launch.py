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

    controller_node = Node(
        package="tomato_control",
        executable="controller_node",
        name="controller_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            robot_description,
            {
                # Disparity/depth filtering
                "min_valid_disparity": 80.0,
                "max_valid_disparity": 220.0,
                "min_valid_ratio": 0.10,
                "roi_shrink": 0.20,

                # Manual eye-to-hand transform
                # Camera center/left optical frame relative to robot base
                "camera_x_m": -0.20,
                "camera_y_m": 0.0524,
                "camera_z_m": 0.65,
                "camera_pitch_down_deg": 45.0,

                # Horizontal approach
                "pregrasp_offset_m": 0.05,
                "retreat_offset_m": 0.05,
                "tool_angle_from_horizontal": 0.0,
                "elbow_solution": "down",

                # Motor command publishing
                "enable_motor_commands": True,
                "joint_command_topic": "/joint_target_positions",
                "command_interval_sec": 2.0,

                # Safety approval
                "require_manual_approval": True,
            }
        ],
    )

    return LaunchDescription([
        motor_node,
        stereo_camera,
        tomato_detection_node,
        tomato_ripeness_node,
        controller_node,
    ])


# [controller_node-9]   pregrasp:
# [controller_node-9]     target_base = x=0.269, y=-0.025, z=0.237
# [controller_node-9]     joints = j1=-0.094, j2=0.503, j3=1.229, j4=-0.161 rad
# [controller_node-9] 
# [controller_node-9]   contact:
# [controller_node-9]     target_base = x=0.319, y=-0.025, z=0.237
# [controller_node-9]     joints = j1=-0.079, j2=0.856, j3=0.619, j4=0.097 rad
# [controller_node-9] 
# [controller_node-9]   retreat:
# [controller_node-9]     target_base = x=0.269, y=-0.025, z=0.237
# [controller_node-9]     joints = j1=-0.094, j2=0.503, j3=1.229, j4=-0.161 rad
# [controller_node-9] 