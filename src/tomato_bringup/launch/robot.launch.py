from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ---------------------------------------------------------
    # Feetech motor hardware node
    # ---------------------------------------------------------
    motor_node = Node(
        package="tomato_motor_control",
        executable="motor_node",
        name="feetech_motor_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "port": "/dev/ttyACM0",
                "motor_config_path": (
                    "/home/ann/tomato_robot_ws/src/"
                    "tomato_motor_control/config/motors.yaml"
                ),
                "goal_time": 100,
                "goal_retry_period_sec": 0.25,
                "goal_tolerance_rad": 0.03,
                "goal_retry_timeout_sec": 10.0,
            }
        ],
    )

    # ---------------------------------------------------------
    # Stereo cameras and stereo depth pipeline
    # ---------------------------------------------------------
    stereo_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("tomato_camera"),
                    "launch",
                    "stereo.launch.py",
                ]
            )
        )
    )

    # ---------------------------------------------------------
    # YOLO tomato detection
    # ---------------------------------------------------------
    tomato_detection_node = Node(
        package="tomato_perception",
        executable="tomato_detection_node",
        name="tomato_detection_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "model_path": (
                    "/home/ann/tomato_robot_ws/src/"
                    "tomato_perception/models/yolo11s_6.pt"
                ),
                "yolo_conf": 0.5,
            }
        ],
    )

    # ---------------------------------------------------------
    # HSV/ripeness validation
    # ---------------------------------------------------------
    tomato_ripeness_node = Node(
        package="tomato_perception",
        executable="tomato_ripeness_node",
        name="tomato_ripeness_node",
        output="screen",
        emulate_tty=True,
    )

    # ---------------------------------------------------------
    # Robot description used by the analytical IK solver
    # ---------------------------------------------------------
    robot_description_content = Command(
        [
            "xacro ",
            PathJoinSubstitution(
                [
                    FindPackageShare("tomato_description"),
                    "urdf",
                    "tomato_arm.urdf.xacro",
                ]
            ),
        ]
    )

    robot_description = {
        "robot_description": ParameterValue(
            robot_description_content,
            value_type=str,
        ),
    }

    # ---------------------------------------------------------
    # Contact-only controller:
    # select -> approve -> contact -> hold -> optional direct home
    # ---------------------------------------------------------
    controller_node = Node(
        package="tomato_control",
        executable="controller_node",
        name="controller_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            robot_description,
            {
                # Disparity filtering.
                "min_valid_disparity": 1.0,
                "max_valid_disparity": 400.0,
                "min_valid_ratio": 0.10,
                "roi_shrink": 0.40,
                "surface_disparity_percentile": 75.0,

                # Left-camera pose in base_link.
                "camera_x_m": -0.20,
                "camera_y_m": 0.051555,
                "camera_z_m": 0.65,
                "camera_pitch_down_deg": 35.0,

                # Contact target and IK configuration.
                "tool_angle_from_horizontal": 0.0,
                "elbow_solution": "up",
                "contact_surface_offset_m": 0.03,
                "contact_y_offset_m": 0.03,
                "contact_z_offset_m": 0.05,

                # Motor output.
                "enable_motor_commands": True,
                "joint_command_topic": "/joint_target_positions",
                "command_interval_sec": 1.0,
                "invert_joint_1_command": True,

                # Optional direct return to the fixed home pose.
                "return_home_service_name": "/controller/return_home",
                "home_joint_positions": [
                    -0.0928058376670813,
                    0.10471975511965978,
                    1.53588974175501,
                    0.32903887900147005,
                ],

                # Manual selection and approval.
                "selection_service_name": "/controller/select_tomato",
                "clear_selection_service_name": (
                    "/controller/clear_selection"
                ),
                "require_manual_approval": True,
                "approval_service_name": (
                    "/controller/set_motion_approval"
                ),
            },
        ],
    )

    return LaunchDescription(
        [
            motor_node,
            stereo_camera,
            tomato_detection_node,
            tomato_ripeness_node,
            controller_node,
        ]
    )