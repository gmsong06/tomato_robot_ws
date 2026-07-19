import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_name = "tomato_description"
    pkg_share = get_package_share_directory(package_name)

    default_model_path = os.path.join(
        pkg_share,
        "urdf",
        "tomato_arm.urdf.xacro",
    )

    rviz_config_path = os.path.join(
        pkg_share,
        "rviz",
        "tomato_robot.rviz",
    )

    model_arg = DeclareLaunchArgument(
        "model",
        default_value=default_model_path,
        description="Absolute path to robot URDF/Xacro file",
    )

    robot_description_content = Command(
        [
            "xacro",
            " ",
            LaunchConfiguration("model"),
        ]
    )

    # Important: force the expanded XML to be treated as a string,
    # not parsed as YAML.
    robot_description = ParameterValue(
        robot_description_content,
        value_type=str,
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
            }
        ],
    )

    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
            }
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=[
            "-d",
            rviz_config_path,
        ],
    )

    return LaunchDescription(
        [
            model_arg,
            robot_state_publisher_node,
            joint_state_publisher_gui_node,
            rviz_node,
        ]
    )