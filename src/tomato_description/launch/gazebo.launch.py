import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    tomato_description_share = get_package_share_directory(
        "tomato_description"
    )

    ros_gz_sim_share = get_package_share_directory(
        "ros_gz_sim"
    )

    xacro_path = os.path.join(
        tomato_description_share,
        "urdf",
        "tomato_arm_live_camera_control.urdf.xacro",
    )

    world_path = os.path.join(
        tomato_description_share,
        "worlds",
        "tomato_world.sdf",
    )

    robot_description_content = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            xacro_path,
        ]
    )

    robot_description = {
        "robot_description": ParameterValue(
            robot_description_content,
            value_type=str,
        ),
        "use_sim_time": True,
    }

    # Start Gazebo with the custom sensor-enabled world.
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                ros_gz_sim_share,
                "launch",
                "gz_sim.launch.py",
            )
        ),
        launch_arguments={
            "gz_args": f"-r {world_path} --render-engine ogre",
        }.items(),
    )

    # Publish the ROS TF tree.
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    # Spawn the Xacro-generated robot.
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        name="spawn_tomato_robot",
        output="screen",
        arguments=[
            "-name",
            "tomato_arm",
            "-topic",
            "robot_description",
            "-z",
            "0.01",
        ],
    )

    # Bridge clock and CameraInfo messages.
    sensor_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="sensor_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",

            "/stereo/left/camera_info"
            "@sensor_msgs/msg/CameraInfo"
            "[gz.msgs.CameraInfo",

            "/stereo/right/camera_info"
            "@sensor_msgs/msg/CameraInfo"
            "[gz.msgs.CameraInfo",
        ],
    )

    # Efficient Gazebo-to-ROS image bridge.
    image_bridge = Node(
        package="ros_gz_image",
        executable="image_bridge",
        name="stereo_image_bridge",
        output="screen",
        arguments=[
            "/stereo/left/image_raw",
            "/stereo/right/image_raw",
        ],
        parameters=[
            {
                "use_sim_time": True,
            }
        ],
    )

    return LaunchDescription(
        [
            gazebo,
            robot_state_publisher,
            spawn_robot,
            sensor_bridge,
            image_bridge,
        ]
    )