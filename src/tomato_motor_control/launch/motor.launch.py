from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription(
        [
            Node(
                package="tomato_motor_control",
                executable="motor_node",
                name="motor_node",
                parameters=[
                    {
                        "port": "/dev/ttyUSB0",
                        "baudrate": 1000000,
                        "motor_id": 1,
                    }
                ],
                output="screen",
            )
        ]
    )