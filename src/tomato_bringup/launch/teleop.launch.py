from launch import LaunchDescription
from launch_ros.actions import Node


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
            }
        ]
    )

    keyboard_teleop_node = Node(
        package="tomato_teleop",
        executable="keyboard_teleop_node",
        name="keyboard_teleop_node",
        output="screen",
        emulate_tty=True,
        prefix="xterm -e",
        parameters=[
            {
                "velocity": 1700.0,
                "num_motors": 6,
                "motor_index": 1,
                "publish_hz": 50.0,
            }
        ],
    )

    return LaunchDescription([
        motor_node,
        keyboard_teleop_node,
    ])