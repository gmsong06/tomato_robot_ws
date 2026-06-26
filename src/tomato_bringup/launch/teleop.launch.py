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
                "num_motors": 6,
                "step_rad": 0.05,
            }
        ],
    )

    return LaunchDescription([
        motor_node,
        keyboard_teleop_node,
    ])