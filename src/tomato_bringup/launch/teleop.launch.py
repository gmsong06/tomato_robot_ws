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
                "goal_time": 100,
            }
        ],
    )

    return LaunchDescription([
        motor_node,
    ])