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
            }
        ],
    )

    camera_node = Node(
        package="tomato_camera",
        executable="camera_node",
        name="camera_node",
        output="screen",
        emulate_tty=True,
        respawn=True,
        parameters=[
            {
                "camera_id": 0,
                "width": 640,
                "height": 480,
                "fps": 30,
                "topic_name": "/camera/image_raw",
                "frame_id": "camera_frame",
            }
        ],
    )

    return LaunchDescription([
        motor_node,
        camera_node,
    ])