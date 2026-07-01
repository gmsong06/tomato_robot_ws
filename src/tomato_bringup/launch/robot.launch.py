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

    tomato_reactive_controller_node = Node(
        package="tomato_perception",
        executable="tomato_reactive_controller_node",
        name="tomato_reactive_controller_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "velocity": 1700.0,
                "num_joints": 1,
            }
        ],
    )

    return LaunchDescription([
        motor_node,
        camera_node,
        # tomato_detection_node,
        # tomato_ripeness_node,
        # tomato_reactive_controller_node,
    ])