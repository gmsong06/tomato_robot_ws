import rclpy
from tomato_interfaces.srv import SetTorque


def call_set_torque(node, torque_client, enable: bool):
    while not torque_client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info("Waiting for /set_torque service...")

    req = SetTorque.Request()
    req.enabled = [enable]

    future = torque_client.call_async(req)

    while rclpy.ok() and not future.done():
        rclpy.spin_once(node, timeout_sec=0.05)

    response = future.result()

    if response is None or not response.success:
        message = "No response" if response is None else response.message
        raise RuntimeError(f"Failed to set torque={enable}: {message}")

    node.get_logger().info(response.message)