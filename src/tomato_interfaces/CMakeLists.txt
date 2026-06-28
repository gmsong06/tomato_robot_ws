cmake_minimum_required(VERSION 3.8)
project(tomato_interfaces)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TomatoDetection.msg"
  "msg/TomatoDetectionArray.msg"
  "msg/TomatoRipeness.msg"
  "msg/TorqueState.msg"
  "srv/SetTorque.srv"
  DEPENDENCIES std_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()