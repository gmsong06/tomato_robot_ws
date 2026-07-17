from __future__ import annotations

import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from stereo_msgs.msg import DisparityImage
from std_srvs.srv import SetBool

from tomato_control.camera_geometry import CameraGeometry
from tomato_control.controller_config import ControllerConfig
from tomato_control.controller_models import TomatoCandidate
from tomato_control.horizontal_approach_planner import HorizontalApproachPlanner
from tomato_control.ik_solver import TomatoArmIK
from tomato_control.motion_executor import MotionExecutor
from tomato_control.tomato_candidate_builder import TomatoCandidateBuilder
from tomato_control.tomato_depth_estimator import TomatoDepthEstimator
from tomato_interfaces.msg import TomatoRipenessArray


class ControllerNode(Node):
    """Coordinate perception, trajectory generation, approval, and motion."""

    def __init__(self):
        super().__init__("controller_node")

        ControllerConfig.declare_parameters(self)
        self.config = ControllerConfig.from_node(self)

        robot_description_xml = str(
            self.get_parameter("robot_description").value
        )
        if not robot_description_xml:
            raise RuntimeError(
                "robot_description parameter is empty. "
                "Pass the URDF/xacro into this node."
            )

        ik_solver = TomatoArmIK.from_robot_description(
            robot_description_xml
        )

        self.camera_geometry = CameraGeometry(self.config)
        self.depth_estimator = TomatoDepthEstimator(self.config)
        self.approach_planner = HorizontalApproachPlanner(
            self.config,
            ik_solver,
            self.get_logger(),
        )
        self.candidate_builder = TomatoCandidateBuilder(
            self.depth_estimator,
            self.camera_geometry,
            self.approach_planner,
            self.get_logger(),
        )
        self.motion_executor = MotionExecutor(self, self.config)

        self.cv_bridge = CvBridge()
        self.pending_approval_candidate: TomatoCandidate | None = None
        self.has_logged_camera_intrinsics = False

        self.motion_timer = self.create_timer(
            self.config.command_interval_seconds,
            self.motion_executor.publish_next_command,
        )

        self.approval_service = self.create_service(
            SetBool,
            self.config.approval_service_name,
            self.motion_approval_callback,
        )

        self.left_camera_info_subscription = self.create_subscription(
            CameraInfo,
            "/stereo/left/camera_info",
            self.left_camera_info_callback,
            qos_profile_sensor_data,
        )

        self.ripeness_subscriber = Subscriber(
            self,
            TomatoRipenessArray,
            "/tomato_ripeness",
        )
        self.disparity_subscriber = Subscriber(
            self,
            DisparityImage,
            "/stereo/disparity",
        )

        self.synchronizer = ApproximateTimeSynchronizer(
            [self.ripeness_subscriber, self.disparity_subscriber],
            queue_size=10,
            slop=0.15,
        )
        self.synchronizer.registerCallback(self.synced_callback)

        self._log_startup_configuration()

    def left_camera_info_callback(self, message: CameraInfo) -> None:
        intrinsics = self.camera_geometry.update_intrinsics(message)

        if not self.has_logged_camera_intrinsics:
            self.get_logger().info(
                "Cached left camera intrinsics: "
                f"fx={intrinsics.focal_x_px:.2f}, "
                f"fy={intrinsics.focal_y_px:.2f}, "
                f"cx={intrinsics.principal_x_px:.2f}, "
                f"cy={intrinsics.principal_y_px:.2f}"
            )
            self.has_logged_camera_intrinsics = True

    def synced_callback(
        self,
        ripeness_message: TomatoRipenessArray,
        disparity_message: DisparityImage,
    ) -> None:
        if (
            self.motion_executor.is_in_progress
            or self.pending_approval_candidate is not None
        ):
            return

        if self.camera_geometry.intrinsics is None:
            self.get_logger().warn(
                "No left camera intrinsics received yet; waiting for "
                "/stereo/left/camera_info"
            )
            return

        disparity_image = self.cv_bridge.imgmsg_to_cv2(
            disparity_message.image,
            desired_encoding="32FC1",
        )

        self.get_logger().info(
            f"Received {len(ripeness_message.ripenesses)} "
            "tomato ripeness result(s)"
        )

        reachable_candidates: list[TomatoCandidate] = []

        for tomato_detection in ripeness_message.ripenesses:
            candidate = self.candidate_builder.build(
                disparity_image,
                disparity_message,
                tomato_detection,
            )

            if candidate is None:
                continue

            reachable_candidates.append(candidate)
            self._log_candidate(candidate)

        if not reachable_candidates:
            self.get_logger().info("No valid tomato depth candidates")
            return

        self.get_logger().info(
            f"Found {len(reachable_candidates)} "
            "reachable tomato candidate(s)"
        )

        # Temporary behavior before manual multi-tomato ID selection:
        # prefer ripeness, then the largest bounding box.
        selected_candidate = max(
            reachable_candidates,
            key=lambda candidate: (
                candidate.ripeness_priority,
                candidate.bounding_box_area_px,
            ),
        )
        self.request_motion_approval(selected_candidate)

    def request_motion_approval(
        self,
        candidate: TomatoCandidate,
    ) -> None:
        if not self.config.manual_approval_required:
            self.motion_executor.start(candidate)
            return

        if self.motion_executor.is_in_progress:
            self.get_logger().info(
                "Motion already in progress; not requesting another approval"
            )
            return

        if self.pending_approval_candidate is not None:
            self.get_logger().info(
                "A tomato approach is already waiting for approval"
            )
            return

        self.pending_approval_candidate = candidate
        self._log_approval_request(candidate)

    def motion_approval_callback(self, request, response):
        if not self.config.manual_approval_required:
            response.success = False
            response.message = (
                "Manual approval is disabled because "
                "require_manual_approval is false"
            )
            return response

        if self.pending_approval_candidate is None:
            response.success = False
            response.message = (
                "There is no motion sequence waiting for approval"
            )
            return response

        candidate = self.pending_approval_candidate
        detection_id = candidate.detection.detection_id

        if not request.data:
            self.pending_approval_candidate = None
            self.motion_executor.reset()
            response.success = True
            response.message = (
                f"Canceled pending motion for detection id={detection_id}"
            )
            self.get_logger().warn(response.message)
            return response

        self.pending_approval_candidate = None
        motion_started = self.motion_executor.start(candidate)

        if not motion_started:
            self.pending_approval_candidate = candidate
            response.success = False
            response.message = (
                f"Could not start motion for detection id={detection_id} "
                "because another motion is still active"
            )
            self.get_logger().warn(response.message)
            return response

        response.success = True
        response.message = (
            f"Approved pending motion for detection id={detection_id}"
        )
        self.get_logger().warn(response.message)
        return response

    def _log_candidate(self, candidate: TomatoCandidate) -> None:
        detection = candidate.detection
        depth = candidate.depth_estimate
        camera_point = candidate.camera_surface_point
        base_point = candidate.estimated_surface_base
        contact = next(
            waypoint
            for waypoint in candidate.waypoints
            if waypoint.name == "contact"
        ).position_base

        self.get_logger().info(
            f"id={detection.detection_id}, "
            f"ripeness={detection.final_ripeness}, "
            f"priority={candidate.ripeness_priority}, "
            f"confidence={detection.yolo_confidence:.2f}, "
            f"bbox=({detection.x1},{detection.y1})-"
            f"({detection.x2},{detection.y2}), "
            f"ROI=({depth.roi.x_min}:{depth.roi.x_max}, "
            f"{depth.roi.y_min}:{depth.roi.y_max}), "
            f"center_px=({depth.roi.center_u_px},"
            f"{depth.roi.center_v_px}), "
            f"valid={depth.valid_pixel_count}/"
            f"{depth.total_pixel_count}, "
            f"median_disp={depth.median_disparity_px:.2f}px, "
            f"surface_disp={depth.surface_disparity_px:.2f}px, "
            f"depth={depth.optical_depth_m:.3f} m, "
            f"camera_surface=(x={camera_point.x_m:.3f}, "
            f"y={camera_point.y_m:.3f}, z={camera_point.z_m:.3f}) m, "
            f"base_surface=(x={base_point.x_m:.3f}, "
            f"y={base_point.y_m:.3f}, z={base_point.z_m:.3f}) m, "
            f"corrected_contact=(x={contact.x_m:.3f}, "
            f"y={contact.y_m:.3f}, z={contact.z_m:.3f}) m"
        )

        for command in candidate.waypoint_commands:
            position = command.waypoint.position_base
            self.get_logger().info(
                f"id={detection.detection_id}, "
                f"{command.name} target_base=("
                f"x={position.x_m:.3f}, "
                f"y={position.y_m:.3f}, "
                f"z={position.z_m:.3f}), "
                f"ros_joints={command.joint_angles}"
            )

    def _log_approval_request(self, candidate: TomatoCandidate) -> None:
        detection = candidate.detection
        camera_point = candidate.camera_surface_point
        base_point = candidate.estimated_surface_base

        self.get_logger().warn("=" * 80)
        self.get_logger().warn(
            "SERVICE APPROVAL REQUIRED BEFORE MOTOR COMMANDS"
        )
        self.get_logger().warn("=" * 80)
        self.get_logger().warn(f"Detection id: {detection.detection_id}")
        self.get_logger().warn(f"Ripeness: {detection.final_ripeness}")
        self.get_logger().warn(
            f"Confidence: {detection.yolo_confidence:.2f}"
        )
        self.get_logger().warn(
            "Estimated camera-facing surface in camera frame: "
            f"x={camera_point.x_m:.3f}, "
            f"y={camera_point.y_m:.3f}, "
            f"z={camera_point.z_m:.3f} m"
        )
        self.get_logger().warn(
            "Estimated camera-facing surface in base_link before offsets: "
            f"x={base_point.x_m:.3f}, "
            f"y={base_point.y_m:.3f}, "
            f"z={base_point.z_m:.3f} m"
        )
        self.get_logger().warn(
            "Applied contact corrections: "
            f"X=-{self.config.contact_standoff_m:.3f} m, "
            f"Y={self.config.contact_lateral_offset_m:+.3f} m, "
            f"Z={self.config.contact_vertical_offset_m:+.3f} m"
        )

        self.get_logger().warn("Planned waypoint joint commands:")
        for command in candidate.waypoint_commands:
            position = command.waypoint.position_base
            motor_angles = (
                self.motion_executor.convert_ros_angles_to_motor_angles(
                    command.joint_angles
                )
            )
            joints = command.joint_angles
            self.get_logger().warn(
                f"  {command.name}: target_base=("
                f"x={position.x_m:.3f}, "
                f"y={position.y_m:.3f}, "
                f"z={position.z_m:.3f}), "
                f"ros_joints=(j1={joints['joint_1']:.3f}, "
                f"j2={joints['joint_2']:.3f}, "
                f"j3={joints['joint_3']:.3f}, "
                f"j4={joints['joint_4']:.3f}), "
                f"motor_j1={motor_angles['joint_1']:.3f} rad"
            )

        self.get_logger().warn(
            "Approve with: ros2 service call "
            f"{self.config.approval_service_name} "
            "std_srvs/srv/SetBool '{data: true}'"
        )
        self.get_logger().warn(
            "Cancel with: ros2 service call "
            f"{self.config.approval_service_name} "
            "std_srvs/srv/SetBool '{data: false}'"
        )

    def _log_startup_configuration(self) -> None:
        self.get_logger().info("CONTROLLER STARTED")
        self.get_logger().info(
            f"Motor command topic: {self.config.joint_command_topic}, "
            f"motor_commands_enabled={self.config.motor_commands_enabled}"
        )
        self.get_logger().info(
            f"Motion approval service: "
            f"{self.config.approval_service_name}, "
            f"manual_approval_required="
            f"{self.config.manual_approval_required}"
        )
        self.get_logger().info(
            "Tomato depth settings: "
            f"roi_total_shrink="
            f"{self.config.roi_total_shrink_fraction:.2f}, "
            f"surface_percentile="
            f"{self.config.surface_disparity_percentile:.1f}"
        )
        self.get_logger().info(
            "Contact corrections in base_link: "
            f"X=-{self.config.contact_standoff_m:.3f} m, "
            f"Y={self.config.contact_lateral_offset_m:+.3f} m, "
            f"Z={self.config.contact_vertical_offset_m:+.3f} m"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
