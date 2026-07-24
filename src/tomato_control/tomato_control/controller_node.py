from __future__ import annotations

import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo
from stereo_msgs.msg import DisparityImage
from std_msgs.msg import Int32MultiArray, String
from std_srvs.srv import SetBool, Trigger

from tomato_control.camera_geometry import CameraGeometry
from tomato_control.controller_config import ControllerConfig
from tomato_control.controller_models import TomatoCandidate, WaypointCommand
from tomato_control.horizontal_approach_planner import HorizontalApproachPlanner
from tomato_control.ik_solver import TomatoArmIK
from tomato_control.motion_executor import MotionExecutor
from tomato_control.tomato_candidate_builder import TomatoCandidateBuilder
from tomato_control.tomato_depth_estimator import TomatoDepthEstimator
from tomato_interfaces.msg import TomatoRipenessArray
from tomato_interfaces.srv import DebugTomato, SelectTomato


class ControllerNode(Node):
    """Coordinate perception, trajectory generation, approval, and motion."""

    STATE_SCANNING = "scanning"
    STATE_SELECTED = "selected"
    STATE_APPROACHING = "approaching"
    STATE_AT_TOMATO = "at_tomato"
    STATE_RETREATING = "retreating"
    STATE_RETURNING_HOME = "returning_home"

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

        # The IK solver now accepts targets expressed in base_link coordinates.
        # It subtracts the URDF joint_2 origin internally before solving the
        # shoulder/elbow geometry.
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

        self.available_tomato_ids_publisher = self.create_publisher(
            Int32MultiArray,
            "/controller/available_tomato_ids",
            10,
        )

        execution_state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.execution_state_publisher = self.create_publisher(
            String,
            "/controller/execution_state",
            execution_state_qos,
        )
        self.execution_state = self.STATE_SCANNING

        self.cv_bridge = CvBridge()
        self.pending_approval_candidate: TomatoCandidate | None = None
        self.selected_candidate: TomatoCandidate | None = None
        self.latest_reachable_candidates: dict[int, TomatoCandidate] = {}
        self.latest_ripeness_message: TomatoRipenessArray | None = None
        self.latest_disparity_message: DisparityImage | None = None
        self.latest_disparity_image = None
        self.has_logged_camera_intrinsics = False
        self.is_holding_at_tomato = False
        self.is_holding_at_retreat = False

        self.motion_timer = self.create_timer(
            self.config.command_interval_seconds,
            self.motion_executor.publish_next_command,
        )

        # Periodic publication keeps browser/rosbridge clients synchronized
        # even if they do not request transient-local durability.
        self.execution_state_timer = self.create_timer(
            1.0,
            self.publish_execution_state,
        )

        self.selection_service = self.create_service(
            SelectTomato,
            self.config.selection_service_name,
            self.select_tomato_callback,
        )

        self.clear_selection_service = self.create_service(
            Trigger,
            self.config.clear_selection_service_name,
            self.clear_selection_callback,
        )

        self.approval_service = self.create_service(
            SetBool,
            self.config.approval_service_name,
            self.motion_approval_callback,
        )

        self.retract_service = self.create_service(
            Trigger,
            self.config.retract_service_name,
            self.retract_callback,
        )

        self.debug_tomato_service = self.create_service(
            DebugTomato,
            "/controller/debug_tomato",
            self.debug_tomato_callback,
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
        self.publish_execution_state()

    def set_execution_state(self, new_state: str) -> None:
        """Update and immediately publish the controller execution state."""

        valid_states = {
            self.STATE_SCANNING,
            self.STATE_SELECTED,
            self.STATE_APPROACHING,
            self.STATE_AT_TOMATO,
            self.STATE_RETREATING,
            self.STATE_RETURNING_HOME,
        }

        if new_state not in valid_states:
            raise ValueError(f"Unknown controller execution state: {new_state}")

        previous_state = self.execution_state
        self.execution_state = new_state
        self.publish_execution_state()

        if previous_state != new_state:
            self.get_logger().info(
                f"Controller execution state: {previous_state} -> {new_state}"
            )

    def publish_execution_state(self) -> None:
        """Publish the current state for dashboard and other ROS clients."""

        message = String()
        message.data = self.execution_state
        self.execution_state_publisher.publish(message)

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
            or self.selected_candidate is not None
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

        # Cache the latest synchronized perception inputs so the on-demand
        # debug service can rerun the exact same calculations for any visible ID.
        self.latest_ripeness_message = ripeness_message
        self.latest_disparity_message = disparity_message
        self.latest_disparity_image = disparity_image.copy()

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
            self.latest_reachable_candidates = {}
            self.publish_available_tomato_ids()
            self.get_logger().info("No reachable tomatoes currently available")
            return

        self.latest_reachable_candidates = {
            int(candidate.detection.detection_id): candidate
            for candidate in reachable_candidates
        }

        self.publish_available_tomato_ids()
        self._log_available_tomatoes()

    def publish_available_tomato_ids(self) -> None:
        """Publish the IDs that are currently reachable and selectable."""

        message = Int32MultiArray()
        message.data = sorted(self.latest_reachable_candidates.keys())
        self.available_tomato_ids_publisher.publish(message)

    def clear_available_tomato_ids(self) -> None:
        """Tell dashboard clients that no tomato is currently selectable."""

        self.latest_reachable_candidates = {}
        self.publish_available_tomato_ids()

    def _log_available_tomatoes(self) -> None:
        """Print the latest reachable tomatoes without selecting one."""

        available_ids = sorted(self.latest_reachable_candidates)

        self.get_logger().warn("=" * 80)
        self.get_logger().warn(
            f"AVAILABLE REACHABLE TOMATOES: {available_ids}"
        )

        for detection_id in available_ids:
            candidate = self.latest_reachable_candidates[detection_id]
            detection = candidate.detection
            base_point = candidate.estimated_surface_base
            contact_waypoint = next(
                waypoint
                for waypoint in candidate.waypoints
                if waypoint.name == "contact"
            )
            contact_point = contact_waypoint.position_base

            self.get_logger().warn(
                f"id={detection_id}, "
                f"ripeness={detection.final_ripeness}, "
                f"confidence={detection.yolo_confidence:.2f}, "
                f"estimated_surface_base=("
                f"x={base_point.x_m:.3f}, "
                f"y={base_point.y_m:.3f}, "
                f"z={base_point.z_m:.3f}) m, "
                f"corrected_contact=("
                f"x={contact_point.x_m:.3f}, "
                f"y={contact_point.y_m:.3f}, "
                f"z={contact_point.z_m:.3f}) m"
            )

        self.get_logger().warn(
            "Select one with: ros2 service call "
            f"{self.config.selection_service_name} "
            "tomato_interfaces/srv/SelectTomato "
            "'{detection_id: 0}'"
        )
        self.get_logger().warn(
            "Selecting an ID freezes the candidate. Approval executes "
            "pregrasp and contact; /controller/retract later sends retreat "
            "and then home."
        )
        self.get_logger().warn("=" * 80)

    def debug_tomato_callback(self, request, response):
        """Return a detailed acceptance/rejection report for one detection ID."""

        requested_id = int(request.detection_id)

        if (
            self.latest_ripeness_message is None
            or self.latest_disparity_message is None
            or self.latest_disparity_image is None
        ):
            response.success = False
            response.report = (
                "No synchronized tomato-ripeness/disparity frame has been "
                "received yet. Wait for perception data and call the service "
                "again."
            )
            return response

        matching_detection = next(
            (
                detection
                for detection in self.latest_ripeness_message.ripenesses
                if int(detection.detection_id) == requested_id
            ),
            None,
        )

        if matching_detection is None:
            visible_ids = sorted(
                int(detection.detection_id)
                for detection in self.latest_ripeness_message.ripenesses
            )
            response.success = False
            response.report = (
                f"Detection id={requested_id} is not present in the latest "
                f"synchronized frame. Visible IDs: {visible_ids}"
            )
            return response

        accepted, report = self.candidate_builder.build_debug_report(
            self.latest_disparity_image,
            self.latest_disparity_message,
            matching_detection,
        )

        response.success = True
        response.report = report

        verdict = "ACCEPTED" if accepted else "REJECTED"
        self.get_logger().warn(
            f"Generated debug report for tomato id={requested_id}: {verdict}"
        )
        self.get_logger().warn("\n" + report)
        return response

    def select_tomato_callback(self, request, response):
        """Freeze one currently reachable tomato by detection ID."""

        if self.motion_executor.is_in_progress:
            response.success = False
            response.message = "Cannot select while motion is active"
            return response

        if self.pending_approval_candidate is not None:
            response.success = False
            response.message = (
                "Cannot select while a motion is waiting for approval"
            )
            return response

        if self.selected_candidate is not None:
            selected_id = int(
                self.selected_candidate.detection.detection_id
            )
            response.success = False
            response.message = (
                f"Tomato id={selected_id} is already selected. "
                f"Call {self.config.clear_selection_service_name} first."
            )
            return response

        requested_id = int(request.detection_id)
        candidate = self.latest_reachable_candidates.get(requested_id)

        if candidate is None:
            available_ids = sorted(self.latest_reachable_candidates)
            response.success = False
            response.message = (
                f"Tomato id={requested_id} is not currently reachable. "
                f"Available IDs: {available_ids}"
            )
            return response

        self.selected_candidate = candidate

        approval_requested = self.request_motion_approval(candidate)
        if not approval_requested:
            self.selected_candidate = None
            response.success = False
            response.message = (
                f"Could not create approval request for tomato "
                f"id={requested_id}"
            )
            return response

        self.clear_available_tomato_ids()
        self.set_execution_state(self.STATE_SELECTED)
        self._log_selected_candidate(candidate)

        response.success = True
        response.message = (
            f"Selected and froze tomato id={requested_id}. "
            "The pregrasp/contact approach is waiting for approval."
        )
        return response

    def clear_selection_callback(self, request, response):
        """Clear a selection only when it is safe to resume scanning."""

        del request

        if self.motion_executor.is_in_progress:
            response.success = False
            response.message = "Cannot clear selection while motion is active"
            return response

        if self.selected_candidate is None:
            response.success = False
            response.message = "No tomato is currently selected"
            return response

        if (
            self.is_holding_at_tomato
            and self.config.motor_commands_enabled
        ):
            response.success = False
            response.message = (
                "The arm is holding at the tomato. Call "
                f"{self.config.retract_service_name} to retreat and return "
                "home before clearing."
            )
            return response

        selected_id = int(
            self.selected_candidate.detection.detection_id
        )
        self.selected_candidate = None
        self.pending_approval_candidate = None
        self.clear_available_tomato_ids()
        self.is_holding_at_tomato = False
        self.is_holding_at_retreat = False
        self.set_execution_state(self.STATE_SCANNING)

        response.success = True
        response.message = (
            f"Cleared tomato id={selected_id}, canceled any pending approval, "
            "and resumed perception updates"
        )
        self.get_logger().warn(response.message)
        return response

    def _log_selected_candidate(
        self,
        candidate: TomatoCandidate,
    ) -> None:
        """Print the frozen candidate without starting motion."""

        detection = candidate.detection
        base_point = candidate.estimated_surface_base
        contact_waypoint = next(
            waypoint
            for waypoint in candidate.waypoints
            if waypoint.name == "contact"
        )
        contact_point = contact_waypoint.position_base

        self.get_logger().warn("=" * 80)
        self.get_logger().warn(
            f"SELECTED AND FROZEN TOMATO ID={detection.detection_id}"
        )
        self.get_logger().warn(
            f"ripeness={detection.final_ripeness}, "
            f"confidence={detection.yolo_confidence:.2f}"
        )
        self.get_logger().warn(
            "estimated_surface_base=("
            f"x={base_point.x_m:.3f}, "
            f"y={base_point.y_m:.3f}, "
            f"z={base_point.z_m:.3f}) m"
        )
        self.get_logger().warn(
            "corrected_contact=("
            f"x={contact_point.x_m:.3f}, "
            f"y={contact_point.y_m:.3f}, "
            f"z={contact_point.z_m:.3f}) m"
        )
        self.get_logger().warn(
            "Perception is frozen for this selection. Approval executes "
            "pregrasp and contact. The retreat service is available only after "
            "contact completes."
        )
        self.get_logger().warn(
            "Clear selection with: ros2 service call "
            f"{self.config.clear_selection_service_name} "
            "std_srvs/srv/Trigger '{}'"
        )
        self.get_logger().warn("=" * 80)

    def request_motion_approval(
        self,
        candidate: TomatoCandidate,
    ) -> bool:
        """Create an approval request for the selected tomato approach."""

        if not self.config.manual_approval_required:
            self.get_logger().warn(
                "Step 4 requires require_manual_approval:=true. "
                "No motion was started."
            )
            return False

        if self.motion_executor.is_in_progress:
            self.get_logger().info(
                "Motion already in progress; not requesting another approval"
            )
            return False

        if self.pending_approval_candidate is not None:
            self.get_logger().info(
                "A tomato approach is already waiting for approval"
            )
            return False

        self.pending_approval_candidate = candidate
        self._log_approval_request(candidate)
        return True

    def motion_approval_callback(self, request, response):
        """Approve or cancel the frozen pregrasp/contact approach."""

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
                "There is no selected tomato waiting for approval"
            )
            return response

        candidate = self.pending_approval_candidate
        detection_id = int(candidate.detection.detection_id)

        if not request.data:
            self.pending_approval_candidate = None
            self.selected_candidate = None
            self.clear_available_tomato_ids()
            self.is_holding_at_tomato = False
            self.is_holding_at_retreat = False
            self.set_execution_state(self.STATE_SCANNING)

            response.success = True
            response.message = (
                f"Canceled tomato id={detection_id}; perception updates resumed"
            )
            self.get_logger().warn(response.message)
            return response

        approach_commands = tuple(
            command
            for command in candidate.waypoint_commands
            if command.name in {"pregrasp", "contact"}
        )

        self._log_approach_start(candidate, approach_commands)

        self.set_execution_state(self.STATE_APPROACHING)

        started = self.motion_executor.start_commands(
            candidate,
            approach_commands,
            sequence_name="pregrasp/contact approach",
            on_complete=self._handle_approach_complete,
        )

        if not started:
            self.set_execution_state(self.STATE_SELECTED)
            response.success = False
            response.message = (
                f"Could not start approach for tomato id={detection_id}"
            )
            return response

        self.pending_approval_candidate = None

        if self.config.motor_commands_enabled:
            response.message = (
                f"Approved tomato id={detection_id}. Executing pregrasp "
                "and contact only."
            )
        else:
            response.message = (
                f"Approved tomato id={detection_id}. Completed a dry run of "
                "pregrasp and contact; no motor commands were published."
            )

        response.success = True
        self.get_logger().warn(response.message)
        return response

    def _log_approach_start(
        self,
        candidate: TomatoCandidate,
        approach_commands,
    ) -> None:
        """Log the exact commands that are about to run."""

        detection_id = int(candidate.detection.detection_id)

        self.get_logger().warn("=" * 80)
        self.get_logger().warn(
            f"STARTING PREGRASP/CONTACT FOR TOMATO ID={detection_id}"
        )

        for command in approach_commands:
            position = command.waypoint.position_base
            motor_angles = (
                self.motion_executor.convert_ros_angles_to_motor_angles(
                    command.joint_angles
                )
            )
            self.get_logger().warn(
                f"  {command.name}: target_base=("
                f"x={position.x_m:.3f}, "
                f"y={position.y_m:.3f}, "
                f"z={position.z_m:.3f}), "
                f"ros_joints={command.joint_angles}, "
                f"motor_joints={motor_angles}"
            )

        self.get_logger().warn(
            "RETREAT IS NOT INCLUDED. The controller will stop and hold "
            "after the contact command."
        )
        self.get_logger().warn("=" * 80)

    def _handle_approach_complete(self) -> None:
        """Enter the held-at-contact state after contact is commanded."""

        if self.selected_candidate is None:
            self.get_logger().error(
                "Approach completed, but no selected tomato is stored"
            )
            return

        self.is_holding_at_tomato = True
        self.is_holding_at_retreat = False
        self.set_execution_state(self.STATE_AT_TOMATO)
        detection_id = int(
            self.selected_candidate.detection.detection_id
        )

        self.get_logger().warn("=" * 80)
        self.get_logger().warn(
            f"AT TOMATO ID={detection_id}: HOLDING CONTACT COMMAND"
        )
        self.get_logger().warn(
            "No retreat command has been sent yet. Retract manually with: "
            "ros2 service call "
            f"{self.config.retract_service_name} std_srvs/srv/Trigger '{{}}'"
        )
        self.get_logger().warn(
            "The retract service sends the stored retreat waypoint and then the fixed "
            "home pose."
        )
        self.get_logger().warn("=" * 80)

    def retract_callback(self, request, response):
        """Move through retreat and then the fixed joint-space home pose."""

        del request

        if self.motion_executor.is_in_progress:
            response.success = False
            response.message = (
                "Cannot retract while another motion is active"
            )
            return response

        if self.selected_candidate is None:
            response.success = False
            response.message = (
                "No tomato execution is currently active"
            )
            return response

        if not self.is_holding_at_tomato:
            response.success = False
            response.message = (
                "Retract is only allowed after pregrasp/contact completes"
            )
            return response

        retreat_command = next(
            (
                command
                for command in self.selected_candidate.waypoint_commands
                if command.name == "retreat"
            ),
            None,
        )

        if retreat_command is None:
            response.success = False
            response.message = (
                "The selected tomato does not contain a retreat command"
            )
            self.get_logger().error(response.message)
            return response

        detection_id = int(
            self.selected_candidate.detection.detection_id
        )

        self._log_retreat_start(retreat_command, detection_id)

        self.set_execution_state(self.STATE_RETREATING)

        started = self.motion_executor.start_commands(
            self.selected_candidate,
            (retreat_command,),
            sequence_name="base-frame retreat",
            on_complete=self._handle_retreat_complete,
        )

        if not started:
            self.set_execution_state(self.STATE_AT_TOMATO)
            response.success = False
            response.message = (
                f"Could not start retreat/home sequence for tomato "
                f"id={detection_id}"
            )
            return response

        self.is_holding_at_tomato = False
        self.is_holding_at_retreat = False
        response.success = True

        if self.config.motor_commands_enabled:
            response.message = (
                f"Started retreat and return-home sequence for tomato "
                f"id={detection_id}"
            )
        else:
            response.message = (
                f"Completed dry-run retreat and home sequence for tomato "
                f"id={detection_id}; no motor command was published"
            )

        self.get_logger().warn(response.message)
        return response

    def _build_home_command(self) -> WaypointCommand:
        """Create the fixed joint-space home command in ROS convention."""

        joint_names = ("joint_1", "joint_2", "joint_3", "joint_4")
        joint_angles = dict(
            zip(joint_names, self.config.home_joint_positions_rad)
        )

        return WaypointCommand(
            name="home",
            waypoint=None,
            joint_angles=joint_angles,
            ik_result=None,
        )

    def _log_retreat_start(
        self,
        retreat_command: WaypointCommand,
        detection_id: int,
    ) -> None:
        retreat_position = retreat_command.waypoint.position_base
        retreat_motor_angles = (
            self.motion_executor.convert_ros_angles_to_motor_angles(
                retreat_command.joint_angles
            )
        )

        self.get_logger().warn("=" * 80)
        self.get_logger().warn(
            f"STARTING RETREAT FOR TOMATO ID={detection_id}"
        )
        self.get_logger().warn(
            "  retreat: target_base=("
            f"x={retreat_position.x_m:.3f}, "
            f"y={retreat_position.y_m:.3f}, "
            f"z={retreat_position.z_m:.3f}), "
            f"ros_joints={retreat_command.joint_angles}, "
            f"motor_joints={retreat_motor_angles}"
        )
        self.get_logger().warn("Execution phase: contact -> retreat")
        self.get_logger().warn("=" * 80)

    def _handle_retreat_complete(self) -> None:
        """Start the separate fixed joint-space return-home phase."""

        if self.selected_candidate is None:
            self.get_logger().error(
                "Retreat completed, but no selected tomato is stored"
            )
            return

        home_command = self._build_home_command()
        home_motor_angles = (
            self.motion_executor.convert_ros_angles_to_motor_angles(
                home_command.joint_angles
            )
        )
        detection_id = int(
            self.selected_candidate.detection.detection_id
        )

        self.set_execution_state(self.STATE_RETURNING_HOME)
        self.get_logger().warn("=" * 80)
        self.get_logger().warn(
            f"RETREAT COMPLETE FOR TOMATO ID={detection_id}; RETURNING HOME"
        )
        self.get_logger().warn(
            "  home: fixed joint-space target, "
            f"ros_joints={home_command.joint_angles}, "
            f"motor_joints={home_motor_angles}"
        )
        self.get_logger().warn("Execution phase: retreat -> home")
        self.get_logger().warn("=" * 80)

        started = self.motion_executor.start_commands(
            self.selected_candidate,
            (home_command,),
            sequence_name="return home",
            on_complete=self._handle_execution_complete,
        )

        if not started:
            self.get_logger().error(
                f"Could not start return-home sequence for tomato "
                f"id={detection_id}"
            )

    def _handle_execution_complete(self) -> None:
        """Clear the frozen execution after home and resume perception."""

        detection_id = None
        if self.selected_candidate is not None:
            detection_id = int(
                self.selected_candidate.detection.detection_id
            )

        self.selected_candidate = None
        self.pending_approval_candidate = None
        self.clear_available_tomato_ids()
        self.is_holding_at_tomato = False
        self.is_holding_at_retreat = False
        self.set_execution_state(self.STATE_SCANNING)

        self.get_logger().warn("=" * 80)
        self.get_logger().warn(
            f"EXECUTION COMPLETE FOR TOMATO ID={detection_id}"
        )
        self.get_logger().warn(
            "Completed pregrasp -> contact -> retreat -> home."
        )
        self.get_logger().warn(
            "Selection was cleared and perception updates resumed."
        )
        self.get_logger().warn("=" * 80)

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
            "APPROVAL REQUIRED FOR PREGRASP/CONTACT MOTION"
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
            "Estimated camera-facing surface in the base frame before offsets: "
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

        self.get_logger().warn("Planned approach commands:")
        for command in candidate.waypoint_commands:
            if command.name not in {"pregrasp", "contact"}:
                continue
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
            "Approval executes only pregrasp and contact. Retreat is not sent."
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
            "Available tomato IDs topic: "
            "/controller/available_tomato_ids"
        )
        self.get_logger().info(
            "Controller execution-state topic: "
            "/controller/execution_state"
        )
        self.get_logger().info(
            f"Tomato selection service: "
            f"{self.config.selection_service_name}"
        )
        self.get_logger().info(
            f"Clear selection service: "
            f"{self.config.clear_selection_service_name}"
        )
        self.get_logger().info(
            f"Motion approval service: "
            f"{self.config.approval_service_name}, "
            f"manual_approval_required="
            f"{self.config.manual_approval_required}"
        )
        self.get_logger().info(
            f"Retreat service: {self.config.retract_service_name}"
        )
        self.get_logger().info(
            "Tomato depth settings: "
            f"roi_total_shrink="
            f"{self.config.roi_total_shrink_fraction:.2f}, "
            f"surface_percentile="
            f"{self.config.surface_disparity_percentile:.1f}"
        )
        self.get_logger().info(
            "Contact corrections in the base frame: "
            f"X=-{self.config.contact_standoff_m:.3f} m, "
            f"Y={self.config.contact_lateral_offset_m:+.3f} m, "
            f"Z={self.config.contact_vertical_offset_m:+.3f} m"
        )
        home = self.config.home_joint_positions_rad
        self.get_logger().info(
            "Home pose in ROS/URDF radians: "
            f"j1={home[0]:.3f}, j2={home[1]:.3f}, "
            f"j3={home[2]:.3f}, j4={home[3]:.3f}"
        )
        self.get_logger().info(
            "DASHBOARD STATE MODE: approval executes pregrasp/contact and "
            "holds. The retract service publishes retreating, then "
            "returning_home, and resumes scanning after the home command."
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