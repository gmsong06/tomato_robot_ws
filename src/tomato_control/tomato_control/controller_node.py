import numpy as np

import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from stereo_msgs.msg import DisparityImage
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from std_srvs.srv import SetBool

from tomato_control.ik_solver import TomatoArmIK
from tomato_interfaces.msg import TomatoRipenessArray


class ControllerNode(Node):
    """
    Convert tomato detections and stereo disparity into robot joint commands.

    Main pipeline:
        ripeness detection + disparity
        -> tomato surface point in left camera optical frame
        -> tomato surface point in base_link
        -> pregrasp/contact/retreat Cartesian waypoints
        -> analytical IK
        -> optional service approval
        -> motor joint commands

    Coordinate conventions:
        Left camera optical frame:
            +X = image right
            +Y = image down
            +Z = forward from the lens

        Robot base frame:
            +X = forward
            +Y = robot left
            +Z = up
    """

    JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4"]

    def __init__(self):
        super().__init__("controller_node")

        self.declare_controller_parameters()
        self.load_controller_parameters()
        self.validate_controller_parameters()

        robot_description_xml = str(
            self.get_parameter("robot_description").value
        )
        if not robot_description_xml:
            raise RuntimeError(
                "robot_description parameter is empty. "
                "Pass the URDF/xacro into this node."
            )

        self.ik_solver = TomatoArmIK.from_robot_description(
            robot_description_xml
        )

        # Motion state
        self.joint_names = list(self.JOINT_NAMES)
        self.queued_waypoint_commands = []
        self.is_motion_in_progress = False
        self.pending_approval_candidate = None
        self.active_candidate = None

        # Camera state
        self.cv_bridge = CvBridge()
        self.left_camera_intrinsics = None
        self.has_logged_camera_intrinsics = False

        # Publishers, services, timers, and subscribers
        self.joint_command_publisher = self.create_publisher(
            Float64MultiArray,
            self.joint_command_topic,
            10,
        )

        self.motion_timer = self.create_timer(
            self.command_interval_seconds,
            self.publish_next_waypoint_command,
        )

        self.approval_service = self.create_service(
            SetBool,
            self.approval_service_name,
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

        self.log_startup_configuration()

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def declare_controller_parameters(self):
        self.declare_parameter("robot_description", "")

        # Disparity filtering
        self.declare_parameter("min_valid_disparity", 1.0)
        self.declare_parameter("max_valid_disparity", 400.0)
        self.declare_parameter("min_valid_ratio", 0.10)

        # Fraction of the full bbox width/height removed in total.
        # Example: 0.40 removes 20% from each side and keeps the center 60%.
        self.declare_parameter("roi_shrink", 0.30)

        # Use a higher disparity percentile to favor the camera-facing surface
        # instead of a depth closer to the center of a curved tomato.
        self.declare_parameter("surface_disparity_percentile", 75.0)

        # Manual eye-to-hand transform for the left rectified camera.
        # Approximate mount: 20 cm behind, 65 cm up, half-baseline left-camera
        # offset, pitched 45 degrees downward.
        self.declare_parameter("camera_x_m", -0.20)
        self.declare_parameter("camera_y_m", 0.0524)
        self.declare_parameter("camera_z_m", 0.65)
        self.declare_parameter("camera_pitch_down_deg", 45.0)

        # Horizontal tomato-relative trajectory
        self.declare_parameter("pregrasp_offset_m", 0.05)
        self.declare_parameter("retreat_offset_m", 0.05)
        self.declare_parameter("tool_angle_from_horizontal", 0.0)
        self.declare_parameter("elbow_solution", "up")

        # Contact-point corrections
        # Positive contact_surface_offset_m stops earlier along robot -X.
        # Positive contact_y_offset_m shifts toward robot left.
        # Positive contact_z_offset_m shifts upward.
        self.declare_parameter("contact_surface_offset_m", 0.015)
        self.declare_parameter("contact_y_offset_m", 0.0)
        self.declare_parameter("contact_z_offset_m", 0.0)

        # Motor command publishing
        self.declare_parameter("enable_motor_commands", False)
        self.declare_parameter(
            "joint_command_topic",
            "/joint_target_positions",
        )
        self.declare_parameter("command_interval_sec", 2.0)

        # Convert the ROS/URDF base-yaw sign to the physical servo sign.
        self.declare_parameter("invert_joint_1_command", True)

        # Manual service approval
        self.declare_parameter("require_manual_approval", True)
        self.declare_parameter(
            "approval_service_name",
            "/controller/set_motion_approval",
        )

    def load_controller_parameters(self):
        # Disparity filtering
        self.minimum_valid_disparity_px = float(
            self.get_parameter("min_valid_disparity").value
        )
        self.maximum_valid_disparity_px = float(
            self.get_parameter("max_valid_disparity").value
        )
        self.minimum_valid_disparity_ratio = float(
            self.get_parameter("min_valid_ratio").value
        )
        self.roi_total_shrink_fraction = float(
            self.get_parameter("roi_shrink").value
        )
        self.surface_disparity_percentile = float(
            self.get_parameter("surface_disparity_percentile").value
        )

        # Camera pose in base_link
        self.camera_base_x_m = float(
            self.get_parameter("camera_x_m").value
        )
        self.camera_base_y_m = float(
            self.get_parameter("camera_y_m").value
        )
        self.camera_base_z_m = float(
            self.get_parameter("camera_z_m").value
        )
        self.camera_pitch_down_degrees = float(
            self.get_parameter("camera_pitch_down_deg").value
        )

        # Tomato-relative trajectory
        self.pregrasp_distance_m = float(
            self.get_parameter("pregrasp_offset_m").value
        )
        self.retreat_distance_m = float(
            self.get_parameter("retreat_offset_m").value
        )
        self.tool_angle_from_horizontal_rad = float(
            self.get_parameter("tool_angle_from_horizontal").value
        )
        self.elbow_configuration = str(
            self.get_parameter("elbow_solution").value
        )

        # Contact corrections
        self.contact_standoff_m = float(
            self.get_parameter("contact_surface_offset_m").value
        )
        self.contact_lateral_offset_m = float(
            self.get_parameter("contact_y_offset_m").value
        )
        self.contact_vertical_offset_m = float(
            self.get_parameter("contact_z_offset_m").value
        )

        # Motor output
        self.motor_commands_enabled = bool(
            self.get_parameter("enable_motor_commands").value
        )
        self.joint_command_topic = str(
            self.get_parameter("joint_command_topic").value
        )
        self.command_interval_seconds = float(
            self.get_parameter("command_interval_sec").value
        )
        self.invert_base_yaw_motor_command = bool(
            self.get_parameter("invert_joint_1_command").value
        )

        # Approval
        self.manual_approval_required = bool(
            self.get_parameter("require_manual_approval").value
        )
        self.approval_service_name = str(
            self.get_parameter("approval_service_name").value
        )

    def validate_controller_parameters(self):
        if self.minimum_valid_disparity_px < 0.0:
            raise ValueError("min_valid_disparity must be nonnegative")

        if (
            self.maximum_valid_disparity_px
            <= self.minimum_valid_disparity_px
        ):
            raise ValueError(
                "max_valid_disparity must be greater than "
                "min_valid_disparity"
            )

        if not 0.0 <= self.minimum_valid_disparity_ratio <= 1.0:
            raise ValueError("min_valid_ratio must be between 0 and 1")

        if not 0.0 <= self.roi_total_shrink_fraction < 1.0:
            raise ValueError("roi_shrink must be in the range [0, 1)")

        if not 0.0 <= self.surface_disparity_percentile <= 100.0:
            raise ValueError(
                "surface_disparity_percentile must be between 0 and 100"
            )

        if self.pregrasp_distance_m < 0.0:
            raise ValueError("pregrasp_offset_m must be nonnegative")

        if self.retreat_distance_m < 0.0:
            raise ValueError("retreat_offset_m must be nonnegative")

        if self.contact_standoff_m < 0.0:
            raise ValueError("contact_surface_offset_m must be nonnegative")

        if self.command_interval_seconds <= 0.0:
            raise ValueError("command_interval_sec must be greater than 0")

        if self.elbow_configuration not in {"up", "down"}:
            raise ValueError("elbow_solution must be 'up' or 'down'")

    def log_startup_configuration(self):
        self.get_logger().info("CONTROLLER STARTED")
        self.get_logger().info(
            f"Motor command topic: {self.joint_command_topic}, "
            f"motor_commands_enabled={self.motor_commands_enabled}"
        )
        self.get_logger().info(
            f"Motion approval service: {self.approval_service_name}, "
            f"manual_approval_required={self.manual_approval_required}"
        )
        self.get_logger().info(
            "Tomato depth settings: "
            f"roi_total_shrink={self.roi_total_shrink_fraction:.2f}, "
            f"surface_percentile={self.surface_disparity_percentile:.1f}"
        )
        self.get_logger().info(
            "Contact corrections in base_link: "
            f"X=-{self.contact_standoff_m:.3f} m, "
            f"Y={self.contact_lateral_offset_m:+.3f} m, "
            f"Z={self.contact_vertical_offset_m:+.3f} m"
        )

    # ------------------------------------------------------------------
    # Camera intrinsics and coordinate transforms
    # ------------------------------------------------------------------

    def left_camera_info_callback(self, message: CameraInfo):
        """Cache left rectified camera intrinsics."""

        projection_matrix = message.p
        intrinsic_matrix = message.k

        if projection_matrix[0] != 0.0 and projection_matrix[5] != 0.0:
            focal_x_px = float(projection_matrix[0])
            focal_y_px = float(projection_matrix[5])
            principal_x_px = float(projection_matrix[2])
            principal_y_px = float(projection_matrix[6])
        else:
            # Fall back to K if the rectified projection matrix is unavailable.
            focal_x_px = float(intrinsic_matrix[0])
            focal_y_px = float(intrinsic_matrix[4])
            principal_x_px = float(intrinsic_matrix[2])
            principal_y_px = float(intrinsic_matrix[5])

        self.left_camera_intrinsics = {
            "focal_x_px": focal_x_px,
            "focal_y_px": focal_y_px,
            "principal_x_px": principal_x_px,
            "principal_y_px": principal_y_px,
        }

        if not self.has_logged_camera_intrinsics:
            self.get_logger().info(
                "Cached left camera intrinsics: "
                f"fx={focal_x_px:.2f}, "
                f"fy={focal_y_px:.2f}, "
                f"cx={principal_x_px:.2f}, "
                f"cy={principal_y_px:.2f}"
            )
            self.has_logged_camera_intrinsics = True

    def back_project_pixel_to_camera_point(
        self,
        pixel_u,
        pixel_v,
        optical_depth_m,
    ):
        """
        Back-project a rectified pixel and optical-axis depth into the left
        camera optical frame.
        """

        if self.left_camera_intrinsics is None:
            return None

        focal_x_px = self.left_camera_intrinsics["focal_x_px"]
        focal_y_px = self.left_camera_intrinsics["focal_y_px"]
        principal_x_px = self.left_camera_intrinsics["principal_x_px"]
        principal_y_px = self.left_camera_intrinsics["principal_y_px"]

        camera_x_m = (
            (float(pixel_u) - principal_x_px)
            * optical_depth_m
            / focal_x_px
        )
        camera_y_m = (
            (float(pixel_v) - principal_y_px)
            * optical_depth_m
            / focal_y_px
        )

        return {
            "x_m": camera_x_m,
            "y_m": camera_y_m,
            "z_m": optical_depth_m,
        }

    def get_camera_to_base_rotation(self):
        """
        Return the rotation from the left rectified camera optical frame to
        robot base_link.
        """

        pitch_down_rad = np.deg2rad(self.camera_pitch_down_degrees)

        # Columns are the camera optical axes expressed in base_link:
        # camera +X = robot right = base -Y
        # camera +Y = image down = base backward/down
        # camera +Z = lens forward = base forward/down
        return np.array(
            [
                [0.0, -np.sin(pitch_down_rad), np.cos(pitch_down_rad)],
                [-1.0, 0.0, 0.0],
                [0.0, -np.cos(pitch_down_rad), -np.sin(pitch_down_rad)],
            ],
            dtype=float,
        )

    def transform_camera_point_to_base(self, camera_point):
        """Transform a 3D point from the camera optical frame to base_link."""

        camera_point_vector = np.array(
            [
                camera_point["x_m"],
                camera_point["y_m"],
                camera_point["z_m"],
            ],
            dtype=float,
        )

        camera_rotation_in_base = self.get_camera_to_base_rotation()
        camera_origin_in_base = np.array(
            [
                self.camera_base_x_m,
                self.camera_base_y_m,
                self.camera_base_z_m,
            ],
            dtype=float,
        )

        base_point_vector = (
            camera_rotation_in_base @ camera_point_vector
            + camera_origin_in_base
        )

        return {
            "x_m": float(base_point_vector[0]),
            "y_m": float(base_point_vector[1]),
            "z_m": float(base_point_vector[2]),
        }

    # ------------------------------------------------------------------
    # Bounding boxes and disparity
    # ------------------------------------------------------------------

    @staticmethod
    def shrink_bounding_box(
        x_min,
        y_min,
        x_max,
        y_max,
        total_shrink_fraction,
    ):
        """
        Shrink a bbox inward to sample the tomato interior instead of its
        edges and background.

        total_shrink_fraction is split equally across both sides.
        For example, 0.40 removes 20% from the left, right, top, and bottom.
        """

        box_width = x_max - x_min
        box_height = y_max - y_min

        horizontal_margin = int(box_width * total_shrink_fraction / 2.0)
        vertical_margin = int(box_height * total_shrink_fraction / 2.0)

        return (
            x_min + horizontal_margin,
            y_min + vertical_margin,
            x_max - horizontal_margin,
            y_max - vertical_margin,
        )

    @staticmethod
    def clamp_bounding_box(
        x_min,
        y_min,
        x_max,
        y_max,
        image_width,
        image_height,
    ):
        """Clamp a bounding box to valid image coordinates."""

        clamped_x_min = max(0, min(int(x_min), image_width - 1))
        clamped_x_max = max(0, min(int(x_max), image_width))
        clamped_y_min = max(0, min(int(y_min), image_height - 1))
        clamped_y_max = max(0, min(int(y_max), image_height))

        return (
            clamped_x_min,
            clamped_y_min,
            clamped_x_max,
            clamped_y_max,
        )

    def estimate_tomato_surface_depth(
        self,
        disparity_image,
        disparity_message,
        tomato_detection,
    ):
        """
        Estimate the camera-facing tomato surface depth from the interior of
        its YOLO bounding box.
        """

        image_height, image_width = disparity_image.shape[:2]

        roi_x_min, roi_y_min, roi_x_max, roi_y_max = (
            self.clamp_bounding_box(
                tomato_detection.x1,
                tomato_detection.y1,
                tomato_detection.x2,
                tomato_detection.y2,
                image_width,
                image_height,
            )
        )

        if roi_x_max <= roi_x_min or roi_y_max <= roi_y_min:
            return None

        roi_x_min, roi_y_min, roi_x_max, roi_y_max = (
            self.shrink_bounding_box(
                roi_x_min,
                roi_y_min,
                roi_x_max,
                roi_y_max,
                self.roi_total_shrink_fraction,
            )
        )

        roi_x_min, roi_y_min, roi_x_max, roi_y_max = (
            self.clamp_bounding_box(
                roi_x_min,
                roi_y_min,
                roi_x_max,
                roi_y_max,
                image_width,
                image_height,
            )
        )

        if roi_x_max <= roi_x_min or roi_y_max <= roi_y_min:
            return None

        disparity_roi = disparity_image[
            roi_y_min:roi_y_max,
            roi_x_min:roi_x_max,
        ]

        valid_disparity_mask = (
            np.isfinite(disparity_roi)
            & (disparity_roi > self.minimum_valid_disparity_px)
            & (disparity_roi < self.maximum_valid_disparity_px)
        )

        valid_pixel_count = int(np.count_nonzero(valid_disparity_mask))
        total_pixel_count = int(disparity_roi.size)

        if total_pixel_count == 0:
            return None

        valid_pixel_ratio = valid_pixel_count / total_pixel_count

        if (
            valid_pixel_count == 0
            or valid_pixel_ratio < self.minimum_valid_disparity_ratio
        ):
            return {
                "is_valid": False,
                "roi_x_min": roi_x_min,
                "roi_y_min": roi_y_min,
                "roi_x_max": roi_x_max,
                "roi_y_max": roi_y_max,
                "valid_pixel_count": valid_pixel_count,
                "total_pixel_count": total_pixel_count,
                "valid_pixel_ratio": valid_pixel_ratio,
            }

        valid_disparities_px = disparity_roi[valid_disparity_mask]

        median_disparity_px = float(np.median(valid_disparities_px))
        mean_disparity_px = float(np.mean(valid_disparities_px))

        # Larger disparity means a closer surface. The selected percentile
        # biases the depth toward the camera-facing side of the tomato.
        surface_disparity_px = float(
            np.percentile(
                valid_disparities_px,
                self.surface_disparity_percentile,
            )
        )

        optical_depth_m = (
            abs(disparity_message.f * disparity_message.t)
            / surface_disparity_px
        )

        roi_center_u_px = int((roi_x_min + roi_x_max) / 2)
        roi_center_v_px = int((roi_y_min + roi_y_max) / 2)

        return {
            "is_valid": True,
            "roi_x_min": roi_x_min,
            "roi_y_min": roi_y_min,
            "roi_x_max": roi_x_max,
            "roi_y_max": roi_y_max,
            "roi_center_u_px": roi_center_u_px,
            "roi_center_v_px": roi_center_v_px,
            "valid_pixel_count": valid_pixel_count,
            "total_pixel_count": total_pixel_count,
            "valid_pixel_ratio": valid_pixel_ratio,
            "median_disparity_px": median_disparity_px,
            "mean_disparity_px": mean_disparity_px,
            "surface_disparity_px": surface_disparity_px,
            "optical_depth_m": optical_depth_m,
        }

    # ------------------------------------------------------------------
    # Waypoints and inverse kinematics
    # ------------------------------------------------------------------

    def create_horizontal_approach_waypoints(self, estimated_surface_base):
        """
        Create the tomato-relative pregrasp, contact, and retreat waypoints.

        contact_surface_offset_m moves contact toward base -X so the tool stops
        before the estimated stereo point.

        contact_y_offset_m and contact_z_offset_m are measured hand-eye/contact
        corrections in base_link.
        """

        estimated_surface_x_m = estimated_surface_base["x_m"]
        estimated_surface_y_m = estimated_surface_base["y_m"]
        estimated_surface_z_m = estimated_surface_base["z_m"]

        contact_x_m = estimated_surface_x_m - self.contact_standoff_m
        contact_y_m = (
            estimated_surface_y_m + self.contact_lateral_offset_m
        )
        contact_z_m = (
            estimated_surface_z_m + self.contact_vertical_offset_m
        )

        return [
            {
                "name": "pregrasp",
                "x_m": contact_x_m - self.pregrasp_distance_m,
                "y_m": contact_y_m,
                "z_m": contact_z_m,
                "tool_angle_rad": self.tool_angle_from_horizontal_rad,
            },
            {
                "name": "contact",
                "x_m": contact_x_m,
                "y_m": contact_y_m,
                "z_m": contact_z_m,
                "tool_angle_rad": self.tool_angle_from_horizontal_rad,
            },
            {
                "name": "retreat",
                "x_m": contact_x_m - self.retreat_distance_m,
                "y_m": contact_y_m,
                "z_m": contact_z_m,
                "tool_angle_rad": self.tool_angle_from_horizontal_rad,
            },
        ]

    def solve_waypoint_sequence(self, waypoints, detection_id):
        """Run IK for every Cartesian waypoint in the tomato trajectory."""

        waypoint_commands = []

        for waypoint in waypoints:
            ik_result = self.ik_solver.solve(
                waypoint["x_m"],
                waypoint["y_m"],
                waypoint["z_m"],
                tool_angle_from_horizontal=waypoint["tool_angle_rad"],
                elbow_solution=self.elbow_configuration,
                target_is_tool_tip=True,
            )

            if not ik_result.success:
                self.get_logger().warn(
                    f"id={detection_id}: IK failed for "
                    f"{waypoint['name']}: {ik_result.reason}"
                )
                return None

            waypoint_commands.append(
                {
                    "name": waypoint["name"],
                    "waypoint": waypoint,
                    "joint_angles": ik_result.joint_angles,
                    "ik_result": ik_result,
                }
            )

        return waypoint_commands

    # ------------------------------------------------------------------
    # Motor command conversion and motion sequencing
    # ------------------------------------------------------------------

    def convert_ros_angles_to_motor_angles(self, ros_joint_angles):
        """
        Convert logical ROS/URDF joint angles into physical motor commands.

        The base servo is mounted with the opposite sign, so joint_1 may be
        inverted only at this hardware boundary. Perception, TF, RViz, and IK
        continue using the ROS convention.
        """

        motor_joint_angles = {}

        for joint_name in self.joint_names:
            motor_angle_rad = float(ros_joint_angles[joint_name])

            if (
                joint_name == "joint_1"
                and self.invert_base_yaw_motor_command
            ):
                motor_angle_rad = -motor_angle_rad

            motor_joint_angles[joint_name] = motor_angle_rad

        return motor_joint_angles

    def publish_joint_angles(self, waypoint_name, ros_joint_angles):
        motor_joint_angles = self.convert_ros_angles_to_motor_angles(
            ros_joint_angles
        )

        message = Float64MultiArray()

        joint_dimension = MultiArrayDimension()
        joint_dimension.label = "joints"
        joint_dimension.size = len(self.joint_names)
        joint_dimension.stride = len(self.joint_names)
        message.layout.dim.append(joint_dimension)

        message.data = [
            motor_joint_angles[joint_name]
            for joint_name in self.joint_names
        ]

        self.joint_command_publisher.publish(message)

        self.get_logger().info(
            f"Published {waypoint_name} motor command: "
            f"{motor_joint_angles}"
        )

    def reset_motion_state(self):
        """Clear all state associated with the current tomato approach."""

        self.queued_waypoint_commands = []
        self.is_motion_in_progress = False
        self.pending_approval_candidate = None
        self.active_candidate = None

    def start_motion_sequence(self, candidate):
        if self.is_motion_in_progress:
            self.get_logger().info(
                "Motion already in progress; not starting another approach"
            )
            return False

        self.active_candidate = candidate

        if not self.motor_commands_enabled:
            self.get_logger().warn(
                "Motor commands are disabled. Set "
                "enable_motor_commands:=true to publish to the motor node."
            )

            for waypoint_command in candidate["waypoint_commands"]:
                waypoint = waypoint_command["waypoint"]
                motor_angles = self.convert_ros_angles_to_motor_angles(
                    waypoint_command["joint_angles"]
                )

                self.get_logger().info(
                    f"DRY RUN {waypoint_command['name']} "
                    f"target_base=("
                    f"x={waypoint['x_m']:.3f}, "
                    f"y={waypoint['y_m']:.3f}, "
                    f"z={waypoint['z_m']:.3f}), "
                    f"ros_joints={waypoint_command['joint_angles']}, "
                    f"motor_joints={motor_angles}"
                )

            self.reset_motion_state()
            self.get_logger().info(
                "Dry run complete. Controller is ready for another tomato."
            )
            return True

        self.queued_waypoint_commands = list(
            candidate["waypoint_commands"]
        )
        self.is_motion_in_progress = True

        tomato_detection = candidate["detection"]
        self.get_logger().info(
            f"Starting horizontal approach for "
            f"id={tomato_detection.detection_id}, "
            f"ripeness={tomato_detection.final_ripeness}"
        )

        # Send the first waypoint immediately. The timer sends the rest.
        self.publish_next_waypoint_command()
        return True

    def finish_motion_sequence(self):
        """Finish the active motion and re-arm for another tomato."""

        completed_detection_id = None

        if self.active_candidate is not None:
            completed_detection_id = self.active_candidate[
                "detection"
            ].detection_id

        self.reset_motion_state()

        if completed_detection_id is None:
            self.get_logger().info("Motion sequence complete")
        else:
            self.get_logger().info(
                "Motion sequence complete for "
                f"detection id={completed_detection_id}"
            )

        self.get_logger().info(
            "Controller is ready for another tomato approach and approval."
        )

    def publish_next_waypoint_command(self):
        if not self.is_motion_in_progress:
            return

        if not self.queued_waypoint_commands:
            self.finish_motion_sequence()
            return

        next_waypoint_command = self.queued_waypoint_commands.pop(0)
        self.publish_joint_angles(
            next_waypoint_command["name"],
            next_waypoint_command["joint_angles"],
        )

    # ------------------------------------------------------------------
    # Candidate selection and approval
    # ------------------------------------------------------------------

    @staticmethod
    def get_ripeness_priority(ripeness):
        if ripeness == "fully_ripened":
            return 3
        if ripeness == "half_ripened":
            return 2
        if ripeness == "green":
            return 1
        return 0

    def request_motion_approval(self, candidate):
        """
        Store one candidate and wait for approval through the SetBool service.
        """

        if not self.manual_approval_required:
            self.start_motion_sequence(candidate)
            return

        if self.is_motion_in_progress:
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

        tomato_detection = candidate["detection"]
        camera_surface_point = candidate["camera_surface_point"]
        estimated_surface_base = candidate["estimated_surface_base"]
        waypoint_commands = candidate["waypoint_commands"]

        self.get_logger().warn("=" * 80)
        self.get_logger().warn(
            "SERVICE APPROVAL REQUIRED BEFORE MOTOR COMMANDS"
        )
        self.get_logger().warn("=" * 80)

        self.get_logger().warn(
            f"Detection id: {tomato_detection.detection_id}"
        )
        self.get_logger().warn(
            f"Ripeness: {tomato_detection.final_ripeness}"
        )
        self.get_logger().warn(
            f"Confidence: {tomato_detection.yolo_confidence:.2f}"
        )

        self.get_logger().warn(
            "Estimated camera-facing surface in camera frame: "
            f"x={camera_surface_point['x_m']:.3f}, "
            f"y={camera_surface_point['y_m']:.3f}, "
            f"z={camera_surface_point['z_m']:.3f} m"
        )

        self.get_logger().warn(
            "Estimated camera-facing surface in base_link before contact "
            "offsets: "
            f"x={estimated_surface_base['x_m']:.3f}, "
            f"y={estimated_surface_base['y_m']:.3f}, "
            f"z={estimated_surface_base['z_m']:.3f} m"
        )

        self.get_logger().warn(
            "Applied contact corrections: "
            f"X=-{self.contact_standoff_m:.3f} m, "
            f"Y={self.contact_lateral_offset_m:+.3f} m, "
            f"Z={self.contact_vertical_offset_m:+.3f} m"
        )

        self.get_logger().warn("Planned waypoint joint commands:")
        for waypoint_command in waypoint_commands:
            waypoint = waypoint_command["waypoint"]
            ros_joint_angles = waypoint_command["joint_angles"]
            motor_joint_angles = self.convert_ros_angles_to_motor_angles(
                ros_joint_angles
            )

            self.get_logger().warn(
                f"  {waypoint['name']}: "
                f"target_base=("
                f"x={waypoint['x_m']:.3f}, "
                f"y={waypoint['y_m']:.3f}, "
                f"z={waypoint['z_m']:.3f}), "
                f"ros_joints=("
                f"j1={ros_joint_angles['joint_1']:.3f}, "
                f"j2={ros_joint_angles['joint_2']:.3f}, "
                f"j3={ros_joint_angles['joint_3']:.3f}, "
                f"j4={ros_joint_angles['joint_4']:.3f}), "
                f"motor_j1={motor_joint_angles['joint_1']:.3f} rad"
            )

        self.get_logger().warn(
            "Approve with: ros2 service call "
            f"{self.approval_service_name} "
            "std_srvs/srv/SetBool '{data: true}'"
        )
        self.get_logger().warn(
            "Cancel with: ros2 service call "
            f"{self.approval_service_name} "
            "std_srvs/srv/SetBool '{data: false}'"
        )

    def motion_approval_callback(self, request, response):
        """Approve or cancel the currently pending tomato approach."""

        if not self.manual_approval_required:
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

        tomato_detection = self.pending_approval_candidate["detection"]
        detection_id = tomato_detection.detection_id

        if not request.data:
            self.reset_motion_state()
            response.success = True
            response.message = (
                f"Canceled pending motion for detection id={detection_id}"
            )
            self.get_logger().warn(response.message)
            return response

        approved_candidate = self.pending_approval_candidate
        self.pending_approval_candidate = None

        motion_started = self.start_motion_sequence(approved_candidate)

        if not motion_started:
            self.pending_approval_candidate = approved_candidate
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

    # ------------------------------------------------------------------
    # Synchronized perception callback
    # ------------------------------------------------------------------

    def synced_callback(
        self,
        ripeness_message: TomatoRipenessArray,
        disparity_message: DisparityImage,
    ):
        if (
            self.is_motion_in_progress
            or self.pending_approval_candidate is not None
        ):
            return

        if self.left_camera_intrinsics is None:
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

        reachable_candidates = []

        for tomato_detection in ripeness_message.ripenesses:
            depth_estimate = self.estimate_tomato_surface_depth(
                disparity_image,
                disparity_message,
                tomato_detection,
            )

            if depth_estimate is None:
                self.get_logger().warn(
                    f"id={tomato_detection.detection_id}: invalid bbox"
                )
                continue

            if not depth_estimate["is_valid"]:
                self.get_logger().warn(
                    f"id={tomato_detection.detection_id}: "
                    "no reliable disparity in ROI "
                    f"valid={depth_estimate['valid_pixel_count']}/"
                    f"{depth_estimate['total_pixel_count']} "
                    f"ratio={depth_estimate['valid_pixel_ratio']:.2f}"
                )
                continue

            camera_surface_point = (
                self.back_project_pixel_to_camera_point(
                    depth_estimate["roi_center_u_px"],
                    depth_estimate["roi_center_v_px"],
                    depth_estimate["optical_depth_m"],
                )
            )

            if camera_surface_point is None:
                self.get_logger().warn(
                    f"id={tomato_detection.detection_id}: "
                    "could not compute 3D point"
                )
                continue

            estimated_surface_base = self.transform_camera_point_to_base(
                camera_surface_point
            )

            waypoints = self.create_horizontal_approach_waypoints(
                estimated_surface_base
            )

            waypoint_commands = self.solve_waypoint_sequence(
                waypoints,
                tomato_detection.detection_id,
            )

            if waypoint_commands is None:
                continue

            bounding_box_area_px = max(
                0,
                tomato_detection.x2 - tomato_detection.x1,
            ) * max(
                0,
                tomato_detection.y2 - tomato_detection.y1,
            )

            candidate = {
                "detection": tomato_detection,
                "depth_estimate": depth_estimate,
                "camera_surface_point": camera_surface_point,
                "estimated_surface_base": estimated_surface_base,
                "waypoints": waypoints,
                "waypoint_commands": waypoint_commands,
                "bounding_box_area_px": bounding_box_area_px,
                "ripeness_priority": self.get_ripeness_priority(
                    tomato_detection.final_ripeness
                ),
            }

            reachable_candidates.append(candidate)

            contact_waypoint = next(
                waypoint
                for waypoint in waypoints
                if waypoint["name"] == "contact"
            )

            self.get_logger().info(
                f"id={tomato_detection.detection_id}, "
                f"ripeness={tomato_detection.final_ripeness}, "
                f"priority={candidate['ripeness_priority']}, "
                f"confidence={tomato_detection.yolo_confidence:.2f}, "
                f"bbox=({tomato_detection.x1},"
                f"{tomato_detection.y1})-({tomato_detection.x2},"
                f"{tomato_detection.y2}), "
                f"ROI=({depth_estimate['roi_x_min']}:"
                f"{depth_estimate['roi_x_max']}, "
                f"{depth_estimate['roi_y_min']}:"
                f"{depth_estimate['roi_y_max']}), "
                f"center_px=({depth_estimate['roi_center_u_px']},"
                f"{depth_estimate['roi_center_v_px']}), "
                f"valid={depth_estimate['valid_pixel_count']}/"
                f"{depth_estimate['total_pixel_count']}, "
                f"median_disp={depth_estimate['median_disparity_px']:.2f}px, "
                f"surface_disp={depth_estimate['surface_disparity_px']:.2f}px, "
                f"depth={depth_estimate['optical_depth_m']:.3f} m, "
                f"camera_surface=("
                f"x={camera_surface_point['x_m']:.3f}, "
                f"y={camera_surface_point['y_m']:.3f}, "
                f"z={camera_surface_point['z_m']:.3f}) m, "
                f"base_surface=("
                f"x={estimated_surface_base['x_m']:.3f}, "
                f"y={estimated_surface_base['y_m']:.3f}, "
                f"z={estimated_surface_base['z_m']:.3f}) m, "
                f"corrected_contact=("
                f"x={contact_waypoint['x_m']:.3f}, "
                f"y={contact_waypoint['y_m']:.3f}, "
                f"z={contact_waypoint['z_m']:.3f}) m"
            )

            for waypoint_command in waypoint_commands:
                waypoint = waypoint_command["waypoint"]
                self.get_logger().info(
                    f"id={tomato_detection.detection_id}, "
                    f"{waypoint_command['name']} target_base=("
                    f"x={waypoint['x_m']:.3f}, "
                    f"y={waypoint['y_m']:.3f}, "
                    f"z={waypoint['z_m']:.3f}), "
                    f"ros_joints={waypoint_command['joint_angles']}"
                )

        if not reachable_candidates:
            self.get_logger().info("No valid tomato depth candidates")
            return

        self.get_logger().info(
            f"Found {len(reachable_candidates)} "
            "reachable tomato candidate(s)"
        )

        # Current behavior: choose the highest ripeness priority, then the
        # largest bounding box. This will later be replaced by manual ID
        # selection in the multi-tomato execution pipeline.
        selected_candidate = max(
            reachable_candidates,
            key=lambda candidate: (
                candidate["ripeness_priority"],
                candidate["bounding_box_area_px"],
            ),
        )

        self.request_motion_approval(selected_candidate)


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
