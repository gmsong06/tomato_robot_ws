import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from stereo_msgs.msg import DisparityImage
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from std_srvs.srv import SetBool
from tomato_interfaces.msg import TomatoRipenessArray

from message_filters import Subscriber, ApproximateTimeSynchronizer
# from urdf_parser_py.urdf import URDF
from rclpy.qos import qos_profile_sensor_data
from tomato_control.ik_solver import TomatoArmIK


class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller_node")

        self.declare_parameter("robot_description", "")
        self.declare_parameter("min_valid_disparity", 1.0)
        self.declare_parameter("max_valid_disparity", 400.0)
        self.declare_parameter("min_valid_ratio", 0.10)
        self.declare_parameter("roi_shrink", 0.20)

        # Manual eye-to-hand transform parameters
        # 20cm behind, 65cm up, half the baseline from calibration as y offset for left camera frame, 45 deg downwards angle
        self.declare_parameter("camera_x_m", -0.20)
        self.declare_parameter("camera_y_m", 0.0524)
        self.declare_parameter("camera_z_m", 0.65)
        self.declare_parameter("camera_pitch_down_deg", 45.0)

        # Horizontal approach test params
        self.declare_parameter("pregrasp_offset_m", 0.05)
        self.declare_parameter("retreat_offset_m", 0.05)
        self.declare_parameter("tool_angle_from_horizontal", 0.0)
        self.declare_parameter("elbow_solution", "up")

        # Motor command publishing params
        self.declare_parameter("enable_motor_commands", False)
        self.declare_parameter("joint_command_topic", "/joint_target_positions")
        self.declare_parameter("command_interval_sec", 2.0)

        self.declare_parameter("require_manual_approval", True)
        self.declare_parameter(
            "approval_service_name",
            "/controller/set_motion_approval",
        )

        self.declare_parameter("surface_disparity_percentile", 75.0)

        self.declare_parameter("contact_surface_offset_m", 0.015)
        
        self.declare_parameter("invert_joint_1_command", True)

        self.declare_parameter("contact_y_offset_m", 0.0)
        self.declare_parameter("contact_z_offset_m", 0.0)


        self.surface_disparity_percentile = float(
            self.get_parameter("surface_disparity_percentile").value
        )

        self.contact_surface_offset_m = float(
            self.get_parameter("contact_surface_offset_m").value
        )

        self.invert_joint_1_command = bool(
            self.get_parameter("invert_joint_1_command").value
        )

        self.contact_y_offset_m = float(
            self.get_parameter("contact_y_offset_m").value
        )

        self.contact_z_offset_m = float(
            self.get_parameter("contact_z_offset_m").value
        )

        robot_description = self.get_parameter("robot_description").value
        if robot_description == "":
            raise RuntimeError("robot_description parameter is empty. Pass the URDF/xacro into this node.")

        self.ik_solver = TomatoArmIK.from_robot_description(robot_description)

        self.min_valid_disparity = float(
            self.get_parameter("min_valid_disparity").value
        )
        self.max_valid_disparity = float(
            self.get_parameter("max_valid_disparity").value
        )
        self.min_valid_ratio = float(
            self.get_parameter("min_valid_ratio").value
        )
        self.roi_shrink = float(
            self.get_parameter("roi_shrink").value
        )

        self.camera_x_m = float(
            self.get_parameter("camera_x_m").value
        )
        self.camera_y_m = float(
            self.get_parameter("camera_y_m").value
        )
        self.camera_z_m = float(
            self.get_parameter("camera_z_m").value
        )
        self.camera_pitch_down_deg = float(
            self.get_parameter("camera_pitch_down_deg").value
        )

        self.pregrasp_offset_m = float(
            self.get_parameter("pregrasp_offset_m").value
        )
        self.retreat_offset_m = float(
            self.get_parameter("retreat_offset_m").value
        )
        self.tool_angle_from_horizontal = float(
            self.get_parameter("tool_angle_from_horizontal").value
        )
        self.elbow_solution = str(
            self.get_parameter("elbow_solution").value
        )

        self.enable_motor_commands = bool(
            self.get_parameter("enable_motor_commands").value
        )
        self.joint_command_topic = str(
            self.get_parameter("joint_command_topic").value
        )
        self.command_interval_sec = float(
            self.get_parameter("command_interval_sec").value
        )

        self.require_manual_approval = bool(
            self.get_parameter("require_manual_approval").value
        )
        self.approval_service_name = str(
            self.get_parameter("approval_service_name").value
        )

        self.joint_order = ["joint_1", "joint_2", "joint_3", "joint_4"]
        self.motion_queue = []
        self.motion_in_progress = False
        self.pending_candidate = None
        self.current_candidate = None

        self.joint_command_pub = self.create_publisher(
            Float64MultiArray,
            self.joint_command_topic,
            10,
        )

        self.motion_timer = self.create_timer(
            self.command_interval_sec,
            self.publish_next_motion_waypoint,
        )

        self.approval_service = self.create_service(
            SetBool,
            self.approval_service_name,
            self.motion_approval_callback,
        )

        self.bridge = CvBridge()

        self.left_intrinsics = None
        self.logged_camera_info = False

        self.left_camera_info_sub = self.create_subscription(
            CameraInfo,
            "/stereo/left/camera_info",
            self.left_camera_info_callback,
            qos_profile_sensor_data,
        )

        self.ripeness_sub = Subscriber(
            self,
            TomatoRipenessArray,
            "/tomato_ripeness",
        )

        self.disparity_sub = Subscriber(
            self,
            DisparityImage,
            "/stereo/disparity",
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.ripeness_sub, self.disparity_sub],
            queue_size=10,
            slop=0.15,
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info("CONTROLLER STARTED")
        self.get_logger().info(
            f"Motor command topic: {self.joint_command_topic}, "
            f"enable_motor_commands={self.enable_motor_commands}"
        )
        self.get_logger().info(
            f"Motion approval service: {self.approval_service_name}, "
            f"require_manual_approval={self.require_manual_approval}"
        )


    def get_joint_xyz(self, robot, joint_name):
        for joint in robot.joints:
            if joint.name == joint_name:
                if joint.origin is None or joint.origin.xyz is None:
                    return [0.0, 0.0, 0.0]
                return [float(v) for v in joint.origin.xyz]

        raise ValueError(f"Joint {joint_name} not found in URDF")

    def left_camera_info_callback(self, msg: CameraInfo):
        """
        Cache left rectified camera intrinsics.
        """

        p = msg.p
        k = msg.k

        if p[0] != 0.0 and p[5] != 0.0:
            fx = float(p[0])
            fy = float(p[5])
            cx = float(p[2])
            cy = float(p[6])
        # Fallback to K if P isn't avail
        else:
            fx = float(k[0])
            fy = float(k[4])
            cx = float(k[2])
            cy = float(k[5])

        self.left_intrinsics = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }

        if not self.logged_camera_info:
            self.get_logger().info(
                f"Cached left camera intrinsics: "
                f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}"
            )
            self.logged_camera_info = True

    def shrink_bbox(self, x1, y1, x2, y2, shrink_ratio):
        """
        Shrink bbox inward so disparity is sampled from the tomato interior,
        not the tomato edge/background. By default, it shrinks to 20%.
        """

        w = x2 - x1
        h = y2 - y1

        dx = int(w * shrink_ratio / 2.0)
        dy = int(h * shrink_ratio / 2.0)

        return x1 + dx, y1 + dy, x2 - dx, y2 - dy

    def clamp_bbox(self, x1, y1, x2, y2, image_w, image_h):
        x1 = max(0, min(int(x1), image_w - 1))
        x2 = max(0, min(int(x2), image_w))
        y1 = max(0, min(int(y1), image_h - 1))
        y2 = max(0, min(int(y2), image_h))

        return x1, y1, x2, y2

    def get_3d_point(self, u, v, depth_m):
        """
        Back-project a 2D pixel and depth into a 3D point.

        Output frame:
        left rectified camera optical frame

        ROS optical camera convention:
        X = right
        Y = down
        Z = forward
        """

        if self.left_intrinsics is None:
            return None

        fx = self.left_intrinsics["fx"]
        fy = self.left_intrinsics["fy"]
        cx = self.left_intrinsics["cx"]
        cy = self.left_intrinsics["cy"]

        x_m = (float(u) - cx) * depth_m / fx
        y_m = (float(v) - cy) * depth_m / fy
        z_m = depth_m

        return {
            "x": x_m,
            "y": y_m,
            "z": z_m,
        }


    def get_camera_to_base_rotation(self):
        """
        Return the rotation from left rectified camera optical frame to robot base frame.

        Camera optical frame:
            +X = right in image
            +Y = down in image
            +Z = forward out of lens

        Robot base frame:
            +X = forward
            +Y = left
            +Z = up
        """

        theta = np.deg2rad(self.camera_pitch_down_deg)

        # Columns are the camera optical axes in the robot base frame
        # camera +X = robot right = -Y
        # camera +Y = image down, which becomes down/back
        # camera +Z = lens forward, which becomes forward/down
        return np.array(
            [
                [0.0, -np.sin(theta),  np.cos(theta)],
                [-1.0, 0.0,            0.0],
                [0.0, -np.cos(theta), -np.sin(theta)],
            ],
            dtype=float,
        )


    def camera_to_base_point(self, point_camera):
        """
        Convert a point from left rectified camera optical frame to robot base frame.

        Camera optical frame:
            +X = right in image
            +Y = down in image
            +Z = forward out of lens

        Robot base frame:
            +X = forward
            +Y = left
            +Z = up
        """

        pc = np.array(
            [
                point_camera["x"],
                point_camera["y"],
                point_camera["z"],
            ],
            dtype=float,
        )

        R_base_camera = self.get_camera_to_base_rotation()

        # Translation matrix
        t_base_camera = np.array(
            [
                self.camera_x_m,
                self.camera_y_m,
                self.camera_z_m,
            ],
            dtype=float,
        )

        pb = R_base_camera @ pc + t_base_camera

        return {
            "x": float(pb[0]),
            "y": float(pb[1]),
            "z": float(pb[2]),
        }
        

    def make_horizontal_approach_waypoints(self, point_base):
        """
        Horizontal approach test

        Input:
            point_base is estimated tomato surface point in robot base frame.

        Robot base frame:
            +X = forward
            +Y = left
            +Z = up

        Horizontal approach:
            pregrasp is slightly behind tomato in -X
            contact stops slightly before the estimated depth point
            retreat backs up in -X
        """

        x = point_base["x"]

        # Apply measured lateral and vertical calibration corrections.
        y = point_base["y"] + self.contact_y_offset_m
        z = point_base["z"] + self.contact_z_offset_m

        # Stop slightly before the estimated stereo point so the commanded
        # tool tip does not penetrate toward the tomato center.
        contact_x = x - self.contact_surface_offset_m

        return [
            {
                "name": "pregrasp",
                "x": contact_x - self.pregrasp_offset_m,
                "y": y,
                "z": z,
                "tool_angle_from_horizontal": self.tool_angle_from_horizontal,
            },
            {
                "name": "contact",
                "x": contact_x,
                "y": y,
                "z": z,
                "tool_angle_from_horizontal": self.tool_angle_from_horizontal,
            },
            {
                "name": "retreat",
                "x": contact_x - self.retreat_offset_m,
                "y": y,
                "z": z,
                "tool_angle_from_horizontal": self.tool_angle_from_horizontal,
            },
        ]

    def compute_ik_sequence(self, waypoints, detection_id):
        ik_sequence = []

        for waypoint in waypoints:
            ik_result = self.ik_solver.solve(
                waypoint["x"],
                waypoint["y"],
                waypoint["z"],
                tool_angle_from_horizontal=waypoint["tool_angle_from_horizontal"],
                elbow_solution=self.elbow_solution,
                target_is_tool_tip=True,
            )

            if not ik_result.success:
                self.get_logger().warn(
                    f"id={detection_id}: IK failed for {waypoint['name']}: "
                    f"{ik_result.reason}"
                )
                return None

            ik_sequence.append(
                {
                    "name": waypoint["name"],
                    "waypoint": waypoint,
                    "joint_angles": ik_result.joint_angles,
                    "ik_result": ik_result,
                }
            )

        return ik_sequence

    def publish_joint_angles(self, waypoint_name, joint_angles):
        msg = Float64MultiArray()

        dim = MultiArrayDimension()
        dim.label = "joints"
        dim.size = len(self.joint_order)
        dim.stride = len(self.joint_order)
        msg.layout.dim.append(dim)

        msg.data = []

        for joint_name in self.joint_order:
            angle = float(joint_angles[joint_name])

            # Convert the ROS/URDF joint_1 convention into the physical
            # servo direction if the motor is mounted in the opposite direction.
            if joint_name == "joint_1" and self.invert_joint_1_command:
                angle = -angle

            msg.data.append(angle)

        self.joint_command_pub.publish(msg)

        self.get_logger().info(
            f"Published {waypoint_name} joint command: "
            f"{dict(zip(self.joint_order, msg.data))}"
        )

    def reset_motion_state(self):
        """
        Clear all state associated with the current or pending tomato approach.

        After this runs, the next synchronized perception callback can select
        another tomato and request a new service approval.
        """

        self.motion_queue = []
        self.motion_in_progress = False
        self.pending_candidate = None
        self.current_candidate = None


    def start_motion_sequence(self, candidate):
        if self.motion_in_progress:
            self.get_logger().info(
                "Motion already in progress, not starting a new tomato approach"
            )
            return False

        self.current_candidate = candidate

        if not self.enable_motor_commands:
            self.get_logger().warn(
                "Motor commands are disabled. Set enable_motor_commands:=true to publish to the motor node."
            )

            for command in candidate["ik_sequence"]:
                waypoint = command["waypoint"]
                self.get_logger().info(
                    f"DRY RUN {command['name']} target_base=("
                    f"x={waypoint['x']:.3f}, "
                    f"y={waypoint['y']:.3f}, "
                    f"z={waypoint['z']:.3f}), "
                    f"joints={command['joint_angles']}"
                )

            self.reset_motion_state()
            self.get_logger().info(
                "Dry run complete. Controller is ready for another tomato approach."
            )
            return True

        self.motion_queue = list(candidate["ik_sequence"])
        self.motion_in_progress = True

        detection = candidate["detection"]
        self.get_logger().info(
            f"Starting horizontal approach for id={detection.detection_id}, "
            f"ripeness={detection.final_ripeness}"
        )

        self.publish_next_motion_waypoint()
        return True


    def finish_motion_sequence(self):
        """
        Finish the active motion and re-arm the controller for the next tomato.
        """

        detection_id = None

        if self.current_candidate is not None:
            detection_id = self.current_candidate["detection"].detection_id

        self.reset_motion_state()

        if detection_id is None:
            self.get_logger().info("Motion sequence complete")
        else:
            self.get_logger().info(
                f"Motion sequence complete for detection id={detection_id}"
            )

        self.get_logger().info(
            "Controller is ready for another tomato approach and service approval."
        )


    def publish_next_motion_waypoint(self):
        if not self.motion_in_progress:
            return

        if not self.motion_queue:
            self.finish_motion_sequence()
            return

        command = self.motion_queue.pop(0)
        self.publish_joint_angles(
            command["name"],
            command["joint_angles"],
        )

    def get_roi_depth(self, disparity_image, disparity_msg, detection):
        # Get image height and width from DisparityImage, it's the number of rows and columns respectively
        h, w = disparity_image.shape[:2]

        # Clamp to stay inside image
        x1, y1, x2, y2 = self.clamp_bbox(
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
            w,
            h,
        )

        if x2 <= x1 or y2 <= y1:
            return None

        # Take only the interior of the tomato (I don't think this is a problem because we only care about the center anyway but might change)
        x1, y1, x2, y2 = self.shrink_bbox(
            x1,
            y1,
            x2,
            y2,
            self.roi_shrink,
        )

        # Clamp again, lowk not necessary because shrinking would only make it smaller but js in case
        x1, y1, x2, y2 = self.clamp_bbox(x1, y1, x2, y2, w, h)

        if x2 <= x1 or y2 <= y1:
            return None

        # Extract region of interest
        roi = disparity_image[y1:y2, x1:x2]

        # Pixel is valid if it is finite and disparity is between min and max valid disparity
        valid = (
            np.isfinite(roi)
            & (roi > self.min_valid_disparity)
            & (roi < self.max_valid_disparity)
        )

        valid_count = int(np.count_nonzero(valid))
        total_count = int(roi.size)

        if total_count == 0:
            return None

        # Computes the fraction of roi that is valid
        valid_ratio = valid_count / total_count

        if valid_count == 0 or valid_ratio < self.min_valid_ratio:
            return {
                "valid": False,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "valid_count": valid_count,
                "total_count": total_count,
                "valid_ratio": valid_ratio,
            }

        # Extracts only valid disparities from roi
        valid_disparities = roi[valid]

        # Mean and median disparity (using median right now to avoid outliers)
        median_disparity = float(np.median(valid_disparities))
        mean_disparity = float(np.mean(valid_disparities))

        # Use a higher disparity percentile to bias the estimate toward
        # the camera-facing tomato surface instead of the ROI's middle depth.
        surface_disparity = float(
            np.percentile(
                valid_disparities,
                self.surface_disparity_percentile,
            )
        )

        depth_m = abs(disparity_msg.f * disparity_msg.t) / surface_disparity

        center_u = int((x1 + x2) / 2)
        center_v = int((y1 + y2) / 2)

        return {
            "valid": True,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_u": center_u,
            "center_v": center_v,
            "valid_count": valid_count,
            "total_count": total_count,
            "valid_ratio": valid_ratio,
            "median_disparity": median_disparity,
            "mean_disparity": mean_disparity,
            "depth_m": depth_m,
            "surface_disparity": surface_disparity,
        }

    def ripeness_priority(self, ripeness):
        if ripeness == "fully_ripened":
            return 3
        if ripeness == "half_ripened":
            return 2
        if ripeness == "green":
            return 1
        return 0

    def request_manual_approval(self, candidate):
        """
        Ask for service approval before sending commands to the motor node.
        This is meant for debugging/testing on the real robot.
        """

        if not self.require_manual_approval:
            self.start_motion_sequence(candidate)
            return

        if self.motion_in_progress:
            self.get_logger().info(
                "Motion already in progress, not requesting another approval"
            )
            return

        if self.pending_candidate is not None:
            self.get_logger().info(
                "A tomato approach is already waiting for approval"
            )
            return

        self.pending_candidate = candidate

        detection = candidate["detection"]
        point_camera = candidate["point_3d"]
        point_base = candidate["point_base"]
        waypoint_commands = candidate["ik_sequence"]

        self.get_logger().warn("=" * 80)
        self.get_logger().warn("SERVICE APPROVAL REQUIRED BEFORE MOTOR COMMANDS")
        self.get_logger().warn("=" * 80)

        self.get_logger().warn(f"Detection id: {detection.detection_id}")
        self.get_logger().warn(f"Ripeness: {detection.final_ripeness}")
        self.get_logger().warn(f"Confidence: {detection.yolo_confidence:.2f}")

        self.get_logger().warn(
            "point_camera = "
            f"x={point_camera['x']:.3f}, "
            f"y={point_camera['y']:.3f}, "
            f"z={point_camera['z']:.3f} m"
        )

        self.get_logger().warn(
            "point_base = "
            f"x={point_base['x']:.3f}, "
            f"y={point_base['y']:.3f}, "
            f"z={point_base['z']:.3f} m"
        )

        self.get_logger().warn("Planned waypoint joint commands:")
        for cmd in waypoint_commands:
            waypoint = cmd["waypoint"]
            joint_angles = cmd["joint_angles"]

            self.get_logger().warn(
                f"  {waypoint['name']}: "
                f"target_base=("
                f"x={waypoint['x']:.3f}, "
                f"y={waypoint['y']:.3f}, "
                f"z={waypoint['z']:.3f}), "
                f"joints=("
                f"j1={joint_angles['joint_1']:.3f}, "
                f"j2={joint_angles['joint_2']:.3f}, "
                f"j3={joint_angles['joint_3']:.3f}, "
                f"j4={joint_angles['joint_4']:.3f}) rad"
            )

        self.get_logger().warn(
            f"Approve from another terminal with: ros2 service call "
            f"{self.approval_service_name} std_srvs/srv/SetBool '{{data: true}}'"
        )
        self.get_logger().warn(
            f"Cancel with: ros2 service call "
            f"{self.approval_service_name} std_srvs/srv/SetBool '{{data: false}}'"
        )


    def motion_approval_callback(self, request, response):
        """
        Approve or cancel the currently pending tomato approach.

        request.data = True:
            Approve and start the pending motion sequence.

        request.data = False:
            Cancel and discard the pending motion sequence.
        """

        if not self.require_manual_approval:
            response.success = False
            response.message = (
                "Manual approval is disabled because require_manual_approval is false"
            )
            return response

        if self.pending_candidate is None:
            response.success = False
            response.message = "There is no motion sequence waiting for approval"
            return response

        detection = self.pending_candidate["detection"]
        detection_id = detection.detection_id

        if not request.data:
            self.reset_motion_state()
            response.success = True
            response.message = (
                f"Canceled pending motion for detection id={detection_id}"
            )
            self.get_logger().warn(response.message)
            return response

        candidate = self.pending_candidate
        self.pending_candidate = None

        started = self.start_motion_sequence(candidate)

        if not started:
            self.pending_candidate = candidate
            response.success = False
            response.message = (
                f"Could not start motion for detection id={detection_id} "
                f"because another motion is still active"
            )
            self.get_logger().warn(response.message)
            return response

        response.success = True
        response.message = (
            f"Approved pending motion for detection id={detection_id}"
        )
        self.get_logger().warn(response.message)
        return response


    def synced_callback(
        self,
        ripeness_msg: TomatoRipenessArray,
        disparity_msg: DisparityImage,
    ):

        if self.motion_in_progress or self.pending_candidate is not None:
            return

        if self.left_intrinsics is None:
            self.get_logger().warn(
                "No left camera intrinsics received yet, waiting for /stereo/left/camera_info"
            )
            return

        # Converts the disparity image from a ROS image message into an opencv image
        # 32 bit and one channel
        disparity_image = self.bridge.imgmsg_to_cv2(
            disparity_msg.image,
            desired_encoding="32FC1",
        )

        self.get_logger().info(
            f"Received {len(ripeness_msg.ripenesses)} tomato ripeness result(s)"
        )

        candidates = []

        for detection in ripeness_msg.ripenesses:
            depth_info = self.get_roi_depth(
                disparity_image,
                disparity_msg,
                detection,
            )

            if depth_info is None:
                self.get_logger().warn(
                    f"id={detection.detection_id}: invalid bbox"
                )
                continue

            if not depth_info["valid"]:
                self.get_logger().warn(
                    f"id={detection.detection_id}: no reliable disparity in ROI "
                    f"valid={depth_info['valid_count']}/{depth_info['total_count']} "
                    f"ratio={depth_info['valid_ratio']:.2f}"
                )
                continue
            
            point_3d = self.get_3d_point(
                depth_info["center_u"],
                depth_info["center_v"],
                depth_info["depth_m"],
            )

            if point_3d is None:
                self.get_logger().warn(
                    f"id={detection.detection_id}: could not compute 3D point"
                )
                continue

            point_base = self.camera_to_base_point(point_3d)
            waypoints = self.make_horizontal_approach_waypoints(point_base)
            ik_sequence = self.compute_ik_sequence(
                waypoints,
                detection.detection_id,
            )

            if ik_sequence is None:
                continue

            # Area of original YOLO box
            area = max(0, detection.x2 - detection.x1) * max(0, detection.y2 - detection.y1)

            candidate = {
                "detection": detection,
                "depth": depth_info,
                "point_3d": point_3d,
                "point_base": point_base,
                "waypoints": waypoints,
                "ik_sequence": ik_sequence,
                "area": area,
                "priority": self.ripeness_priority(detection.final_ripeness),
            }

            candidates.append(candidate)

            self.get_logger().info(
                f"id={detection.detection_id}, "
                f"ripeness={detection.final_ripeness}, "
                f"priority={candidate['priority']}, "
                f"confidence={detection.yolo_confidence:.2f}, "
                f"bbox=({detection.x1},{detection.y1})-({detection.x2},{detection.y2}), "
                f"ROI=({depth_info['x1']}:{depth_info['x2']}, "
                f"{depth_info['y1']}:{depth_info['y2']}), "
                f"center_px=({depth_info['center_u']},{depth_info['center_v']}), "
                f"valid={depth_info['valid_count']}/{depth_info['total_count']}, "
                f"median_disp={depth_info['median_disparity']:.2f}px, "
                f"depth={depth_info['depth_m']:.3f} m, "
                f"point_camera=("
                f"x={point_3d['x']:.3f}, "
                f"y={point_3d['y']:.3f}, "
                f"z={point_3d['z']:.3f}) m, "
                f"point_base=("
                f"x={point_base['x']:.3f}, "
                f"y={point_base['y']:.3f}, "
                f"z={point_base['z']:.3f}) m"
            )

            for command in ik_sequence:
                waypoint = command["waypoint"]
                self.get_logger().info(
                    f"id={detection.detection_id}, "
                    f"{command['name']} target_base=("
                    f"x={waypoint['x']:.3f}, "
                    f"y={waypoint['y']:.3f}, "
                    f"z={waypoint['z']:.3f}), "
                    f"joints={command['joint_angles']}"
                )

        if not candidates:
            self.get_logger().info("No valid tomato depth candidates")
            return

        self.get_logger().info(
            f"Logged {len(candidates)} valid tomato depth candidate(s)"
        )

        # Pick the best reachable tomato candidate for this first horizontal approach test.
        # For now, choose highest ripeness priority, then largest detected area.
        best_candidate = max(
            candidates,
            key=lambda candidate: (
                candidate["priority"],
                candidate["area"],
            ),
        )

        self.request_manual_approval(best_candidate)


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
