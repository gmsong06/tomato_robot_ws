import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from stereo_msgs.msg import DisparityImage
from tomato_interfaces.msg import TomatoRipenessArray

from message_filters import Subscriber, ApproximateTimeSynchronizer


class TomatoReactiveControllerNode(Node):
    def __init__(self):
        super().__init__("tomato_reactive_controller_node")

        self.declare_parameter("min_valid_disparity", 1.0)
        self.declare_parameter("max_valid_disparity", 400.0)
        self.declare_parameter("min_valid_ratio", 0.10)
        self.declare_parameter("roi_shrink", 0.20)

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

        self.bridge = CvBridge()

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

        self.motor_pub = self.create_publisher(
            Float64MultiArray,
            "/joint_target_positions",
            10,
        )

        self.get_logger().info("Tomato reactive controller started")

    
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

        # Converts median disparity to depth with stereo formula f * B / disparity
        depth_m = abs(disparity_msg.f * disparity_msg.t) / median_disparity

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
        }


    def ripeness_priority(self, ripeness):
        if ripeness == "fully_ripened":
            return 3
        if ripeness == "half_ripened":
            return 2
        if ripeness == "green":
            return 1
        return 0


    def synced_callback(
        self,
        ripeness_msg: TomatoRipenessArray,
        disparity_msg: DisparityImage,
    ):

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
            
            # Area of original YOLO box
            area = max(0, detection.x2 - detection.x1) * max(0, detection.y2 - detection.y1)

            candidate = {
                "detection": detection,
                "depth": depth_info,
                "area": area,
                "priority": self.ripeness_priority(detection.final_ripeness),
            }

            candidates.append(candidate)

            self.get_logger().info(
                f"id={detection.detection_id}, "
                f"ripeness={detection.final_ripeness}, "
                f"bbox=({detection.x1},{detection.y1})-({detection.x2},{detection.y2}), "
                f"ROI=({depth_info['x1']}:{depth_info['x2']}, "
                f"{depth_info['y1']}:{depth_info['y2']}), "
                f"valid={depth_info['valid_count']}/{depth_info['total_count']}, "
                f"median_disp={depth_info['median_disparity']:.2f}px, "
                f"depth={depth_info['depth_m']:.3f} m"
            )

        if not candidates:
            self.get_logger().info("No valid tomato depth candidates")
            return

        self.get_logger().info(
            f"Logged {len(candidates)} valid tomato depth candidate(s)"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TomatoReactiveControllerNode()

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