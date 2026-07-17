from __future__ import annotations

import numpy as np
from stereo_msgs.msg import DisparityImage

from tomato_control.camera_geometry import CameraGeometry
from tomato_control.controller_models import TomatoCandidate
from tomato_control.horizontal_approach_planner import HorizontalApproachPlanner
from tomato_control.tomato_depth_estimator import TomatoDepthEstimator


class TomatoCandidateBuilder:
    """Build one reachable tomato candidate from detection and disparity data."""

    def __init__(
        self,
        depth_estimator: TomatoDepthEstimator,
        camera_geometry: CameraGeometry,
        approach_planner: HorizontalApproachPlanner,
        logger,
    ):
        self.depth_estimator = depth_estimator
        self.camera_geometry = camera_geometry
        self.approach_planner = approach_planner
        self.logger = logger

    def build(
        self,
        disparity_image: np.ndarray,
        disparity_message: DisparityImage,
        tomato_detection,
    ) -> TomatoCandidate | None:
        detection_id = int(tomato_detection.detection_id)

        depth_estimate = self.depth_estimator.estimate(
            disparity_image,
            disparity_message,
            tomato_detection,
        )

        if depth_estimate is None:
            self.logger.warn(f"id={detection_id}: invalid bounding box")
            return None

        if not depth_estimate.is_valid:
            self.logger.warn(
                f"id={detection_id}: no reliable disparity in ROI "
                f"valid={depth_estimate.valid_pixel_count}/"
                f"{depth_estimate.total_pixel_count} "
                f"ratio={depth_estimate.valid_pixel_ratio:.2f}"
            )
            return None

        if depth_estimate.optical_depth_m is None:
            self.logger.warn(f"id={detection_id}: missing optical depth")
            return None

        camera_surface_point = self.camera_geometry.back_project_pixel(
            depth_estimate.roi.center_u_px,
            depth_estimate.roi.center_v_px,
            depth_estimate.optical_depth_m,
        )

        if camera_surface_point is None:
            self.logger.warn(
                f"id={detection_id}: camera intrinsics are unavailable"
            )
            return None

        estimated_surface_base = (
            self.camera_geometry.transform_camera_point_to_base(
                camera_surface_point
            )
        )

        waypoints = self.approach_planner.create_waypoints(
            estimated_surface_base
        )
        waypoint_commands = self.approach_planner.solve_waypoints(
            waypoints,
            detection_id,
        )

        if waypoint_commands is None:
            return None

        bounding_box_area_px = max(
            0,
            tomato_detection.x2 - tomato_detection.x1,
        ) * max(
            0,
            tomato_detection.y2 - tomato_detection.y1,
        )

        return TomatoCandidate(
            detection=tomato_detection,
            depth_estimate=depth_estimate,
            camera_surface_point=camera_surface_point,
            estimated_surface_base=estimated_surface_base,
            waypoints=waypoints,
            waypoint_commands=waypoint_commands,
            bounding_box_area_px=bounding_box_area_px,
            ripeness_priority=self._ripeness_priority(
                tomato_detection.final_ripeness
            ),
        )

    @staticmethod
    def _ripeness_priority(ripeness: str) -> int:
        priorities = {
            "fully_ripened": 3,
            "half_ripened": 2,
            "green": 1,
        }
        return priorities.get(ripeness, 0)
