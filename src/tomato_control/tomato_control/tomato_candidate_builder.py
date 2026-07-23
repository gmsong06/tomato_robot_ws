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

        estimated_surface_origin = (
            self.camera_geometry.transform_camera_point_to_origin(
                camera_surface_point
            )
        )

        waypoints = self.approach_planner.create_waypoints(
            estimated_surface_origin
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
            estimated_surface_origin=estimated_surface_origin,
            waypoints=waypoints,
            waypoint_commands=waypoint_commands,
            bounding_box_area_px=bounding_box_area_px,
            ripeness_priority=self._ripeness_priority(
                tomato_detection.final_ripeness
            ),
        )

    def build_debug_report(
        self,
        disparity_image: np.ndarray,
        disparity_message: DisparityImage,
        tomato_detection,
    ) -> tuple[bool, str]:
        """Run the same perception/IK calculations and explain each decision."""

        detection_id = int(tomato_detection.detection_id)
        lines: list[str] = [
            "=" * 72,
            f"TOMATO DEBUG REPORT: ID {detection_id}",
            "=" * 72,
            "Detection:",
            f"  bbox: x1={int(tomato_detection.x1)}, "
            f"y1={int(tomato_detection.y1)}, "
            f"x2={int(tomato_detection.x2)}, "
            f"y2={int(tomato_detection.y2)}",
            f"  ripeness: {getattr(tomato_detection, 'final_ripeness', '(unknown)')}",
            f"  YOLO confidence: {float(getattr(tomato_detection, 'yolo_confidence', 0.0)):.4f}",
            "",
            "DEPTH / DISPARITY",
        ]

        depth_estimate = self.depth_estimator.estimate(
            disparity_image,
            disparity_message,
            tomato_detection,
        )

        if depth_estimate is None:
            lines.extend(
                [
                    "  result: FAIL",
                    "  reason: invalid or zero-area bounding box/ROI",
                    "",
                    "FINAL VERDICT: REJECTED BEFORE IK",
                ]
            )
            return False, "\n".join(lines)

        lines.extend(
            [
                "  interior ROI: "
                f"x=[{depth_estimate.roi.x_min}, {depth_estimate.roi.x_max}), "
                f"y=[{depth_estimate.roi.y_min}, {depth_estimate.roi.y_max})",
                f"  valid disparity pixels: {depth_estimate.valid_pixel_count}/"
                f"{depth_estimate.total_pixel_count}",
                f"  valid disparity ratio: {depth_estimate.valid_pixel_ratio:.4f}",
                "  required valid ratio: "
                f"{self.depth_estimator.config.minimum_valid_disparity_ratio:.4f}",
            ]
        )

        if not depth_estimate.is_valid:
            lines.extend(
                [
                    "  result: FAIL",
                    "  reason: insufficient reliable disparity pixels",
                    "",
                    "FINAL VERDICT: REJECTED BEFORE IK",
                ]
            )
            return False, "\n".join(lines)

        lines.extend(
            [
                f"  median disparity: {depth_estimate.median_disparity_px:.6f} px",
                f"  mean disparity: {depth_estimate.mean_disparity_px:.6f} px",
                f"  selected surface disparity: {depth_estimate.surface_disparity_px:.6f} px",
                "  surface disparity percentile: "
                f"{self.depth_estimator.config.surface_disparity_percentile:.2f}",
            ]
        )

        if depth_estimate.optical_depth_m is None:
            lines.extend(
                [
                    "  result: FAIL",
                    "  reason: optical depth was not produced",
                    "",
                    "FINAL VERDICT: REJECTED BEFORE IK",
                ]
            )
            return False, "\n".join(lines)

        lines.extend(
            [
                f"  optical depth: {depth_estimate.optical_depth_m:.6f} m",
                "  result: PASS",
                "",
                "CAMERA BACK-PROJECTION",
                f"  ROI center pixel u: {depth_estimate.roi.center_u_px}",
                f"  ROI center pixel v: {depth_estimate.roi.center_v_px}",
            ]
        )

        intrinsics = self.camera_geometry.intrinsics
        if intrinsics is None:
            lines.extend(
                [
                    "  result: FAIL",
                    "  reason: left camera intrinsics are unavailable",
                    "",
                    "FINAL VERDICT: REJECTED BEFORE IK",
                ]
            )
            return False, "\n".join(lines)

        lines.extend(
            [
                f"  fx: {intrinsics.focal_x_px:.6f} px",
                f"  fy: {intrinsics.focal_y_px:.6f} px",
                f"  cx: {intrinsics.principal_x_px:.6f} px",
                f"  cy: {intrinsics.principal_y_px:.6f} px",
            ]
        )

        camera_surface_point = self.camera_geometry.back_project_pixel(
            depth_estimate.roi.center_u_px,
            depth_estimate.roi.center_v_px,
            depth_estimate.optical_depth_m,
        )
        if camera_surface_point is None:
            lines.extend(
                [
                    "  result: FAIL",
                    "  reason: back-projection returned no point",
                    "",
                    "FINAL VERDICT: REJECTED BEFORE IK",
                ]
            )
            return False, "\n".join(lines)

        lines.extend(
            [
                "  camera optical point:",
                f"    x: {camera_surface_point.x_m:.6f} m",
                f"    y: {camera_surface_point.y_m:.6f} m",
                f"    z: {camera_surface_point.z_m:.6f} m",
                "  result: PASS",
                "",
                "CAMERA-TO-ORIGIN TRANSFORM",
                "  configured camera position relative to robot origin:",
                f"    x: {self.camera_geometry.config.camera_origin_x_m:.6f} m",
                f"    y: {self.camera_geometry.config.camera_origin_y_m:.6f} m",
                f"    z: {self.camera_geometry.config.camera_origin_z_m:.6f} m",
                "  configured downward pitch: "
                f"{self.camera_geometry.config.camera_pitch_down_degrees:.3f} deg",
            ]
        )

        estimated_surface_origin = (
            self.camera_geometry.transform_camera_point_to_origin(
                camera_surface_point
            )
        )
        lines.extend(
            [
                "  estimated tomato surface relative to robot origin:",
                f"    x: {estimated_surface_origin.x_m:.6f} m",
                f"    y: {estimated_surface_origin.y_m:.6f} m",
                f"    z: {estimated_surface_origin.z_m:.6f} m",
                "",
                "WAYPOINT GENERATION",
                f"  contact standoff: {self.approach_planner.config.contact_standoff_m:.6f} m",
                f"  contact lateral offset: {self.approach_planner.config.contact_lateral_offset_m:.6f} m",
                f"  contact vertical offset: {self.approach_planner.config.contact_vertical_offset_m:.6f} m",
                f"  pregrasp distance: {self.approach_planner.config.pregrasp_distance_m:.6f} m",
                f"  retreat distance: {self.approach_planner.config.retreat_distance_m:.6f} m",
            ]
        )

        waypoints = self.approach_planner.create_waypoints(
            estimated_surface_origin
        )
        for waypoint in waypoints:
            p = waypoint.position_origin
            lines.append(
                f"  {waypoint.name}: "
                f"({p.x_m:.6f}, {p.y_m:.6f}, {p.z_m:.6f}) m"
            )

        lines.append("")
        accepted, ik_report = (
            self.approach_planner.build_ik_diagnostic_report(waypoints)
        )
        lines.append(ik_report)
        lines.extend(
            [
                "",
                "=" * 72,
                f"FINAL CONTROLLER VERDICT: {'ACCEPTED' if accepted else 'REJECTED'}",
                "=" * 72,
            ]
        )
        return accepted, "\n".join(lines)

    @staticmethod
    def _ripeness_priority(ripeness: str) -> int:
        priorities = {
            "fully_ripened": 3,
            "half_ripened": 2,
            "green": 1,
        }
        return priorities.get(ripeness, 0)
