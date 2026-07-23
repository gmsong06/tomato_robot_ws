from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Point3D:
    """A 3D point expressed in meters."""

    x_m: float
    y_m: float
    z_m: float


@dataclass(frozen=True)
class CameraIntrinsics:
    """Rectified pinhole-camera intrinsics in pixels."""

    focal_x_px: float
    focal_y_px: float
    principal_x_px: float
    principal_y_px: float


@dataclass(frozen=True)
class BoundingBox:
    """Integer image-space bounding box using half-open max coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return max(0, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        return max(0, self.y_max - self.y_min)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center_u_px(self) -> int:
        return int((self.x_min + self.x_max) / 2)

    @property
    def center_v_px(self) -> int:
        return int((self.y_min + self.y_max) / 2)


@dataclass(frozen=True)
class DepthEstimate:
    """Disparity statistics and the resulting optical-axis depth."""

    is_valid: bool
    roi: BoundingBox
    valid_pixel_count: int
    total_pixel_count: int
    valid_pixel_ratio: float
    median_disparity_px: float | None = None
    mean_disparity_px: float | None = None
    surface_disparity_px: float | None = None
    optical_depth_m: float | None = None


@dataclass(frozen=True)
class CartesianWaypoint:
    """A tool-tip pose in the fixed joint_2-origin target frame."""

    name: str
    position_joint_2: Point3D
    tool_angle_rad: float

    @property
    def position_base(self) -> Point3D:
        """Compatibility alias; the returned point is joint_2-relative."""

        return self.position_joint_2


@dataclass(frozen=True)
class WaypointCommand:
    """A Cartesian waypoint paired with its IK joint solution."""

    name: str
    # Tomato-relative commands have a Cartesian waypoint. Joint-only commands,
    # such as the fixed home pose, use None.
    waypoint: CartesianWaypoint | None
    joint_angles: dict[str, float]
    ik_result: Any | None


@dataclass(frozen=True)
class TomatoCandidate:
    """One reachable tomato and its fully solved three-point trajectory."""

    detection: Any
    depth_estimate: DepthEstimate
    camera_surface_point: Point3D
    estimated_surface_joint_2: Point3D
    waypoints: tuple[CartesianWaypoint, ...]
    waypoint_commands: tuple[WaypointCommand, ...]
    bounding_box_area_px: int
    ripeness_priority: int

    @property
    def estimated_surface_base(self) -> Point3D:
        """Compatibility alias; the returned point is joint_2-relative."""

        return self.estimated_surface_joint_2