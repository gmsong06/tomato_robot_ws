from __future__ import annotations

import numpy as np
from sensor_msgs.msg import CameraInfo

from tomato_control.controller_config import ControllerConfig
from tomato_control.controller_models import CameraIntrinsics, Point3D


class CameraGeometry:
    """Camera intrinsics and the transform into the joint_2 target frame."""

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.intrinsics: CameraIntrinsics | None = None

    def update_intrinsics(self, message: CameraInfo) -> CameraIntrinsics:
        projection_matrix = message.p
        intrinsic_matrix = message.k

        if projection_matrix[0] != 0.0 and projection_matrix[5] != 0.0:
            focal_x_px = float(projection_matrix[0])
            focal_y_px = float(projection_matrix[5])
            principal_x_px = float(projection_matrix[2])
            principal_y_px = float(projection_matrix[6])
        else:
            focal_x_px = float(intrinsic_matrix[0])
            focal_y_px = float(intrinsic_matrix[4])
            principal_x_px = float(intrinsic_matrix[2])
            principal_y_px = float(intrinsic_matrix[5])

        self.intrinsics = CameraIntrinsics(
            focal_x_px=focal_x_px,
            focal_y_px=focal_y_px,
            principal_x_px=principal_x_px,
            principal_y_px=principal_y_px,
        )
        return self.intrinsics

    def back_project_pixel(
        self,
        pixel_u: int,
        pixel_v: int,
        optical_depth_m: float,
    ) -> Point3D | None:
        """Back-project a rectified pixel into the left optical frame."""

        if self.intrinsics is None:
            return None

        camera_x_m = (
            (float(pixel_u) - self.intrinsics.principal_x_px)
            * optical_depth_m
            / self.intrinsics.focal_x_px
        )
        camera_y_m = (
            (float(pixel_v) - self.intrinsics.principal_y_px)
            * optical_depth_m
            / self.intrinsics.focal_y_px
        )

        return Point3D(
            x_m=camera_x_m,
            y_m=camera_y_m,
            z_m=optical_depth_m,
        )

    def transform_camera_point_to_joint_2(
        self,
        camera_point: Point3D,
    ) -> Point3D:
        """Transform a camera point into fixed axes at the joint_2 origin."""

        camera_point_vector = np.array(
            [camera_point.x_m, camera_point.y_m, camera_point.z_m],
            dtype=float,
        )

        camera_origin_in_joint_2 = np.array(
            [
                self.config.camera_joint_2_x_m,
                self.config.camera_joint_2_y_m,
                self.config.camera_joint_2_z_m,
            ],
            dtype=float,
        )

        joint_2_point_vector = (
            self._camera_to_joint_2_rotation() @ camera_point_vector
            + camera_origin_in_joint_2
        )

        return Point3D(
            x_m=float(joint_2_point_vector[0]),
            y_m=float(joint_2_point_vector[1]),
            z_m=float(joint_2_point_vector[2]),
        )

    def transform_camera_point_to_base(self, camera_point: Point3D) -> Point3D:
        """Compatibility alias for the former, misleading method name."""

        return self.transform_camera_point_to_joint_2(camera_point)

    def _camera_to_joint_2_rotation(self) -> np.ndarray:
        pitch_down_rad = np.deg2rad(
            self.config.camera_pitch_down_degrees
        )

        # Axes are parallel to base_link:
        # camera +X = robot right = joint_2-frame -Y
        # camera +Y = image down = joint_2-frame backward/down
        # camera +Z = lens forward = joint_2-frame forward/down
        return np.array(
            [
                [0.0, -np.sin(pitch_down_rad), np.cos(pitch_down_rad)],
                [-1.0, 0.0, 0.0],
                [0.0, -np.cos(pitch_down_rad), -np.sin(pitch_down_rad)],
            ],
            dtype=float,
        )