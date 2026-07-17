from __future__ import annotations

import numpy as np
from stereo_msgs.msg import DisparityImage

from tomato_control.controller_config import ControllerConfig
from tomato_control.controller_models import BoundingBox, DepthEstimate


class TomatoDepthEstimator:
    """Estimate the camera-facing surface depth of one tomato detection."""

    def __init__(self, config: ControllerConfig):
        self.config = config

    def estimate(
        self,
        disparity_image: np.ndarray,
        disparity_message: DisparityImage,
        tomato_detection,
    ) -> DepthEstimate | None:
        image_height, image_width = disparity_image.shape[:2]

        original_box = self._clamp_box(
            BoundingBox(
                x_min=tomato_detection.x1,
                y_min=tomato_detection.y1,
                x_max=tomato_detection.x2,
                y_max=tomato_detection.y2,
            ),
            image_width,
            image_height,
        )

        if original_box.area == 0:
            return None

        interior_box = self._shrink_box(
            original_box,
            self.config.roi_total_shrink_fraction,
        )
        interior_box = self._clamp_box(
            interior_box,
            image_width,
            image_height,
        )

        if interior_box.area == 0:
            return None

        disparity_roi = disparity_image[
            interior_box.y_min:interior_box.y_max,
            interior_box.x_min:interior_box.x_max,
        ]

        valid_mask = (
            np.isfinite(disparity_roi)
            & (
                disparity_roi
                > self.config.minimum_valid_disparity_px
            )
            & (
                disparity_roi
                < self.config.maximum_valid_disparity_px
            )
        )

        valid_pixel_count = int(np.count_nonzero(valid_mask))
        total_pixel_count = int(disparity_roi.size)

        if total_pixel_count == 0:
            return None

        valid_pixel_ratio = valid_pixel_count / total_pixel_count

        if (
            valid_pixel_count == 0
            or valid_pixel_ratio
            < self.config.minimum_valid_disparity_ratio
        ):
            return DepthEstimate(
                is_valid=False,
                roi=interior_box,
                valid_pixel_count=valid_pixel_count,
                total_pixel_count=total_pixel_count,
                valid_pixel_ratio=valid_pixel_ratio,
            )

        valid_disparities_px = disparity_roi[valid_mask]
        median_disparity_px = float(np.median(valid_disparities_px))
        mean_disparity_px = float(np.mean(valid_disparities_px))
        surface_disparity_px = float(
            np.percentile(
                valid_disparities_px,
                self.config.surface_disparity_percentile,
            )
        )

        if surface_disparity_px <= 0.0:
            return None

        optical_depth_m = (
            abs(disparity_message.f * disparity_message.t)
            / surface_disparity_px
        )

        return DepthEstimate(
            is_valid=True,
            roi=interior_box,
            valid_pixel_count=valid_pixel_count,
            total_pixel_count=total_pixel_count,
            valid_pixel_ratio=valid_pixel_ratio,
            median_disparity_px=median_disparity_px,
            mean_disparity_px=mean_disparity_px,
            surface_disparity_px=surface_disparity_px,
            optical_depth_m=optical_depth_m,
        )

    @staticmethod
    def _shrink_box(
        box: BoundingBox,
        total_shrink_fraction: float,
    ) -> BoundingBox:
        horizontal_margin = int(
            box.width * total_shrink_fraction / 2.0
        )
        vertical_margin = int(
            box.height * total_shrink_fraction / 2.0
        )

        return BoundingBox(
            x_min=box.x_min + horizontal_margin,
            y_min=box.y_min + vertical_margin,
            x_max=box.x_max - horizontal_margin,
            y_max=box.y_max - vertical_margin,
        )

    @staticmethod
    def _clamp_box(
        box: BoundingBox,
        image_width: int,
        image_height: int,
    ) -> BoundingBox:
        return BoundingBox(
            x_min=max(0, min(int(box.x_min), image_width - 1)),
            y_min=max(0, min(int(box.y_min), image_height - 1)),
            x_max=max(0, min(int(box.x_max), image_width)),
            y_max=max(0, min(int(box.y_max), image_height)),
        )
