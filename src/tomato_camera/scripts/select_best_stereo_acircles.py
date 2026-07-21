#!/usr/bin/env python3
"""
Select strong nested subsets of stereo asymmetric-circle-grid image pairs.

Designed for:
  - OpenCV asymmetric circles grid: 4 x 11
  - center spacing parameter: 0.01905 m
  - expected physical stereo baseline: 0.100 m

The script:
  1. Extracts a .tar.gz/.tgz archive (or scans a directory).
  2. Pairs left/right images by filename/path tokens.
  3. Detects the asymmetric circles grid in both views.
  4. Reserves a small, diverse validation set when enough pairs exist.
  5. Repeatedly removes one harmful/redundant training pair.
  6. Saves nested best_40, best_30, and best_20 subsets.
  7. Writes a ROS 2 disparity_node parameter file (disparity.yaml).

"Best" is judged using held-out rectified vertical alignment, held-out
reprojection error, stereo RMS, closeness to the known 0.100 m baseline,
and pose/image coverage. The baseline is a penalty/sanity check, not the
only optimization target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class StereoPair:
    key: str
    left_path: Path
    right_path: Path
    left_points: np.ndarray       # (N, 1, 2), float32
    right_points: np.ndarray      # (N, 1, 2), float32
    feature: np.ndarray           # diversity feature vector
    sharpness: float


@dataclass
class CalibrationModel:
    image_size: tuple[int, int]   # (width, height)
    rms_left: float
    rms_right: float
    stereo_rms: float
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.T.reshape(-1)))

    @property
    def rectified_baseline_m(self) -> float:
        fx = float(self.P2[0, 0])
        return abs(float(self.P2[0, 3]) / fx) if abs(fx) > 1e-12 else math.nan


@dataclass
class Evaluation:
    score: float
    baseline_m: float
    baseline_error_mm: float
    rectified_baseline_m: float
    stereo_rms_px: float
    validation_reprojection_mean_px: float
    validation_vertical_mean_px: float
    validation_vertical_median_px: float
    validation_vertical_max_px: float
    diversity_penalty: float

    def to_dict(self) -> dict[str, float]:
        return {
            "score": self.score,
            "baseline_m": self.baseline_m,
            "baseline_error_mm": self.baseline_error_mm,
            "rectified_baseline_m": self.rectified_baseline_m,
            "stereo_rms_px": self.stereo_rms_px,
            "validation_reprojection_mean_px": self.validation_reprojection_mean_px,
            "validation_vertical_mean_px": self.validation_vertical_mean_px,
            "validation_vertical_median_px": self.validation_vertical_median_px,
            "validation_vertical_max_px": self.validation_vertical_max_px,
            "diversity_penalty": self.diversity_penalty,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find nested best 20/30/40 stereo asymmetric-grid subsets."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a .tar.gz/.tgz archive or an already-extracted directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stereo_subset_results"),
        help="Output directory (default: stereo_subset_results).",
    )
    parser.add_argument("--cols", type=int, default=4, help="Grid columns (default: 4).")
    parser.add_argument("--rows", type=int, default=11, help="Grid rows (default: 11).")
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.01905,
        help="Asymmetric-grid spacing parameter in meters (default: 0.01905).",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=0.100,
        help="Measured stereo baseline in meters (default: 0.100).",
    )
    parser.add_argument(
        "--left-tokens",
        default="left,cam0",
        help="Comma-separated tokens identifying left images (default: left,cam0).",
    )
    parser.add_argument(
        "--right-tokens",
        default="right,cam1",
        help="Comma-separated tokens identifying right images (default: right,cam1).",
    )
    parser.add_argument(
        "--validation-count",
        type=int,
        default=8,
        help="Maximum held-out validation pairs (default: 8).",
    )
    parser.add_argument(
        "--targets",
        default="40,30,20",
        help="Comma-separated subset sizes to save (default: 40,30,20).",
    )
    parser.add_argument(
        "--estimate-k3",
        action="store_true",
        help="Estimate k3. Default fixes k3=0 to match common ROS calibration output.",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Write manifests/calibration only; do not copy selected images.",
    )
    parser.add_argument(
        "--debug-detections",
        action="store_true",
        help="Save annotated circle detections for inspection.",
    )
    parser.add_argument(
        "--disparity-node-name",
        default="/disparity_node",
        help=(
            "Fully qualified ROS 2 node name used as the top-level key in "
            "disparity.yaml (default: /disparity_node)."
        ),
    )
    return parser.parse_args()


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:*") as tf:
        for member in tf.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"Archive contains a link, refusing to extract: {member.name}")
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe archive path: {member.name}")
        tf.extractall(destination)


def token_list(value: str) -> list[str]:
    tokens = [x.strip().lower() for x in value.split(",") if x.strip()]
    if not tokens:
        raise ValueError("At least one left/right token is required.")
    return tokens


def contains_token(text: str, token: str) -> bool:
    pattern = rf"(^|[^a-z0-9]){re.escape(token)}([^a-z0-9]|$)"
    return re.search(pattern, text.lower()) is not None


def classify_side(path: Path, root: Path, left_tokens: Sequence[str], right_tokens: Sequence[str]) -> str | None:
    rel = path.relative_to(root)
    text = "/".join(part.lower() for part in rel.parts)
    left_hit = any(contains_token(text, token) for token in left_tokens)
    right_hit = any(contains_token(text, token) for token in right_tokens)

    if left_hit and not right_hit:
        return "left"
    if right_hit and not left_hit:
        return "right"
    return None


def canonical_pair_key(
    path: Path,
    root: Path,
    left_tokens: Sequence[str],
    right_tokens: Sequence[str],
) -> str:
    rel = path.relative_to(root).with_suffix("")
    components: list[str] = []
    all_tokens = [*left_tokens, *right_tokens]

    for component in rel.parts:
        value = component.lower()
        for token in all_tokens:
            value = re.sub(
                rf"(^|[^a-z0-9]){re.escape(token)}(?=[^a-z0-9]|$)",
                r"\1",
                value,
            )
        value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
        if value:
            components.append(value)

    return "/".join(components)


def discover_pairs(
    root: Path,
    left_tokens: Sequence[str],
    right_tokens: Sequence[str],
) -> tuple[list[tuple[str, Path, Path]], list[str]]:
    left_by_key: dict[str, Path] = {}
    right_by_key: dict[str, Path] = {}
    warnings: list[str] = []

    images = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    for path in images:
        side = classify_side(path, root, left_tokens, right_tokens)
        if side is None:
            continue
        key = canonical_pair_key(path, root, left_tokens, right_tokens)
        if not key:
            key = path.stem.lower()

        target = left_by_key if side == "left" else right_by_key
        if key in target:
            warnings.append(
                f"Duplicate {side} key '{key}': keeping {target[key]}, ignoring {path}"
            )
            continue
        target[key] = path

    shared = sorted(set(left_by_key) & set(right_by_key))
    for key in sorted(set(left_by_key) - set(right_by_key)):
        warnings.append(f"No right image for key '{key}': {left_by_key[key]}")
    for key in sorted(set(right_by_key) - set(left_by_key)):
        warnings.append(f"No left image for key '{key}': {right_by_key[key]}")

    pairs = [(key, left_by_key[key], right_by_key[key]) for key in shared]
    return pairs, warnings


def make_blob_detector(blob_color: int | None) -> cv2.SimpleBlobDetector:
    """Create a permissive detector for calibration circles.

    blob_color=None disables polarity filtering, which is useful when exposure
    or target polarity differs across captures.
    """
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 2
    params.maxThreshold = 253
    params.thresholdStep = 5
    params.filterByArea = True
    params.minArea = 8
    params.maxArea = 200000
    params.filterByColor = blob_color is not None
    if blob_color is not None:
        params.blobColor = int(blob_color)
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minDistBetweenBlobs = 3
    return cv2.SimpleBlobDetector_create(params)


def detect_asymmetric_grid(gray: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray | None:
    """Try several conservative OpenCV detection paths.

    The original script always enabled CALIB_CB_CLUSTERING and always used a
    color-filtered custom detector. Either choice can cause otherwise valid
    asymmetric grids to be rejected, depending on the target and background.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = cv2.equalizeHist(gray)
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    images = [gray, blurred, equalized, enhanced]
    flag_options = [
        cv2.CALIB_CB_ASYMMETRIC_GRID,
        cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING,
    ]

    # First try OpenCV's own default blob detector. This most closely matches
    # the standard findCirclesGrid usage and often works better than an overly
    # constrained custom detector.
    for image in images:
        for flags in flag_options:
            found, centers = cv2.findCirclesGrid(
                image,
                pattern_size,
                flags=flags,
            )
            if found and centers is not None:
                return centers.astype(np.float32)

    # Then try permissive and polarity-specific detectors, including inverted
    # images for white-on-black targets.
    detector_options = [
        make_blob_detector(None),
        make_blob_detector(0),
        make_blob_detector(255),
    ]
    image_options = images + [cv2.bitwise_not(image) for image in images]

    for image in image_options:
        for detector in detector_options:
            for flags in flag_options:
                found, centers = cv2.findCirclesGrid(
                    image,
                    pattern_size,
                    flags=flags,
                    blobDetector=detector,
                )
                if found and centers is not None:
                    return centers.astype(np.float32)

    return None


def image_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def grid_feature(points: np.ndarray, width: int, height: int, cols: int) -> np.ndarray:
    pts = points.reshape(-1, 2)
    center = pts.mean(axis=0)
    hull = cv2.convexHull(pts.astype(np.float32))
    area_fraction = max(float(cv2.contourArea(hull)) / float(width * height), 1e-9)

    if len(pts) >= cols:
        row_vector = pts[cols - 1] - pts[0]
    else:
        row_vector = pts[-1] - pts[0]
    angle = math.atan2(float(row_vector[1]), float(row_vector[0]))

    return np.array(
        [
            center[0] / width,
            center[1] / height,
            math.sqrt(area_fraction) * 2.0,
            math.cos(angle) * 0.35,
            math.sin(angle) * 0.35,
        ],
        dtype=np.float64,
    )


def save_detection_debug(
    out_dir: Path,
    key: str,
    left: np.ndarray,
    right: np.ndarray,
    left_points: np.ndarray | None,
    right_points: np.ndarray | None,
    pattern_size: tuple[int, int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    left_vis = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    right_vis = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    if left_points is not None:
        cv2.drawChessboardCorners(left_vis, pattern_size, left_points, True)
    if right_points is not None:
        cv2.drawChessboardCorners(right_vis, pattern_size, right_points, True)
    cv2.imwrite(str(out_dir / f"{safe_key}_left.png"), left_vis)
    cv2.imwrite(str(out_dir / f"{safe_key}_right.png"), right_vis)


def load_and_detect_pairs(
    raw_pairs: Sequence[tuple[str, Path, Path]],
    pattern_size: tuple[int, int],
    debug_dir: Path | None,
) -> tuple[list[StereoPair], tuple[int, int], list[str]]:
    valid: list[StereoPair] = []
    rejected: list[str] = []
    expected_size: tuple[int, int] | None = None
    cols = pattern_size[0]

    for index, (key, left_path, right_path) in enumerate(raw_pairs, start=1):
        left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            rejected.append(f"{key}: unreadable image")
            continue
        if left.shape != right.shape:
            rejected.append(f"{key}: left/right dimensions differ: {left.shape} vs {right.shape}")
            continue

        height, width = left.shape[:2]
        image_size = (width, height)
        if expected_size is None:
            expected_size = image_size
        elif image_size != expected_size:
            rejected.append(f"{key}: size {image_size} differs from expected {expected_size}")
            continue

        left_points = detect_asymmetric_grid(left, pattern_size)
        right_points = detect_asymmetric_grid(right, pattern_size)

        if debug_dir is not None:
            save_detection_debug(
                debug_dir,
                key,
                left,
                right,
                left_points,
                right_points,
                pattern_size,
            )

        if left_points is None or right_points is None:
            missing = []
            if left_points is None:
                missing.append("left")
            if right_points is None:
                missing.append("right")
            missing_text = ", ".join(missing)
            rejected.append(f"{key}: grid not detected in {missing_text}")
            print(f"[{index:03d}/{len(raw_pairs):03d}] rejected: {key} ({missing_text})")
            continue

        left_feature = grid_feature(left_points, width, height, cols)
        right_feature = grid_feature(right_points, width, height, cols)
        feature = 0.5 * (left_feature + right_feature)
        sharpness = 0.5 * (image_sharpness(left) + image_sharpness(right))

        valid.append(
            StereoPair(
                key=key,
                left_path=left_path,
                right_path=right_path,
                left_points=left_points,
                right_points=right_points,
                feature=feature,
                sharpness=sharpness,
            )
        )
        print(f"[{index:03d}/{len(raw_pairs):03d}] detected: {key}")

    if expected_size is None:
        raise RuntimeError("No readable stereo image pairs were found.")
    return valid, expected_size, rejected


def asymmetric_object_points(cols: int, rows: int, spacing_m: float) -> np.ndarray:
    points = np.zeros((cols * rows, 3), dtype=np.float32)
    points[:, :2] = np.array(
        [
            ((2 * col + (row % 2)) * spacing_m, row * spacing_m)
            for row in range(rows)
            for col in range(cols)
        ],
        dtype=np.float32,
    )
    return points


def calibrate_subset(
    pairs: Sequence[StereoPair],
    image_size: tuple[int, int],
    object_points: np.ndarray,
    estimate_k3: bool,
) -> CalibrationModel:
    if len(pairs) < 8:
        raise ValueError("At least 8 valid pairs are required for a stable calibration.")

    obj = [object_points.copy() for _ in pairs]
    left_points = [pair.left_points for pair in pairs]
    right_points = [pair.right_points for pair in pairs]

    mono_flags = 0 if estimate_k3 else cv2.CALIB_FIX_K3
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        50,
        1e-6,
    )

    rms_left, K1, D1, _, _ = cv2.calibrateCamera(
        obj,
        left_points,
        image_size,
        None,
        None,
        flags=mono_flags,
        criteria=criteria,
    )
    rms_right, K2, D2, _, _ = cv2.calibrateCamera(
        obj,
        right_points,
        image_size,
        None,
        None,
        flags=mono_flags,
        criteria=criteria,
    )

    (
        stereo_rms,
        K1,
        D1,
        K2,
        D2,
        R,
        T,
        E,
        F,
    ) = cv2.stereoCalibrate(
        obj,
        left_points,
        right_points,
        K1,
        D1,
        K2,
        D2,
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1,
        D1,
        K2,
        D2,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    arrays = [K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q]
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise RuntimeError("Calibration produced non-finite values.")

    return CalibrationModel(
        image_size=image_size,
        rms_left=float(rms_left),
        rms_right=float(rms_right),
        stereo_rms=float(stereo_rms),
        K1=K1,
        D1=D1,
        K2=K2,
        D2=D2,
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
    )


def solvepnp_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> float:
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        D,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return math.inf
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
    diff = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def pair_metrics(
    pair: StereoPair,
    model: CalibrationModel,
    object_points: np.ndarray,
) -> tuple[float, float, float]:
    left_error = solvepnp_reprojection_error(
        object_points, pair.left_points, model.K1, model.D1
    )
    right_error = solvepnp_reprojection_error(
        object_points, pair.right_points, model.K2, model.D2
    )
    reprojection = 0.5 * (left_error + right_error)

    left_rect = cv2.undistortPoints(
        pair.left_points,
        model.K1,
        model.D1,
        R=model.R1,
        P=model.P1,
    ).reshape(-1, 2)
    right_rect = cv2.undistortPoints(
        pair.right_points,
        model.K2,
        model.D2,
        R=model.R2,
        P=model.P2,
    ).reshape(-1, 2)

    vertical = np.abs(left_rect[:, 1] - right_rect[:, 1])
    return reprojection, float(np.mean(vertical)), float(np.max(vertical))


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad < 1e-12:
        std = float(np.std(values))
        if std < 1e-12:
            return np.zeros_like(values)
        return (values - median) / std
    return 0.67448975 * (values - median) / mad


def nearest_neighbor_redundancy(features: np.ndarray) -> np.ndarray:
    if len(features) <= 1:
        return np.zeros(len(features), dtype=np.float64)
    diff = features[:, None, :] - features[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    # Higher means more redundant/replaceable.
    return 1.0 / np.maximum(nearest, 1e-6)


def coverage_bins(pairs: Sequence[StereoPair]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.array([p.feature for p in pairs])
    x_bins = np.clip((features[:, 0] * 3).astype(int), 0, 2)
    y_bins = np.clip((features[:, 1] * 3).astype(int), 0, 2)
    center_bins = y_bins * 3 + x_bins

    scales = features[:, 2]
    if len(np.unique(scales)) >= 3:
        q1, q2 = np.quantile(scales, [1 / 3, 2 / 3])
        scale_bins = np.digitize(scales, [q1, q2], right=False)
    else:
        scale_bins = np.zeros(len(pairs), dtype=int)

    angles = np.arctan2(features[:, 4], features[:, 3])
    normalized = (angles + math.pi) / (2 * math.pi)
    angle_bins = np.clip((normalized * 4).astype(int), 0, 3)
    return center_bins, scale_bins, angle_bins


def diversity_penalty(pairs: Sequence[StereoPair]) -> float:
    if not pairs:
        return 100.0
    center_bins, scale_bins, angle_bins = coverage_bins(pairs)
    occupied_center = len(np.unique(center_bins))
    occupied_scale = len(np.unique(scale_bins))
    occupied_angle = len(np.unique(angle_bins))

    return (
        0.045 * (9 - occupied_center)
        + 0.070 * (3 - occupied_scale)
        + 0.025 * (4 - occupied_angle)
    )


def choose_diverse_validation(
    pairs: Sequence[StereoPair],
    count: int,
) -> tuple[list[StereoPair], list[StereoPair]]:
    if count <= 0:
        return list(pairs), []

    sharpness = np.array([p.sharpness for p in pairs], dtype=np.float64)
    cutoff = float(np.quantile(sharpness, 0.15)) if len(pairs) >= 10 else float(np.min(sharpness))
    eligible = [i for i, p in enumerate(pairs) if p.sharpness >= cutoff]
    if len(eligible) < count:
        eligible = list(range(len(pairs)))

    features = np.array([pairs[i].feature for i in eligible], dtype=np.float64)
    centroid = np.mean(features, axis=0)
    selected_local = [int(np.argmax(np.linalg.norm(features - centroid, axis=1)))]

    while len(selected_local) < count:
        selected_features = features[selected_local]
        distances = np.linalg.norm(
            features[:, None, :] - selected_features[None, :, :], axis=2
        )
        min_distance = np.min(distances, axis=1)
        min_distance[selected_local] = -np.inf
        selected_local.append(int(np.argmax(min_distance)))

    selected_indices = {eligible[i] for i in selected_local}
    validation = [p for i, p in enumerate(pairs) if i in selected_indices]
    training = [p for i, p in enumerate(pairs) if i not in selected_indices]
    return training, validation


def evaluate_model(
    model: CalibrationModel,
    validation_pairs: Sequence[StereoPair],
    fallback_pairs: Sequence[StereoPair],
    object_points: np.ndarray,
    expected_baseline_m: float,
) -> Evaluation:
    evaluation_pairs = validation_pairs if validation_pairs else fallback_pairs
    reprojection_errors: list[float] = []
    vertical_means: list[float] = []
    vertical_maxima: list[float] = []

    for pair in evaluation_pairs:
        reproj, vertical_mean, vertical_max = pair_metrics(pair, model, object_points)
        reprojection_errors.append(reproj)
        vertical_means.append(vertical_mean)
        vertical_maxima.append(vertical_max)

    reproj_mean = float(np.mean(reprojection_errors))
    vertical_mean = float(np.mean(vertical_means))
    vertical_median = float(np.median(vertical_means))
    vertical_max = float(np.max(vertical_maxima))
    baseline_error_mm = abs(model.baseline_m - expected_baseline_m) * 1000.0
    div_penalty = diversity_penalty(fallback_pairs)

    # Vertical rectification error is weighted most heavily. Baseline closeness is
    # deliberately a moderate sanity penalty rather than the sole objective.
    score = (
        1.00 * vertical_mean
        + 0.35 * reproj_mean
        + 0.10 * model.stereo_rms
        + 0.08 * baseline_error_mm
        + div_penalty
    )

    return Evaluation(
        score=float(score),
        baseline_m=model.baseline_m,
        baseline_error_mm=float(baseline_error_mm),
        rectified_baseline_m=model.rectified_baseline_m,
        stereo_rms_px=model.stereo_rms,
        validation_reprojection_mean_px=reproj_mean,
        validation_vertical_mean_px=vertical_mean,
        validation_vertical_median_px=vertical_median,
        validation_vertical_max_px=vertical_max,
        diversity_penalty=float(div_penalty),
    )


def reduce_to_target(
    pairs: Sequence[StereoPair],
    model: CalibrationModel,
    object_points: np.ndarray,
    target_size: int,
) -> tuple[list[StereoPair], list[dict[str, object]]]:
    """
    Reduce a calibrated pool to target_size without recalibrating after every removal.

    Calibration-dependent errors are measured once for the current stage. As pairs
    are removed, redundancy and coverage protection are recomputed dynamically.
    """
    if target_size > len(pairs):
        raise ValueError(f"Target {target_size} exceeds current pool {len(pairs)}.")

    current = list(pairs)
    fixed_metrics: dict[str, tuple[float, float, float]] = {}
    for pair in current:
        fixed_metrics[pair.key] = pair_metrics(pair, model, object_points)

    removals: list[dict[str, object]] = []
    while len(current) > target_size:
        reproj = np.array([fixed_metrics[p.key][0] for p in current], dtype=np.float64)
        vertical = np.array([fixed_metrics[p.key][1] for p in current], dtype=np.float64)
        sharpness = np.array([p.sharpness for p in current], dtype=np.float64)
        features = np.array([p.feature for p in current], dtype=np.float64)
        redundancy = nearest_neighbor_redundancy(features)

        badness = (
            robust_z(reproj)
            + 1.5 * robust_z(vertical)
            + 0.35 * robust_z(redundancy)
            + 0.20 * robust_z(-sharpness)
        )

        center_bins, scale_bins, angle_bins = coverage_bins(current)
        for i in range(len(current)):
            # Strongly protect views that are the sole representative of a region,
            # distance scale, or board orientation.
            if np.count_nonzero(center_bins == center_bins[i]) == 1:
                badness[i] -= 3.0
            if np.count_nonzero(scale_bins == scale_bins[i]) == 1:
                badness[i] -= 1.0
            if np.count_nonzero(angle_bins == angle_bins[i]) == 1:
                badness[i] -= 0.5

        remove_index = int(np.argmax(badness))
        removed = current.pop(remove_index)
        r, vmean, vmax = fixed_metrics[removed.key]
        removals.append(
            {
                "removed_key": removed.key,
                "remaining": len(current),
                "badness": float(badness[remove_index]),
                "reprojection_px": float(r),
                "vertical_mean_px": float(vmean),
                "vertical_max_px": float(vmax),
                "sharpness": float(removed.sharpness),
            }
        )

    return current, removals


def matrix_data(matrix: np.ndarray) -> list[float]:
    return [float(x) for x in matrix.reshape(-1)]


def write_ros_camera_yaml(
    path: Path,
    camera_name: str,
    image_size: tuple[int, int],
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    P: np.ndarray,
) -> None:
    width, height = image_size
    distortion = D.reshape(-1)
    if len(distortion) < 5:
        distortion = np.pad(distortion, (0, 5 - len(distortion)))
    distortion = distortion[:5]

    def fmt(values: Iterable[float]) -> str:
        return ", ".join(f"{float(v):.12g}" for v in values)

    text = f"""image_width: {width}
image_height: {height}
camera_name: {camera_name}
camera_matrix:
  rows: 3
  cols: 3
  data: [{fmt(matrix_data(K))}]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [{fmt(distortion)}]
rectification_matrix:
  rows: 3
  cols: 3
  data: [{fmt(matrix_data(R))}]
projection_matrix:
  rows: 3
  cols: 4
  data: [{fmt(matrix_data(P))}]
"""
    path.write_text(text)


def write_disparity_yaml(path: Path, node_name: str = "/disparity_node") -> None:
    """Write ROS 2 parameters for stereo_image_proc's DisparityNode.

    These are disparity-matching/timing/QoS parameters, not camera calibration
    matrices. The node still obtains focal length and baseline from the incoming
    left/right CameraInfo messages generated from left.yaml and right.yaml.
    """
    node_name = node_name.strip() or "/disparity_node"
    if not node_name.startswith("/"):
        node_name = f"/{node_name}"

    text = f"""{node_name}:
  ros__parameters:
    P1: 600.0
    P2: 2400.0
    approximate_sync: true
    approximate_sync_tolerance_seconds: 0.0
    correlation_window_size: 7
    disp12_max_diff: 1
    disparity_range: 192
    image_transport: raw
    min_disparity: 0
    prefilter_cap: 31
    prefilter_size: 9
    qos_overrides:
      /parameter_events:
        publisher:
          depth: 1000
          durability: volatile
          history: keep_last
          reliability: reliable
      /stereo/disparity:
        publisher:
          depth: 1
          history: keep_last
          reliability: reliable
      /stereo/left/camera_info:
        subscription:
          depth: 5
          history: keep_last
          reliability: best_effort
      /stereo/left/image_rect:
        subscription:
          depth: 5
          history: keep_last
          reliability: best_effort
      /stereo/right/camera_info:
        subscription:
          depth: 5
          history: keep_last
          reliability: best_effort
      /stereo/right/image_rect:
        subscription:
          depth: 5
          history: keep_last
          reliability: best_effort
    queue_size: 10
    sgbm_mode: 0
    speckle_range: 2
    speckle_size: 200
    start_type_description_service: true
    stereo_algorithm: 1
    texture_threshold: 10
    uniqueness_ratio: 5.0
    use_sim_time: false
    use_system_default_qos: false
"""
    path.write_text(text)


def copy_subset_images(subset_dir: Path, pairs: Sequence[StereoPair]) -> None:
    left_dir = subset_dir / "left"
    right_dir = subset_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    for index, pair in enumerate(pairs, start=1):
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", pair.key)
        left_name = f"{index:03d}_{safe_key}{pair.left_path.suffix.lower()}"
        right_name = f"{index:03d}_{safe_key}{pair.right_path.suffix.lower()}"
        shutil.copy2(pair.left_path, left_dir / left_name)
        shutil.copy2(pair.right_path, right_dir / right_name)


def save_subset(
    output_root: Path,
    subset_size: int,
    pairs: Sequence[StereoPair],
    validation_pairs: Sequence[StereoPair],
    model: CalibrationModel,
    evaluation: Evaluation,
    copy_images: bool,
    disparity_node_name: str,
) -> None:
    subset_dir = output_root / f"best_{subset_size}"
    subset_dir.mkdir(parents=True, exist_ok=True)

    with (subset_dir / "selected_pairs.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "key", "left_path", "right_path", "sharpness"])
        for index, pair in enumerate(pairs, start=1):
            writer.writerow(
                [index, pair.key, str(pair.left_path), str(pair.right_path), pair.sharpness]
            )

    with (subset_dir / "validation_pairs.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "left_path", "right_path"])
        for pair in validation_pairs:
            writer.writerow([pair.key, str(pair.left_path), str(pair.right_path)])

    metrics = evaluation.to_dict()
    metrics.update(
        {
            "subset_size": subset_size,
            "left_rms_px": model.rms_left,
            "right_rms_px": model.rms_right,
            "translation_m": matrix_data(model.T),
            "rotation": matrix_data(model.R),
        }
    )
    (subset_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    write_ros_camera_yaml(
        subset_dir / "left.yaml",
        "narrow_stereo/left",
        model.image_size,
        model.K1,
        model.D1,
        model.R1,
        model.P1,
    )
    write_ros_camera_yaml(
        subset_dir / "right.yaml",
        "narrow_stereo/right",
        model.image_size,
        model.K2,
        model.D2,
        model.R2,
        model.P2,
    )
    write_disparity_yaml(
        subset_dir / "disparity.yaml",
        disparity_node_name,
    )

    np.savez_compressed(
        subset_dir / "calibration.npz",
        K1=model.K1,
        D1=model.D1,
        K2=model.K2,
        D2=model.D2,
        R=model.R,
        T=model.T,
        E=model.E,
        F=model.F,
        R1=model.R1,
        R2=model.R2,
        P1=model.P1,
        P2=model.P2,
        Q=model.Q,
    )

    if copy_images:
        copy_subset_images(subset_dir, pairs)


def main() -> int:
    args = parse_args()
    targets = sorted(
        {int(x.strip()) for x in args.targets.split(",") if x.strip()}, reverse=True
    )
    if not targets or min(targets) < 8:
        raise ValueError("Targets must contain subset sizes of at least 8.")

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    left_tokens = token_list(args.left_tokens)
    right_tokens = token_list(args.right_tokens)
    pattern_size = (args.cols, args.rows)
    object_points = asymmetric_object_points(args.cols, args.rows, args.spacing)

    with tempfile.TemporaryDirectory(prefix="stereo_subset_") as tmp:
        tmp_root = Path(tmp)
        if args.input.is_dir():
            image_root = args.input.resolve()
        elif args.input.is_file() and (
            args.input.name.endswith(".tar.gz")
            or args.input.name.endswith(".tgz")
            or args.input.suffix == ".tar"
        ):
            image_root = tmp_root / "extracted"
            print(f"Extracting {args.input} ...")
            safe_extract_tar(args.input.resolve(), image_root)
        else:
            raise FileNotFoundError(
                f"Input must be a directory or .tar/.tar.gz/.tgz archive: {args.input}"
            )

        raw_pairs, pairing_warnings = discover_pairs(
            image_root, left_tokens, right_tokens
        )
        for warning in pairing_warnings:
            print(f"PAIRING WARNING: {warning}", file=sys.stderr)

        print(f"Found {len(raw_pairs)} left/right filename pairs.")
        if len(raw_pairs) == 0:
            raise RuntimeError(
                "No pairs found. Adjust --left-tokens/--right-tokens to match your filenames."
            )

        debug_dir = args.output / "detection_debug" if args.debug_detections else None
        valid_pairs, image_size, rejected = load_and_detect_pairs(
            raw_pairs, pattern_size, debug_dir
        )

        (args.output / "rejected_pairs.txt").write_text(
            "\n".join(rejected) + ("\n" if rejected else "")
        )
        (args.output / "pairing_warnings.txt").write_text(
            "\n".join(pairing_warnings) + ("\n" if pairing_warnings else "")
        )

        print(f"Valid detected stereo pairs: {len(valid_pairs)}")
        print(f"Rejected pairs: {len(rejected)}")

        max_target = max(targets)
        if len(valid_pairs) < max_target:
            raise RuntimeError(
                f"Only {len(valid_pairs)} valid stereo pairs remain, but target {max_target} "
                "was requested. If '52 images' means 26 left/right pairs, 30- and 40-pair "
                "subsets are impossible; use --targets 20 or capture more pairs."
            )

        validation_count = min(
            max(0, args.validation_count),
            max(0, len(valid_pairs) - max_target),
        )
        training, validation = choose_diverse_validation(valid_pairs, validation_count)
        print(
            f"Training pool: {len(training)} pairs; held-out validation: {len(validation)} pairs."
        )

        if len(training) < max_target:
            raise RuntimeError(
                f"Validation split left only {len(training)} training pairs for target {max_target}."
            )

        removal_log: list[dict[str, object]] = []
        snapshots: dict[int, tuple[list[StereoPair], CalibrationModel, Evaluation]] = {}

        current = list(training)
        current_model = calibrate_subset(
            current, image_size, object_points, args.estimate_k3
        )
        current_eval = evaluate_model(
            current_model, validation, current, object_points, args.baseline
        )
        print(
            f"Initial {len(current)}: score={current_eval.score:.4f}, "
            f"stereo RMS={current_model.stereo_rms:.4f}px, "
            f"vertical={current_eval.validation_vertical_mean_px:.4f}px, "
            f"baseline={current_model.baseline_m * 100:.3f}cm"
        )

        for target in targets:
            if target > len(current):
                print(
                    f"WARNING: skipping target {target}; current pool has {len(current)} pairs.",
                    file=sys.stderr,
                )
                continue

            if len(current) > target:
                current, stage_removals = reduce_to_target(
                    current, current_model, object_points, target
                )
                for row in stage_removals:
                    row["stage_target"] = target
                    removal_log.append(row)
                    print(
                        f"  removed {row['removed_key']!r} -> {row['remaining']} pairs "
                        f"(reproj={row['reprojection_px']:.4f}px, "
                        f"vertical={row['vertical_mean_px']:.4f}px)"
                    )

            # Recalibrate exactly at each requested target. This target model is also
            # used to rank removals for the next smaller target.
            current_model = calibrate_subset(
                current, image_size, object_points, args.estimate_k3
            )
            current_eval = evaluate_model(
                current_model, validation, current, object_points, args.baseline
            )
            snapshots[target] = (list(current), current_model, current_eval)

            print(
                f"Target {target}: score={current_eval.score:.4f}, "
                f"stereo RMS={current_model.stereo_rms:.4f}px, "
                f"validation vertical={current_eval.validation_vertical_mean_px:.4f}px, "
                f"baseline={current_model.baseline_m * 100:.3f}cm"
            )

        with (args.output / "removal_log.csv").open("w", newline="") as f:
            if removal_log:
                writer = csv.DictWriter(f, fieldnames=list(removal_log[0].keys()))
                writer.writeheader()
                writer.writerows(removal_log)

        if validation:
            validation_dir = args.output / "validation"
            validation_dir.mkdir(parents=True, exist_ok=True)
            with (validation_dir / "validation_pairs.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["key", "left_path", "right_path", "sharpness"])
                for pair in validation:
                    writer.writerow(
                        [pair.key, str(pair.left_path), str(pair.right_path), pair.sharpness]
                    )
            if not args.no_copy_images:
                copy_subset_images(validation_dir, validation)

        summary_rows: list[dict[str, object]] = []
        for target in targets:
            if target not in snapshots:
                print(f"WARNING: target {target} was not reached.", file=sys.stderr)
                continue
            pairs, saved_model, saved_eval = snapshots[target]
            save_subset(
                args.output,
                target,
                pairs,
                validation,
                saved_model,
                saved_eval,
                copy_images=not args.no_copy_images,
                disparity_node_name=args.disparity_node_name,
            )
            summary_rows.append(
                {
                    "subset_size": target,
                    "left_rms_px": saved_model.rms_left,
                    "right_rms_px": saved_model.rms_right,
                    **saved_eval.to_dict(),
                }
            )

        with (args.output / "summary.csv").open("w", newline="") as f:
            if summary_rows:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)

        # The disparity parameters do not depend on which calibration subset wins,
        # so also place a convenient copy at the output root. Each best_N folder
        # receives the same file alongside left.yaml and right.yaml.
        write_disparity_yaml(
            args.output / "disparity.yaml",
            args.disparity_node_name,
        )

        config = {
            "pattern": "asymmetric_circles_grid",
            "cols": args.cols,
            "rows": args.rows,
            "spacing_m": args.spacing,
            "expected_baseline_m": args.baseline,
            "image_width": image_size[0],
            "image_height": image_size[1],
            "raw_filename_pairs": len(raw_pairs),
            "valid_detected_pairs": len(valid_pairs),
            "rejected_pairs": len(rejected),
            "training_pairs": len(training),
            "validation_pairs": len(validation),
            "targets": targets,
            "disparity_node_name": args.disparity_node_name,
            "disparity_yaml": str(args.output / "disparity.yaml"),
        }
        (args.output / "run_config.json").write_text(json.dumps(config, indent=2))

        print(f"\nFinished. Results written to: {args.output}")
        print(f"  disparity params: {args.output / 'disparity.yaml'}")
        for row in summary_rows:
            print(
                f"  best_{row['subset_size']}: score={row['score']:.4f}, "
                f"vertical={row['validation_vertical_mean_px']:.4f}px, "
                f"baseline={float(row['baseline_m']) * 100:.3f}cm"
            )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, cv2.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
