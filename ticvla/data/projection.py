"""Image-space trace ground truth for TIC-VLA.

Converts the metric future waypoints stored in the dataset JSONs into a trace of
pixel coordinates on the current camera image: the path the robot will drive,
drawn on the ground plane in front of it.

Conventions
-----------
- Dataset offsets are FLU (x forward, y left, z up) in the *current* frame's
  body frame, and the body origin coincides with the camera (``t_bc = 0`` in
  ``data/s01_batch_json_generation.py``).
- The robot drives on the ground, so the trace is drawn ``camera_height`` metres
  below the camera: ``p_ground = (x, y, -camera_height)``.
- Camera axes are OpenCV style (x right, y down, z forward), so
  ``X_cv = -y_flu``, ``Y_cv = -z_flu``, ``Z_cv = x_flu``.
- Output traces are normalized to [0, 1] (x/width, y/height), matching the
  NaviTrace prompt contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Each dataset JSON covers a 20 s window; the trace runs from the current
# timestamp to the end of that window.
WINDOW_SECONDS = 20.0
# Number of points in a ground-truth trace.
TRACE_POINTS = 10
# Fixed look-ahead for a trace, in seconds. Independent of the 20 s window so
# every sample gets the same amount of future (the JSONs hold up to 40 s).
HORIZON_SECONDS = 10.0
# Points closer than this (metres in front of the camera) are numerically
# unstable to project.
MIN_DEPTH_M = 0.15


@dataclass(frozen=True)
class CameraModel:
    """Pinhole intrinsics plus the camera's height above the ground plane."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    camera_height_m: float
    # Downward tilt of the camera in degrees. A positive pitch raises the
    # horizon to ``cy - fy*tan(pitch)`` instead of leaving it at ``cy``.
    pitch_deg: float = 0.0
    # True when intrinsics/height are estimated rather than read from the dataset.
    estimated: bool = False
    label: str = ""

    @property
    def horizon_v(self) -> float:
        """Image row where the ground plane vanishes."""
        return self.cy - self.fy * np.tan(np.radians(self.pitch_deg))

    @classmethod
    def from_hfov(
        cls,
        width: int,
        height: int,
        hfov_deg: float,
        camera_height_m: float,
        label: str = "",
        pitch_deg: float = 0.0,
        estimated: bool = True,
    ) -> "CameraModel":
        """Square-pixel pinhole from a horizontal field of view."""
        focal = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        return cls(
            fx=focal, fy=focal, cx=width / 2.0, cy=height / 2.0,
            width=width, height=height, camera_height_m=camera_height_m,
            pitch_deg=pitch_deg, estimated=estimated, label=label,
        )

    def describe(self) -> str:
        tag = "ESTIMATED" if self.estimated else "from dataset"
        hfov = 2 * np.degrees(np.arctan((self.width / 2.0) / self.fx))
        return (
            f"{self.label or 'camera'} [{tag}]: {self.width}x{self.height}, "
            f"fx=fy={self.fx:.0f} (HFOV {hfov:.0f}°), "
            f"cx={self.cx:.0f} cy={self.cy:.0f}, height={self.camera_height_m:.2f} m, "
            f"pitch={self.pitch_deg:.1f}° (horizon v={self.horizon_v:.0f})"
        )

    @classmethod
    def from_isaac_camera_params(cls, params_path: str | Path, camera_height_m: float) -> "CameraModel":
        """Build intrinsics from an Isaac Sim ``camera_params_*.json`` file.

        The OpenGL-style ``cameraProjection`` matrix stores 2*fx/width and
        2*fy/height in its first two diagonal entries.
        """
        params = json.load(open(params_path, "r"))
        width, height = (int(v) for v in params["renderProductResolution"])
        projection = params["cameraProjection"]
        return cls(
            fx=float(projection[0]) * width / 2.0,
            fy=float(projection[5]) * height / 2.0,
            cx=width / 2.0,
            cy=height / 2.0,
            width=width,
            height=height,
            camera_height_m=camera_height_m,
        )

    def project_ground(self, offsets_flu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project FLU offsets onto the ground plane in pixel coordinates.

        Args:
            offsets_flu: (N, 3) or (N, 2) forward/left(/up) offsets in metres.

        Returns:
            uv: (N, 2) pixel coordinates (may lie outside the image).
            in_front: (N,) bool mask, True where the point is in front of the camera.
        """
        offsets = np.asarray(offsets_flu, dtype=float)
        forward = offsets[:, 0]
        left = offsets[:, 1]

        up = offsets[:, 2] if offsets.shape[1] > 2 else np.zeros_like(forward)

        x_cv = -left
        y_cv = self.camera_height_m - up
        z_cv = forward

        # Tilt the camera down by pitch_deg (rotation about the camera x-axis).
        if self.pitch_deg:
            angle = np.radians(self.pitch_deg)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            y_cv, z_cv = y_cv * cos_a - z_cv * sin_a, y_cv * sin_a + z_cv * cos_a

        in_front = z_cv > MIN_DEPTH_M
        safe_z = np.where(in_front, z_cv, 1.0)
        u = self.fx * x_cv / safe_z + self.cx
        v = self.fy * y_cv / safe_z + self.cy
        return np.column_stack([u, v]), in_front


# DynaNav: Nova Carter front Hawk camera, 1920x1080, ~90 deg horizontal FOV.
# Intrinsics verified against the per-frame camera_params JSONs shipped with the
# dataset; the camera height is the constant z in each recording's trajectory.csv.
DYNANAV_CAMERA = CameraModel(
    fx=957.8,
    fy=957.8,
    cx=960.0,
    cy=540.0,
    width=1920,
    height=1080,
    camera_height_m=0.346,
    estimated=False,
    label="DynaNav / Nova Carter front Hawk",
)

# SCAND and GND ship no intrinsics and no camera height, so the models below are
# ESTIMATES from the platform and image geometry, pending verification:
#   - SCAND Spot   (1280x720):  Azure Kinect RGB, ~90 deg HFOV, sensor on the body.
#   - SCAND Jackal (1280x1024): front stereo camera, ~90 deg HFOV, sensor mast.
#   - GND          (640x360/480): Jackal RGB camera, ~70 deg HFOV.
# Tune fx (spread/curvature of the trace) and camera_height_m (how fast the trace
# rises toward the horizon) if the overlays look off.
SCAND_SPOT = CameraModel.from_hfov(1280, 720, 90.0, 0.60, "SCAND / Spot Azure Kinect")
SCAND_JACKAL = CameraModel.from_hfov(1280, 1024, 90.0, 0.55, "SCAND / Jackal stereo")
# GND: measured from the raw AU.bag (scripts/extract_bag_calibration.py, 2026-07-17).
#   zed_node/rgb/camera_info -> ZED2 left, 640x360 rectified (plumb_bob, D all zero),
#     fx=fy=263.80, cx=328.88, cy=178.55  => 101 deg HFOV, not the 70 deg we assumed.
#   /tf_static zed2_base_link -> zed2_camera_center -> pitch +2.86 deg (nose down),
#     which independently confirms the ~3 deg picked by eye.
# The bag's TF puts zed2_base_link at the base_link origin, so it gives no usable
# mount height; camera_height_m stays the visually calibrated value.
GND_ZED2 = CameraModel(
    fx=263.80, fy=263.80, cx=328.88, cy=178.55,
    width=640, height=360, camera_height_m=1.00, pitch_deg=2.86,
    estimated=False, label="GND / ZED2 left (from AU.bag)",
)
GND_WIDE = GND_ZED2
# The 640x480 recordings are ALL and ONLY the UMD sites, and they are a
# different rig entirely (measured from UMD_map1_1_trail_chunk01.bag): topic
# /camera_processed, frame narrow_stereo, LiDAR is an Ouster not a Velodyne,
# and there are no zed2 frames at all.
#
# Its /camera_info carries two different matrices and only P applies here:
#   K = [372.25, 368.58, 329.00, 227.90], D = [-0.297, 0.067, ...]  raw sensor
#   P = [265.18, 312.35, 332.65, 223.35]                            rectified
# The frames stored in the dataset are byte-for-byte the /camera_processed
# images (checked against the bag: mean abs diff 1.6, i.e. JPEG noise only) and
# they are already rectified. Measured on 342 near-straight peripheral edges in
# lot9: a true straight line would bow ~3.8% of its length if the frames were
# raw, but they bow 1.18%, and undistorting with (K, D) does not reduce it. So
# D must not be applied, and P -- not K -- is the intrinsic for these images.
#
# Independent check: yaw-rate optical flow (agreeing across the EKF odometry and
# the raw gyro) gives fx ~ 225 here, against ~283 on the AU bag whose true fx is
# 263.8. That estimator runs ~7-9% high, so the corrected UMD figure is ~210 --
# same ballpark as P's 265, and nowhere near K's 372.
#
# Its TF tree has no usable camera transform (world->camera_frame is identity),
# so pitch and height still need visual calibration.
# NOTE: project_ground is a pinhole and ignores D; with k1 = -0.30 straight
# ground lines bow noticeably toward the image edges. Undistortion is the next
# correctness fix for UMD specifically.
GND_UMD = CameraModel(
    fx=265.18, fy=312.35, cx=332.65, cy=223.35,
    width=640, height=480, camera_height_m=1.00, pitch_deg=2.86,
    estimated=True, label="GND / UMD narrow_stereo (rectified P; pitch+height estimated)",
)
GND_43 = GND_UMD

CAMERA_REGISTRY: Dict[str, Optional[CameraModel]] = {
    "DynaNav": DYNANAV_CAMERA,
    "SCAND": SCAND_SPOT,
    "GND": GND_WIDE,
}


def get_camera(
    dataset: str,
    image_size: Optional[Tuple[int, int]] = None,
    recording: str = "",
) -> CameraModel:
    """Camera model for a dataset, refined by image resolution and platform.

    SCAND and GND mix resolutions and platforms across recordings, so the model
    is chosen per sample rather than per dataset.
    """
    if dataset == "DynaNav":
        return DYNANAV_CAMERA

    if dataset == "SCAND":
        if image_size == (1280, 1024) or "jackal" in recording.lower():
            base = SCAND_JACKAL
        else:
            base = SCAND_SPOT
    elif dataset == "GND":
        base = GND_43 if image_size == (640, 480) else GND_WIDE
    else:
        raise KeyError(f"Unknown dataset '{dataset}'; expected one of {sorted(CAMERA_REGISTRY)}")

    # Rescale to the actual image if a recording uses an unexpected resolution.
    if image_size is not None and image_size != (base.width, base.height):
        width, height = image_size
        scale = width / base.width
        base = CameraModel(
            fx=base.fx * scale, fy=base.fy * scale,
            cx=width / 2.0, cy=height / 2.0,
            width=width, height=height,
            camera_height_m=base.camera_height_m, pitch_deg=base.pitch_deg,
            estimated=True, label=f"{base.label} (rescaled)",
        )
    return base


def future_offsets_in_window(
    sample: Dict,
    window_seconds: float = WINDOW_SECONDS,
    horizon_seconds: Optional[float] = HORIZON_SECONDS,
) -> np.ndarray:
    """Future FLU offsets ahead of the current frame.

    With ``horizon_seconds`` set (the default), every sample gets the same
    ``horizon_seconds`` of future regardless of where it sits in its window —
    the JSONs store up to 40 s of future, so the 20 s window edge is not a real
    limit. Traces then have a consistent scale, which matters because all of
    them are resampled to the same number of points.

    Pass ``horizon_seconds=None`` to fall back to the old behaviour of stopping
    at the end of the sample's own window.
    """
    if horizon_seconds is not None:
        cutoff = float(sample.get("timestamp", 0.0) or 0.0) + float(horizon_seconds)
    else:
        cutoff = window_seconds
    offsets = [
        w["offset"]
        for w in sample.get("future", [])
        if isinstance(w, dict) and "offset" in w and float(w.get("time", np.inf)) <= cutoff
    ]
    if not offsets:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(offsets, dtype=float)


def smooth_polyline(points: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred moving average that keeps the endpoints anchored."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3 or window < 3:
        return points
    window = min(window, len(points) if len(points) % 2 else len(points) - 1)
    if window < 3:
        return points
    pad = window // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.column_stack([
        np.convolve(padded[:, i], kernel, mode="valid") for i in range(points.shape[1])
    ])
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return smoothed


def resample_polyline(points: np.ndarray, n_points: int = TRACE_POINTS) -> np.ndarray:
    """Arc-length resample to exactly ``n_points`` (uniform spacing along the path)."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        raise ValueError("Cannot resample an empty polyline")
    if len(points) == 1:
        return np.tile(points, (n_points, 1))

    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    if dist[-1] <= 0:
        return np.tile(points[:1], (n_points, 1))
    dist = dist / dist[-1]
    targets = np.linspace(0.0, 1.0, n_points)
    return np.column_stack([np.interp(targets, dist, points[:, i]) for i in range(points.shape[1])])


def _densify(points: np.ndarray, max_step_px: float = 4.0) -> np.ndarray:
    """Insert intermediate points so the polyline can be clipped precisely."""
    if len(points) < 2:
        return points
    out: List[np.ndarray] = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        steps = int(np.ceil(np.linalg.norm(end - start) / max_step_px))
        if steps > 1:
            for k in range(1, steps):
                out.append(start + (end - start) * (k / steps))
        out.append(end)
    return np.asarray(out)


def _first_visible_run(uv: np.ndarray, width: int, height: int, margin: float = 1.0) -> np.ndarray:
    """Return the first contiguous run of points inside the image rectangle.

    The trace starts at the robot, which projects *below* the image, so the run
    begins where the path enters the frame at the bottom edge. Taking only the
    first run avoids stitching across a segment that leaves and re-enters view.
    """
    inside = (
        (uv[:, 0] >= -margin)
        & (uv[:, 0] <= width - 1 + margin)
        & (uv[:, 1] >= -margin)
        & (uv[:, 1] <= height - 1 + margin)
    )
    if not inside.any():
        return np.zeros((0, 2), dtype=float)
    start = int(np.argmax(inside))
    end = start
    while end < len(inside) and inside[end]:
        end += 1
    return uv[start:end]


def image_trace_from_sample(
    sample: Dict,
    camera: CameraModel,
    n_points: int = TRACE_POINTS,
    window_seconds: float = WINDOW_SECONDS,
    horizon_seconds: Optional[float] = HORIZON_SECONDS,
    smooth_window: int = 5,
    min_visible_points: int = 2,
    min_path_length_px: float = 40.0,
    normalize: bool = True,
    use_elevation: bool = True,
) -> Optional[np.ndarray]:
    """Build the image-space trace ground truth for one sample.

    Pipeline: take the future offsets inside the window, smooth them in metric
    space, project onto the ground plane, keep the visible part of the path, and
    arc-length resample to exactly ``n_points``.

    Returns:
        (n_points, 2) array, normalized to [0, 1] unless ``normalize`` is False,
        or None when the sample yields no usable trace (robot nearly stationary,
        or the path is not visible in the current frame).
    """
    offsets = future_offsets_in_window(sample, window_seconds, horizon_seconds)
    if len(offsets) < 2:
        return None

    if not use_elevation:
        # Treat the path as perfectly flat, ignoring logged elevation change.
        offsets = offsets.copy()
        offsets[:, 2] = 0.0

    offsets = smooth_polyline(offsets, window=smooth_window)
    uv, in_front = camera.project_ground(offsets)
    if in_front.sum() < 2:
        return None

    # Keep the leading run of points in front of the camera; a path that passes
    # behind the camera cannot continue as a single image-space curve.
    first_behind = np.argmax(~in_front) if (~in_front).any() else len(in_front)
    if not in_front[0]:
        first_behind = len(in_front)
        uv = uv[in_front]
    else:
        uv = uv[:first_behind]
    if len(uv) < 2:
        return None

    visible = _first_visible_run(_densify(uv), camera.width, camera.height)
    if len(visible) < min_visible_points:
        return None

    path_length = float(np.linalg.norm(np.diff(visible, axis=0), axis=1).sum())
    if path_length < min_path_length_px:
        return None

    trace = resample_polyline(visible, n_points)
    trace[:, 0] = np.clip(trace[:, 0], 0.0, camera.width - 1)
    trace[:, 1] = np.clip(trace[:, 1], 0.0, camera.height - 1)
    if normalize:
        trace = trace / np.array([camera.width, camera.height], dtype=float)
    return trace


def _quat_to_rotation(quat_xyzw: Sequence[float]) -> np.ndarray:
    """Rotation matrix (body -> world) from an [x, y, z, w] quaternion."""
    x, y, z, w = (float(v) for v in quat_xyzw[:4])
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def history_path_in_current_frame(sample: Dict) -> np.ndarray:
    """Reconstruct the travelled path, expressed in the current frame's FLU axes.

    Each history entry stores its displacement from the previous entry in *that*
    entry's body frame, plus its own body->world orientation, so the path is
    integrated in world coordinates and then rotated into the current frame.

    Returns:
        (N, 3) array of past positions; the last row is ~0.1 s before now.
    """
    history = [h for h in sample.get("history", []) if isinstance(h, dict)]
    if len(history) < 2:
        return np.zeros((0, 3), dtype=float)

    position = np.zeros(3)
    positions = [position.copy()]
    for previous, current in zip(history[:-1], history[1:]):
        delta = np.asarray(current.get("trajectory", [0.0, 0.0, 0.0]), dtype=float)
        rotation = _quat_to_rotation(previous.get("orientation", [0, 0, 0, 1]))
        position = position + rotation @ delta
        positions.append(position.copy())
    positions = np.asarray(positions)

    current_quat = (sample.get("current", {}) or {}).get("orientation") or history[-1].get(
        "orientation", [0, 0, 0, 1]
    )
    rotation_current = _quat_to_rotation(current_quat)
    # World offsets from the newest history sample, rotated into the current frame.
    return (positions - positions[-1]) @ rotation_current


def orientation_rpy_deg(quat_xyzw: Sequence[float]) -> Tuple[float, float, float]:
    """Roll, pitch, yaw in degrees from an [x, y, z, w] quaternion."""
    x, y, z, w = (float(v) for v in quat_xyzw[:4])
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return 0.0, 0.0, 0.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(float(v) for v in np.degrees([roll, pitch, yaw]))


def camera_for_frame(
    camera: CameraModel,
    sample: Dict,
    pitch_offset_deg: float = 0.0,
    max_pitch_deg: float = 25.0,
) -> CameraModel:
    """Camera with this frame's *measured* pitch instead of a fixed estimate.

    Outdoor platforms pitch as they drive over slopes and bumps, so one constant
    tilt cannot fit a whole recording. The per-frame orientation stored in the
    JSON gives the actual tilt; the waypoint offsets are gravity-aligned, so the
    tilt has to be applied here at projection time.

    ``pitch_offset_deg`` adds a fixed camera-mount bias on top of the measured
    body pitch. Implausible values are ignored rather than producing a wild trace.

    NOTE: off by default. Once GND's mount pitch was measured from the bag
    (2.86 deg), an A/B on flagged frames showed the odometry deviation made
    things worse, not better: a -2.25 deg deviation dropped the horizon and
    compressed the distance bars, a +1.35 deg one lifted the trace off the
    ground. The deviation is dominated by odometry noise, and since the trace is
    drawn on the ground the robot itself drives over, the fixed mount angle is
    the right quantity. Kept for datasets with genuinely reliable pitch.
    """
    quat = (sample.get("current", {}) or {}).get("orientation")
    if not quat or len(quat) < 4:
        return camera
    _, pitch_now, _ = orientation_rpy_deg(quat)

    # The absolute pitch carries an unknown per-recording bias (the pose frame is
    # not gravity-aligned the same way in every log), so only the *deviation*
    # from the window's own baseline is trusted. On a sustained slope the robot,
    # the camera and the ground tilt together, and the angle that matters for a
    # trace drawn on that ground is the mount angle — which the baseline keeps.
    baseline_pitches = [
        orientation_rpy_deg(h["orientation"])[1]
        for h in sample.get("history", [])
        if isinstance(h, dict) and h.get("orientation") and len(h["orientation"]) >= 4
    ]
    baseline = float(np.median(baseline_pitches)) if baseline_pitches else pitch_now

    pitch = camera.pitch_deg + (pitch_now - baseline) + pitch_offset_deg
    if not np.isfinite(pitch) or abs(pitch) > max_pitch_deg:
        return camera
    return replace(camera, pitch_deg=pitch, label=f"{camera.label} · per-frame pitch")


def history_frames_at(
    sample: Dict,
    lookbacks_s: Sequence[float] = (9.0, 6.0, 3.0),
    clamp_to_start: bool = True,
) -> List[Dict]:
    """Pick historical frames at the requested look-backs from the current time.

    Look-backs that would land before the window start are clamped to 0 s, so
    early samples repeat the first frame rather than dropping context.

    Returns one dict per look-back: ``{requested_s, actual_s, img}``.
    """
    history = [h for h in sample.get("history", []) if isinstance(h, dict) and "img" in h]
    if not history:
        return []
    times = np.array([float(h.get("time", 0.0)) for h in history])
    current_t = float(sample.get("timestamp", 0.0) or 0.0)

    frames: List[Dict] = []
    for lookback in lookbacks_s:
        target = current_t - float(lookback)
        if target < times.min():
            if not clamp_to_start:
                continue
            target = times.min()
        idx = int(np.argmin(np.abs(times - target)))
        frames.append({
            "requested_s": float(lookback),
            "actual_s": float(times[idx]),
            "img": history[idx]["img"],
        })
    return frames
