"""Per-rig sensor contracts for the Wild VLN pipeline.

Every fact here was measured from the raw bags (sessions of 2026-07-22/23),
not assumed. A bag must match its rig contract exactly to enter the pipeline;
a mismatch means a new rig variant and demands investigation, not a fallback.

The two traps this file exists to prevent from recurring:
  - UMD publishes TWO intrinsic matrices; the stored frames are the rectified
    /camera_processed stream, so P applies and the distortion D must NOT be
    applied. Using K overstates fx by 40%.
  - The Ouster stamps message headers with sensor uptime, not epoch, so UMD
    topics can only be associated on bag-receive time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class RigContract:
    name: str
    image_topic: str
    image_size: Tuple[int, int]              # (width, height)
    camera_info_topic: str
    # Projection intrinsics of the stored images: fx, fy, cx, cy.
    # AUTHORITATIVE SOURCE: the GND authors' calibration files (user-provided
    # 2026-07-25), NOT the bag camera_info — UMD's bag P is stale.
    intrinsics: Tuple[float, float, float, float]
    # What the bag's camera_info is expected to report (rig identity check).
    caminfo_expected: Tuple[float, float, float, float]
    # Which camera_info matrix must match `caminfo_expected` ("K" or "P").
    intrinsics_source: str
    # LiDAR -> camera extrinsic, 4x4 row-major (authors' calibration).
    T_cam_lidar: Tuple[Tuple[float, float, float, float], ...]
    cloud_topic: str
    odom_topic: str
    gps_topic: str
    # Cross-topic association: "header" only when every sensor stamps epoch
    # time; "bag" when any sensor (e.g. Ouster) stamps uptime.
    time_base: str
    # EKF pose positions teleport on some rigs (GPS fusion resets); the twist
    # field is always the one to integrate.
    odom_pose_trustworthy: bool
    lidar_height_m: float                    # sensor plate above ground (measured)
    compressed_images: bool
    # Cloud->pose association time offset (s) and LiDAR accumulation window
    # for depth lifting, both measured on held-out LiDAR (2026-07-24).
    cloud_time_offset_s: float = 0.0
    depth_accum_window_s: float = 0.0
    notes: str = ""
    extra_topics: Dict[str, str] = field(default_factory=dict)


ZED2 = RigContract(
    name="gnd-zed2",
    image_topic="zed_node/rgb/image_rect_color/compressed",
    image_size=(640, 360),
    camera_info_topic="zed_node/rgb/camera_info",
    intrinsics=(263.799377, 263.799377, 328.877686, 178.553421),
    caminfo_expected=(263.80, 263.80, 328.88, 178.55),
    intrinsics_source="K",                   # K == P, D == 0 on this rig
    T_cam_lidar=((0., -1., 0., 0.08), (0., 0., -1., -0.15),
                 (1., 0., 0., -0.25), (0., 0., 0., 1.)),
    cloud_topic="/velodyne_points",
    odom_topic="/odometry/filtered",
    gps_topic="/f9p_rover/navpvt",
    time_base="header",
    odom_pose_trustworthy=False,             # AU peaks at ~3e4 m/s on resets
    lidar_height_m=0.39,
    compressed_images=True,
    cloud_time_offset_s=0.0,     # -0.10 was fit under the wrong cardinal
    # Official-calib re-test 2026-07-26: accumulation costs absrel here
    # (0.150 single -> 0.190 at +-1 s; relative poses noisier than UMD's)
    # but buys 38 -> 61% patch coverage. User call: +-1 s, same as UMD.
    depth_accum_window_s=1.0,
    notes="14 sites; mount pitch 2.86 deg measured from TF",
    extra_topics={"imu": "zed_node/imu/data"},
)

UMD = RigContract(
    name="gnd-umd",
    image_topic="/camera_processed",
    image_size=(640, 480),
    camera_info_topic="/camera_info",
    intrinsics=(357.77508, 358.53616, 334.31282, 222.21926),
    caminfo_expected=(265.18, 312.35, 332.65, 223.35),   # bag P is STALE
    intrinsics_source="P",                   # frames are rectified; never apply D
    T_cam_lidar=((0., -1., 0., 0.), (0., 0., -1., 0.15),
                 (1., 0., 0., -0.02), (0., 0., 0., 1.)),
    cloud_topic="/ouster/points",
    odom_topic="/odometry/filtered",
    gps_topic="/mavros/global_position/raw/fix",
    time_base="bag",                         # Ouster headers are sensor uptime
    odom_pose_trustworthy=False,             # clean at lot9, but trail peaks at
                                             # ~4e3 m/s -> never use pose anywhere
    lidar_height_m=0.41,
    compressed_images=False,
    cloud_time_offset_s=0.0,
    # Official-calib re-test 2026-07-26: absrel holds to +-1 s
    # (0.058 -> 0.065, coverage 40 -> 75%); +-2 s breaks down (0.142).
    depth_accum_window_s=1.0,
    notes="11 sites, all and only UMD; no usable camera TF (pitch/height fitted)",
    extra_topics={"imu": "/imu/data"},
)


def rig_for_site(site: str) -> RigContract:
    return UMD if site.startswith("UMD") else ZED2


def check_camera_info(rig: RigContract, K, P, D, width, height,
                      tol: float = 0.5) -> Optional[str]:
    """Return None if camera_info matches the contract, else a reason string."""
    if (width, height) != rig.image_size:
        return f"image size {width}x{height} != {rig.image_size}"
    src = K if rig.intrinsics_source == "K" else P
    got = (src[0][0], src[1][1], src[0][2], src[1][2])
    for name, g, want in zip(("fx", "fy", "cx", "cy"), got, rig.caminfo_expected):
        if abs(g - want) > tol:
            return f"{name}={g:.2f} != {want:.2f} (from {rig.intrinsics_source})"
    if rig.intrinsics_source == "K" and max(abs(d) for d in D) > 1e-9:
        return "expected zero distortion on this rig"
    return None
