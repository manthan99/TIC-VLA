"""
Consolidated configuration for mobile navigation environments.
"""
import os
import numpy as np
from pathlib import Path
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.sensors import ContactSensorCfg, CameraCfg, RayCasterCfg, patterns
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from typing import Sequence, Union, Tuple
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg

from .observations import ObservationsCfg
from .rewards import RewardsCfg
from .termination import TerminationsCfg
from .commands import CommandsCfg
from .actions import ActionsCfg
from .event import EventCfg
from .curriculum import CurriculumCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
ArrayLike = Union[Sequence[float], Tuple[float, float], Tuple[float, float, float]]
PACKAGE_ROOT = Path(__file__).resolve().parent

# Asset paths
ENV_USD = f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd"
ROBOT_USD = os.getenv(
    "TICVLA_RL_ROBOT_USD",
    str(PACKAGE_ROOT / "assets" / "coco_one" / "coco_one.usd"),
)
GO2_USD = f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"
IDLE_CLIP_USD = os.getenv(
    "TICVLA_RL_IDLE_CLIP_USD",
    str(PACKAGE_ROOT / "assets" / "ped_actions" / "synbody_idle426.fbx" / "synbody_idle426.fbx.usd"),
)
WALK_CLIP_USD = os.getenv(
    "TICVLA_RL_WALK_CLIP_USD",
    str(PACKAGE_ROOT / "assets" / "ped_actions" / "synbody_walking426.fbx" / "synbody_walking426.fbx.usd"),
)
PEDESTRIAN_ASSETS_PATH = os.getenv(
    "TICVLA_RL_PEDESTRIAN_ASSETS_PATH",
    str(PACKAGE_ROOT / "assets" / "peds"),
)
LENGTH = 40.0
ENV_ORIGINS = [
    [0.00, 0.00, 0.05],
    # [  2.00,  0.00, 0.05],
    # [ -6.10,  0.00, 0.05],
    # [  1.12, -8.80, 0.05],
    # [ -13.83, -2.90, 0.05],
    # [ -19.51,  8.50, 0.05],
    # [ -15.00, 20.00, 0.05],
    # [ -16.00, 43.00, 0.05],
    # [  2.47, 29.87, 0.05],
    # [ -18.00, 54.00, 0.05],
    # [ -0.67, 53.00, 0.05],
]

@configclass
class TICVLANavSceneCfg(InteractiveSceneCfg):
    num_envs: int = 1 #Must be reflected in env_origins
    env_spacing: float = 5.0

    @property
    def env_origins(self) -> list[list[float]]:
        return ENV_ORIGINS

    # Terrain configuration
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=ENV_USD,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # Coco One Robot configuration
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path=ROBOT_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0., 0., 0.3),
            joint_pos={
                ".*wheel_joint*":0.0,"base_to_front_axle_joint":0.0,
            },
            joint_vel={
                ".*wheel_joint":0.0,"base_to_front_axle_joint":0.0,
            },
        ),
        actuators={"wheels": DelayedPDActuatorCfg(
            joint_names_expr=[".*wheel_joint"],
            velocity_limit=100.0,
            min_delay=0,  
            max_delay=4, 
            stiffness={
                ".*_wheel_joint": 0.0,
            },
            damping={".*_wheel_joint": 0.3},
            friction={
                ".*_wheel_joint": 0.0,
            },
            armature={
                ".*_wheel_joint": 0.0,
            },

        ), "axle": 
            DCMotorCfg(
            joint_names_expr=["base_to_front_axle_joint"],
            saturation_effort=64.0,
            effort_limit=64.0,
            velocity_limit=20.,
            stiffness=25.0,
            damping=0.5,
            friction=0.
        ),
        "shock": ImplicitActuatorCfg(
            joint_names_expr=[".*shock_joint"],
            stiffness=0.0,
            damping=0.0,
        )
        
        },
    soft_joint_pos_limit_factor=1.0,#0.95
    )

    # Unitree Go2 Robot configuration
    # robot = ArticulationCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot",
    #     spawn=UsdFileCfg(
    #         usd_path=GO2_USD,
    #         activate_contact_sensors=True,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=False,
    #             retain_accelerations=False,
    #             linear_damping=0.0,
    #             angular_damping=0.0,
    #             max_linear_velocity=1000.0,
    #             max_angular_velocity=1000.0,
    #             max_depenetration_velocity=1.0,
    #         ),
    #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #             enabled_self_collisions=False,
    #             solver_position_iteration_count=4,
    #             solver_velocity_iteration_count=0,
    #         ),
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.0, 0.0, 0.54),
    #         joint_pos={
    #             ".*L_hip_joint": 0.1,
    #             ".*R_hip_joint": -0.1,
    #             "F[L,R]_thigh_joint": 0.8,
    #             "R[L,R]_thigh_joint": 1.0,
    #             ".*_calf_joint": -1.5,
    #         },
    #         joint_vel={".*": 0.0},
    #     ),
    #     soft_joint_pos_limit_factor=0.9,
    #     actuators={
    #         "base_legs": DCMotorCfg(
    #             joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    #             effort_limit=23.5,
    #             saturation_effort=23.5,
    #             velocity_limit=30.0,
    #             stiffness=25.0,
    #             damping=0.5,
    #             friction=0.0,
    #         ),
    #     },
    # )

    # # # sensors
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )
    height_scanner = None
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3, 
        track_air_time=True)

    # Coco One Robot camera configuration
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_cam",
        update_period=0.1,
        height=1080,
        width=1920,
        data_types=['rgb'],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=400.0, horizontal_aperture=30.0, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.310, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    # Unitree Go2 Robot camera configuration
    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=['rgb'],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=12.0, focus_distance=400.0, horizontal_aperture=30.0, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )

@configclass
class TICVLANavEnvCfg(ManagerBasedRLEnvCfg):
    # Simulation configuration
    scene: TICVLANavSceneCfg = TICVLANavSceneCfg()

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    #curriculum: CurriculumCfg = CurriculumCfg()

    # Asset paths
    env_usd: str = ENV_USD
    robot_usd: str = ROBOT_USD

    # Pedestrian configuration
    gltf_scale: float = 0.01
    walkable_radius: float = 40.0
    idle_clip_usd: str = IDLE_CLIP_USD
    walk_clip_usd: str = WALK_CLIP_USD
    dynamic_assets_path: str = PEDESTRIAN_ASSETS_PATH
    maximum_dynamic_object_number: int = 35
    
    # Walkable position sampling parameters
    person_radius: float = 0.6
    walkable_probe_top: float = 1.2
    walkable_probe_depth: float = 3.0

    # Task parameters
    num_people: int = 100
    disable_gpu_physics: bool = False

    robot_spawn_yaw: float = 0.0

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 20 
        self.episode_length_s = LENGTH
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # Enable scene query support for headless mode (needed for raycasting)
        self.sim.enable_scene_query_support = True
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

# Environment presets
@configclass
class HospitalNavEnvCfg(TICVLANavEnvCfg):
    """Hospital environment configuration."""
    env_usd: str = f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd"

@configclass
class WarehouseNavEnvCfg(TICVLANavEnvCfg):
    """Warehouse environment configuration."""
    env_usd: str = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd"

@configclass
class OfficeNavEnvCfg(TICVLANavEnvCfg):
    """Office environment configuration."""
    env_usd: str = f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd"

@configclass
class TownNavEnvCfg(TICVLANavEnvCfg):
    """Town environment configuration."""
    env_usd: str = f"{ISAAC_NUCLEUS_DIR}/Environments/Outdoor/Rivermark/rivermark.usd"
