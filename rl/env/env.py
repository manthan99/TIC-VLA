"""
TIC-VLA navigation environment with pedestrian crowds.
"""
import math
import numpy as np
import os
import torch
import random

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from pxr import Usd, UsdGeom, UsdSkel, Sdf, Gf
from omni.physx import get_physx_scene_query_interface
from isaacsim.core.utils.stage import add_reference_to_stage
import isaacsim.core.utils.prims as prims_utils
import rl.env.navigation_mdp as mdp
import omni.usd
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import SphereCfg, PreviewSurfaceCfg

from .config import PEDESTRIAN_ASSETS_PATH, TICVLANavEnvCfg
from .utils import _is_path_clear, _contact_normal, _get_or_create_tos_ops

class TICVLANavEnv(ManagerBasedRLEnv):
    """TIC-VLA navigation environment with pedestrian crowds."""
    
    cfg: TICVLANavEnvCfg

    def __init__(self, cfg: TICVLANavEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Set up stage
        UsdGeom.SetStageUpAxis(self.sim.stage, UsdGeom.Tokens.z)
        if not self.sim.stage.GetPrimAtPath("/World/Crowd").IsValid():
            prims_utils.create_prim("/World/Crowd", "Xform")
        
        # Set up goal visualization markers
        self._setup_goal_markers()
        
        # Pedestrian state
        self.extras = {}
        self.num_people = int(getattr(self.cfg, "num_people", 30))
        self.gltf_scale = getattr(self.cfg, "gltf_scale", 0.01)
        
        self.people_xy = torch.zeros(self.num_people, 2, device=self.device)           # [N,2]
        self.people_goal_xy = torch.zeros_like(self.people_xy)                          # [N,2]
        self.people_speed = torch.zeros(self.num_people, device=self.device)            # [N]
        self._heading2d = torch.randn(self.num_people, 2, device=self.device)          # [N,2]
        self._heading2d /= (torch.linalg.norm(self._heading2d, dim=-1, keepdim=True) + 1e-9)
        self._wall_side = torch.zeros(self.num_people, dtype=torch.int8, device=self.device)
        self._yaw_face  = torch.zeros(self.num_people, device=self.device)
        self._people_paths = []    # list[str] under /World/Crowd (or /World)
        self._crowd_spawned = False

        # Pedestrian physics parameters
        self._avoid_gain = 2.2        
        self._yaw_offset = math.pi                        # how strongly we steer away from neighbors

        # Environment setup
        self._make_env_collidable("/World/", approximation="mesh")
        self._index_walkable_bounds("/World/ground")

        # Build dynamic assets
        self.dynamic_asset_animatable_state()  # (loads walk/idle clips)
        self.dynamic_assets_path = getattr(cfg, "dynamic_assets_path", PEDESTRIAN_ASSETS_PATH)
        print(f"[INFO] dynamic_assets_path: {self.dynamic_assets_path}")

        if not os.path.isdir(self.dynamic_assets_path):
            raise FileNotFoundError(
                f"Pedestrian asset directory does not exist: {self.dynamic_assets_path}. "
                "Set TICVLA_RL_PEDESTRIAN_ASSETS_PATH to a directory containing pedestrian USD assets."
            )

        self.unique_dynamic_asset_path = []
        for asset in sorted(os.listdir(self.dynamic_assets_path)):
            asset_path = os.path.join(self.dynamic_assets_path, asset)
            if os.path.isfile(asset_path) and asset_path.endswith(".usd"):
                self.unique_dynamic_asset_path.append(asset_path)
            elif os.path.isdir(asset_path):
                nested_usd = os.path.join(asset_path, asset.split("_")[-1] + ".usd")
                if os.path.isfile(nested_usd):
                    self.unique_dynamic_asset_path.append(nested_usd)

        self.max_n = getattr(self.cfg, "maximum_dynamic_object_number", len(self.unique_dynamic_asset_path))
        self.unique_dynamic_asset_path = self.unique_dynamic_asset_path[:min(len(self.unique_dynamic_asset_path), self.max_n)]

        if not self.unique_dynamic_asset_path:
            raise FileNotFoundError(f"No pedestrian USD assets found in: {self.dynamic_assets_path}")

        print(f"[INFO] number of unique dynamic assets: {len(self.unique_dynamic_asset_path)}")
        self._build_dynamic_prototypes(self.unique_dynamic_asset_path, self.gltf_scale)

        # Termination statistics tracking
        self._termination_history = []  # List of recent termination reasons
        self._max_history_length = 10  # Keep track of last 10 episodes
        self._termination_counts = {}  # Count of each termination type

        self.extras = {}
        self._last_done_reason = ["boot"] * int(self.num_envs)
        self._initial_distances = torch.zeros(self.num_envs, device=self.device)

        # VLM guidance buffers (initialized lazily)
        self._vlm_has_guidance = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._vlm_origin_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._vlm_origin_yaw_quat_w = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self._vlm_origin_yaw_quat_w[:, 0] = 1.0  # identity yaw
        self._vlm_waypoint_b = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)
        self._vlm_latency_s = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Optional: episode-level natural language instructions tied to start–goal pairs.
        # These correspond to the five unique office start–goal sets used in _random_spawn_robot.
        self._office_instructions: list[str] = [
            "Turn right and follow the blue carpet to the end of the hallway. Enter the room with the reception desk.",
            "Go straight down the hallway, turn left at the end, then continue forward to enter the office room. Move inside and stop in front of the desk on the right.",
            "Go straight ahead, passing through the hallway behind the black glass doors, and then enter the office desk area and stop at the white desk with the plant on it, located on the right side of the wall at the end.",
        ]
        # Per-env index into _office_instructions / episode pair list.
        self._episode_pair_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _ensure_scene_queries_ready(self):
        import omni.physx as physx, omni.usd
        # 1) push authored physics to PhysX
        try:
            physx.get_physx_interface().force_load_physics_from_usd()
        except Exception as e:
            print(f"[WARN] force_load_physics_from_usd failed: {e}")
        # 2) flush USD composition
        try:
            omni.usd.get_context().wait_for_pending_updates()
        except Exception:
            pass
        # 3) tick physics a few frames so scene-query structures are built
        if hasattr(self.sim, "step_physics"):
            for _ in range(3):
                self.sim.step_physics(1.0 / 60.0)

    #========================================== ROBOT PARAMETERS ===========================================
    def _robots_xyz(self, env_ids) -> torch.Tensor:
        """World (x,y,z) of the Go2 root for a batch of envs. Returns [E,3] on self.device."""
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        data = self.scene["robot"].data
        pos = data.root_pos_w[env_ids]  # [E, 3]
        return pos.to(self.device)

    #========================================= PEOPLE PARAMETERS ==========================================
    def _body_xy_to_world_xy(self, body_xy: torch.Tensor, yaw: torch.Tensor):
        c = torch.cos(yaw); s = torch.sin(yaw)
        # [2x2] rot per env: [[c,-s],[s,c]] @ [vx,vy]
        vxw =  c * body_xy[..., 0] - s * body_xy[..., 1]
        vyw =  s * body_xy[..., 0] + c * body_xy[..., 1]
        return torch.stack([vxw, vyw], dim=-1)


    def move_with_slide_xy(self, p0_xy, delta_xy, radius, z, max_step, wall_side=0):
        """Move with sliding along walls."""
        p0 = np.array(p0_xy, np.float32)
        d = np.array(delta_xy, np.float32)
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            return float(p0[0]), float(p0[1])

        dir2 = d / L
        step_len = min(L, max_step)

        # forward
        fwd = dir2 * step_len
        if _is_path_clear(p0, p0 + fwd, radius, z):
            p = p0 + fwd
            return float(p[0]), float(p[1])

        # left, right, back (prefer wall_side if set)
        left = np.array([-dir2[1], dir2[0]], np.float32)
        right = -left
        back = -dir2
        cand = [left, right, back]
        if wall_side in (-1, 1):
            cand = [left if wall_side == 1 else right,
                    right if wall_side == 1 else left,
                    back]

        for v in cand:
            alt = v * step_len
            if _is_path_clear(p0, p0 + alt, radius, z):
                p = p0 + alt
                return float(p[0]), float(p[1])

        # tiny slide along wall tangent as last try
        n = _contact_normal(p0, z)  # 2D
        t = np.array([-n[1], n[0]], np.float32)
        side = float(wall_side) if wall_side in (-1, 1) else 1.0
        slide = t * side * (0.6 * step_len)
        if np.linalg.norm(slide) > 1e-6 and _is_path_clear(p0, p0 + slide, radius, z):
            p = p0 + slide
            return float(p[0]), float(p[1])

        # give up this frame
        return float(p0[0]), float(p0[1])
    
    def _min_wall_clearance(self, x: float, y: float, z: float | None = None,
                            num_dirs: int = 16, max_check: float = 3.0) -> float:
        """Approximate min XY clearance to obstacles around (x,y) at height z.

        Returns the smallest distance (meters) to a wall/obstacle within max_check.
        If no wall is hit in a direction, that direction contributes max_check.
        """
        if z is None:
            # pelvis-ish height is good for indoor obstacles
            z_top = float(getattr(self.cfg, "walkable_probe_top", 1.2))
            depth = float(getattr(self.cfg, "walkable_probe_depth", 3.0))
            person_radius = float(getattr(self.cfg, "person_radius", 0.6))
            z = self._ground_z(x, y, z_top=z_top, depth=depth) + person_radius + 0.05

        sq = get_physx_scene_query_interface()
        origin = (float(x), float(y), float(z))
        min_clear = float("inf")

        # sample evenly around the circle
        for i in range(num_dirs):
            ang = 2.0 * math.pi * (i / num_dirs)
            d = (math.cos(ang), math.sin(ang), 0.0)

            dist_i = max_check  # default if nothing hit
            if hasattr(sq, "raycast_closest"):
                hit = sq.raycast_closest(origin, d, float(max_check))
                if self._sq_hit_truthy(hit):
                    # extract hit position robustly
                    pos = None
                    if isinstance(hit, dict):
                        pos = hit.get("position") or hit.get("point")
                    elif hasattr(hit, "position"):
                        pos = hit.position
                    if pos is not None:
                        try:
                            dx = float(pos[0]) - x
                            dy = float(pos[1]) - y
                        except Exception:
                            # some bindings expose .x/.y
                            dx = float(pos.x) - x
                            dy = float(pos.y) - y
                        dist_i = (dx * dx + dy * dy) ** 0.5

            if dist_i < min_clear:
                min_clear = dist_i

        return float(min_clear)

    def _apply_people_to_stage(self):
        stage = self.sim.stage
        for i, path in enumerate(self._people_paths):
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                prim = UsdGeom.Xform.Define(stage, Sdf.Path(path)).GetPrim()
            xf = UsdGeom.Xformable(prim)
            t, o, s = _get_or_create_tos_ops(xf)

            x = float(self.people_xy[i,0]); y = float(self.people_xy[i,1])
            z = self._ground_z(x, y) + 0.02
            t.Set(Gf.Vec3f(x, y, z))

            yaw = float(self._yaw_face[i].item()) + float(self._yaw_offset)
            half = 0.5 * yaw
            o.Set(Gf.Quatf(math.cos(half), 0.0, 0.0, math.sin(half)))

    def dynamic_asset_animatable_state(self):
        """Import idle/walk animation USDs under /World, set looping timeline."""
        add_reference_to_stage(
            usd_path=getattr(self.cfg, "idle_clip_usd"),
            prim_path="/World/idle"
        )
        add_reference_to_stage(
            usd_path=getattr(self.cfg, "walk_clip_usd"),
            prim_path="/World/walk"
        )

        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_start_time(0)
        timeline.set_end_time(1.1)
        timeline.set_target_framerate(40)
        timeline.set_looping(True)

        # Cache actual SkelAnimation prims
        self._clip_walk = "/World/walk/SMPLX_neutral/root/pelvis0/SMPLX_neutral_Scene"
        print("[CLIPS] walk:", self._clip_walk)

    def _build_dynamic_prototypes(self, usd_paths: list[str], scale=0.01):
        """Build dynamic prototypes for pedestrians."""
        prims_utils.create_prim("/World/DatasetDynamic", "Xform")

        self.dynamic_proto_prim_paths = list()
        for i, usd in enumerate(usd_paths):
            proto_prim_path = f"/World/DatasetDynamic/Human_{i:04d}"
            cfg = UsdFileCfg(usd_path=usd, scale=(1.0, 1.0, 1.0))
            spawn_from_usd(proto_prim_path, cfg)
            proto_prim = self.sim.stage.GetPrimAtPath(proto_prim_path)
            UsdGeom.Imageable(proto_prim).MakeInvisible() # hide the prototype
            prim = self.sim.stage.GetPrimAtPath(proto_prim_path)
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
            for p in Usd.PrimRange(prim):
                if p.IsInstanceable():
                    p.SetInstanceable(False)

            self.dynamic_proto_prim_paths.append(proto_prim_path)

        print(f"[PROTOS] dynamic prototypes: {len(self.dynamic_proto_prim_paths)} at /World/DatasetDynamic")

    def spawn_dynamic_population(
        self,
        num_pedestrian: int,
        positions_xyz=None,   # shape: [N,3] world
        headings_wxyz=None,   # shape: [N,4]
        default_height: float = 1.30,
        default_heading_wxyz=(1.0, 0.0, 0.0, 0.0),
    ):
        """Spawn pedestrians by referencing pre-built prototypes."""
        assert getattr(self, "dynamic_proto_prim_paths", None), \
            "Call _build_dynamic_prototypes(...) first."

        stage = self.sim.stage

        # helpers
        def _to_np(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        pos_np = _to_np(positions_xyz)
        head_np = _to_np(headings_wxyz)

        for i in range(num_pedestrian):
            dst_path  = f"/World/Crowd/Person_{i:04d}"
            proto = random.choice(self.dynamic_proto_prim_paths)

            # --- position
            if pos_np is not None:
                px, py, pz = pos_np[i]
                # if caller only provided XY or z is NaN/None, use default height
                pz = float(pz) if (pz is not None and np.isfinite(pz)) else float(default_height)
            else:
                px, py, pz = 0.0, 0.0, float(default_height)

            # --- heading
            if head_np is not None:
                qw, qx, qy, qz = head_np[i]
            else:
                qw, qx, qy, qz = default_heading_wxyz

            # --- define destination xform and reference the PROTOTYPE prim
            parent_prim = UsdGeom.Xform.Define(stage, Sdf.Path(dst_path)).GetPrim()
            parent_xf = UsdGeom.Xformable(parent_prim)
            parent_xf.ClearXformOpOrder()
            t = parent_xf.AddTranslateOp(); o = parent_xf.AddOrientOp(); s = parent_xf.AddScaleOp()
            t.Set(Gf.Vec3f(float(px), float(py), float(pz)))
            o.Set(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))
            s.Set(Gf.Vec3f(1.0, 1.0, 1.0))

            # --- Body child: holds prototype reference + Y-up→Z-up fix + instance scale
            body_path = dst_path + "/Body"
            body_prim = UsdGeom.Xform.Define(stage, Sdf.Path(body_path)).GetPrim()
            body_xf = UsdGeom.Xformable(body_prim)
            body_xf.ClearXformOpOrder()
            bt = body_xf.AddTranslateOp(); brx = body_xf.AddRotateXOp(); bs = body_xf.AddScaleOp()
            bt.Set(Gf.Vec3f(0.0, 0.0, 1.3))   # <-- USE the lift so feet sit on the floor
            brx.Set(90.0)
            bs.Set(Gf.Vec3f(self.gltf_scale, self.gltf_scale, self.gltf_scale))

            # --- reference the prototype
            refs = body_prim.GetReferences()
            refs.ClearReferences()
            refs.AddReference(Sdf.Reference(primPath=str(proto)))

            # --- force visibility on Body AND all children
            img = UsdGeom.Imageable(body_prim)
            img.MakeVisible()
            img.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)

            for p in Usd.PrimRange(body_prim):
                if UsdGeom.Imageable(p):
                    UsdGeom.Imageable(p).MakeVisible()
                    UsdGeom.Imageable(p).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
                if p.IsInstanceable():
                    p.SetInstanceable(False)

            # --- animation binding (under Body)
            try:
                skel_prim = stage.GetPrimAtPath(body_path + "/root/pelvis0/Skeleton")
                if skel_prim and skel_prim.IsValid() and getattr(self, "_clip_walk", ""):
                    UsdSkel.BindingAPI.Apply(skel_prim)
                    binding = UsdSkel.BindingAPI(skel_prim)
                    rel = binding.CreateAnimationSourceRel()
                    rel.ClearTargets(True)
                    rel.AddTarget(self._clip_walk)
                else:
                    print(f"[WARN] Could not bind clip for {dst_path}: missing skeleton or clip.")
            except Exception as e:
                print(f"[WARN] Anim bind failed for {dst_path}: {e}")

            self._people_paths.append(dst_path)

        print(f"[SPAWN] Spawned {len(self._people_paths)} pedestrians")

    def _index_walkable_bounds(self, root_prim="/World/ground"):
        """Index walkable bounds for the environment."""
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(root_prim)
        if not root or not root.IsValid():
            self._floor_lo = (-5.0, -10.0); self._floor_hi = (25.0, 10.0)
            return

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True)
        lo = [float("inf"), float("inf")]
        hi = [-float("inf"), -float("inf")]

        for p in Usd.PrimRange(root):
            if p.IsA(UsdGeom.Mesh):
                name = p.GetName().lower()
                if "floor" in name or "corridor" in name or "ramp" in name:
                    box = bbox_cache.ComputeWorldBound(p).GetBox()
                    mn, mx = box.GetMin(), box.GetMax()
                    lo[0] = min(lo[0], mn[0]); lo[1] = min(lo[1], mn[1])
                    hi[0] = max(hi[0], mx[0]); hi[1] = max(hi[1], mx[1])

        if not all(np.isfinite(lo)) or not all(np.isfinite(hi)):
            self._floor_lo = (-5.0, -10.0); self._floor_hi = (25.0, 10.0)
        else:
            self._floor_lo = (lo[0], lo[1]); self._floor_hi = (hi[0], hi[1])

    def _make_env_collidable(self, root_prim="/World", approximation="mesh"):
        """Make environment collidable for physics."""
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(root_prim)
        if not root or not root.IsValid():
            print(f"[WARN] No env prim at {root_prim}")
            return

        from pxr import Usd, UsdGeom, UsdPhysics, Sdf, PhysxSchema
        # turn off instancing
        for p in Usd.PrimRange(root):
            if p.IsInstanceable():
                p.SetInstanceable(False)
        
        for p in Usd.PrimRange(root):
            if p.IsA(UsdGeom.Mesh):
                # Always apply collision API (so attributes exist)
                UsdPhysics.CollisionAPI.Apply(p)

                # keep as collidable walkable geometry
                try:
                    phys = PhysxSchema.PhysxCollisionAPI.Apply(p)
                    phys.CreateApproximationAttr().Set(approximation)  # "mesh"/"convexHull"/"convexDecomposition"
                    try:
                        tri = PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(p)
                        tri.CreateDoubleSidedAttr().Set(True)
                    except Exception:
                        pass
                except Exception:
                    pass

                try:
                    UsdGeom.Mesh(p).CreateDoubleSidedAttr(True)
                except Exception:
                    pass
                
    def _build_initial_people_positions_xyz(self) -> torch.Tensor:
        N = self.num_people
        pos = torch.zeros((N, 3), device=self.device, dtype=torch.float32)
        for i in range(N):
            x, y = self._sample_free_xy()
            self.people_xy[i] = torch.tensor([x, y], device=self.device)
            z = self._ground_z(x, y) + 0.02
            pos[i] = torch.tensor([x, y, z], device=self.device)
        return pos

    def _build_initial_people_headings(self) -> torch.Tensor:
        N = self.num_people
        heads = torch.zeros((N, 4), device=self.device, dtype=torch.float32)
        for i in range(N):
            gx, gy = self._sample_free_xy()
            self.people_goal_xy[i] = torch.tensor([gx, gy], device=self.device)
            dx, dy = gx - float(self.people_xy[i,0]), gy - float(self.people_xy[i,1])
            yaw = math.atan2(dy, dx); half = 0.5*yaw
            heads[i] = torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)], device=self.device)
        return heads

    def _rand_xy_box(self, lo=(-5.0, -10.0), hi=(25.0, 10.0)):
        """Generate random XY position within bounds."""
        import random
        return (random.uniform(lo[0], hi[0]), random.uniform(lo[1], hi[1]))
    
    def _sample_free_xy(self, lo=None, hi=None, tries=128):
        """Sample free XY position for pedestrians."""
        # Choose a reasonable indoor center/radius. Defaults tie to robot spawn.
        cx, cy = (float(0), float(0))
        R = float(getattr(self.cfg, "walkable_radius", 20.0))
        person_radius = float(getattr(self.cfg, "person_radius", 0.6))
        required_margin = 1.0 + person_radius   # <-- 1 m from wall (outer body edge)

        for attempt in range(tries):
            r = (random.random() ** 0.5) * R
            ang = random.uniform(0.0, 2.0 * math.pi)
            x = cx + r * math.cos(ang)
            y = cy + r * math.sin(ang)

            # existing checks
            path_clear = _is_path_clear((x, y), (x + 1e-3, y), person_radius, person_radius + 0.05)
            indoor_ok  = self._is_indoor(x, y)

            if not (path_clear and indoor_ok):
                continue

            # NEW: wall clearance check (sample 16 rays out to ~3 m)
            min_clear = self._min_wall_clearance(x, y, num_dirs=16, max_check=3.0)

            # Ensure the **edge** of the body stays 1 m from walls
            if min_clear >= required_margin:
                return x, y

        # fallback: if the indoor sampler fails (e.g., wrong center), return spawn
        return cx, cy
    
    def get_walkable_positions_in_rect_cached(
        self,
        num_positions: int,
        robot_radius: float,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        tries_per_position: int = 200,
    ) -> torch.Tensor:
        """
        Sample N walkable positions inside an axis-aligned rectangle centered at (0,0),
        then translate per-env later. Points are indoor and pass _is_path_clear.
        Cached by (N, radius, rect).
        Returns [N,2] on self.device in WORLD coords around (0,0).
        """
        key = f"rect_{num_positions}_{robot_radius}_{x_range}_{y_range}"
        if not hasattr(self, "_walkable_rect_cache"):
            self._walkable_rect_cache = {}
        if key in self._walkable_rect_cache:
            return self._walkable_rect_cache[key]

        import random, math, numpy as np
        xs, xe = float(x_range[0]), float(x_range[1])
        ys, ye = float(y_range[0]), float(y_range[1])
        person_radius = float(getattr(self.cfg, "person_radius", 0.6))
        z_top = float(getattr(self.cfg, "walkable_probe_top", 1.2))
        depth = float(getattr(self.cfg, "walkable_probe_depth", 3.0))

        pts = []
        for _ in range(num_positions):
            ok = False
            for _try in range(tries_per_position):
                x = random.uniform(xs, xe)
                y = random.uniform(ys, ye)
                # indoor + clearance checks (reusing your existing utilities)
                if self._is_indoor(x, y) and _is_path_clear((x, y), (x + 1e-3, y), robot_radius, robot_radius + 0.05):
                    # optional: keep 1m-from-wall margin using your _min_wall_clearance
                    min_clear = self._min_wall_clearance(x, y, num_dirs=16, max_check=3.0)
                    if min_clear >= 1.0 + person_radius:
                        pts.append([x, y])
                        ok = True
                        break
            if not ok:
                # fallback: center
                pts.append([0.0, 0.0])

        out = torch.tensor(pts, device=self.device, dtype=torch.float32)
        self._walkable_rect_cache[key] = out
        return out

    
    def _is_indoor(self, x, y):
        """Check if position is indoors."""
        # 1) find the floor height below (use config values)
        z_top = float(getattr(self.cfg, "walkable_probe_top", 1.2))
        depth = float(getattr(self.cfg, "walkable_probe_depth", 3.0))
        z_floor = self._ground_z(x, y, z_top=z_top, depth=depth)
        if not np.isfinite(z_floor):
            return False
        # 2) find a ceiling within ~2–5 m above that floor
        dh = self._up_hit_distance(x, y, z_floor + 0.2, max_up=40.0)
        is_indoor = (dh is not None) and (0.8 <= dh <= 6.0)  # ceiling exists not too far away
        
        # if not is_indoor:
        #     print(f"[DEBUG] _is_indoor({x:.2f}, {y:.2f}, z_floor={z_floor:.2f}): Not indoor - ground Z={z_floor:.2f}, ceiling distance={dh}")
        
        return is_indoor

    def _ground_z(self, x, y, z_top=None, depth=None):
        """Get ground Z coordinate at given XY position."""
        if z_top is None:
            z_top = float(getattr(self.cfg, "walkable_probe_top", 1.2))
        if depth is None:
            depth = float(getattr(self.cfg, "walkable_probe_depth", 3.0))
        sq = get_physx_scene_query_interface()
        origin = (float(x), float(y), float(z_top))
        down = (0.0, 0.0, -1.0)
        dist = float(depth)

        # 1) Prefer 'closest' because it returns a hit struct/dict
        if hasattr(sq, "raycast_closest"):
            ret = sq.raycast_closest(origin, down, dist)
            if self._sq_hit_truthy(ret):
                pos = None
                if isinstance(ret, dict):
                    pos = ret.get("position") or ret.get("point")
                elif hasattr(ret, "position"):
                    pos = ret.position
                if pos is not None:
                    try:
                        return float(pos[2])
                    except Exception:
                        return float(pos.z)

        # 2) Fallback: 'all' with callback — pick the highest hit below z_top
        if hasattr(sq, "raycast_all"):
            best = None  # highest z <= z_top
            def _cb(hit):
                nonlocal best
                try:
                    p = hit.position
                    hz = float(p[2] if hasattr(p, "__getitem__") else p.z)
                except Exception:
                    return True
                if hz <= z_top + 1e-3 and (best is None or hz > best):
                    best = hz
                return True  # continue iterating
            sq.raycast_all(origin, down, dist, _cb, False)
            if best is not None:
                return best

        # 3) Last resort
        return 0.0

    def _up_hit_distance(self, x, y, z0, max_up=5.0):
        """Distance to the first collider above (x,y,z0). Returns None if no ceiling is found."""
        sq = get_physx_scene_query_interface()
        if max_up is None or max_up <= 0:
            return None

        # Nudge origin upward so we don't self-hit the floor surface we started from.
        origin = (float(x), float(y), float(z0) + 1e-3)
        up = (0.0, 0.0, 1.0)
        dist = float(max_up)

        # --- 1) Try 'closest' and accept either .distance/.t or position
        if hasattr(sq, "raycast_closest"):
            ret = sq.raycast_closest(origin, up, dist)
            if self._sq_hit_truthy(ret):
                # Prefer an explicit distance/t if present
                d = None
                try:
                    if isinstance(ret, dict):
                        d = ret.get("distance") or ret.get("t")
                    else:
                        d = getattr(ret, "distance", None) or getattr(ret, "t", None)
                except Exception:
                    d = None
                if d is not None:
                    try:
                        d = float(d)
                        if 0.0 < d <= dist:
                            return d
                    except Exception:
                        pass

                # Fall back to computing from hit position
                pos = None
                if isinstance(ret, dict):
                    pos = ret.get("position") or ret.get("point")
                elif hasattr(ret, "position"):
                    pos = ret.position
                if pos is not None:
                    try:
                        hz = float(pos[2] if hasattr(pos, "__getitem__") else pos.z)
                        d = hz - float(z0)
                        if 0.0 < d <= dist:
                            return d
                    except Exception:
                        pass
            # IMPORTANT: do NOT return here; fall through to raycast_all as a second attempt

        # --- 2) Fallback: 'all' -> choose the smallest positive t
        if hasattr(sq, "raycast_all"):
            best_t = None
            def _cb(hit):
                nonlocal best_t
                try:
                    p = hit.position
                    hz = float(p[2] if hasattr(p, "__getitem__") else p.z)
                except Exception:
                    return True  # keep iterating
                t = hz - float(z0)
                if 0.0 < t <= dist and (best_t is None or t < best_t):
                    best_t = t
                return True
            sq.raycast_all(origin, up, dist, _cb, False)
            if best_t is not None:
                return best_t

        # --- 3) Optional: auto-expand search height from cfg if provided
        extra = float(getattr(self.cfg, "ceiling_probe_height", 0.0))
        if extra > dist:
            return self._up_hit_distance(x, y, z0, max_up=extra)

        return None

    def _sq_hit_truthy(self, ret):
        """Try to convert various return types to a boolean 'any hit?'."""
        try:
            # dict-like
            if isinstance(ret, dict):
                if "hit" in ret:
                    return bool(ret["hit"])
                if "count" in ret:
                    return ret["count"] > 0
            # sequence-like (list of hits)
            if hasattr(ret, "__len__"):
                return len(ret) > 0
            # object with 'has_hits' or 'hit' or 'count'
            for attr in ("has_hits", "hit", "count"):
                if hasattr(ret, attr):
                    v = getattr(ret, attr)
                    return bool(v() if callable(v) else v)
            # fall back to truthiness
            return bool(ret)
        except Exception:
            return False

        
    def _robot_yaw(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Yaw (rad) for the robot(s) in env_ids. Returns [E] on self.device."""
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Access articulation data across IsaacLab versions
        try:
            data = self.scene["robot"].data
        except Exception:
            data = self.scene.articulations["robot"].data

        # Prefer explicit quaternion if available; otherwise slice from root_state_w
        if hasattr(data, "root_quat_w"):
            q = data.root_quat_w[env_ids]           # [E,4] (w,x,y,z)
        else:
            q = data.root_state_w[env_ids, 3:7]     # [E,4] (w,x,y,z)

        q = q.to(self.device)
        w, x, y, z = q.unbind(-1)
        # Standard ZYX yaw from (w,x,y,z)
        yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return yaw  # [E]

    def _random_spawn_robot(self, env_ids: torch.Tensor):
        """Randomly choose one of the predefined start–goal pairs for each episode.

        For every reset, we:
          - Sample a start pose (x, y, yaw) from a fixed list of office episodes.
          - Spawn the robot at that start.
          - Set the navigation goal to the corresponding fixed goal (x, y).

        This ensures we *only* use the authored start/goal pairs (no random goal points),
        while still randomizing which pair is used each episode.
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        robot = self.scene["robot"]
        E = len(env_ids)
        ids_t = env_ids

        # ------------------------------------------------------------------
        # Authored office start/goal pairs (compressed to unique pairs)
        # ------------------------------------------------------------------
        # Each entry: (start_x, start_y, start_z), start_yaw_deg, (goal_x, goal_y, goal_z)
        episode_starts = torch.tensor(
            [
                [-8.2, 52.2, 0.01],   
                [-3.04, 48.1, 0.01],  
                [-15.2, 12.7, 0.01],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        episode_yaws_deg = torch.tensor(
            [
                3.14,  
                -270.0,   
                90.0,     
            ],
            device=self.device,
            dtype=torch.float32,
        )
        episode_goals = torch.tensor(
            [
                [-3.00, 10.87, 0.01],  
                [-15.5, 54.7, 0.01],
                [-12.5, 49, 0.01],   
            ],
            device=self.device,
            dtype=torch.float32,
        )

        # Sample an authored pair index for each environment
        num_pairs = episode_starts.shape[0]
        idx = torch.randint(0, num_pairs, (E,), device=self.device)
        # Remember which pair was chosen so we can optionally retrieve its instruction.
        self._episode_pair_idx[ids_t] = idx

        spawn_positions = episode_starts[idx]          # [E,3]
        spawn_yaws = episode_yaws_deg[idx] * math.pi / 180.0  # [E] -> radians
        goal_positions = episode_goals[idx]            # [E,3]

        # Build positions with correct z height (use ground z + robot height offset)
        spawn_positions_3d = spawn_positions.clone()

        # Get ground height for each spawn position
        for i, env_id in enumerate(env_ids.tolist()):
            ground_z = self._ground_z(float(spawn_positions[i, 0]), float(spawn_positions[i, 1]))
            # Use default root state z offset (0.3 from config)
            spawn_positions_3d[i, 2] = ground_z + 0.3

        # Build quaternions from yaw
        from isaaclab.utils.math import quat_from_euler_xyz

        quaternions = quat_from_euler_xyz(
            torch.zeros(E, device=self.device),  # roll
            torch.zeros(E, device=self.device),  # pitch
            spawn_yaws,  # yaw
        )  # [E, 4]

        # Set robot positions and orientations
        root_states = torch.cat([spawn_positions_3d, quaternions], dim=-1)  # [E, 7]
        robot.write_root_pose_to_sim(root_states, env_ids=ids_t)

        # Reset velocities to zero
        velocities = torch.zeros((E, 6), device=self.device)
        robot.write_root_velocity_to_sim(velocities, env_ids=ids_t)

        # ------------------------------------------------------------------
        # Set deterministic goals for the selected start–goal pairs
        # ------------------------------------------------------------------
        if hasattr(self, "command_manager") and self.command_manager is not None:
            try:
                command_term = self.command_manager.get_term("pose_command")
                env_ids_list = env_ids.tolist()

                # World-frame goal position for each env
                for i, env_id in enumerate(env_ids_list):
                    gx, gy, gz = goal_positions[i]
                    command_term.pos_command_w[env_id, 0] = gx
                    command_term.pos_command_w[env_id, 1] = gy
                    # Use robot's default root height for z to keep things stable
                    default_z = float(robot.data.default_root_state[env_id, 2])
                    command_term.pos_command_w[env_id, 2] = default_z if not math.isnan(default_z) else gz

                # Align heading towards the goal if simple_heading is enabled
                if getattr(command_term.cfg, "simple_heading", False):
                    # Compute target direction in world frame for all envs
                    target_vec = command_term.pos_command_w[:, :2] - robot.data.root_pos_w[:, :2]
                    target_dir = torch.atan2(target_vec[:, 1], target_vec[:, 0])
                    command_term.heading_command_w[:] = target_dir

                # Refresh body-frame command and metrics to stay consistent
                if hasattr(command_term, "_update_command"):
                    command_term._update_command()
                if hasattr(command_term, "_update_metrics"):
                    command_term._update_metrics()
            except (AttributeError, KeyError):
                # If anything fails here, fall back to default goal behavior next step
                pass

    def get_current_instruction(self, env_id: int = 0) -> str | None:
        """Return the natural-language instruction for the current episode of the given env.

        This is optional: trainers/scripts can ignore it and supply their own instruction.
        """
        if not hasattr(self, "_office_instructions") or not hasattr(self, "_episode_pair_idx"):
            return None
        if env_id < 0 or env_id >= int(self.num_envs):
            return None
        pair_idx = int(self._episode_pair_idx[env_id].item())
        if pair_idx < 0 or pair_idx >= len(self._office_instructions):
            return None
        return self._office_instructions[pair_idx]

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environment indices."""
        
        super()._reset_idx(env_ids)
            
        # Spawn the global crowd once
        if not self._crowd_spawned:
            # >>> ensure PhysX raycasts will return hits
            self._ensure_scene_queries_ready()
            
            pos = self._build_initial_people_positions_xyz()  # (N,3)
            heads = self._build_initial_people_headings()     # (N,4)
            self.people_speed.uniform_(0.8, 1.4)
            self.spawn_dynamic_population(num_pedestrian=self.num_people, positions_xyz=pos, headings_wxyz=heads)

            self._prev_people_xy = self.people_xy.clone().detach()
            self._yaw_face.zero_()
    
            self._apply_people_to_stage()
            self._crowd_spawned = True

        # Randomly spawn robot at walkable positions for each episode
        if len(env_ids) > 0:
            self._random_spawn_robot(env_ids)

        # Keep a copy of the natural-language instruction for each env so that
        # trainers (e.g., TICVLA) can refresh prompts immediately after a reset.
        try:
            self.extras.setdefault("instruction", [""] * int(self.num_envs))
            for eid in env_ids.tolist():
                inst = self.get_current_instruction(int(eid))
                if inst:
                    self.extras["instruction"][int(eid)] = inst
        except Exception:
            # Logging not required here; best-effort cache for training scripts
            pass

        # Reset actions if available
        if hasattr(self, "action") and getattr(self, "action", None) is not None:
            getattr(self, "action")[env_ids] = 0.0
        

    def step(self, actions: torch.Tensor):
        # Capture episode lengths BEFORE reset happens
        pre_reset_episode_lengths = self.episode_length_buf.clone()
        obs, rewards, terminated, truncated, info = super().step(actions)

        # Update goal visualization markers AFTER command manager has potentially resampled
        # This ensures we show the current (possibly new) goal positions
        self._update_goal_markers()

        done = (terminated | truncated)
        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1).tolist()
        if not isinstance(done_ids, list):
            done_ids = [int(done_ids)] if bool(done) else []

        # Only show termination info if there are actually terminated environments
        if done_ids:
            # collect per-term flags once
            term_flags = self._collect_term_flags()

            # classify and store
            reasons = []
            for eid in done_ids:
                r = self._classify_reason_for_env(eid, terminated, truncated, info, term_flags)
                self._last_done_reason[eid] = r
                reasons.append(r)
                
                # Track termination statistics with episode number
                episode_num = int(pre_reset_episode_lengths[eid].item())
                self._track_termination_statistics(r, episode_num)

            # Log termination statistics after processing all terminations
            self._log_termination_percentages()
            
            # optional: attach to info so trainers/loggers can pick it up
            info.setdefault("extras", {})
            info["extras"]["done_env_ids"] = done_ids
            info["extras"]["done_reasons"] = reasons
            # Surface the freshly sampled instructions for envs that were reset.
            instruction_map = {}
            for env_id in done_ids:
                inst = self.get_current_instruction(int(env_id))
                if inst:
                    instruction_map[int(env_id)] = inst
            if instruction_map:
                info["extras"]["instruction"] = instruction_map

        return obs, rewards, terminated, truncated, info
    
    def _collect_term_flags(self) -> dict[str, torch.Tensor]:
        """Return {term_name: [E] bool tensor} for all active termination terms."""
        tmgr = getattr(self, "termination_manager", None)
        if tmgr is None:
            return {}

        # figure out names (works across isaaclab versions)
        names = []
        for attr in ("active_terms", "terms", "_active_terms", "_terms"):
            obj = getattr(tmgr, attr, None)
            if isinstance(obj, dict):
                names = list(obj.keys()); break
            if isinstance(obj, (list, tuple)):
                names = list(obj); break

        out = {}
        for name in names:
            term = tmgr.get_term(name)
            
            # If the term is already a tensor, use it directly
            if torch.is_tensor(term):
                if term.shape[0] == self.num_envs:
                    out[name] = term.to(self.device)
                elif term.numel() == self.num_envs:
                    # Reshape if needed
                    reshaped = term.reshape(self.num_envs).to(self.device)
                    out[name] = reshaped
            else:
                # Try to get the tensor from the term object
                cand = None
                for attr in ("terminated", "_terminated", "terminated_buf", "_terminated_buf", "is_terminated", "value", "_value"):
                    v = getattr(term, attr, None)
                    if v is not None:
                        if callable(v):
                            try:
                                v = v()
                            except Exception:
                                v = None
                        if torch.is_tensor(v) and v.shape[0] == self.num_envs:
                            cand = v.to(self.device)
                            break
                        elif torch.is_tensor(v) and v.numel() == self.num_envs:
                            cand = v.reshape(self.num_envs).to(self.device)
                            break
                
                if cand is not None:
                    out[name] = cand
        return out


    def _classify_reason_for_env(
        self,
        eid: int,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        info: dict,
        term_flags: dict[str, torch.Tensor],
    ) -> str:
        """Return a single string reason for env `eid` (with reasonable priority)."""
        
        # 1) explicit runner timeout
        tmo = info.get("time_out", None)
        if tmo is not None and bool(tmo[eid]):
            return "time_out"

        # 3) any named termination terms from termination_manager
        hits = [name for name, flag in term_flags.items() if bool(flag[eid])]
        if hits:
            return "+".join(hits)

        # 4) fallback on the gym-style flags
        if bool(truncated[eid]):
            return "truncated"
        if bool(terminated[eid]):
            return "terminated"

        return "unknown"

    def _track_termination_statistics(self, reason: str, episode_num: int = None):
        """Track termination statistics for recent episodes."""
        # Add to history
        self._termination_history.append(reason)
        
        # Keep only the last max_history_length episodes
        if len(self._termination_history) > self._max_history_length:
            self._termination_history.pop(0)
        
        # Update counts
        if reason not in self._termination_counts:
            self._termination_counts[reason] = 0
        self._termination_counts[reason] += 1

    def _log_termination_percentages(self):
        """Track termination statistics for recent episodes (no console output)."""
        return

    def get_termination_statistics(self):
        """Get current termination statistics for external access."""
        if len(self._termination_history) == 0:
            return {"recent_percentages": {}, "overall_counts": {}}
        
        # Calculate recent percentages
        recent_counts = {}
        for reason in self._termination_history:
            recent_counts[reason] = recent_counts.get(reason, 0) + 1
        
        total_episodes = len(self._termination_history)
        recent_percentages = {}
        for reason, count in recent_counts.items():
            recent_percentages[reason] = (count / total_episodes) * 100.0
        
        return {
            "recent_percentages": recent_percentages,
            "overall_counts": self._termination_counts.copy(),
            "total_episodes_tracked": total_episodes,
            "total_terminations": sum(self._termination_counts.values())
        }

    def _step_people(self, dt: float):
        """Step pedestrian simulation."""
        xy = self.people_xy      # [N,2]
        speed = self.people_speed   # [N]
        N = self.num_people

        # persistent heading + tiny drift
        heading = self._heading2d   # [N,2]
        drift = 0.05 * torch.randn_like(heading)
        base_h = heading + drift

        # separation
        delta = xy.unsqueeze(1) - xy.unsqueeze(0)             # [N,N,2]
        dmat = torch.linalg.norm(delta + 1e-6, dim=-1)       # [N,N]
        person_radius = float(getattr(self.cfg, "person_radius", 0.6))
        R = person_radius * 5.0 + 1  # desired center-to-center spacing
        mask = (dmat > 1e-6) & (dmat < R)
        dir_away = delta / (dmat.unsqueeze(-1) + 1e-6)
        strength = ((R - dmat).clamp(min=0.0) / R) * mask
        repel = (strength.unsqueeze(-1) * dir_away).sum(dim=1)  # [N,2]

        # blend & normalize
        blended = base_h + self._avoid_gain * repel
        blended = blended / (torch.linalg.norm(blended, dim=-1, keepdim=True) + 1e-6)

        step = (speed.unsqueeze(-1) * dt) * blended           # [N,2]

        new_xy = xy.clone()
        for i in range(N):
            p0 = (float(xy[i,0]), float(xy[i,1]))
            d = (float(step[i,0]), float(step[i,1]))
            z_top = float(getattr(self.cfg, "walkable_probe_top", 1.2))
            depth = float(getattr(self.cfg, "walkable_probe_depth", 3.0))
            person_radius = float(getattr(self.cfg, "person_radius", 0.6))
            max_step_xy = 0.5 * person_radius  # anti-tunneling
            zq = self._ground_z(p0[0], p0[1],
                                z_top=z_top,
                                depth=depth) + person_radius + 0.02
            side = int(self._wall_side[i].item())
            nx, ny = self.move_with_slide_xy(p0, d, person_radius, zq, max_step_xy, wall_side=side)

            # wall-side hysteresis
            near_wall = not _is_path_clear((nx, ny), (nx + d[0]*0.2, ny + d[1]*0.2), person_radius, zq)
            if near_wall and self._wall_side[i] == 0:
                g = np.array([d[0], d[1]], np.float32)
                n = _contact_normal((nx, ny), zq)
                t = np.array([-n[1], n[0]], np.float32)
                self._wall_side[i] = 1 if (g @ t) >= 0 else -1
            if not near_wall:
                self._wall_side[i] = 0

            # update heading memory to what actually happened (prevents oscillation)
            new_xy[i, 0], new_xy[i, 1] = nx, ny

            # now compute actual move and update heading memory
            mv_t = new_xy[i] - xy[i]                      # torch, same device
            mL = torch.linalg.norm(mv_t)
            if mL > 1e-5:
                self._heading2d[i] = mv_t / (mL + 1e-9)
            else:
                rnd = torch.randn(2, device=self.device, dtype=self._heading2d.dtype)
                rnd = rnd / (torch.linalg.norm(rnd) + 1e-6)
                self._heading2d[i] = 0.7 * self._heading2d[i] + 0.3 * rnd

            new_xy[i,0], new_xy[i,1] = nx, ny

        # animation facing
        vel = new_xy - self._prev_people_xy
        spd = torch.linalg.norm(vel, dim=-1)
        moving = spd > 1e-3
        new_yaw = torch.atan2(vel[:,1], vel[:,0])
        self._yaw_face[moving] = new_yaw[moving]

        self._prev_people_xy = new_xy.detach().clone()
        self.people_xy = new_xy

    def _setup_goal_markers(self):
        """Set up goal visualization markers."""
        # Create goal marker configuration
        goal_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/GoalMarkers",
            markers={
                "goal": SphereCfg(
                    radius=0.2,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red sphere
                ),
            }
        )
        
        # Initialize goal markers for each environment
        self.goal_markers = VisualizationMarkers(goal_marker_cfg)

    def _update_goal_markers(self):
        if self.goal_markers is None:
            return

        command_term = self.command_manager.get_term("pose_command")
        world_goal_positions = command_term.pos_command_w[:, :2]  # [E,2], absolute

        # change-detect (optional)
        if not hasattr(self, '_prev_world_goal_positions'):
            self._prev_world_goal_positions = world_goal_positions.clone()
        elif not torch.allclose(world_goal_positions, self._prev_world_goal_positions, atol=1e-6):
            self._prev_world_goal_positions = world_goal_positions.clone()

        # add z and draw
        goal_positions_3d = torch.zeros(self.num_envs, 3, device=self.device)
        goal_positions_3d[:, :2] = world_goal_positions
        goal_positions_3d[:, 2] = 0.5
        self.goal_markers.visualize(translations=goal_positions_3d)

    def set_vlm_guidance(self, waypoint_b: torch.Tensor, latency_s: torch.Tensor, env_ids: torch.Tensor | None = None, waypoint_frame: str = "body"):
        """
        Set VLM guidance for reward shaping.
        
        Args:
            waypoint_b: [E,2] or [E,9] (3 waypoints: (x3,y3,x6,y6,x9,y9) per env) or single [2]/[9].
            latency_s: [E] or [] latency in seconds between observation and guidance availability.
            env_ids: Optional env indices to set; defaults to all.
            waypoint_frame: "body" if provided in robot body frame at guidance time; "world" if already global.
        """
        import torch as _torch
        from isaaclab.utils.math import yaw_quat as _yaw_quat, quat_apply as _quat_apply
        # Normalize inputs to batched tensors
        if env_ids is None:
            env_ids = _torch.arange(self.num_envs, device=self.device, dtype=_torch.long)
        elif not isinstance(env_ids, _torch.Tensor):
            env_ids = _torch.as_tensor(env_ids, device=self.device, dtype=_torch.long)
        ids = env_ids

        if waypoint_b.dim() == 1:
            waypoint_b = waypoint_b.unsqueeze(0)
        # Broadcast singleton row to ids
        if waypoint_b.shape[0] == 1 and ids.numel() > 1:
            waypoint_b = waypoint_b.repeat(ids.numel(), 1)
        waypoint_b = waypoint_b.to(device=self.device, dtype=_torch.float32)

        if latency_s.dim() == 0:
            latency_s = latency_s.unsqueeze(0)
        if latency_s.shape[0] == 1 and ids.numel() > 1:
            latency_s = latency_s.repeat(ids.numel())
        latency_s = latency_s.to(device=self.device, dtype=_torch.float32)

        # Select (x,y) from 3s/6s/9s bins depending on latency
        # waypoint_b shape can be [...,2] or [...,9] laid out as (x3,y3,x6,y6,x9,y9)
        if waypoint_b.shape[-1] >= 9:
            w3 = waypoint_b[:, 0:2]
            w6 = waypoint_b[:, 2:4]
            w9 = waypoint_b[:, 4:6]
            # Map latency to nearest bin among 3,6,9 seconds
            bins = _torch.tensor([3.0, 6.0, 9.0], device=self.device, dtype=_torch.float32)
            lat = latency_s.clamp_min(0.0)
            dists = _torch.stack([_torch.abs(lat - b) for b in bins], dim=1)
            idx = _torch.argmin(dists, dim=1)  # [B]
            chosen = _torch.zeros(ids.numel(), 2, device=self.device, dtype=_torch.float32)
            mask3 = idx == 0
            mask6 = idx == 1
            mask9 = idx == 2
            if mask3.any():
                chosen[mask3] = w3[mask3]
            if mask6.any():
                chosen[mask6] = w6[mask6]
            if mask9.any():
                chosen[mask9] = w9[mask9]
            sel_xy = chosen
        else:
            # Already a single (x,y)
            sel_xy = waypoint_b[:, :2]

        # Lazily create world waypoint buffer if absent
        if not hasattr(self, "_vlm_waypoint_w") or self._vlm_waypoint_w is None:
            self._vlm_waypoint_w = _torch.zeros(self.num_envs, 2, dtype=_torch.float32, device=self.device)

        # Only recompute world-frame waypoint if the selected body-frame waypoint changed
        # Compare against stored body-frame waypoint buffer
        if not hasattr(self, "_vlm_waypoint_b") or self._vlm_waypoint_b is None:
            self._vlm_waypoint_b = _torch.zeros(self.num_envs, 2, dtype=_torch.float32, device=self.device)

        prev_b = self._vlm_waypoint_b[ids]
        diff = (sel_xy - prev_b).abs()
        changed_mask = diff.max(dim=1).values > 1e-5

        if changed_mask.any():
            ch_ids = ids[changed_mask]
            # Capture origin pose at guidance arrival ONLY for changed envs
            robot = self.scene["robot"]
            self._vlm_origin_pos_w[ch_ids] = robot.data.root_state_w[ch_ids, 0:3]
            self._vlm_origin_yaw_quat_w[ch_ids] = _yaw_quat(robot.data.root_quat_w[ch_ids])

            if waypoint_frame == "body":
                # Promote to 3D and rotate by yaw into world, add origin position
                sel_xy3 = _torch.zeros(ch_ids.numel(), 3, device=self.device, dtype=_torch.float32)
                sel_xy3[:, :2] = sel_xy[changed_mask]
                wpt_w = _quat_apply(self._vlm_origin_yaw_quat_w[ch_ids], sel_xy3) + self._vlm_origin_pos_w[ch_ids]
                self._vlm_waypoint_w[ch_ids] = wpt_w[:, :2]
            else:
                # Already in world coordinates
                self._vlm_waypoint_w[ch_ids] = sel_xy[changed_mask]

            # Update stored body-frame waypoint for changed envs
            self._vlm_waypoint_b[ch_ids] = sel_xy[changed_mask]

        # Always refresh latency and has-guidance flag
        self._vlm_latency_s[ids] = latency_s
        self._vlm_has_guidance[ids] = True

        # handle invalid waypoints
        invalid_mask = _torch.isclose(sel_xy, _torch.tensor(-100.0, device=self.device, dtype=_torch.float32), atol=1e-4)
        if invalid_mask.any():
            self._vlm_has_guidance[ids] = False

        # Expose in extras for logging/inspection
        self.extras.setdefault("vlm", {})
        self.extras["vlm"]["latency_s_mean"] = float(self._vlm_latency_s[ids].mean().item())
        self.extras["vlm"]["waypoint_w_norm_mean"] = float(self._vlm_waypoint_w[ids].norm(dim=-1).mean().item())