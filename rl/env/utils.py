"""
Utility functions for mobile navigation environment.
"""
import math
import numpy as np
from omni.physx import get_physx_scene_query_interface


def _sq_has(obj, name):
    """Check if object has attribute."""
    return hasattr(obj, name)


def _sq_hit_truthy(ret):
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


def _sweep_sphere_any(radius, p0, dir3, dist):
    """Sphere sweep with robust fallbacks."""
    sq = get_physx_scene_query_interface()
    # Prefer 'any' or 'closest', else 'all', else old 'sweep_sphere'
    if _sq_has(sq, "sweep_sphere_any"):
        return _sq_hit_truthy(sq.sweep_sphere_any(radius, tuple(p0), tuple(dir3), float(dist)))
    if _sq_has(sq, "sweep_sphere_closest"):
        return _sq_hit_truthy(sq.sweep_sphere_closest(radius, tuple(p0), tuple(dir3), float(dist)))
    if _sq_has(sq, "sweep_sphere_all"):
        return _sq_hit_truthy(sq.sweep_sphere_all(radius, tuple(p0), tuple(dir3), float(dist)))
    if _sq_has(sq, "sweep_sphere"):
        return _sq_hit_truthy(sq.sweep_sphere(radius, tuple(p0), tuple(dir3), float(dist)))
    return False  # no sweep available


def _raycast_any(p0, dir3, dist):
    """Raycast with robust fallbacks."""
    sq = get_physx_scene_query_interface()
    # Prefer 'any', then 'closest', then 'all', then legacy
    if _sq_has(sq, "raycast_any"):
        return _sq_hit_truthy(sq.raycast_any(tuple(p0), tuple(dir3), float(dist)))
    if _sq_has(sq, "raycast_closest"):
        return _sq_hit_truthy(sq.raycast_closest(tuple(p0), tuple(dir3), float(dist)))
    if _sq_has(sq, "raycast_all"):
        return _sq_hit_truthy(sq.raycast_all(tuple(p0), tuple(dir3), float(dist)))
    if _sq_has(sq, "raycast"):
        return _sq_hit_truthy(sq.raycast(tuple(p0), tuple(dir3), float(dist)))
    return False


def _is_path_clear(p0_xy, p1_xy, radius, z) -> bool:
    """Sphere-sweep between p0 and p1 at height z, with robust fallbacks."""
    p0 = np.array([p0_xy[0], p0_xy[1], z], dtype=np.float32)
    d = np.array([p1_xy[0] - p0_xy[0], p1_xy[1] - p0_xy[1], 0.0], dtype=np.float32)
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return True
    dir3 = d / L

    # 1) try a real sphere sweep if available
    try:
        if _sweep_sphere_any(radius, p0, dir3, L):
            return False
        return True
    except Exception:
        pass

    # 2) fallback: 3-ray "thick" raycast if raycast exists
    try:
        right = np.array([-dir3[1], dir3[0], 0.0], np.float32) * radius * 0.8
        for off in (np.zeros(3, np.float32), right, -right):
            if _raycast_any(p0 + off, dir3, L):
                return False
        return True
    except Exception:
        pass

    # 3) last resort: discrete overlap samples along the path
    sq = get_physx_scene_query_interface()
    # find some overlap function
    overlap_fn = None
    for nm in ("overlap_sphere_any", "overlap_sphere_all", "overlap_sphere"):
        if hasattr(sq, nm):
            overlap_fn = getattr(sq, nm)
            break
    n = max(2, int(np.ceil(L / max(radius * 0.5, 1e-3))))
    for i in range(1, n + 1):
        pi = p0 + dir3 * (L * (i / n))
        if overlap_fn is not None:
            if _sq_hit_truthy(overlap_fn(float(radius), tuple(pi))):
                return False
    return True


def _contact_normal(p_xy, z, probe=0.8, rays=16):
    """Sample around the agent and return the *surface normal* of the closest hit."""
    sq = get_physx_scene_query_interface()
    origin = (float(p_xy[0]), float(p_xy[1]), float(z))
    best_t = float("inf")
    best_n = None

    # Prefer closest so we can read normal & distance
    for k in range(rays):
        a = 2.0 * math.pi * (k / rays)
        dir3 = (math.cos(a), math.sin(a), 0.0)
        hit = None
        if hasattr(sq, "raycast_closest"):
            hit = sq.raycast_closest(origin, dir3, float(probe))
        elif hasattr(sq, "raycast_all"):
            # fallback: approximate "closest" with raycast_all
            nearest = {"distance": float("inf"), "normal": None}
            def _cb(h):
                try:
                    d = float(getattr(h, "distance", None) or h.get("distance", None) or 0.0)
                except Exception:
                    d = 0.0
                if 0.0 <= d < nearest["distance"]:
                    nearest["distance"] = d
                    n = getattr(h, "normal", None) if hasattr(h, "normal") else h.get("normal", None)
                    nearest["normal"] = n
                return True
            sq.raycast_all(origin, dir3, float(probe), _cb, False)
            hit = nearest

        if hit:
            # read distance
            dist = None
            if isinstance(hit, dict):
                dist = hit.get("distance")
                n = hit.get("normal")
            else:
                dist = getattr(hit, "distance", None)
                n = getattr(hit, "normal", None)

            if dist is not None and n is not None and dist < best_t:
                best_t = float(dist)
                best_n = n

    if best_n is None:
        return np.array([0.0, 0.0], np.float32)

    nx, ny = (float(best_n[0]), float(best_n[1]))
    v = np.array([nx, ny], np.float32)
    nrm = np.linalg.norm(v)
    return v / (nrm + 1e-9)


def try_move_xy(p0_xy, delta_xy, radius=0.30, z=0.42):
    """Try to move from p0_xy by delta_xy, with collision detection."""
    p0 = np.array([float(p0_xy[0]), float(p0_xy[1]), float(z)], np.float32)
    d = np.array([float(delta_xy[0]), float(delta_xy[1]), 0.0], np.float32)
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return float(p0_xy[0]), float(p0_xy[1])
    dir3 = d / L
    # use robust wrappers
    blocked = _sweep_sphere_any(radius, p0, dir3, L) or _raycast_any(p0, dir3, L)
    return (float(p0[0]), float(p0[1])) if blocked else (float(p0[0] + d[0]), float(p0[1] + d[1]))


def _get_or_create_tos_ops(xformable):
    """Get or create translate/orient/scale operations for USD transforms."""
    from pxr import UsdGeom, Gf
    
    ops = list(xformable.GetOrderedXformOps())
    def _find(typ): 
        return next((o for o in ops if o.GetOpType() == typ), None)

    t = _find(UsdGeom.XformOp.TypeTranslate) or xformable.AddTranslateOp()
    o = _find(UsdGeom.XformOp.TypeOrient) or xformable.AddOrientOp()
    s = _find(UsdGeom.XformOp.TypeScale) or xformable.AddScaleOp()

    xformable.SetXformOpOrder([t, o, s])
    if o.Get() is None:
        o.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return t, o, s