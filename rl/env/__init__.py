# rl/env/__init__.py
# Auto-register Gym env IDs when this package is imported.
try:
    from . import register_envs as _register_envs  # side-effect: registers
except Exception as e:
    import warnings
    warnings.warn(f"TICVLA env registration skipped: {e}")
