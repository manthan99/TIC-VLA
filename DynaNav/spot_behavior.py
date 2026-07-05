"""
Standalone keyboard teleoperation for Boston Dynamics Spot (no command queue),
with smooth ramping to prevent jerk on key release.

Controls:
  W/S = forward/back (v_x)
  Q/E = strafe left/right (v_y)
  A/D = yaw left/right (w_z)
  Space = EMERGENCY STOP (immediate zero)
  T = toggle teleop on/off

Notes:
- Directly drives SpotFlatTerrainPolicy using [v_x, v_y, w_z].
- Keeps RobotNavigationManager only to publish position for dynamic avoidance.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import carb
import carb.events
import carb.input
import omni

from omni.kit.scripting import BehaviorScript
from omni.anim.people.settings import PeopleSettings, AgentEvent

# Spot policy (direct control)
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy

# Optional: publish robot pose for avoidance
from robot_navigation_manager import RobotNavigationManager


class SpotBehavior(BehaviorScript):
    # === Speed setpoints ===
    _lin_speed_fwd   = 1.0   # m/s for W
    _lin_speed_back  = 0.5   # m/s for S
    _lat_speed_left  = 0.5   # m/s for Q
    _lat_speed_right = 0.5   # m/s for E
    _ang_speed_ccw   = 1.0   # rad/s for A
    _ang_speed_cw    = 1.0   # rad/s for D

    # === Smoothing (slew-rate limiters) ===
    _max_accel_vx = 2.0   # m/s^2
    _max_decel_vx = 2.5   # m/s^2
    _max_accel_vy = 2.0   # m/s^2
    _max_decel_vy = 2.5   # m/s^2
    _max_accel_wz = 3.0   # rad/s^2
    _max_decel_wz = 3.5   # rad/s^2

    _deadband_v = 0.02    # m/s   -> snap to 0 near zero target
    _deadband_w = 0.02    # rad/s

    def on_init(self):
        # Policy controller for Spot
        self._policy: SpotFlatTerrainPolicy = SpotFlatTerrainPolicy(
            prim_path=str(self.prim_path),
            name=self.get_agent_name(),
        )
        # Ensure every call to forward updates the action
        self._policy._decimation = 1
       
        # Command state (smoothed)
        self._cmd_vx = 0.0
        self._cmd_vy = 0.0
        self._cmd_wz = 0.0
        self._base_command = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Input state
        self._pressed = set()
        self._kb_sub = None
        self._keyboard = None
        self._input = None
        self._teleop_enabled = True
        self._initialized = False

        # Settings / avoidance
        self.setting = carb.settings.get_settings()
        self._avoidanceOn = False
        self._navigation_manager: Optional[RobotNavigationManager] = None

        # Agent registration (optional)
        self._bus = None
        self._agent_registered_event = None

        carb.log_info(f"[{self.get_agent_name()}] SpotBehavior initialized (teleop only, smooth ramps).")

    # --- Lifecycle ------------------------------------------------------------
    def on_play(self):
        self._setup_for_play()
        self._register_to_agent_manager()
        self._attach_keyboard()

    def on_stop(self):
        self._detach_keyboard()
        self._navigation_manager = None
        self._initialized = False
        self._bus = None
        self._agent_registered_event = None
        self._cmd_vx = self._cmd_vy = self._cmd_wz = 0.0
        self._base_command[:] = 0.0

    def on_destroy(self):
        self._detach_keyboard()
        if self._navigation_manager is not None:
            self._navigation_manager.destroy()
            self._navigation_manager = None
        self._bus = None
        self._agent_registered_event = None

    # --- Setup / registration -------------------------------------------------
    def _setup_for_play(self):
        self._avoidanceOn = self.setting.get(PeopleSettings.DYNAMIC_AVOIDANCE_ENABLED)

        # Robot handle for nav manager: use policy.robot if exposed
        spot_robot = getattr(self._policy, "robot", None)
        if spot_robot is not None:
            self._navigation_manager = RobotNavigationManager(
                str(self.prim_path), spot_robot, navmesh_enabled=False, dynamic_avoidance_enabled=self._avoidanceOn
            )

        # Agent registration bus (optional)
        self._agent_registered_event = carb.events.type_from_string(AgentEvent.AgentRegistered)
        self._bus = omni.kit.app.get_app().get_message_bus_event_stream()

        carb.log_info("[SpotTeleop] Controls: W/S fwd/back | Q/E strafe | A/D yaw | Space EMERGENCY STOP | T toggle")

    def _register_to_agent_manager(self):
        if self._bus and self._agent_registered_event:
            info = {"agent_name": str(self.get_agent_name()), "prim_path": str(self.prim_path)}
            self._bus.push(self._agent_registered_event, payload=info)

    # --- Keyboard wiring ------------------------------------------------------
    def _attach_keyboard(self):
        import omni.appwindow
        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            carb.log_warn("[SpotTeleop] No app window (headless). Keyboard control disabled.")
            return
        self._keyboard = app_window.get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._kb_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)

    def _detach_keyboard(self):
        if self._input and self._keyboard and self._kb_sub:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._kb_sub)
            except Exception:
                pass
        self._kb_sub = None
        self._keyboard = None
        self._input = None

    def _on_key(self, event):
        from carb.input import KeyboardEventType as KET, KeyboardInput as KI
        if event.type in (KET.KEY_PRESS, KET.KEY_REPEAT):
            self._pressed.add(event.input)
        elif event.type == KET.KEY_RELEASE:
            self._pressed.discard(event.input)

        # Emergency stop (immediate zero)
        if event.input == KI.SPACE and event.type in (KET.KEY_PRESS, KET.KEY_REPEAT):
            self._emergency_stop()

        # Toggle teleop
        if event.input == KI.T and event.type == KET.KEY_PRESS:
            self._teleop_enabled = not self._teleop_enabled
            carb.log_info(f"[SpotTeleop] Teleop {'ENABLED' if self._teleop_enabled else 'DISABLED'}")

    # --- Frame update ---------------------------------------------------------
    def on_update(self, current_time: float, delta_time: float):
        if not self._initialized:
            # Initialize policy if it exposes an initialize() method
            try:
                init_fn = getattr(self._policy, "initialize", None)
                if callable(init_fn):
                    self._policy.initialize()
            except Exception as e:
                carb.log_warn(f"[SpotTeleop] Policy initialize skipped/failed: {e}")
            self._initialized = True

        if self._teleop_enabled:
            self._apply_keyboard_velocity(delta_time)
        else:
            self._smooth_stop(delta_time)

        # Push command to policy
        self._base_command[0] = self._cmd_vx
        self._base_command[1] = self._cmd_vy
        self._base_command[2] = self._cmd_wz
        self._policy.forward(delta_time, self._base_command)

        # Optional: publish position for avoidance
        if self._avoidanceOn and self._navigation_manager is not None:
            self._navigation_manager.publish_robot_position(1.0 / 30.0, radius=3.0)

    # --- Velocity helpers -----------------------------------------------------
    def _apply_keyboard_velocity(self, dt: float):
        from carb.input import KeyboardInput as KI

        # Targets from keys
        vx_tgt = 0.0
        vy_tgt = 0.0
        wz_tgt = 0.0

        # X (forward/back)
        if (KI.W in self._pressed) ^ (KI.S in self._pressed):
            vx_tgt = self._lin_speed_fwd if (KI.W in self._pressed) else -self._lin_speed_back

        # Y (strafe)
        if (KI.Q in self._pressed) ^ (KI.E in self._pressed):
            vy_tgt = self._lat_speed_left if (KI.Q in self._pressed) else -self._lat_speed_right

        # Z (yaw)
        if (KI.A in self._pressed) ^ (KI.D in self._pressed):
            wz_tgt = self._ang_speed_ccw if (KI.A in self._pressed) else -self._ang_speed_cw

        # Slew toward targets
        self._cmd_vx = self._slew(self._cmd_vx, vx_tgt, self._max_accel_vx, self._max_decel_vx, dt, self._deadband_v)
        self._cmd_vy = self._slew(self._cmd_vy, vy_tgt, self._max_accel_vy, self._max_decel_vy, dt, self._deadband_v)
        self._cmd_wz = self._slew(self._cmd_wz, wz_tgt, self._max_accel_wz, self._max_decel_wz, dt, self._deadband_w)

    def _smooth_stop(self, dt: float):
        # Ramp all DOFs down to zero (gentle stop)
        self._cmd_vx = self._slew(self._cmd_vx, 0.0, self._max_accel_vx, self._max_decel_vx, dt, self._deadband_v)
        self._cmd_vy = self._slew(self._cmd_vy, 0.0, self._max_accel_vy, self._max_decel_vy, dt, self._deadband_v)
        self._cmd_wz = self._slew(self._cmd_wz, 0.0, self._max_accel_wz, self._max_decel_wz, dt, self._deadband_w)

    def _emergency_stop(self):
        # Immediate zero; keeps teleop enabled
        self._pressed.clear()
        self._cmd_vx = self._cmd_vy = self._cmd_wz = 0.0
        self._base_command[:] = 0.0
        # Push zero right away
        self._policy.forward(0.0, self._base_command)

    @staticmethod
    def _slew(cur: float, tgt: float, accel: float, decel: float, dt: float, deadband: float) -> float:
        """
        Rate-limit 'cur' toward 'tgt'.
        Positive delta uses 'accel', negative delta uses 'decel'.
        Apply a small deadband near zero when target is also near zero.
        """
        dt = max(1e-4, float(dt))
        dv = tgt - cur
        limit = accel if dv > 0.0 else decel
        max_step = limit * dt
        if dv > max_step:
            cur += max_step
        elif dv < -max_step:
            cur -= max_step
        else:
            cur = tgt

        if abs(cur) < deadband and abs(tgt) < deadband:
            cur = 0.0
        return cur

    # --- Helpers --------------------------------------------------------------
    def get_agent_name(self):
        return str(self.prim_path).split("/")[-1]
