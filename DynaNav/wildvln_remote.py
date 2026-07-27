"""Client shim: query a remote Wild VLN trace server from the DynaNav sim.

Copy this file (only stdlib + numpy needed) to the sim machine next to
the behavior scripts, start the server on the training box:

    CUDA_VISIBLE_DEVICES=3 python -m wildvln.dn_server --port 8121

then in the behavior script replace the in-process TICVLA model with:

    from wildvln_remote import WildVLNRemote
    model = WildVLNRemote("http://<training-box-ip>:8121")
    response, waypoints = model.predict(image_path, instruction)

predict() returns (response_text, waypoints) where waypoints is a
(30, 3) float32 array of [dx, dy, dtheta] cumulative offsets relative
to the current frame at 10 Hz — the same convention as the TIC-VLA
action expert — produced by resampling the predicted metric trace
(x fwd / y left, up to 10 m) at CRUISE_V m/s. The optional rolling
`memory` dict is maintained internally across calls (reset() between
episodes).
"""

from __future__ import annotations

import base64
import json
import urllib.request

import numpy as np

CRUISE_V = 0.5          # m/s — DynaNav teleop median is 0.45
DT = 0.1
N_STEPS = 30


def trace_to_waypoints(trace, v=CRUISE_V, n=N_STEPS, dt=DT):
    """Resample a metric trace polyline into n cumulative (dx,dy,dtheta)
    waypoints at speed v — TIC-VLA action-expert convention."""
    p = np.vstack([[0.0, 0.0], np.asarray(trace, np.float32)])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.minimum(np.arange(1, n + 1) * v * dt, s[-1])
    xy = np.stack([np.interp(t, s, p[:, k]) for k in (0, 1)], 1)
    d = np.diff(np.vstack([[0.0, 0.0], xy]), axis=0)
    theta = np.arctan2(d[:, 1], d[:, 0])
    theta[np.linalg.norm(d, axis=1) < 1e-4] = 0.0
    return np.concatenate([xy, theta[:, None]], 1).astype(np.float32)


class WildVLNRemote:
    def __init__(self, url: str, timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.memory = None

    def reset(self):
        self.memory = None

    def predict(self, image, instruction: str):
        """image: path to a jpeg/png OR raw bytes. Returns
        (response_text, (30,3) waypoints ndarray)."""
        raw = image if isinstance(image, (bytes, bytearray)) else \
            open(image, "rb").read()
        body = json.dumps({
            "image_b64": base64.b64encode(raw).decode(),
            "instruction": instruction,
            "memory": self.memory}).encode()
        req = urllib.request.Request(
            self.url + "/act", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        if out.get("memory_out"):
            self.memory = out["memory_out"]
        trace = out.get("trace")
        if not trace:
            return out.get("raw", ""), np.zeros((N_STEPS, 3), np.float32)
        return out.get("cot") or out.get("raw", ""), \
            trace_to_waypoints(trace)
