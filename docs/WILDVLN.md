# Wild VLN — BEV trace-prediction VLM (pipeline & reproduction notes)

Instruction-guided navigation as **metric BEV trace prediction**: a VLM
(RynnBrain1.1-2B + LoRA) takes the front camera image + language
instruction (+ rolling memory + driven history) and emits
`<think>…</think><memory>{…}</memory><trace_bev>(x, y), …</trace_bev>` —
10 points, meters, robot frame (x fwd / y left), up to 10 m.
All heavy data lives under `/data/patelm/ticvla/`; code in `wildvln/`
(conda env `wildvln`). The TIC-VLA baseline uses env `tic-vla` +
`source .env.training`.

## Datasets

| dataset | samples (train/val/test) | parquet |
|---|---|---|
| GND (campus, wheeled) | 9,043 / 960 / — | `wildvln/p5/samples.parquet` |
| GrandTour (ANYmal, 48 missions) | 8,660 / 870 / 2,247 | `grandtour/p5/samples.parquet` |
| DynaNav (Isaac sim, TIC-VLA bench) | 10,530 / 1,054 / 1,967 | `dnav/p5/samples.parquet` |

Splits are **location-level**: GrandTour holds out the ARCHE and SBB
environments entirely (`test_site`); DynaNav test = the recordings in
`eval_split_manifest.json`; GND holds out GTown.

### GND (original pipeline, `p*.py`)
`p2b*` index/keyframes → `p2c_vitfeat/p2c_depthcache/p2c_semantics`
caches → `p4_episodes/p4_fullann/p4_farm` (mining + 122B annotation
farm, port 8118) → `p5_package` → `p6_sft` and friends.

### GrandTour (`gt_*.py`)
1. `gt_explore` / `scratchpad gt_download_full` — HF download
   (leggedrobotics/grand_tour_dataset; **HF_HUB_DISABLE_XET=1** + retry
   loop mandatory on this box), zarr-v2 extraction (write missing root
   `.zgroup`).
2. `gt_undistort` — equidistant fisheye → pinhole (balance=0,
   1280×853, fx≈540) → `images/hdr_front_rect/` + `intrinsics.json`.
3. `gt_wavemap` — pywavemap occupancy (0.1 m, XT-32 spherical
   projector, continuous_beam) → `wavemap/<bag>.wvmp`. Not shipped by
   the dataset; built from `hesai_points_undistorted` + dlio poses
   (`dlio_map_odometry` pose IS T_dlio_map→hesai directly).
4. `gt_prepare` — orchestrates 1–3 for all missions (48/51 usable).
5. `gt_p2index` — `p2b/grandtour/<bag>/{index.npz, keyframes/, rig.json}`.
   **Devkit TF attrs are parent/child-swapped**: `pq(entry)` =
   T_child→base, so `T_cam_base = static_to_base(tf, "hdr_front")`
   with NO inversion.
6. `gt_farm` — runs the unchanged GND farm wholesale
   (`WILDVLN_P2B_ROOT` + monkeypatched `rig_for_site`); 205 episodes,
   184 kept. `gt_p5_package` → parquet (11,777 samples).
7. QC/visuals: `gt_ep_bev` (wavemap BEV per step), `gt_reoverlay`
   (repaint overlays without re-annotating).

**Overlay projection is pitch-aware** (`overlay_trace(pts3d=…)`):
path stations are expressed in the full-SE3 keyframe pose before
projecting — yaw-only ego + static T_cam_base mispaints whenever the
base pitches (stairs). Same fix applies to GND (poses are full SE3).

### BEV-token input caches (GA-VLN recipe, GrandTour port)
- `gt_splatcache` — wavemap-**raycast** patch depth (32-px grid,
  batched early-exit marching, 0.6–25 m) at 0.7 m-spaced splat kfs →
  `p2c/depth/<bag>.npz`.
- `gt_vitcache` — merged ViT tokens (LLM image-token space, 2048-d)
  at the same kfs → `p2c/vit/<bag>.npz` (~43 GB).
- `gt_splat.GtBevSplatter` — 28 m distance-windowed causal splat,
  slope-aware z-band (gated on local *path* elevation, not current
  pose). QC: `python -m wildvln.gt_splat <bag> <ep_id> <step>`.

### DynaNav (`dn_*.py`, TIC-VLA benchmark)
- **Horizon**: teleop median 0.45 m/s → TIC-VLA's 3 s action horizon
  ≈ 1.4 m. Our traces are capped at **5 m** (88 % of samples have that
  much future; 10 m would need ~22 s — stale in dynamic scenes).
- `dn_episodes` — jsons → parquet, instructions verbatim, t0 mode
  (history frame is window-relative → skipped).
- `scripts/dn_baseline_bev.py` — base TIC-VLA ckpt + MDE
  (DAv2-metric-outdoor) BEV plots. Ground-plane fit vs known camera
  height (trajectory.csv z) rescales MDE; per-scene scale is
  near-constant (indoor ≈ 0.11×, outdoor ≈ 2.11×, IQR ±5 %).
  Handles Spot pinhole and Carter f-theta fisheye (nominal 1920×1200
  rendered 1080 → cy−60).
- **Sim prototype**: `wildvln/dn_server.py` (HTTP, default :8121,
  serves any adapter ckpt; merge_and_unload for speed) +
  `DynaNav/wildvln_remote.py` (stdlib client shim for the behavior
  scripts; trace → 30×(dx,dy,dθ) @10 Hz at 0.5 m/s cruise, rolling
  memory, sanity clamp). Loop smoke-tested; **zero-shot GND/GrandTour
  ckpts emit pixel-scale garbage on DynaNav imagery** — fine-tune with
  the dn parquet before sim runs.

## Model runs (`/data/patelm/ticvla/wildvln/p6/<run>`)

| run | recipe | key held-out numbers |
|---|---|---|
| m0/m1/m1b/m2 | ablations (BEV tokens, CoT, …) | see p6 logs |
| m3 | `--epochs 2 --turn-boost 3` | GTown turn-exec 32 %, false-turn 10 % |
| m4 | m3 + `--maneuver` token | GTown false-turn 5 %, turn ADE 1.99 |
| grpo1 | GRPO on m3 (reward −min(ADE,4), G=8, k3-KL β=0.02 vs frozen SFT ref adapter) | GTown clean 0.874, turn-exec 40 %, place-err 1.4 m; val sharp-exec 53 % |
| g1 | m3 recipe on GND+GrandTour (`--samples` multi-parquet) | training 2026-07-27 |
| TIC-VLA baseline | pretrained ckpt, 400 held-out | ADE 0.207 / FDE 0.449 |

GRPO gotchas: use a frozen copy of the SFT adapter as KL reference
(`disable_adapter()` references the pre-SFT base — wrong); sampled
reward can *decline* while greedy improves — evaluate checkpoints,
never the reward curve.

## Inference latency (H100, HF generate, measured 2026-07-27)
~22 ms/token uncontended (53 ms/token with training on the box);
t0 ≈ 2.5–6 s, chained CoT ≈ 7–11 s end-to-end. The fla
(flash-linear-attention) Triton path is ~2× *slower* for batch-1
decode — do not install it in the serving env. Real fixes, in order:
vLLM serving, shorter/no CoT at inference, compact coords.

## Eval / QC artifacts (claude.ai)
- GrandTour episode viewer (overlays + wavemap BEV): `6537419d-…d2d`
- TIC-VLA baseline on DynaNav BEV: `f807359f-…9b1`

## Known environment gotchas
- `HF_HUB_DISABLE_XET=1` for all HF downloads on this machine.
- pywavemap: `pip install "git+https://github.com/ethz-asl/wavemap@v2.1.0#subdirectory=library/python"`
  with `CMAKE_ARGS=-DCMAKE_POLICY_VERSION_MINIMUM=3.5`.
- No CUDA toolkit (nvcc) on the box → no source builds of CUDA exts.
- Shared GPUs: check `nvidia-smi` before multi-GPU launches; the 122B
  annotation server occupies GPUs 4–7 (port 8118).
