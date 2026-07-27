#!/usr/bin/env python3
"""GrandTour HDR camera undistortion (equidistant fisheye -> pinhole).

caminfo distortion_model is 'equidistant' (Kannala-Brandt, 4 coeffs) ->
cv2.fisheye. Rectified pinhole K chosen with balance=0.0 (crop to valid
FOV, no black borders) at a configurable output size; the rectified K
is what stage-B overlays / BEV lifting must use downstream.

Usage:
  python -m wildvln.gt_undistort --mission 2024-11-04-10-57-34 --qc
  python -m wildvln.gt_undistort --mission ... --batch   (all frames)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path("/data/patelm/ticvla/grandtour/raw")
OUT_QC = Path("/data/patelm/ticvla/grandtour/qc/undist")
OUT_W, OUT_H = 1280, 853          # ~2/3 scale, keeps detail, VLM-friendly


def load_caminfo(mission, cam="hdr_front"):
    y = yaml.safe_load(open(ROOT / mission / "metadata"
                            / f"{cam}_caminfo.yaml"))["camera_info"]
    assert y["distortion_model"] == "equidistant", y["distortion_model"]
    K = np.array(y["K"], float).reshape(3, 3)
    D = np.array(y["D"], float).reshape(4, 1)
    return K, D, (y["width"], y["height"])


class Undistorter:
    def __init__(self, mission, cam="hdr_front", out_wh=(OUT_W, OUT_H),
                 balance=0.0):
        K, D, (W, H) = load_caminfo(mission, cam)
        self.K, self.D, self.in_wh = K, D, (W, H)
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (W, H), np.eye(3), balance=balance,
            new_size=out_wh)
        self.newK = newK
        self.out_wh = out_wh
        self.m1, self.m2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), newK, out_wh, cv2.CV_16SC2)

    def __call__(self, img):
        # images on disk may be stored downscaled vs caminfo resolution
        if (img.shape[1], img.shape[0]) != self.in_wh:
            img = cv2.resize(img, self.in_wh,
                             interpolation=cv2.INTER_LINEAR)
        return cv2.remap(img, self.m1, self.m2, cv2.INTER_LINEAR)

    def intrinsics(self):
        fx, fy = self.newK[0, 0], self.newK[1, 1]
        cx, cy = self.newK[0, 2], self.newK[1, 2]
        return dict(fx=float(fx), fy=float(fy), cx=float(cx),
                    cy=float(cy), width=self.out_wh[0],
                    height=self.out_wh[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", required=True)
    ap.add_argument("--cam", default="hdr_front")
    ap.add_argument("--qc", action="store_true")
    ap.add_argument("--batch", action="store_true")
    args = ap.parse_args()

    u = Undistorter(args.mission, args.cam)
    print("rectified intrinsics:", json.dumps(u.intrinsics()))
    src = ROOT / args.mission / "images" / args.cam
    frames = sorted(src.glob("*.jpeg"))
    print(f"{len(frames)} frames at {src}")

    if args.qc:
        OUT_QC.mkdir(parents=True, exist_ok=True)
        for k in (0, len(frames) // 2, len(frames) - 1):
            im = cv2.imread(str(frames[k]))
            und = u(im)
            side = np.concatenate(
                [cv2.resize(im, u.out_wh), und], 1)
            p = OUT_QC / f"{args.mission}_{args.cam}_{k:06d}.jpg"
            cv2.imwrite(str(p), side, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print("qc ->", p)

    if args.batch:
        dst = ROOT / args.mission / "images" / f"{args.cam}_rect"
        dst.mkdir(exist_ok=True)
        json.dump(u.intrinsics(), open(dst / "intrinsics.json", "w"))
        for f in frames:
            o = dst / f.name
            if o.exists():
                continue
            cv2.imwrite(str(o), u(cv2.imread(str(f))),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("BATCH_DONE", dst)


if __name__ == "__main__":
    main()
