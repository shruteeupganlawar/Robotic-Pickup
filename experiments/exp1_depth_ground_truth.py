"""Experiment 1: is the depth-buffer -> metres conversion correct?

A large flat wall is placed perpendicular to the camera's optical axis at a KNOWN distance D.
Every pixel then has a true metric depth of exactly D, so we can compute MAE/RMSE for
(a) the correct nonlinear conversion and (b) the naive linear conversion.

Run:  python -m experiments.exp1_depth_ground_truth          (headless)
      python -m experiments.exp1_depth_ground_truth --gui     (also checks the OpenGL renderer)
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pybullet as p

from experiments.sim_env import connect
from safe_rgbd import DepthConverter
from safe_rgbd.metrics import mae, rmse

W, H = 320, 240
DISTANCES = [0.3, 0.5, 1.0, 1.5, 2.0, 2.5]
CONFIGS = [(0.1, 3.0), (0.05, 5.0)]          # (near, far)


def wall_depth_buffer(distance, near, far, renderer):
    p.resetSimulation()
    half = [0.01, 20.0, 20.0]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.8, 0.2, 0.2, 1])
    # front face of the wall sits exactly at x = distance
    p.createMultiBody(0, col, vis, basePosition=[distance + half[0], 0, 1.0])
    view = p.computeViewMatrix([0, 0, 1.0], [1, 0, 1.0], [0, 0, 1])   # eye, target, up
    proj = p.computeProjectionMatrixFOV(60, W / H, near, far)
    _, _, _, depth, _ = p.getCameraImage(W, H, view, proj, renderer=renderer)
    return np.asarray(depth, dtype=np.float32).reshape(H, W), proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true", help="use the OpenGL hardware renderer")
    args = ap.parse_args()
    connect(gui=args.gui)
    renderer = p.ER_BULLET_HARDWARE_OPENGL if args.gui else p.ER_TINY_RENDERER
    rows = []
    for near, far in CONFIGS:
        for dist in DISTANCES:
            buf, proj = wall_depth_buffer(dist, near, far, renderer)
            conv = DepthConverter.from_proj_matrix(proj)
            truth = np.full((H, W), dist)
            proper = conv.buffer_to_metric(buf)
            naive = conv.linear_to_metric(buf)
            rows.append({
                "near": near, "far": far, "true_m": dist,
                "buffer_median": float(np.median(buf)),
                "nonlinear_mae_mm": mae(proper, truth) * 1000,
                "nonlinear_rmse_mm": rmse(proper, truth) * 1000,
                "linear_mae_mm": mae(naive, truth) * 1000,
                "linear_rmse_mm": rmse(naive, truth) * 1000,
            })

    out = Path("results/tables")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "depth_ground_truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"{'near':>5} {'far':>4} {'true(m)':>8} {'buffer':>8} "
          f"{'nonlin MAE(mm)':>15} {'linear MAE(mm)':>15}")
    for r in rows:
        print(f"{r['near']:>5} {r['far']:>4} {r['true_m']:>8.2f} {r['buffer_median']:>8.4f} "
              f"{r['nonlinear_mae_mm']:>15.4f} {r['linear_mae_mm']:>15.1f}")
    print("Saved results/tables/depth_ground_truth.csv")


if __name__ == "__main__":
    main()
