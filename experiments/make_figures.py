"""Make the README figures.

  results/figures/recovery_demo.png       clean vs corrupted vs recovered (+ error, confidence)
  results/figures/pointcloud.png          3D point cloud from the repaired frame
  results/figures/depth_error_vs_severity.png   (needs Experiment 2 CSV)
  results/figures/grasp_success.png       (needs Experiment 3 CSV)

Run:  python -m experiments.make_figures
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.sim_env import HEIGHT, WIDTH, build_scene, capture, connect
from safe_rgbd import SafePerception, camera_to_world, depth_to_camera_points
from safe_rgbd.corruption import inject_nan, zero_region

FIG = Path("results/figures")
TAB = Path("results/tables")


def recovery_figure():
    connect(gui=False)
    build_scene((0.5, 0.1))
    rgb, depth, seg, view, proj = capture()
    sp = SafePerception(HEIGHT, WIDTH, proj)
    conv = sp.converter

    rng = np.random.default_rng(0)
    bad = inject_nan(depth, 0.05, rng)
    bad = zero_region(bad, 0.03, rng, center=(340, 250))
    frame = sp.process(rgb, bad)
    print(frame.report.summary())

    clean_m = conv.buffer_to_metric(depth)
    bad_m = np.where((bad > 0) & np.isfinite(bad), conv.buffer_to_metric(np.nan_to_num(bad, nan=0.5)), np.nan)
    rec_m = frame.depth_metric
    err_mm = np.abs(rec_m - clean_m) * 1000

    vmin, vmax = float(np.nanmin(clean_m)), float(np.nanmax(clean_m))
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].imshow(rgb[..., :3]); ax[0, 0].set_title("RGB")
    ax[0, 1].imshow(clean_m, cmap="viridis", vmin=vmin, vmax=vmax); ax[0, 1].set_title("Clean depth (m)")
    cmap = plt.cm.viridis.copy(); cmap.set_bad("red")
    ax[0, 2].imshow(np.ma.masked_invalid(bad_m), cmap=cmap, vmin=vmin, vmax=vmax)
    ax[0, 2].set_title("Corrupted (red = invalid)")
    ax[1, 0].imshow(rec_m, cmap="viridis", vmin=vmin, vmax=vmax); ax[1, 0].set_title("Recovered depth (m)")
    im = ax[1, 1].imshow(err_mm, cmap="magma", vmin=0, vmax=50); ax[1, 1].set_title("|recovered - clean| (mm)")
    plt.colorbar(im, ax=ax[1, 1], fraction=0.046)
    im = ax[1, 2].imshow(frame.confidence, cmap="gray", vmin=0, vmax=1); ax[1, 2].set_title("Confidence")
    plt.colorbar(im, ax=ax[1, 2], fraction=0.046)
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG / "recovery_demo.png", dpi=110)
    plt.close()
    print("Saved results/figures/recovery_demo.png")

    # ---- point cloud (world frame), subsampled
    fx, fy, cx, cy = sp.intrinsics
    pts = camera_to_world(depth_to_camera_points(rec_m, fx, fy, cx, cy), view)
    step = 6
    P = pts[::step, ::step].reshape(-1, 3)
    C = rgb[::step, ::step, :3].reshape(-1, 3) / 255.0
    keep = (P[:, 2] > -0.1) & (np.linalg.norm(P[:, :2] - [0.5, 0], axis=1) < 0.6)
    fig = plt.figure(figsize=(8, 7))
    a3 = fig.add_subplot(111, projection="3d")
    a3.scatter(P[keep, 0], P[keep, 1], P[keep, 2], c=C[keep], s=2)
    a3.set_xlabel("x (m)"); a3.set_ylabel("y (m)"); a3.set_zlabel("z (m)")
    a3.set_title("Point cloud from the repaired frame (world frame)")
    a3.view_init(elev=35, azim=-60)
    plt.tight_layout()
    plt.savefig(FIG / "pointcloud.png", dpi=110)
    plt.close()
    print("Saved results/figures/pointcloud.png")


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def depth_error_figure():
    path = TAB / "corruption_benchmark.csv"
    if not path.exists():
        print("skip depth_error_vs_severity (run experiment 2 first)")
        return
    rows = read_csv(path)
    names = ["nan", "inf", "zero_region", "out_of_range", "noise", "depth_bias", "mixed"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in names:
        rs = [r for r in rows if r["corruption"] == n and r["accepted_pct"] != "0.0"]
        xs = [float(r["severity"]) for r in rs if r["mae_all_px_mm"] not in ("nan", "")]
        ys = [float(r["mae_all_px_mm"]) for r in rs if r["mae_all_px_mm"] not in ("nan", "")]
        if xs:
            ax.plot(xs, ys, marker="o", label=n)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("corruption severity"); ax.set_ylabel("mean depth error over the whole image (mm)")
    ax.set_title("Depth error of ACCEPTED frames vs corruption severity\n(rejected frames are not plotted)")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "depth_error_vs_severity.png", dpi=110)
    plt.close()
    print("Saved results/figures/depth_error_vs_severity.png")


def grasp_figure():
    path = TAB / "grasp_summary.csv"
    if not path.exists():
        print("skip grasp_success (run experiment 3 first)")
        return
    rows = read_csv(path)
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    x = np.arange(len(conds))
    raw = [float(next(r for r in rows if r["condition"] == c and r["pipeline"] == "raw")["success_pct"]) for c in conds]
    safe = [float(next(r for r in rows if r["condition"] == c and r["pipeline"] == "safe")["success_pct"]) for c in conds]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.2, raw, 0.4, label="raw")
    ax.bar(x + 0.2, safe, 0.4, label="safe RGB-D")
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=35, ha="right")
    ax.set_ylabel("grasp success (%)"); ax.set_ylim(0, 105)
    ax.set_title("Grasp success: raw vs safe pipeline"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "grasp_success.png", dpi=110)
    plt.close()
    print("Saved results/figures/grasp_success.png")


if __name__ == "__main__":
    recovery_figure()
    depth_error_figure()
    grasp_figure()
