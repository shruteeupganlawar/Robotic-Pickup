"""Experiment 3: grasp success, RAW pipeline vs SAFE pipeline, under corrupted depth.

Both pipelines receive the SAME scene and the SAME corrupted frame in every trial.
Object detection is the simulator's segmentation mask (a "perfect detector"), so the
experiment isolates the effect of depth/frame quality.

Outcomes per trial
  success       arm executed and the cube ended up lifted (z > 0.15 m)
  wrong_motion  arm executed but the cube was not lifted  (unsafe: acted on bad data)
  no_motion     raw: crashed / non-finite target,   safe: frame or target rejected

Run:  python -m experiments.exp3_grasp_trials --trials 20
"""

import argparse
import contextlib
import csv
import io
import time
import warnings
from pathlib import Path

import numpy as np
import pybullet as p

time.sleep = lambda *_: None

from experiments.sim_env import (HEIGHT, WIDTH, build_scene, capture, connect,  # noqa: E402
                                 random_cube_xy)
from safe_rgbd import SafePerception, localize_object  
from safe_rgbd.corruption import apply_corruption  
from safe_rgbd.metrics import wilson_interval  
from src.detect_seg import detect_objects_by_segmentation  
from src.execute_grasp import execute_pick  
from src.grasp_planner import get_grasp_point_from_mask  

LIFT_Z = 0.15

CONDITIONS = [
    ("clean",              "clean",          0.00, False),
    ("nan_10%",            "nan",            0.10, False),
    ("nan_25%",            "nan",            0.25, False),
    ("nan_50%",            "nan",            0.50, False),
    ("dropout_5%_on_cube", "zero_region",    0.05, True),
    ("dropout_10%_on_cube", "zero_region",   0.10, True),
    ("dropout_25%_on_cube", "zero_region",   0.25, True),
    ("truncated_depth",    "truncate_depth", 0.10, False),
    ("mixed_10%",          "mixed",          0.10, False),
    ("mixed_25%",          "mixed",          0.25, False),
    ("depth_bias_blind",   "depth_bias",     0.50, False),
]


def execute_and_score(scene, grasp_point):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            execute_pick(scene["robot"], grasp_point)
    except Exception as e:                       # arm was commanded, then something broke
        return "wrong_motion", f"execution error: {type(e).__name__}"
    z = p.getBasePositionAndOrientation(scene["cube"])[0][2]
    return ("success" if z > LIFT_Z else "wrong_motion"), f"cube_z={z:.3f}"


def run_raw(scene, rgb_c, depth_c, seg, view, proj):
    try:
        np.reshape(rgb_c, (HEIGHT, WIDTH, 4))               # a naive consumer of RGB
        depth = np.reshape(depth_c, (HEIGHT, WIDTH))
        dets = detect_objects_by_segmentation(
            seg, [scene["cube"], scene["duck"]], ["cube", "duck"])
        target = next(d for d in dets if d["class_name"] == "cube")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = get_grasp_point_from_mask(target, seg, depth, view, proj, WIDTH, HEIGHT)
    except Exception as e:
        return "no_motion", f"crash: {type(e).__name__}", None
    if not np.all(np.isfinite(gp)):
        return "no_motion", "non-finite grasp point", None
    outcome, note = execute_and_score(scene, gp)
    return outcome, note, gp


def run_safe(scene, sp, rgb_c, depth_c, seg, view, min_conf):
    frame = sp.process(rgb_c, depth_c)
    if not frame.accepted:
        return "no_motion", "frame rejected: " + frame.reason[:60], None
    mask = (seg.astype(np.int64) & 0xFFFFFF) == scene["cube"]
    loc = localize_object(mask, frame.depth_metric, frame.confidence, view,
                          sp.intrinsics, min_confidence=min_conf)
    if not loc.ok:
        return "no_motion", "target rejected: " + loc.reason, None
    outcome, note = execute_and_score(scene, loc.point)
    return outcome, note, loc.point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20, help="trials per condition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--max-invalid", type=float, default=0.20,
                    help="reject frames with more invalid depth pixels than this fraction")
    ap.add_argument("--roughness-reject", type=float, default=None,
                    help="reject frames whose depth roughness exceeds this (buffer units)")
    ap.add_argument("--tag", default="", help="suffix for the output CSV names, e.g. _maxinv0.6")
    ap.add_argument("--conditions", nargs="*", help="only run these condition labels")
    args = ap.parse_args()

    connect(gui=False)
    conditions = [c for c in CONDITIONS if not args.conditions or c[0] in args.conditions]
    trial_rows = []

    for label, name, sev, on_target in conditions:
        print(f"\n== {label}")
        for t in range(args.trials):
            rng = np.random.default_rng([args.seed, t])
            cube_xy = random_cube_xy(rng)

            scene = build_scene(cube_xy)
            rgb, depth, seg, view, proj = capture()
            cube_true = np.array(p.getBasePositionAndOrientation(scene["cube"])[0])

            ys, xs = np.nonzero((seg.astype(np.int64) & 0xFFFFFF) == scene["cube"])
            centre = (xs.mean() + rng.integers(-10, 11), ys.mean() + rng.integers(-10, 11))
            c_rng = np.random.default_rng([args.seed, t, 999])
            rgb_c, depth_c = apply_corruption(
                name, rgb, depth, sev, c_rng, center=centre if on_target else None)

            sp = SafePerception(HEIGHT, WIDTH, proj, max_invalid_fraction=args.max_invalid,
                                roughness_warn=1e-5,   # or calibrate from clean frames (Exp. 2)
                                roughness_reject=args.roughness_reject)

            out_raw, note_raw, gp_raw = run_raw(scene, rgb_c, depth_c, seg, view, proj)

            scene = build_scene(cube_xy)            # identical fresh scene for the safe run
            out_safe, note_safe, gp_safe = run_safe(scene, sp, rgb_c, depth_c, seg, view,
                                                    args.min_confidence)

            for pipe, out, note, gp in (("raw", out_raw, note_raw, gp_raw),
                                        ("safe", out_safe, note_safe, gp_safe)):
                err = (float(np.linalg.norm(gp[:2] - cube_true[:2]) * 100)
                       if gp is not None else float("nan"))
                trial_rows.append({"condition": label, "trial": t, "pipeline": pipe,
                                   "outcome": out, "xy_error_cm": err, "note": note})
            print(f"  trial {t:>2}: raw={out_raw:<12} safe={out_safe:<12}")

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"grasp_trials{args.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=trial_rows[0].keys())
        w.writeheader()
        w.writerows(trial_rows)

    summary = []
    print(f"\n{'condition':<22}{'pipeline':<6}{'success %':>10}{'95% CI':>14}"
          f"{'wrong %':>9}{'no-motion %':>13}{'succ|executed':>15}")
    for label, *_ in conditions:
        for pipe in ("raw", "safe"):
            rows = [r for r in trial_rows if r["condition"] == label and r["pipeline"] == pipe]
            n = len(rows)
            k = sum(r["outcome"] == "success" for r in rows)
            wrong = sum(r["outcome"] == "wrong_motion" for r in rows)
            nomo = sum(r["outcome"] == "no_motion" for r in rows)
            lo, hi = wilson_interval(k, n)
            cond = k / (k + wrong) * 100 if (k + wrong) else float("nan")
            summary.append({"condition": label, "pipeline": pipe, "n": n,
                            "success_pct": 100 * k / n, "ci_low": 100 * lo, "ci_high": 100 * hi,
                            "wrong_motion_pct": 100 * wrong / n, "no_motion_pct": 100 * nomo / n,
                            "success_given_executed_pct": cond})
            print(f"{label:<22}{pipe:<6}{100*k/n:>10.0f}{f'[{100*lo:.0f}-{100*hi:.0f}]':>14}"
                  f"{100*wrong/n:>9.0f}{100*nomo/n:>13.0f}{cond:>15.0f}")

    with open(out_dir / f"grasp_summary{args.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)
    print(f"\nSaved results/tables/grasp_trials{args.tag}.csv and grasp_summary{args.tag}.csv")


if __name__ == "__main__":
    main()
