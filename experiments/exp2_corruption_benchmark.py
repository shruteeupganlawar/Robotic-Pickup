"""Experiment 2: corruption detection + recovery benchmark.

For many clean PyBullet frames, apply every corruption at every severity and record:
  * was it detected?  (precision / recall / F1, with clean frames as negatives)
  * was it accepted, repaired, or rejected?
  * metric depth error (mm) of the accepted output vs the CLEAN frame
  * undetected-but-harmful frames ("silent failures")

Run:  python -m experiments.exp2_corruption_benchmark --frames 16
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.sim_env import HEIGHT, WIDTH, build_scene, capture, connect, random_cube_xy
from safe_rgbd import SafePerception
from safe_rgbd.corruption import CORRUPTION_NAMES, SEVERITIES, apply_corruption
from safe_rgbd.metrics import confusion, mae, precision_recall_f1

NO_SEVERITY = {"flatten_rgb", "flatten_depth", "constant_depth"}


def collect_frames(n, seed):
    frames = []
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        build_scene(random_cube_xy(rng))
        rgb, depth, _, _, proj = capture()
        frames.append((rgb, depth))
    return frames, proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=16, help="test frames")
    ap.add_argument("--calib", type=int, default=8, help="clean frames used to calibrate thresholds")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    connect(gui=False)
    all_frames, proj = collect_frames(args.calib + args.frames, args.seed)
    calib, test = all_frames[:args.calib], all_frames[args.calib:]

    sp = SafePerception(HEIGHT, WIDTH, proj)
    thr = sp.validator.calibrate([d for _, d in calib])
    print(f"Calibrated roughness threshold from {len(calib)} clean frames: {thr:.3e}")
    conv = sp.converter

    rows = []
    det_true, det_pred = {n: [] for n in CORRUPTION_NAMES}, {n: [] for n in CORRUPTION_NAMES}
    clean_flags = []

    for fi, (rgb, depth) in enumerate(test):
        clean_metric = conv.buffer_to_metric(depth)
        clean_flags.append(sp.process(rgb, depth).report.issues_detected)

    for ci, name in enumerate(CORRUPTION_NAMES):
        sevs = [0.0] if name in NO_SEVERITY else SEVERITIES
        for si, sev in enumerate(sevs):
            n = detected = accepted = repaired = 0
            err_all, err_px, silent_err = [], [], []
            for fi, (rgb, depth) in enumerate(test):
                rng = np.random.default_rng([args.seed, fi, ci, si])
                rgb_c, depth_c = apply_corruption(name, rgb, depth, sev, rng)
                frame = sp.process(rgb_c, depth_c)
                flagged = frame.report.issues_detected
                n += 1
                detected += flagged
                det_true[name].append(True)
                det_pred[name].append(flagged)
                if frame.accepted:
                    accepted += 1
                    repaired += bool(frame.actions)
                    clean_metric = conv.buffer_to_metric(depth)
                    e = mae(frame.depth_metric, clean_metric) * 1000
                    err_all.append(e)
                    if depth_c.shape == depth.shape:
                        changed = ~(depth_c == depth)
                        if changed.any():
                            err_px.append(mae(frame.depth_metric, clean_metric, changed) * 1000)
                    if not flagged:
                        silent_err.append(e)
            rows.append({
                "corruption": name, "severity": sev, "n": n,
                "detected_pct": 100 * detected / n,
                "accepted_pct": 100 * accepted / n,
                "repaired_pct": 100 * repaired / n,
                "rejected_pct": 100 * (n - accepted) / n,
                "mae_all_px_mm": float(np.mean(err_all)) if err_all else float("nan"),
                "mae_corrupted_px_mm": float(np.mean(err_px)) if err_px else float("nan"),
                "silent_accept_n": len(silent_err),
                "silent_accept_mae_mm": float(np.mean(silent_err)) if silent_err else float("nan"),
            })
        print(f"  done: {name}")

    out = Path("results/tables")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "corruption_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    det_rows = []
    for name in CORRUPTION_NAMES:
        y_true = det_true[name] + [False] * len(clean_flags)
        y_pred = det_pred[name] + clean_flags
        tp, tn, fp, fn = confusion(y_true, y_pred)
        pr, rc, f1 = precision_recall_f1(tp, fp, fn)
        det_rows.append({"corruption": name, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                         "precision": pr, "recall": rc, "f1": f1})
    with open(out / "detection_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=det_rows[0].keys())
        w.writeheader()
        w.writerows(det_rows)

    print(f"\nClean frames flagged (false positives): {sum(clean_flags)}/{len(clean_flags)}")
    print(f"\n{'corruption':<16}{'TP':>5}{'FP':>4}{'FN':>5}{'prec':>7}{'recall':>8}{'F1':>7}")
    for r in det_rows:
        print(f"{r['corruption']:<16}{r['TP']:>5}{r['FP']:>4}{r['FN']:>5}"
              f"{r['precision']:>7.2f}{r['recall']:>8.2f}{r['f1']:>7.2f}")
    print(f"\n{'corruption':<16}{'sev':>5}{'det%':>6}{'acc%':>6}{'rej%':>6}"
          f"{'MAE all(mm)':>13}{'MAE corr.px(mm)':>17}{'silent':>8}")
    for r in rows:
        print(f"{r['corruption']:<16}{r['severity']:>5.2f}{r['detected_pct']:>6.0f}"
              f"{r['accepted_pct']:>6.0f}{r['rejected_pct']:>6.0f}"
              f"{r['mae_all_px_mm']:>13.3f}{r['mae_corrupted_px_mm']:>17.2f}"
              f"{r['silent_accept_n']:>8}")
    print("\nSaved results/tables/corruption_benchmark.csv and detection_metrics.csv")


if __name__ == "__main__":
    main()
