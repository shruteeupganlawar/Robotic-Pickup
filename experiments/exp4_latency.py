"""Experiment 4: latency of the safety layer (validate + repair + metric conversion).

Run:  python -m experiments.exp4_latency --iters 1000
"""

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np

from experiments.sim_env import HEIGHT, WIDTH, build_scene, capture, connect
from safe_rgbd import SafePerception
from safe_rgbd.corruption import apply_corruption
from safe_rgbd.metrics import latency_stats

SCENARIOS = [
    ("clean",            "clean",          0.0),
    ("flatten_depth",    "flatten_depth",  0.0),
    ("nan_1%",           "nan",            0.01),
    ("nan_10%",          "nan",            0.10),
    ("zero_region_10%",  "zero_region",    0.10),
    ("mixed_10%",        "mixed",          0.10),
    ("truncated (reject)", "truncate_depth", 0.10),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    args = ap.parse_args()

    connect(gui=False)
    build_scene((0.5, 0.1))
    rgb, depth, _, _, proj = capture()

    sp = SafePerception(HEIGHT, WIDTH, proj)
    sp.validator.calibrate([depth])

    rows = []
    for label, name, sev in SCENARIOS:
        rng = np.random.default_rng(0)
        rgb_c, depth_c = apply_corruption(name, rgb, depth, sev, rng)
        for _ in range(20):                       # warm-up
            sp.process(rgb_c, depth_c)
        times = [sp.process(rgb_c, depth_c).latency_ms for _ in range(args.iters)]
        s = latency_stats(times)
        s["scenario"] = label
        rows.append(s)

    hw = {"machine": platform.machine(), "processor": platform.processor(),
          "python": platform.python_version(), "system": platform.system()}
    out = Path("results/tables")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "latency_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "n", "mean_ms", "median_ms",
                                          "p95_ms", "p99_ms", "max_ms"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    (out / "latency_hardware.json").write_text(json.dumps(hw, indent=2))

    print(f"Hardware: {hw}")
    print(f"{'scenario':<20}{'mean':>8}{'median':>8}{'p95':>8}{'p99':>8}{'max':>8}   (ms)")
    for r in rows:
        print(f"{r['scenario']:<20}{r['mean_ms']:>8.2f}{r['median_ms']:>8.2f}"
              f"{r['p95_ms']:>8.2f}{r['p99_ms']:>8.2f}{r['max_ms']:>8.2f}")
    print("Saved results/tables/latency_benchmark.csv")


if __name__ == "__main__":
    main()
