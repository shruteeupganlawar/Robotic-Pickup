# Perception-to-Grasp 

![Demo GIF](assets/videos/demo.gif)

A simulated robotic pick pipeline in PyBullet. A virtual RGB-D camera looks at a table, the frame is validated, objects are detected, a 3D grasp point is computed from the detection, and a Franka Panda arm attempts the pick.

It connects four stages into one runnable pipeline:

- **RGB-D frame validation** rejects corrupted frames (NaN/Inf depth, zero-size or truncated depth, mostly-empty depth) before they reach planning.
- **Object detection** finds the graspable objects in the camera image.
- **Grasp point computation** converts a 2D detection into a 3D world-frame target using camera projection geometry.
- **Motion execution** drives the simulated Panda arm through hover, descend, grasp and lift using inverse kinematics.

In the recorded demo, the arm moves to a ready pose, hovers over the cube, descends, closes the gripper, and lifts the cube off the table while the duck stays where it is.

On top of that pipeline there is now a **`safe_rgbd` safety layer**: a validator, repair ("recovery") step and confidence-weighted localizer that sit between the camera and the grasp planner, plus a set of experiments (`experiments/`) that measure what that layer actually buys over the naive ("raw") pipeline — see [Measured results](#measured-results-safe_rgbd-experiments) below. It is exercised by `demo_safe.py` and 57 unit tests (`tests/`).

There is also a small Gradio dashboard for trying the detector on your own images.

---

## Pipeline

```text
Virtual RGB-D camera
        |
        v
RGB-D validator  --(corrupt frame)--> abort, no motion
        |
        v
Object detection (segmentation mask)
        |
        v
Grasp planner (mask -> 3D top-surface point)
        |
        v
Panda arm: init pose -> hover -> descend -> close -> lift
```

---

## What the demo shows

The recorded run (about 7 seconds) goes through these stages:

1. **Setup:** the Panda stands upright behind a table with a cube and a duck on it, and the camera frame is captured and validated.
2. **Ready pose and approach:** the arm moves to a ready pose and hovers over the cube.
3. **Descend and grasp:** it descends in small steps, and the gripper closes around the cube.
4. **Lift:** the arm lifts slowly, and the cube stays in the gripper through the end of the clip.

The terminal prints the detected boxes, the computed grasp point, the position error of each arm move, and the cube's final position, so you can verify a pick numerically as well as visually.

### Measured result

From the recorded run:

| Check | Result |
|---|---|
| Frame validation | Passed |
| Objects detected | 2 (cube, duck) |
| Estimated cube grasp point | (0.503, 0.097, 0.024) m |
| Cube spawn position (x, y) | (0.500, 0.100) m |
| Grasp point error (x, y) | about 0.5 cm |
| Arm move error (hover, grasp, lift) | 0.9 cm each |
| Cube height after lift | 0.294 m (started at about 0 m on the table) |

The cube's final height matches the lift target (0.304 m) to within about 1 cm, which shows it was carried by the gripper rather than pushed or dropped. This is a single run on fixed object positions.

---

## Measured results (safe_rgbd experiments)

The results below come from four experiments in `experiments/`, run headless in PyBullet on Windows / Python 3.10 (`results/tables/*.csv`, figures in `results/figures/`). They compare the **raw** pipeline (numpy `reshape`, direct depth use, no checks) against the **safe** pipeline (`safe_rgbd.SafePerception` + `localize_object`) under the same corrupted frames.

### 1. Depth conversion is correct; the naive linear shortcut is not

`experiments/exp1_depth_ground_truth.py` places a wall at a known distance and compares PyBullet's nonlinear depth-buffer-to-metres conversion (`safe_rgbd.DepthConverter`) against a naive linear one.

| near/far | true depth (m) | nonlinear MAE (mm) | linear MAE (mm) |
|---|---|---|---|
| 0.1 / 3.0 | 0.3 – 2.5 | **≤ 0.002** | 480 – 2000 |
| 0.05 / 5.0 | 0.3 – 2.5 | **≤ 0.01** | 2450 – 4050 |

The correct conversion is accurate to micrometres/hundredths of a millimetre; a naive linear conversion of the same buffer is off by half a metre to 4 metres, i.e. unusable. (`results/tables/depth_ground_truth.csv`)

### 2. Corruption detection: 100% on 11 of 12 corruption types, one confirmed blind spot

`experiments/exp2_corruption_benchmark.py` runs 16 clean calibration frames and 16 test frames through every corruption type at every severity and checks whether the validator flags it (`results/tables/detection_metrics.csv`, `corruption_benchmark.csv`).

| corruption | detected | precision / recall / F1 |
|---|---|---|
| nan, inf, zero_region, out_of_range, noise, constant_depth, flatten_rgb/depth, truncate_rgb/depth, mixed | 100% | 1.00 / 1.00 / 1.00 |
| **depth_bias** (a smooth multiplicative bias on depth) | **0%** | 0.00 recall — never flagged |

`depth_bias` is a smooth, statistically invisible corruption (no NaNs, no discontinuities, no out-of-range values) — nothing in the current validator looks for it, so it is **silently accepted every time**, with metric depth error growing from ~4 mm to ~182 mm as severity increases from 0.01 to 0.5 (`silent_accept_n` = 16/16 at every severity). This is the one confirmed gap in the safety layer: the validator catches missing/invalid/noisy data but not a plausible-looking systematic bias. Everything else that is detectable is either repaired (at low severity, ≤ 0.1) or rejected outright (at severity ≥ 0.25, where accepted/repaired drop to 0% and rejected rises to 100%) — see `results/figures/depth_error_vs_severity.png`.

### 3. Grasp success: raw pipeline fails almost every corrupted condition; safe pipeline recovers most of them

`experiments/exp3_grasp_trials.py` runs 20 trials per condition (`--max-invalid 0.20` default reject threshold) through both pipelines on the same corrupted frame and scores success (cube lifted), wrong motion (arm moved on bad data), or no motion (rejected) (`results/tables/grasp_summary.csv`, `results/figures/grasp_success.png`):

| condition | raw success | safe success | notes |
|---|---|---|---|
| clean | 100% | 100% | baseline, both equal |
| nan 10% | 0% (100% no-motion — raw crashes on NaNs) | **100%** | safe repairs and picks |
| nan 25% / 50% | 0% | 0% (both reject) | above the default 20% invalid-pixel threshold |
| dropout 5–25% on cube | 0% (**100% wrong motion** — raw grasps a hallucinated point) | 0% (no motion — safe rejects) | raw is not just wrong, it *moves the arm* on bad data |
| truncated depth | 0% (no motion) | 0% (no motion) | both correctly refuse a malformed buffer |
| mixed corruption 10% | 0% (95% no-motion, 5% wrong motion) | **90%** | safe repairs the recoverable parts |
| mixed corruption 25% | 0% (75% no-motion, 25% wrong motion) | 40% | partial recovery as damage increases |
| depth_bias (blind spot) | 0% wrong motion | 0% wrong motion | **safe pipeline also fails here** — consistent with the detection gap above |

The raw pipeline's failure mode under `zero_region` dropout is the dangerous one: it doesn't crash, it computes a plausible-looking but wrong grasp point and **commands the arm to move**, 100% of the time. The safe pipeline instead rejects the frame or the target and takes no motion. Loosening the reject threshold (`grasp_summary_maxinv0.6.csv`, `--max-invalid 0.6`) recovers `nan_25%`/`nan_50%` to 100% safe success, showing the threshold is a tunable precision/recall trade-off, not a hard limit. Enabling a roughness-based reject filter (`grasp_summary_rr1e-3.csv` vs `grasp_summary_rr3e-4.csv`) changes how `mixed_25%` is handled — a stricter roughness cutoff (3e-4) rejects it outright (0% success, 0% wrong motion) instead of attempting a 40% success rate, trading recall for safety.

### 4. Safety-layer latency

`experiments/exp4_latency.py`, 1000 iterations per scenario on the test machine (Intel, `results/tables/latency_hardware.json`), measuring `SafePerception.process()` time (`results/tables/latency_benchmark.csv`):

| scenario | mean | median | p95 | p99 |
|---|---|---|---|---|
| clean | 8.0 ms | 7.5 ms | 11.3 ms | 13.1 ms |
| flatten_depth | 8.3 ms | 7.7 ms | 11.5 ms | 12.9 ms |
| nan 1% | 37.4 ms | 46.1 ms | 55.6 ms | 65.3 ms |
| nan 10% | 163.9 ms | 161.7 ms | 195.5 ms | 216.6 ms |
| zero_region 10% | 76.9 ms | 77.0 ms | 84.9 ms | 101.0 ms |
| mixed 10% | 104.4 ms | 104.8 ms | 115.8 ms | 135.7 ms |
| truncated (rejected early) | **0.017 ms** | 0.014 ms | 0.019 ms | 0.054 ms |

Clean frames cost ~8 ms. Repair cost scales with how much of the frame is invalid (inpainting more NaN pixels is the dominant cost — 10% NaN is ~20x slower than 1% NaN). A malformed frame that fails a cheap shape/dtype check is rejected in microseconds, before any repair work starts. All numbers are single-machine, single-threaded, and pre-JIT-warmup; they characterise relative cost, not an absolute real-time budget.

### Tests

`tests/` has 57 unit tests covering the validator, recovery, depth conversion, point-cloud back-projection, corruption injectors, the reshaper and the end-to-end pipeline (`python -m pytest tests/`); all pass.

---

## Screenshots

| RGB + Depth | Detection |
|---|---|
| ![Camera test](assets/screenshots/camera_test.png) | ![Detection](assets/screenshots/demo_detection.png) |

---

## Setup

Tested on Windows with Python 3.10.

```bash
git clone <your-repo-url>
cd perception_to_grasp

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install directly:

```bash
pip install pybullet ultralytics opencv-python gradio numpy pillow imageio moviepy matplotlib
```

---

## Usage

Run all commands from the project root.

**Run the full simulation demo** (opens a PyBullet window):

```bash
python -m src.main_demo
```

**Record the demo to `assets/videos/demo.mp4`:**

```bash
python -m src.record_demo
```

**Convert the recording to a GIF:**

```bash
python src/mp4_to_gif.py
# or with custom paths:
python src/mp4_to_gif.py assets/videos/demo.mp4 assets/videos/demo.gif
```

**Launch the detection dashboard** (Gradio, opens at http://127.0.0.1:7860):

```bash
python app.py
```

**Test the validator:**

```bash
python -m src.test_validator
```

**Test the detection pipeline and save a screenshot:**

```bash
python -m src.test_detection_pipeline
```

**Run the demo through the `safe_rgbd` safety layer, optionally with a corrupted frame:**

```bash
python demo_safe.py                                       # clean frame
python demo_safe.py --corrupt nan --severity 0.10         # repaired, then picked
python demo_safe.py --corrupt zero_region --severity 0.10 --on-target   # rejected: no motion
python demo_safe.py --gui                                 # watch it in the PyBullet window
```

**Run the unit tests (57 tests, covers `safe_rgbd`):**

```bash
python -m pytest tests/
```

**See why `reshape` and `resize` are not interchangeable for a raw sensor buffer:**

```bash
python examples/reshape_vs_resize.py
```

**Reproduce the measured results in `results/`:**

```bash
python -m experiments.exp1_depth_ground_truth
python -m experiments.exp2_corruption_benchmark --frames 16
python -m experiments.exp3_grasp_trials --trials 20
python -m experiments.exp4_latency --iters 1000
python -m experiments.make_figures      # regenerate results/figures from the CSVs
```

---

## How it works

### 1. Scene and camera (`src/scene_setup.py`, `src/camera.py`)
A Franka Panda, a table, a cube and a duck are loaded into PyBullet. The camera returns an RGB image, a depth buffer, a segmentation mask and the view/projection matrices.

### 2. RGB-D validation (`src/rgbd_validator.py`)
Before anything is trusted, each frame is checked for missing data, NaN/Inf depth, zero-size depth arrays, and too few valid depth pixels. `inject_corruption()` can deliberately corrupt a frame to show the validator rejecting it.

### 3. Detection (`src/detect_seg.py`)
In simulation, objects are detected from PyBullet's segmentation mask, which gives an exact bounding box per object. This is used because a COCO-pretrained YOLO model does not reliably recognise simple synthetic shapes like the PyBullet cube and duck. A colour-threshold fallback is available in `src/detect_fallback.py`.

### 4. Grasp point (`src/grasp_planner.py`)
Each pixel of the target's mask is back-projected to world coordinates using the depth buffer and the inverse view-projection matrix. The grasp point is the centre of the object's top surface.

### 5. Execution (`src/execute_grasp.py`)
The arm starts from a ready pose, then moves in small straight-line steps using inverse kinematics with joint limits: hover, descend, close the gripper, then lift slowly.

### 6. Dashboard (`app.py`)
A Gradio app runs a pretrained YOLOv8n detector on any uploaded image and draws boxes and labels. This stage works on real photos, unlike the simulation detector.

### 7. Safety layer (`safe_rgbd/`, driven by `demo_safe.py`)
A second, more rigorous perception path that sits in front of the grasp planner:

- **`validator.py`** — flags NaN/Inf/out-of-range/zero depth per pixel and measures depth "roughness" to catch noise, without needing ground truth.
- **`recovery.py`** — repairs recoverable pixels (e.g. inpainting small NaN/dropout regions) without ever touching pixels that were already valid, and assigns each repaired pixel a confidence that decays with distance from the nearest valid pixel. Frames that are too damaged are rejected instead of guessed at.
- **`depth.py`** — the correct nonlinear depth-buffer-to-metres conversion (see [Measured results](#measured-results-safe_rgbd-experiments), experiment 1).
- **`pointcloud.py`** — back-projects a depth image to a world-frame point cloud using the camera intrinsics.
- **`localization.py`** — turns an object mask + repaired depth + per-pixel confidence into a single 3D grasp point, and refuses to return one if confidence is too low.
- **`pipeline.py`** (`SafePerception`) — strings validate → recover → convert → report into one call, producing a `SafeFrame` with an accept/reject decision, the repaired depth, a confidence map and a latency measurement.
- **`corruption.py`** — synthetic corruption injectors (nan, inf, zero_region, out_of_range, noise, depth_bias, constant_depth, flatten_rgb/depth, truncate_rgb/depth, mixed) used by the experiments and by `demo_safe.py --corrupt`.
- **`metrics.py`** — precision/recall/F1, MAE/RMSE, Wilson confidence intervals and latency stats used by `experiments/`.

`demo_safe.py` runs this path end-to-end against the same PyBullet scene as `main_demo.py`, so you can compare it directly with the raw pipeline described above.

---

## Project structure

```text
perception_to_grasp/
├── app.py                      # Gradio detection dashboard
├── demo_safe.py                # demo run through the safe_rgbd safety layer
├── requirements.txt
├── src/
│   ├── scene_setup.py          # PyBullet scene (robot, table, objects)
│   ├── camera.py               # RGB, depth and segmentation capture
│   ├── rgbd_validator.py       # frame validation + corruption injection
│   ├── detect.py               # YOLO detector (real images)
│   ├── detect_seg.py           # segmentation-mask detector (simulation)
│   ├── detect_fallback.py      # colour-threshold fallback
│   ├── visualize_detections.py # draw boxes and labels
│   ├── grasp_planner.py        # pixel + depth -> 3D grasp point
│   ├── execute_grasp.py        # IK motion + gripper control
│   ├── main_demo.py            # full pipeline
│   ├── record_demo.py          # record the run to MP4
│   ├── mp4_to_gif.py           # MP4 -> GIF
│   ├── test_validator.py
│   └── test_detection_pipeline.py
├── safe_rgbd/                  # validation + repair + confidence-weighted localization
│   ├── validator.py            # per-pixel invalid-depth detection + roughness (noise) check
│   ├── recovery.py             # traceable repair with distance-decayed confidence
│   ├── depth.py                # correct nonlinear depth-buffer -> metres conversion
│   ├── pointcloud.py           # depth -> camera/world-frame point cloud
│   ├── localization.py         # mask + depth + confidence -> 3D grasp point (or reject)
│   ├── pipeline.py             # SafePerception: validate -> recover -> convert -> report
│   ├── corruption.py           # synthetic corruption injectors
│   ├── metrics.py              # precision/recall/F1, MAE/RMSE, Wilson CI, latency stats
│   └── report.py               # FrameReport / Status (VALID / RECOVERABLE / UNRECOVERABLE)
├── experiments/                 # reproduces results/*
│   ├── sim_env.py               # shared headless PyBullet scene helper
│   ├── exp1_depth_ground_truth.py
│   ├── exp2_corruption_benchmark.py
│   ├── exp3_grasp_trials.py
│   ├── exp4_latency.py
│   └── make_figures.py          # regenerates results/figures from the CSVs
├── results/
│   ├── tables/                  # CSV/JSON output of each experiment
│   └── figures/                 # depth_error_vs_severity, grasp_success, pointcloud, recovery_demo
├── examples/
│   └── reshape_vs_resize.py     # why reshape != resize for a raw sensor buffer
├── tests/                        # 57 unit tests, incl. safe_rgbd
└── assets/
    ├── screenshots/
    └── videos/
```

---

## Known limitations

- **Simulation detection uses ground-truth segmentation.** The detector in the PyBullet run reads the simulator's segmentation mask, which is not available on a real robot. A real deployment would need a trained detector or segmenter here.
- **The YOLO dashboard is separate from the simulation.** `app.py` uses COCO-pretrained YOLOv8n on uploaded photos. It does not detect the PyBullet cube or duck reliably.
- **Grasping is basic.** Grasp points come from the centre of the top surface, with a top-down gripper orientation. There is no grasp-quality scoring, collision checking or object-part reasoning.
- **One object, one run.** The demo targets the first detected object (the cube). The duck is detected but not picked, and there is no place step. The successful pick was verified in a recorded run on fixed object positions, so results on other layouts or other objects are untested.
- **Single-shot planning.** The scene is captured once. There is no re-detection during the approach and no recovery if a grasp fails.
- **`safe_rgbd`'s validator has one confirmed blind spot: smooth depth bias.** It catches missing, invalid and noisy depth (measured 100% detection across 11 corruption types), but a smooth multiplicative bias with no NaNs, no discontinuities and no out-of-range values is never flagged — it is silently accepted at 100% and reaches the grasp planner with growing metric error (see [Measured results](#measured-results-safe_rgbd-experiments), experiment 2). This is the one condition where the safe pipeline performs no better than the raw one.
- **The reject thresholds are tuned, not derived.** `--max-invalid` and `--roughness-reject` in `experiments/exp3_grasp_trials.py` trade recall for safety (looser thresholds recover more corrupted frames but let more bad ones through); the values used for the headline numbers are defaults, not the result of a principled search.
- **All measurements are simulation-only, single-machine, and single-object (the cube).** Latency numbers in particular are for one Windows/Intel machine and are not validated on target robot hardware.

## Possible next steps

- Swap in a detector trained on the simulated objects.
- Turn the object-height check after the lift (and the finger opening after closing) into automatic grasp verification with retry.
- Pick the duck as well, and place objects at a target location.
- Add pose-aware grasp orientation instead of a fixed top-down grasp.
- Add a depth-bias detector to `safe_rgbd.validator` (e.g. cross-checking depth against a second cue, or a learned residual check) to close the one confirmed blind spot from experiment 2.
- Search the `--max-invalid` / `--roughness-reject` thresholds systematically instead of using the defaults, and report the resulting precision/recall trade-off curve.


