import cv2
import numpy as np
import pybullet as p
from src.main_demo import run_demo

WIDTH, HEIGHT, FPS = 640, 480, 30
CAPTURE_EVERY = 8  

writer = cv2.VideoWriter(
    "assets/videos/demo.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (WIDTH, HEIGHT),
)

view = p.computeViewMatrixFromYawPitchRoll([0.4, 0, 0.2], 1.2, 50, -30, 0, 2)
proj = p.computeProjectionMatrixFOV(60, WIDTH / HEIGHT, 0.1, 5.0)

_original_step = p.stepSimulation
_count = 0


def _step_and_capture(*args, **kwargs):
    global _count
    _original_step(*args, **kwargs)
    _count += 1
    if _count % CAPTURE_EVERY == 0:
        _, _, rgb, _, _ = p.getCameraImage(
            WIDTH, HEIGHT, view, proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        frame = np.reshape(rgb, (HEIGHT, WIDTH, 4))[:, :, :3].astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


p.stepSimulation = _step_and_capture

run_demo()
writer.release()
print("Video saved to assets/videos/demo.mp4")