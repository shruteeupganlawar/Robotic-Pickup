import numpy as np
import pybullet as p
import pybullet_data

WIDTH, HEIGHT = 640, 480
NEAR, FAR, FOV = 0.1, 3.0, 60
CAM = dict(target=[0.5, 0, 0.1], distance=1.0, yaw=50, pitch=-35)

_connected = False


def connect(gui=False):
    global _connected
    if not _connected:
        p.connect(p.GUI if gui else p.DIRECT)
        _connected = True


def random_cube_xy(rng):
    return float(rng.uniform(0.45, 0.55)), float(rng.uniform(0.06, 0.14))


def build_scene(cube_xy=(0.5, 0.1), settle_steps=240):
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf", basePosition=[0, 0, -0.65])
    p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, basePosition=[0, 0, 0])
    cube = p.loadURDF("cube_small.urdf", basePosition=[cube_xy[0], cube_xy[1], 0.0])
    duck = p.loadURDF("duck_vhacd.urdf", basePosition=[0.45, -0.1, 0.0], globalScaling=1.2)
    p.changeDynamics(cube, -1, mass=0.1, lateralFriction=1.5)
    for _ in range(settle_steps):
        p.stepSimulation()
    return {"robot": robot, "cube": cube, "duck": duck}


def camera_matrices():
    view = p.computeViewMatrixFromYawPitchRoll(
        CAM["target"], CAM["distance"], CAM["yaw"], CAM["pitch"], 0, 2)
    proj = p.computeProjectionMatrixFOV(FOV, WIDTH / HEIGHT, NEAR, FAR)
    return view, proj


def capture():
    """Returns rgb (H,W,4) uint8, depth (H,W) float32 buffer, seg (H,W) int32, view, proj."""
    view, proj = camera_matrices()
    _, _, rgb, depth, seg = p.getCameraImage(
        WIDTH, HEIGHT, view, proj, renderer=p.ER_TINY_RENDERER)
    return (np.asarray(rgb, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4),
            np.asarray(depth, dtype=np.float32).reshape(HEIGHT, WIDTH),
            np.asarray(seg).reshape(HEIGHT, WIDTH), view, proj)
