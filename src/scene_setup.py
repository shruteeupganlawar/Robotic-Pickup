import pybullet as p
import pybullet_data
import numpy as np


def setup_scene(gui=True):
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    plane_id = p.loadURDF("plane.urdf", basePosition=[0, 0, -0.65])
    table_id = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])
    robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True,
                          basePosition=[0, 0, 0])

    cube_id = p.loadURDF("cube_small.urdf", basePosition=[0.5, 0.1, 0.0])
    duck_id = p.loadURDF("duck_vhacd.urdf", basePosition=[0.45, -0.1, 0.0],
                         globalScaling=1.2)

    return {
        "robot": robot_id,
        "objects": {"cube": cube_id, "duck": duck_id},
        "table": table_id,
    }


if __name__ == "__main__":
    import time
    scene = setup_scene(gui=True)
    for _ in range(2000):
        p.stepSimulation()
        time.sleep(1 / 240)