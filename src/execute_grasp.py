import time
import numpy as np
import pybullet as p

EE_INDEX = 11          
ARM_JOINTS = list(range(7))
FINGERS = [9, 10]
DOWN_ORN = [0, 1, 0, 0]  

HOME_POSE = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
LOWER = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
UPPER = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]

LOWER9 = LOWER + [0.0, 0.0]
UPPER9 = UPPER + [0.04, 0.04]
RANGES9 = [u - l for l, u in zip(LOWER9, UPPER9)]
REST9 = HOME_POSE + [0.04, 0.04]


def _step(n=1):
    for _ in range(n):
        p.stepSimulation()
        time.sleep(1 / 240)


def _ee_pos(robot_id):
    return np.array(p.getLinkState(robot_id, EE_INDEX)[4])


def _ik(robot_id, pos):
    try:
        return p.calculateInverseKinematics(
            robot_id, EE_INDEX, list(pos), DOWN_ORN,
            lowerLimits=LOWER9, upperLimits=UPPER9,
            jointRanges=RANGES9, restPoses=REST9,
            maxNumIterations=200, residualThreshold=1e-4)
    except Exception:
        return p.calculateInverseKinematics(
            robot_id, EE_INDEX, list(pos), DOWN_ORN,
            maxNumIterations=200, residualThreshold=1e-4)


def _set_arm_targets(robot_id, q):
    for j in ARM_JOINTS:
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                targetPosition=q[j], force=240,
                                maxVelocity=1.5)


def init_arm(robot_id):
    """Snap the arm into a sensible ready pose instead of swinging from upright."""
    for j, q in zip(ARM_JOINTS, HOME_POSE):
        p.resetJointState(robot_id, j, q)
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                targetPosition=q, force=240)
    _step(60)


def move_to(robot_id, target_pos, spacing=0.02, steps_per_wp=12,
            settle_steps=120, tol=0.01):
    """Move the end effector in a straight line using small waypoints."""
    start = _ee_pos(robot_id)
    target = np.array(target_pos, dtype=float)
    n_wp = max(int(np.linalg.norm(target - start) / spacing), 1)

    for k in range(1, n_wp + 1):
        wp = start + (target - start) * k / n_wp
        _set_arm_targets(robot_id, _ik(robot_id, wp))
        _step(steps_per_wp)

    for _ in range(settle_steps):
        if np.linalg.norm(_ee_pos(robot_id) - target) < tol:
            break
        _step(1)

    err = np.linalg.norm(_ee_pos(robot_id) - target)
    print(f"  moved to {np.round(target, 3)}  (error {err*100:.1f} cm)")


def control_gripper(robot_id, open_gripper=True, steps=120):
    target = 0.04 if open_gripper else 0.0
    for f in FINGERS:
        p.setJointMotorControl2(robot_id, f, p.POSITION_CONTROL,
                                targetPosition=target, force=100)
    _step(steps)


def execute_pick(robot_id, grasp_point, table_top_z=-0.025):
    x, y, z_top = [float(v) for v in grasp_point]

    grasp_z = max(z_top - 0.02, table_top_z + 0.015)
    hover_z = grasp_z + 0.20
    lift_z = grasp_z + 0.30

    for f in FINGERS:
        p.changeDynamics(robot_id, f, lateralFriction=2.0)

    print("  initialising arm pose...")
    init_arm(robot_id)

    print("  opening gripper...")
    control_gripper(robot_id, open_gripper=True)

    print("  approaching...")
    move_to(robot_id, [x, y, hover_z])

    print("  descending...")
    move_to(robot_id, [x, y, grasp_z])

    print("  closing gripper...")
    control_gripper(robot_id, open_gripper=False)

    print("  lifting...")
    move_to(robot_id, [x, y, lift_z])

    print(f"Pick sequence complete. Target was: {grasp_point}")