import pybullet as p
import numpy as np


def get_camera_image(cam_target=[0.5, 0, 0.1], cam_distance=1.0,
                     cam_yaw=50, cam_pitch=-35, width=640, height=480):
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target,
        distance=cam_distance,
        yaw=cam_yaw,
        pitch=cam_pitch,
        roll=0,
        upAxisIndex=2
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=width / height, nearVal=0.1, farVal=3.0
    )
    _, _, rgb_img, depth_img, seg_img = p.getCameraImage(
        width, height, view_matrix, proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    rgb_array = np.reshape(rgb_img, (height, width, 4))[:, :, :3]
    depth_array = np.reshape(depth_img, (height, width))
    seg_array = np.reshape(seg_img, (height, width))
    return (rgb_array.astype(np.uint8), depth_array,
            view_matrix, proj_matrix, seg_array)


if __name__ == "__main__":
    from scene_setup import setup_scene
    import matplotlib.pyplot as plt

    setup_scene(gui=False)
    rgb, depth, _, _, seg = get_camera_image()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[1].imshow(depth, cmap="viridis")
    axes[1].set_title("Depth")
    axes[2].imshow(seg)
    axes[2].set_title("Segmentation")
    plt.savefig("assets/screenshots/camera_test.png")
    print("Saved assets/screenshots/camera_test.png")