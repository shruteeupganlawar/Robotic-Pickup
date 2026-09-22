# test_validator.py
from src.scene_setup import setup_scene
from src.camera import get_camera_image
from src.rgbd_validator import validate_frame, inject_corruption, FrameValidationError

setup_scene(gui=False)
rgb, depth, _, _, _ = get_camera_image()

print("Testing clean frame:")
try:
    validate_frame(rgb, depth)
    print("  PASSED — frame is valid")
except FrameValidationError as e:
    print(f"  FAILED — {e}")

print("\nTesting corrupted frame (NaN injection):")
bad_depth = inject_corruption(depth, mode="nan")
try:
    validate_frame(rgb, bad_depth)
    print("  PASSED — frame is valid")
except FrameValidationError as e:
    print(f"  CORRECTLY REJECTED — {e}")

print("\nTesting corrupted frame (all-zero depth):")
zero_depth = inject_corruption(depth, mode="zero")
try:
    validate_frame(rgb, zero_depth)
    print("  PASSED — frame is valid")
except FrameValidationError as e:
    print(f"  CORRECTLY REJECTED — {e}")
