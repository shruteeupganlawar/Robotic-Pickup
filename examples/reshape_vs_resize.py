import cv2
import numpy as np

from safe_rgbd import SafeReshaper

H, W = 480, 640
rgb = np.random.default_rng(0).integers(0, 256, (H, W, 4), dtype=np.uint8)
flat = rgb.flatten()
reshaper = SafeReshaper(H, W, rgb_channels=4)

print("1) flat buffer:", flat.shape, "-> reshape ->", reshaper.reshape_rgb(flat).shape)
print("   identical to the original:", np.array_equal(reshaper.reshape_rgb(flat), rgb))

try:
    reshaper.reshape_rgb(flat[:-100])
except ValueError as e:
    print("2) 100 elements missing -> blocked:", e)

resized = cv2.resize(rgb, (W // 2, H // 2))
print(f"3) resize {rgb.shape} -> {resized.shape}: it RESAMPLES, so pixel values are invented.")
print("   reshape only relabels the same elements; never use resize to 'fix' a bad buffer.")

trap = flat[: H * W * 3]     
print(f"4) truncated RGBA has {trap.size} elements = exactly H*W*3, the size of an RGB image!")
try:
    reshaper.reshape_rgb(trap)
except ValueError:
    print("   blocked, because the channel count comes from the sensor format, never from the size.")
