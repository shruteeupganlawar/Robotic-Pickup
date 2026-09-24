import numpy as np

class SafeReshaper:
    def __init__(self, height: int, width: int, rgb_channels: int=4):
        self.height = height
        self.width = width
        self.rgb_channels = rgb_channels

    def reshape_rgb(self, rgb):
        a = np.asarray(rgb)
        h, w, c = self.height, self.width, self.rgb_channels
        if a.shape == (h, w, c):
            return a
        if a.ndim == 1 and a.size == h*w*c:
            return a.reshape(h,w,c)
        raise ValueError(f"Cannot reshape RGB buffer of shape {a.shape} (size {a.size}) to " f"({h}, {w}, {c}): expected {h*w*c} elements")

    def reshape_depth(self, depth):
        a = np.asarray(depth)
        h, w = self.height, self.width
        if a.shape == (h,w):
            return a
        if a.ndim == 1 and a.size == h*w:
            return a.reshape(h,w)
        raise ValueError(f"Cannot reshape depth buffer of shape {a.shape} (size {a.size}) to " f"({h}, {w}): expected {h*w} elements")
    