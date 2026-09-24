import numpy as np


class DepthConverter:
    def __init__(self, near: float, far: float):
        if near <= 0:
            raise ValueError("Near plane must be positive.")
        if far <= near:
            raise ValueError("Far plane must be greater than near plane.")
        self.near = float(near)
        self.far = float(far)

    @classmethod
    def from_proj_matrix(cls, proj_matrix):
        """Recover near/far from a PyBullet (column-major, 16-value) projection matrix."""
        m = np.asarray(proj_matrix, dtype=np.float64).ravel()
        a, b = m[10], m[14]         
        near = b / (a - 1.0)
        far = b / (a + 1.0)
        return cls(near, far)

    def _check_range(self, d):
        finite = d[np.isfinite(d)]
        if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
            raise ValueError("Depth-buffer values must lie in [0, 1].")

    def buffer_to_metric(self, depth):
        """Nonlinear depth buffer -> metres along the optical axis. NaNs pass through."""
        d = np.asarray(depth, dtype=np.float64)
        self._check_range(d)
        n, f = self.near, self.far
        return (f * n / (f - d * (f - n))).astype(np.float32)

    def metric_to_buffer(self, z):
        z = np.asarray(z, dtype=np.float64)
        n, f = self.near, self.far
        return ((f - f * n / z) / (f - n)).astype(np.float32)

    def linear_to_metric(self, depth):
        """WRONG for PyBullet buffers. Only for comparison experiments."""
        d = np.asarray(depth, dtype=np.float64)
        self._check_range(d)
        return (self.near + d * (self.far - self.near)).astype(np.float32)
