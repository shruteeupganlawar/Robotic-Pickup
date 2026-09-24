import numpy as np
import pytest

from safe_rgbd import DepthConverter


def test_endpoints():
    c = DepthConverter(0.1, 3.0)
    assert c.buffer_to_metric(np.array([0.0]))[0] == pytest.approx(0.1, rel=1e-5)
    assert c.buffer_to_metric(np.array([1.0]))[0] == pytest.approx(3.0, rel=1e-5)


def test_roundtrip():
    c = DepthConverter(0.1, 3.0)
    z = np.array([0.2, 0.5, 1.0, 2.0, 2.9], dtype=np.float32)
    assert np.allclose(c.buffer_to_metric(c.metric_to_buffer(z)), z, rtol=1e-4)


def test_nonlinear_differs_from_linear():
    c = DepthConverter(0.1, 3.0)
    d = np.array([0.5])
    assert c.buffer_to_metric(d)[0] == pytest.approx(0.1 * 3.0 / (3.0 - 0.5 * 2.9), rel=1e-5)
    assert abs(c.buffer_to_metric(d)[0] - c.linear_to_metric(d)[0]) > 1.0   # very different


def test_from_proj_matrix(proj_matrix):
    c = DepthConverter.from_proj_matrix(proj_matrix)
    assert c.near == pytest.approx(0.1, rel=1e-4)
    assert c.far == pytest.approx(3.0, rel=1e-4)


def test_out_of_range_raises():
    c = DepthConverter(0.1, 3.0)
    with pytest.raises(ValueError):
        c.buffer_to_metric(np.array([1.2]))


def test_nan_passes_through():
    c = DepthConverter(0.1, 3.0)
    out = c.buffer_to_metric(np.array([np.nan, 0.5]))
    assert np.isnan(out[0]) and np.isfinite(out[1])


def test_bad_planes():
    with pytest.raises(ValueError):
        DepthConverter(0.0, 3.0)
    with pytest.raises(ValueError):
        DepthConverter(1.0, 0.5)
