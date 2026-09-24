import numpy as np
import pytest

from safe_rgbd.corruption import CORRUPTION_NAMES, apply_corruption, inject_nan, zero_region


def test_nan_fraction(clean_depth):
    out = inject_nan(clean_depth, 0.10, np.random.default_rng(0))
    assert np.isnan(out).mean() == pytest.approx(0.10, abs=0.001)


def test_inputs_not_modified(clean_rgb, clean_depth):
    rgb0, d0 = clean_rgb.copy(), clean_depth.copy()
    rng = np.random.default_rng(0)
    for name in CORRUPTION_NAMES:
        apply_corruption(name, clean_rgb, clean_depth, 0.25, rng)
    assert np.array_equal(clean_rgb, rgb0)
    assert np.array_equal(clean_depth, d0)


def test_deterministic_with_seed(clean_rgb, clean_depth):
    a = apply_corruption("nan", clean_rgb, clean_depth, 0.1, np.random.default_rng(5))[1]
    b = apply_corruption("nan", clean_rgb, clean_depth, 0.1, np.random.default_rng(5))[1]
    assert np.array_equal(a, b, equal_nan=True)


def test_zero_region_area(clean_depth):
    out = zero_region(clean_depth, 0.25, np.random.default_rng(0), center=(320, 240))
    assert (out == 0).mean() == pytest.approx(0.25, abs=0.01)


def test_flatten_and_truncate_shapes(hw, clean_rgb, clean_depth):
    rng = np.random.default_rng(0)
    assert apply_corruption("flatten_rgb", clean_rgb, clean_depth, 0.1, rng)[0].shape == (hw[0] * hw[1] * 4,)
    assert apply_corruption("flatten_depth", clean_rgb, clean_depth, 0.1, rng)[1].shape == (hw[0] * hw[1],)
    assert apply_corruption("truncate_depth", clean_rgb, clean_depth, 0.1, rng)[1].size < hw[0] * hw[1]


def test_unknown_raises(clean_rgb, clean_depth):
    with pytest.raises(ValueError):
        apply_corruption("banana", clean_rgb, clean_depth, 0.1, np.random.default_rng(0))
