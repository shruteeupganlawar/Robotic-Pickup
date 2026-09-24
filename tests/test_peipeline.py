import numpy as np

from safe_rgbd import SafePerception, localize_object
from safe_rgbd.corruption import flatten, inject_nan, truncate, zero_region


def make(hw, proj):
    return SafePerception(hw[0], hw[1], proj)


def test_clean_frame_accepted_no_actions(hw, proj_matrix, clean_rgb, clean_depth):
    f = make(hw, proj_matrix).process(clean_rgb, clean_depth)
    assert f.accepted and f.actions == [] and f.report.status.value == "VALID"
    assert f.depth_metric.shape == hw
    assert np.all(f.confidence == 1.0)


def test_flat_frame_recovered(hw, proj_matrix, clean_rgb, clean_depth):
    f = make(hw, proj_matrix).process(flatten(clean_rgb), flatten(clean_depth))
    assert f.accepted
    assert f.actions == ["reshape_rgb", "reshape_depth"]


def test_nan_frame_repaired_no_nan_left(hw, proj_matrix, clean_rgb, clean_depth):
    bad = inject_nan(clean_depth, 0.1, np.random.default_rng(0))
    f = make(hw, proj_matrix).process(clean_rgb, bad)
    assert f.accepted and np.isfinite(f.depth_metric).all()
    assert f.confidence.min() < 1.0


def test_truncated_frame_rejected(hw, proj_matrix, clean_rgb, clean_depth):
    f = make(hw, proj_matrix).process(clean_rgb, truncate(clean_depth, 0.1))
    assert not f.accepted and f.reason


def test_huge_hole_rejected(hw, proj_matrix, clean_rgb, clean_depth):
    bad = zero_region(clean_depth, 0.5, np.random.default_rng(0), center=(320, 240))
    assert not make(hw, proj_matrix).process(clean_rgb, bad).accepted


def test_localize_refuses_low_confidence(hw, proj_matrix, view_matrix, clean_rgb, clean_depth):
    sp = make(hw, proj_matrix)
    mask = np.zeros(hw, bool)
    mask[220:260, 300:340] = True
    bad = zero_region(clean_depth, 0.10, np.random.default_rng(0), center=(320, 240))
    f = sp.process(clean_rgb, bad)
    assert f.accepted
    loc = localize_object(mask, f.depth_metric, f.confidence, view_matrix, sp.intrinsics)
    assert not loc.ok and "confidence" in loc.reason


def test_localize_ok_on_clean(hw, proj_matrix, view_matrix, clean_rgb, clean_depth):
    sp = make(hw, proj_matrix)
    mask = np.zeros(hw, bool)
    mask[220:260, 300:340] = True
    f = sp.process(clean_rgb, clean_depth)
    loc = localize_object(mask, f.depth_metric, f.confidence, view_matrix, sp.intrinsics)
    assert loc.ok and loc.point.shape == (3,) and np.isfinite(loc.point).all()
