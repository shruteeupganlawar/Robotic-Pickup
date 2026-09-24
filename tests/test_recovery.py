import numpy as np

from safe_rgbd import FrameValidator, Recovery
from safe_rgbd.corruption import flatten, inject_nan, truncate, zero_region


def run(hw, rgb, depth):
    rep = FrameValidator(*hw).validate(rgb, depth)
    return rep, Recovery(*hw).repair(rgb, depth, rep)


def test_nan_filled_and_valid_pixels_untouched(hw, clean_rgb, clean_depth):
    bad = inject_nan(clean_depth, 0.05, np.random.default_rng(0))
    rep, res = run(hw, clean_rgb, bad)
    assert res.accepted
    assert not np.isnan(res.depth).any()
    ok = ~np.isnan(bad)
    assert np.array_equal(res.depth[ok], bad[ok])      # never touches valid pixels


def test_repair_is_close_to_truth(hw, clean_rgb, clean_depth):
    bad = inject_nan(clean_depth, 0.05, np.random.default_rng(0))
    _, res = run(hw, clean_rgb, bad)
    mask = np.isnan(bad)
    assert np.abs(res.depth[mask] - clean_depth[mask]).mean() < 5e-4


def test_confidence_valid_is_one_repaired_is_lower(hw, clean_rgb, clean_depth):
    bad = zero_region(clean_depth, 0.05, np.random.default_rng(0), center=(320, 240))
    _, res = run(hw, clean_rgb, bad)
    mask = bad == 0
    assert np.all(res.confidence[~mask] == 1.0)
    assert res.confidence[mask].max() <= 0.6 + 1e-6
    assert res.confidence[240, 320] < res.confidence[mask].max()   # deeper in the hole = lower


def test_flat_buffers_recovered_exactly(hw, clean_rgb, clean_depth):
    _, res = run(hw, flatten(clean_rgb), flatten(clean_depth))
    assert res.accepted
    assert np.array_equal(res.depth, clean_depth)
    assert np.array_equal(res.rgb, clean_rgb)


def test_unrecoverable_is_rejected(hw, clean_rgb, clean_depth):
    _, res = run(hw, clean_rgb, truncate(clean_depth, 0.1))
    assert not res.accepted and res.reason
