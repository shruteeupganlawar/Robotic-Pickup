import numpy as np

from safe_rgbd import FrameValidator, Status
from safe_rgbd.corruption import (constant_depth, flatten, inject_inf, inject_nan,
                                  out_of_range, truncate, zero_region)

RNG = lambda: np.random.default_rng(0)


def make(hw, **kw):
    return FrameValidator(*hw, **kw)


def test_clean_frame_valid(hw, clean_rgb, clean_depth):
    rep = make(hw).validate(clean_rgb, clean_depth)
    assert rep.status == Status.VALID
    assert not rep.issues_detected


def test_flat_rgb_recoverable(hw, clean_rgb, clean_depth):
    rep = make(hw).validate(flatten(clean_rgb), clean_depth)
    assert rep.status == Status.RECOVERABLE
    assert "reshape_rgb" in rep.actions


def test_flat_depth_recoverable(hw, clean_rgb, clean_depth):
    rep = make(hw).validate(clean_rgb, flatten(clean_depth))
    assert rep.status == Status.RECOVERABLE
    assert "reshape_depth" in rep.actions


def test_truncated_rgb_unrecoverable(hw, clean_rgb, clean_depth):
    rep = make(hw).validate(truncate(clean_rgb, 0.1), clean_depth)
    assert rep.status == Status.UNRECOVERABLE


def test_truncated_depth_unrecoverable(hw, clean_rgb, clean_depth):
    rep = make(hw).validate(clean_rgb, truncate(clean_depth, 0.1))
    assert rep.status == Status.UNRECOVERABLE


def test_truncated_rgba_that_looks_like_rgb_is_rejected(hw, clean_rgb, clean_depth):
    rep = make(hw).validate(truncate(clean_rgb, 0.25), clean_depth)
    assert rep.status == Status.UNRECOVERABLE


def test_depth_bias_is_a_known_blind_spot(hw, clean_rgb, clean_depth):
    """Plausible-but-wrong depth passes structural checks. Documented limitation."""
    from safe_rgbd.corruption import depth_bias
    rep = make(hw).validate(clean_rgb, depth_bias(clean_depth, 0.5))
    assert rep.status == Status.VALID and not rep.issues_detected


def test_small_nan_recoverable(hw, clean_rgb, clean_depth):
    rep = make(hw).validate(clean_rgb, inject_nan(clean_depth, 0.05, RNG()))
    assert rep.status == Status.RECOVERABLE
    assert "inpaint_depth" in rep.actions
    assert 0.04 < rep.stats["nan_frac"] < 0.06


def test_large_nan_unrecoverable(hw, clean_rgb, clean_depth):
    rep = make(hw).validate(clean_rgb, inject_nan(clean_depth, 0.5, RNG()))
    assert rep.status == Status.UNRECOVERABLE


def test_inf_detected(hw, clean_rgb, clean_depth):
    rep = make(hw).validate(clean_rgb, inject_inf(clean_depth, 0.02, RNG()))
    assert rep.status == Status.RECOVERABLE
    assert rep.stats["inf_frac"] > 0


def test_zero_region_detected(hw, clean_rgb, clean_depth):
    rep = make(hw).validate(clean_rgb, zero_region(clean_depth, 0.05, RNG()))
    assert rep.status == Status.RECOVERABLE
    assert rep.stats["zero_frac"] > 0


def test_out_of_range_detected(hw, clean_rgb, clean_depth):
    rep = make(hw).validate(clean_rgb, out_of_range(clean_depth, 0.02, RNG()))
    assert rep.stats["out_of_range_frac"] > 0
    assert rep.status == Status.RECOVERABLE


def test_constant_depth_rejected_by_default(hw, clean_rgb, clean_depth):
    rep = make(hw).validate(clean_rgb, constant_depth(clean_depth))
    assert rep.status == Status.UNRECOVERABLE


def test_constant_depth_can_be_warning_only(hw, clean_rgb, clean_depth):
    rep = make(hw, reject_constant_depth=False).validate(clean_rgb, constant_depth(clean_depth))
    assert rep.status == Status.VALID
    assert rep.warnings


def test_none_inputs(hw, clean_rgb, clean_depth):
    assert make(hw).validate(None, clean_depth).status == Status.UNRECOVERABLE
    assert make(hw).validate(clean_rgb, None).status == Status.UNRECOVERABLE


def test_integer_depth_rejected(hw, clean_rgb):
    rep = make(hw).validate(clean_rgb, np.zeros(hw, dtype=np.uint16))
    assert rep.status == Status.UNRECOVERABLE


def test_validator_does_not_modify_inputs(hw, clean_rgb, clean_depth):
    bad = inject_nan(clean_depth, 0.05, RNG())
    before = bad.copy()
    make(hw).validate(clean_rgb, bad)
    assert np.array_equal(bad, before, equal_nan=True)


def test_speckle_noise_flagged_after_calibration(hw, clean_rgb, clean_depth):
    from safe_rgbd.corruption import gaussian_noise
    v = make(hw)
    v.calibrate([clean_depth])
    assert v.validate(clean_rgb, clean_depth).status == Status.VALID
    noisy = gaussian_noise(clean_depth, 0.5, RNG())
    rep = v.validate(clean_rgb, noisy)
    assert rep.warnings and "median_filter" in rep.actions


def test_roughness_reject_threshold(hw, clean_rgb, clean_depth):
    from safe_rgbd.corruption import gaussian_noise
    v = make(hw, roughness_reject=1e-3)
    v.calibrate([clean_depth])
    mild = v.validate(clean_rgb, gaussian_noise(clean_depth, 0.05, RNG()))     # sigma 5e-4
    heavy = v.validate(clean_rgb, gaussian_noise(clean_depth, 0.5, RNG()))     # sigma 5e-3
    assert mild.status == Status.RECOVERABLE
    assert heavy.status == Status.UNRECOVERABLE


def test_noise_is_still_flagged_when_frame_also_has_holes(hw, clean_rgb, clean_depth):
    from safe_rgbd.corruption import gaussian_noise
    v = make(hw)
    v.calibrate([clean_depth])
    bad = inject_nan(gaussian_noise(clean_depth, 0.5, RNG()), 0.02, RNG())
    rep = v.validate(clean_rgb, bad)
    assert any("noise" in w for w in rep.warnings)


def test_holes_alone_do_not_look_like_noise(hw, clean_rgb, clean_depth):
    v = make(hw)
    v.calibrate([clean_depth])
    rep = v.validate(clean_rgb, inject_nan(clean_depth, 0.05, RNG()))
    assert not any("noise" in w for w in rep.warnings)
