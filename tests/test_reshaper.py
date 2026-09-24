import numpy as np
import pytest

from safe_rgbd import SafeReshaper


def test_flatten_roundtrip_rgb(hw, clean_rgb):
    r = SafeReshaper(*hw)
    out = r.reshape_rgb(clean_rgb.flatten())
    assert out.shape == (*hw, 4)
    assert np.array_equal(out, clean_rgb)


def test_flatten_roundtrip_depth(hw, clean_depth):
    r = SafeReshaper(*hw)
    out = r.reshape_depth(clean_depth.flatten())
    assert out.shape == hw
    assert np.array_equal(out, clean_depth)


def test_rgb_three_channels_inferred(hw):
    r = SafeReshaper(*hw, rgb_channels=3)
    assert r.reshape_rgb(np.zeros(hw[0] * hw[1] * 3, np.uint8)).shape == (*hw, 3)


def test_channel_count_is_never_guessed(hw, clean_rgb):
    r = SafeReshaper(*hw, rgb_channels=4)
    three_channel_sized = clean_rgb.flatten()[: hw[0] * hw[1] * 3]
    with pytest.raises(ValueError):
        r.reshape_rgb(three_channel_sized)


def test_wrong_size_rgb_raises(hw, clean_rgb):
    r = SafeReshaper(*hw)
    with pytest.raises(ValueError):
        r.reshape_rgb(clean_rgb.flatten()[:-100])


def test_wrong_size_depth_raises(hw, clean_depth):
    r = SafeReshaper(*hw)
    with pytest.raises(ValueError):
        r.reshape_depth(clean_depth.flatten()[:-1])


def test_already_shaped_is_passthrough(hw, clean_depth):
    r = SafeReshaper(*hw)
    assert r.reshape_depth(clean_depth) is clean_depth
