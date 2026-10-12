import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lesson1  # noqa: E402

IMG = Path(__file__).resolve().parents[1] / "data" / "coffee.png"


def test_load_is_rgb_uint8():
    rgb = lesson1.load_rgb(IMG)
    assert rgb.shape == (400, 600, 3) and rgb.dtype == np.uint8
    # The image is warm: red must dominate on average, which it would not in BGR order.
    r, g, b = rgb.reshape(-1, 3).mean(axis=0)
    assert r > g > b


def test_uint8_wraps_and_fix_saturates():
    a = np.uint8([200])
    with np.errstate(over="ignore"):
        assert lesson1.brighten_wrong(a, 100)[0] == 44
    assert lesson1.brighten_right(a, 100)[0] == 255
    assert lesson1.brighten_right(np.uint8([10]), -30)[0] == 0


def test_manual_luma_matches_opencv_within_rounding():
    rgb = lesson1.load_rgb(IMG)
    diff = np.abs(lesson1.to_gray_manual(rgb).astype(int) - cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    assert diff.max() <= 1


def test_histogram_counts_every_pixel():
    rgb = lesson1.load_rgb(IMG)
    h = lesson1.histogram(rgb[..., 1])
    assert h.shape == (256,) and h.sum() == 400 * 600


def test_hue_valley_and_mask_on_the_lesson_image():
    rgb = lesson1.load_rgb(IMG)
    valley = lesson1.hue_valley(lesson1.to_hsv(rgb)[..., 0])
    assert valley == 8
    mask = lesson1.saucer_mask_hue(rgb, valley)
    # Saucer patches are almost entirely selected, wood patches almost not.
    for name, (r, c) in lesson1.PATCHES.items():
        share = mask[r, c].mean()
        if name.startswith("saucer"):
            assert share > 0.9, (name, share)
        else:
            assert share < 0.1, (name, share)


def test_percentile_stretch_uses_full_range():
    x = np.linspace(40, 90, 1000).astype(np.uint8)
    y = lesson1.percentile_stretch(x)
    assert y.min() == 0 and y.max() == 255


def test_exercise5_numbers():
    """Exercise 5's answer quotes these: a V-only stretch or equalisation barely moves
    the hue valley and the mask fraction, because H survives the uint8 round trip."""
    rgb = lesson1.load_rgb(IMG)
    h, s, v = cv2.split(lesson1.to_hsv(rgb))
    base = lesson1.saucer_mask_hue(rgb, 8).mean()
    assert round(base, 3) == 0.355
    results = {}
    for name, v2 in [("stretch", lesson1.percentile_stretch(v)), ("equalize", cv2.equalizeHist(v))]:
        back = cv2.cvtColor(cv2.merge([h, s, v2]), cv2.COLOR_HSV2RGB)
        valley = lesson1.hue_valley(lesson1.to_hsv(back)[..., 0])
        results[name] = (valley, round(lesson1.saucer_mask_hue(back, 8).mean(), 3))
    assert results == {"stretch": (7, 0.349), "equalize": (8, 0.359)}
