import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lesson1  # noqa: E402
import lesson2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_belt_is_deterministic_and_matches_the_committed_png():
    a, b = lesson2.make_belt(), lesson2.make_belt()
    assert np.array_equal(a.rgb, b.rgb) and a.n_eggs == 12
    on_disk = lesson1.load_rgb(ROOT / "data" / "belt_synthetic.png")
    assert np.array_equal(on_disk, a.rgb)


def test_saucer_cleanup_numbers():
    rgb = lesson1.load_rgb(ROOT / "data" / "coffee.png")
    stages = lesson2.clean_saucer_mask(lesson1.saucer_mask_hue(rgb, 8))
    counts = {k: lesson2.count_components(v)[0] for k, v in stages.items()}
    assert counts == {"raw": 715, "opened": 14, "closed": 2, "largest": 1}
    assert round(stages["largest"].mean(), 3) == 0.347
    for name, (r, c) in lesson1.PATCHES.items():
        share = stages["largest"][r, c].mean()
        assert share > 0.9 if name.startswith("saucer") else share == 0.0, (name, share)


def test_kernel_sums_and_separability():
    assert lesson2.box_kernel(5).sum() == 1.0 and lesson2.SOBEL_X.sum() == 0.0
    k = cv2.getGaussianKernel(5, 1.0).ravel()
    assert np.allclose(np.round(k, 3), [0.054, 0.244, 0.403, 0.244, 0.054])


def test_second_derivative_kernel_exercise_1():
    row = np.array([[10, 10, 10, 50, 50, 50]], np.float32)
    out = lesson2.convolve(row, np.array([[1, -2, 1]], np.float32))
    assert out.ravel().tolist() == [0, 0, 40, -40, 0, 0]


def test_median_beats_blur_on_dead_pixels():
    belt = lesson2.make_belt()
    gray = cv2.cvtColor(belt.rgb, cv2.COLOR_RGB2GRAY)
    table = {name: (a, e) for name, a, e in lesson2.denoise_table(gray, belt.clean)}
    assert round(table["no filter"][0], 2) == 8.67
    assert table["median 5x5"][0] < table["Gaussian 5x5, sigma 1"][0] < table["no filter"][0]
    # Every blur hurts the outlines; the median hurts them least.
    for name in ("box 5x5", "Gaussian 5x5, sigma 1", "median 5x5"):
        assert table[name][1] > table["no filter"][1]
    assert table["median 5x5"][1] < table["box 5x5"][1]


def test_canny_blur_removes_more_than_thresholds():
    gray = cv2.cvtColor(lesson1.load_rgb(ROOT / "data" / "coffee.png"), cv2.COLOR_RGB2GRAY)
    base = lesson2.canny(gray, 50, 150).mean()
    assert round(base, 4) == 0.1304
    assert lesson2.canny(gray, 100, 200).mean() > lesson2.canny(gray, 50, 150, 2.0).mean()


def test_toy_morphology_table():
    stages = lesson2.morphology_stages(lesson2.toy_shape())
    on = {k: int(v.sum()) for k, v in stages.items()}
    assert on == {"input": 1807, "erode": 1337, "dilate": 2324, "open": 1785, "close": 1848}
    regions = {k: lesson2.count_components(v)[0] for k, v in stages.items()}
    assert regions == {"input": 4, "erode": 2, "dilate": 4, "open": 2, "close": 4}


def test_egg_count_is_exact_and_global_threshold_is_not():
    belt = lesson2.make_belt()
    res = lesson2.count_eggs(belt.rgb)
    assert res.n_eggs == belt.n_eggs == 12
    assert res.n_blobs == 10  # two touching pairs before the distance transform
    assert res.global_threshold == 76 and res.tophat_threshold == 34
    n_global, _, areas = lesson2.count_components(res.global_mask, 400)
    assert n_global == 7 and areas.max() > 100_000  # the lit half of the belt
    assert sorted(int(a) for a in res.areas)[-2:] == [2980, 3678]
