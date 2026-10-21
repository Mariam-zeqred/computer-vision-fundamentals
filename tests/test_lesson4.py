import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lesson1  # noqa: E402
import lesson2  # noqa: E402
import lesson4  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sets():
    return lesson4.make_patch_set(lesson4.N_TRAIN, seed=1), lesson4.make_patch_set(
        lesson4.N_VAL, seed=2
    )


@pytest.fixture(scope="module")
def trained(sets):
    """Everything the lesson trains, once per test session (about 20 s)."""
    train, val = sets
    template = lesson4.mean_template(train)
    ncc_thr, _ = lesson4.best_threshold(lesson4.ncc_scores(train.x, template), train.y)
    w, b, _ = lesson4.train_linear(train)
    net, history = lesson4.train_cnn(train, val)
    return template, ncc_thr, w, b, net, history


def test_patch_set_is_deterministic_and_balanced():
    a = lesson4.make_patch_set(200, seed=5)
    b = lesson4.make_patch_set(200, seed=5)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.kind, b.kind)
    assert a.x.shape == (200, 64, 64) and a.x.dtype == np.uint8
    assert a.y.sum() == 100 and set(a.kind) <= set(lesson4.KINDS)
    assert np.array_equal(a.y == 1, a.kind == "egg")
    assert not np.array_equal(a.x, lesson4.make_patch_set(200, seed=6).x)


def test_standardise_and_ncc_match_opencv(sets):
    train, _ = sets
    z = lesson4.standardise(train.x[:20])
    assert np.allclose(z.mean(axis=1), 0, atol=1e-4) and np.allclose(z.std(axis=1), 1, atol=1e-3)
    template = lesson4.mean_template(train)
    ours = lesson4.ncc_scores(train.x[:20], template)
    theirs = [
        cv2.matchTemplate(p.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)[0, 0]
        for p in train.x[:20]
    ]
    assert np.allclose(ours, theirs, atol=1e-4)


def test_best_threshold_picks_the_gap():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1])
    cut, acc = lesson4.best_threshold(scores, y)
    assert 0.3 < cut < 0.8 and acc == 1.0


def test_detect_peaks_merges_close_maxima_and_scoring_counts():
    m = np.zeros((200, 200), np.float32)
    m[50, 50], m[60, 60], m[150, 150] = 3.0, 2.0, 1.0  # the second is within NMS_RADIUS
    d = lesson4.detect_peaks(m, threshold=0.5)
    assert d.shape == (2, 3) and d[0, 2] == 3.0
    assert tuple(d[0, :2]) == (50 + 32, 50 + 32)  # x, y of the window centre
    s = lesson4.score_detections(d, [(82, 82), (182, 182), (10, 10)])
    assert (s.tp, s.fp, s.fn) == (2, 0, 1) and s.precision == 1.0 and s.recall == 2 / 3


def test_learned_kernel_recovers_sobel():
    rgb = lesson1.load_rgb(ROOT / "data" / "coffee.png")
    coffee = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    coffee -= coffee.mean()
    noise = np.random.default_rng(0).random(coffee.shape, dtype=np.float32) - 0.5
    for x, steps, tol in ((noise, 20, 1e-3), (coffee, 300, 0.25)):
        target = lesson4.correlate_valid(x, lesson2.SOBEL_X)
        k, losses = lesson4.learn_kernel(x, target, steps=steps)
        assert np.abs(k - lesson2.SOBEL_X).max() < tol
        assert losses[-1] < losses[0]
        assert np.abs(lesson4.solve_kernel(x, target) - lesson2.SOBEL_X).max() < 1e-6
    # The lesson's reason for the slow run: neighbouring pixels of a photograph are
    # nearly equal, so the curvature ratio is in the hundreds; for noise it is 1.
    ev_c = np.linalg.eigvalsh(lesson4.kernel_hessian(coffee))
    ev_n = np.linalg.eigvalsh(lesson4.kernel_hessian(noise))
    assert ev_c.max() / ev_c.min() > 500 and ev_n.max() / ev_n.min() < 1.1
    # Gradient descent on the photograph does get there, slowly.
    target = lesson4.correlate_valid(coffee, lesson2.SOBEL_X)
    k, _ = lesson4.learn_kernel(coffee, target, steps=1000)
    assert np.abs(k - lesson2.SOBEL_X).max() < 0.1


def test_linear_map_equals_per_patch_scores(trained):
    _, _, w, b, _, _ = trained
    gray = lesson4.gray_of(lesson2.make_belt())
    m = lesson4.linear_map(gray, w, b)
    assert m.shape == (480 - 63, 640 - 63)
    for i, j in ((0, 0), (100, 200), (300, 500), (416, 576)):
        direct = lesson4.linear_scores(gray[None, i : i + 64, j : j + 64], w, b)[0]
        assert abs(m[i, j] - direct) < 0.05


def test_patch_accuracies(sets, trained):
    """The table in lesson 4, section 4: labels separate the three detectors."""
    _, val = sets
    template, ncc_thr, w, b, net, history = trained
    ncc = lesson4.accuracy_by_kind(lesson4.ncc_scores(val.x, template), val, ncc_thr)
    lin = lesson4.accuracy_by_kind(lesson4.linear_scores(val.x, w, b), val, 0.0)
    cnn = lesson4.accuracy_by_kind(lesson4.cnn_scores(net, val.x), val, 0.0)
    assert 0.85 < ncc["all"] < 0.92 and ncc["label"] < 0.6
    assert 0.85 < lin["all"] < 0.93 and lin["label"] < 0.4
    assert cnn["all"] > 0.97 and cnn["label"] > 0.85
    assert history[-1][1] == cnn["all"]
    assert lesson4.n_parameters(net) == 6129


def test_cnn_is_fully_convolutional(trained):
    *_, net, _ = trained
    gray = lesson4.gray_of(lesson2.make_belt())
    m = lesson4.cnn_map(net, gray)
    assert m.shape == (105, 145)
    s = lesson4.CNN_STRIDE
    for i, j in ((0, 0), (30, 40), (104, 144)):
        direct = lesson4.cnn_scores(net, gray[None, s * i : s * i + 64, s * j : s * j + 64])[0]
        assert abs(m[i, j] - direct) < 1e-4


def test_frame_detections(trained):
    """Section 5's table: twelve eggs on every frame; the three labels fool the two
    linear detectors and not the CNN."""
    template, ncc_thr, w, b, net, _ = trained
    dets = lesson4.make_detectors(template, ncc_thr, w, b, net)
    for title, belt in lesson4.frames_for_scoring():
        gray = lesson4.gray_of(belt)
        for d in dets:
            s = lesson4.score_detections(lesson4.detect(d, gray), belt.centres)
            assert s.tp == 12 and s.fn == 0, (title, d.name)
            expected_fp = 3 if title.startswith("t2") and d.name != "CNN" else 0
            assert s.fp == expected_fp, (title, d.name, s)


def test_scale_breaks_detection_and_the_pyramid_restores_it(trained):
    template, ncc_thr, w, b, net, _ = trained
    dets = lesson4.make_detectors(template, ncc_thr, w, b, net)
    belt = lesson2.make_belt()
    g, centres = lesson4.rescaled(belt, 0.5)
    up = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    up_centres = [(2 * r, 2 * c) for r, c in centres]
    found = {}
    for d in dets:
        found[d.name] = lesson4.score_detections(lesson4.detect(d, g), centres).tp
        s = lesson4.score_detections(lesson4.detect(d, up), up_centres)
        assert s.tp == 12 and s.fp == 0, d.name
    assert found["template NCC"] <= 7 and found["learned filter"] <= 4
    assert found["CNN"] >= 8 and found["CNN"] > found["template NCC"]


def test_decorate_hook_leaves_the_default_belt_unchanged():
    plain = lesson2.make_belt(seed=21, shift=40, noise_seed=22)
    marked = lesson2.make_belt(seed=21, shift=40, noise_seed=22, decorate=lesson4.distractors)
    assert plain.centres == marked.centres
    diff = np.abs(plain.clean.astype(int) - marked.clean.astype(int)) > 0
    assert diff.any()
    # Only the labels and blobs changed, nowhere near an egg centre.
    ys, xs = np.nonzero(diff)
    for r, c in marked.centres:
        assert np.hypot(ys - r, xs - c).min() > 20
