import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lesson6  # noqa: E402

# Everything here runs without the network, the COCO weights or a trained model:
# it pins the pieces of lesson 6 that are written out in NumPy, and the data set.


def test_ellipse_box_axis_aligned_and_rotated():
    assert np.allclose(lesson6.ellipse_box(100, 200, 30, 20, 0), [170, 80, 230, 120])
    assert np.allclose(lesson6.ellipse_box(100, 200, 30, 20, 90), [180, 70, 220, 130])
    x1, y1, x2, y2 = lesson6.ellipse_box(100, 200, 30, 20, 45)
    half = np.sqrt((30**2 + 20**2) / 2)  # both extents at 45 degrees
    assert np.isclose(x2 - x1, 2 * half) and np.isclose(y2 - y1, 2 * half)


def test_make_scene_is_deterministic_and_boxes_lie_inside_the_frame():
    a, b = lesson6.make_scene(1234), lesson6.make_scene(1234)
    assert np.array_equal(a.rgb, b.rgb) and np.array_equal(a.boxes, b.boxes)
    assert a.rgb.shape == (480, 640, 3) and a.rgb.dtype == np.uint8
    assert 2 <= a.n_eggs <= 12
    assert (
        (a.boxes[:, :2] >= 0).all()
        and (a.boxes[:, 2] <= 640).all()
        and (a.boxes[:, 3] <= 480).all()
    )
    wh = a.boxes[:, 2:] - a.boxes[:, :2]
    assert (wh >= 38).all() and (wh <= 69).all()  # half-axes 19-34 px
    assert not np.array_equal(a.rgb, lesson6.make_scene(1235).rgb)


def test_scene_eggs_are_brighter_than_the_belt_around_them():
    s = lesson6.make_scene(1000)
    gray = s.rgb.mean(axis=2)
    for x1, y1, x2, y2 in s.boxes.astype(int):
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        egg = gray[cy - 5 : cy + 5, cx - 5 : cx + 5].mean()
        ring = np.concatenate(
            [gray[y1 - 12 : y1 - 4, x1:x2].ravel(), gray[y2 + 4 : y2 + 12, x1:x2].ravel()]
        )
        assert egg > ring.mean() + 30


def test_labels_round_trip(tmp_path):
    s = lesson6.make_scene(2000)
    path = tmp_path / "x.txt"
    lesson6.write_labels(path, s.boxes, s.rgb.shape[:2])
    lines = path.read_text().splitlines()
    assert len(lines) == s.n_eggs and all(line.startswith("0 ") for line in lines)
    back = lesson6.read_labels(path, s.rgb.shape[:2])
    assert np.abs(back - s.boxes).max() < 0.01
    lesson6.write_labels(path, np.zeros((0, 4)), s.rgb.shape[:2])
    assert path.read_text() == "" and lesson6.read_labels(path, (480, 640)).shape == (0, 4)


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30], [0, 0, 5, 10]], np.float32)
    assert np.allclose(lesson6.iou_matrix(a, b), [[1.0, 25 / 175, 0.0, 0.5]])
    assert lesson6.iou_matrix(b, b).shape == (4, 4)


def test_nms_keeps_the_strongest_of_an_overlapping_group():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 10, 10]], np.float32)
    scores = np.array([0.5, 0.9, 0.7, 0.1], np.float32)
    assert lesson6.nms(boxes, scores, 0.5).tolist() == [1, 2]
    assert lesson6.nms(boxes, scores, 0.99).tolist() == [1, 2, 0]  # only the exact twin merges
    assert lesson6.nms(np.zeros((0, 4)), np.zeros(0)).shape == (0,)


def test_average_precision_by_hand():
    # Two truths in one image. Rank: TP, FP, TP -> precision 1, 1/2, 2/3 at recall 1/2, 1/2, 1.
    truths = [np.array([[0, 0, 10, 10], [50, 50, 60, 60]], np.float32)]
    dets = [
        np.array(
            [[0, 0, 10, 10, 0.9], [100, 100, 110, 110, 0.8], [51, 51, 61, 61, 0.7]], np.float32
        )
    ]
    scores, hits, n = lesson6.match_detections(dets, truths, 0.5)
    assert hits.tolist() == [True, False, True] and n == 2 and np.allclose(scores, [0.9, 0.8, 0.7])
    p, r = lesson6.precision_recall(hits, n)
    assert np.allclose(p, [1, 0.5, 2 / 3]) and np.allclose(r, [0.5, 0.5, 1])
    # 101-point COCO AP: precision 1 for recall levels <= 0.5 (51 of them), 2/3 for the other 50.
    assert np.isclose(lesson6.average_precision(p, r), (51 * 1 + 50 * 2 / 3) / 101)
    assert np.isclose(lesson6.ap_at(dets, truths, 0.5), (51 + 50 * 2 / 3) / 101)
    # A second detection on an already claimed truth is a false positive, not a second hit.
    twice = [np.array([[0, 0, 10, 10, 0.9], [0, 0, 10, 10, 0.8]], np.float32)]
    _, hits, _ = lesson6.match_detections(twice, truths, 0.5)
    assert hits.tolist() == [True, False]
    # Asking for a tighter overlap turns the slightly-off third box into a miss.
    assert lesson6.ap_at(dets, truths, 0.9) < lesson6.ap_at(dets, truths, 0.5)
    assert lesson6.ap_at([np.zeros((0, 5))], truths, 0.5) == 0.0


def test_counts_at_threshold():
    truths = [np.array([[0, 0, 10, 10]], np.float32), np.array([[0, 0, 10, 10]], np.float32)]
    dets = [
        np.array([[0, 0, 10, 10, 0.9], [30, 30, 40, 40, 0.3]], np.float32),
        np.array([[0, 0, 10, 10, 0.2]], np.float32),
    ]
    p, r, err = lesson6.counts_at(dets, truths, 0.5)
    assert (p, r, err) == (1.0, 0.5, 0.5)  # frame 1: 1 of 1; frame 2: 0 of 1
    p, r, err = lesson6.counts_at(dets, truths, 0.1)
    assert (p, r, err) == (2 / 3, 1.0, 0.5)  # frame 1 over-counts by one


def test_letterbox_and_decode_undo_each_other():
    rgb = lesson6.make_scene(3000).rgb  # 480 x 640
    square, scale, (px, py) = lesson6.letterbox(rgb, 320)
    assert square.shape == (320, 320, 3) and np.isclose(scale, 0.5) and (px, py) == (0, 40)
    assert (square[:40] == 114).all() and (square[280:] == 114).all()
    # One anchor row predicting a box at letterboxed (cx, cy, w, h) = (160, 160, 40, 20):
    raw = np.zeros((1, 5, 3), np.float32)
    raw[0, :, 0] = [160, 160, 40, 20, 0.8]
    raw[0, :, 1] = [160, 160, 40, 20, 0.6]  # a weaker twin, removed by NMS
    raw[0, :, 2] = [10, 10, 4, 4, 0.0005]  # below the score floor
    out = lesson6.decode(raw, scale, (px, py))
    assert out.shape == (1, 5)
    assert np.allclose(out[0], [280, 220, 360, 260, 0.8])  # back in 640 x 480 pixels


def test_centres_of_boxes():
    d = np.array([[10, 20, 30, 40, 0.5]], np.float32)
    assert np.allclose(lesson6.centres_of(d), [[20, 30, 0.5]])


def test_ensure_weights_refuses_a_wrong_hash(tmp_path):
    (tmp_path / "yolov8n.pt").write_bytes(b"not a model")
    with pytest.raises(ValueError, match="SHA-256"):
        lesson6.ensure_weights(root=tmp_path, expected="0" * 64)
    assert lesson6.ensure_weights(
        root=tmp_path, expected=lesson6.sha256_of(tmp_path / "yolov8n.pt")
    )
