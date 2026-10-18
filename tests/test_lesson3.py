import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lesson1  # noqa: E402
import lesson2  # noqa: E402
import lesson3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _coffee():
    gray = cv2.cvtColor(lesson1.load_rgb(ROOT / "data" / "coffee.png"), cv2.COLOR_RGB2GRAY)
    H = lesson3.known_homography(gray.shape)
    return gray, H, lesson3.second_view(gray, H)


def test_known_homography_is_normalised_and_invertible():
    gray, H, _ = _coffee()
    assert H[2, 2] == 1.0
    corners = lesson3.frame_corners(gray.shape)
    back = lesson3.project(lesson3.project(corners, H), np.linalg.inv(H))
    assert np.allclose(back, corners, atol=1e-3)


def test_structure_tensor_classifies_flat_edge_corner():
    gray, _, _ = _coffee()
    flat = lesson3.tensor_eigenvalues(gray, 35, 50)
    edge = lesson3.tensor_eigenvalues(gray, 300, 359)
    corner = lesson3.tensor_eigenvalues(gray, 241, 351)
    assert flat.max() < 100  # both tiny
    assert edge[0] > 50_000 and edge[1] < 0.2 * edge[0]  # one large, one small
    assert corner[1] > 50_000 and corner[1] > 0.5 * corner[0]  # both large
    R = lesson3.harris_response(gray)
    assert R[241, 351] > R[300, 359] > R[35, 50]
    assert abs(R[35, 50]) < 1e3


def test_harris_matches_opencv_up_to_scale():
    gray, _, _ = _coffee()
    ours = lesson3.harris_response(gray, sigma=1.0)
    theirs = cv2.cornerHarris(np.float32(gray), 5, 3, 0.04)  # 5x5 box window, not Gaussian
    # Different windows, so compare only where both rank the strongest corners.
    top_ours = set(map(tuple, np.argwhere(ours > 0.05 * ours.max())))
    top_theirs = set(map(tuple, np.argwhere(theirs > 0.05 * theirs.max())))
    assert len(top_ours & top_theirs) > 0.5 * min(len(top_ours), len(top_theirs))


def test_repeatability_and_match_precision():
    gray, H, warped = _coffee()
    for name, min_rep in (("ORB", 0.8), ("SIFT", 0.4)):
        a, b = lesson3.detect(name, gray), lesson3.detect(name, warped)
        assert lesson3.repeatability(a, b, H, gray.shape) > min_rep
        nn = lesson3.match_nearest(name, a, b).correct(H).mean()
        ratio = lesson3.match_ratio(name, a, b).correct(H).mean()
        cross = lesson3.match_crosscheck(name, a, b).correct(H).mean()
        assert ratio > 0.9 and ratio > cross > nn


def test_orb_numbers_quoted_in_the_lesson():
    gray, H, warped = _coffee()
    a, b = lesson3.detect("ORB", gray), lesson3.detect("ORB", warped)
    assert (len(a.keypoints), len(b.keypoints)) == (976, 939)
    m = lesson3.match_ratio("ORB", a, b)
    assert (len(m), int(m.correct(H).sum())) == (511, 489)
    nn = lesson3.match_nearest("ORB", a, b)
    ok = nn.correct(H)
    assert np.median(nn.distance[ok]) == 25 and np.median(nn.distance[~ok]) == 61


def test_ransac_beats_least_squares_and_sift_is_subpixel():
    gray, H, warped = _coffee()
    corners = lesson3.frame_corners(gray.shape)
    worst = {}
    for name in ("ORB", "SIFT"):
        m = lesson3.match_ratio(name, lesson3.detect(name, gray), lesson3.detect(name, warped))
        H_ls, _ = lesson3.fit_homography(m, ransac=False)
        H_ra, inl = lesson3.fit_homography(m, ransac=True)
        e_ls = lesson3.transfer_error(H_ls, H, corners).max()
        e_ra = lesson3.transfer_error(H_ra, H, corners).max()
        assert e_ra < e_ls and e_ra < 1.5 and inl.sum() > 0.9 * len(m)
        worst[name] = e_ra
        trials = lesson3.four_point_trials(m, H, gray.shape, trials=50)
        assert np.median(trials) > e_ra  # four points alone are worse than RANSAC's re-fit
    assert worst["SIFT"] < 0.3 < worst["ORB"]


def test_four_exact_points_recover_h():
    gray, H, _ = _coffee()
    pts = np.float32([[50, 60], [540, 40], [500, 350], [80, 300]])
    H4 = lesson3.homography_from_four(pts, lesson3.project(pts, H))
    assert np.allclose(H4 / H4[2, 2], H, atol=1e-4)


def test_shifted_belt_is_the_rolled_belt():
    base, moved = lesson2.make_belt(), lesson2.make_belt(shift=37, noise_seed=11)
    # The belt (reflectance) rolls 37 px to the right; the lamp does not move, so
    # a pixel's new brightness is its old one scaled by the lamp ratio.
    light = np.linspace(0.30, 1.0, 640)
    cols = np.arange(100, 600)
    expected = base.clean[:, cols - 37] * (light[cols] / light[cols - 37])
    assert np.abs(expected - moved.clean[:, cols]).max() < 2.0  # uint8 rounding
    assert moved.centres == [(r, c + 37) for r, c in base.centres]
    assert not np.array_equal(base.rgb, moved.rgb)


def test_markers_define_h_cam_exactly():
    scene = lesson3.make_scene()
    img = lesson3.project(lesson3.frame_corners(lesson3.BELT_SHAPE), scene.H_cam)
    assert np.allclose(img, lesson3.MARKERS_IMAGE, atol=1e-3)
    back = lesson3.project(lesson3.MARKERS_IMAGE, scene.H_rect)
    assert np.allclose(back, lesson3.frame_corners(lesson3.BELT_SHAPE), atol=1e-2)


def test_rectification_restores_the_egg_count():
    scene = lesson3.make_scene()
    for frame in scene.frames:
        assert lesson2.count_eggs(frame).n_eggs < 12
        assert lesson2.count_eggs(lesson3.rectify(frame, scene.H_rect)).n_eggs == 12


def test_belt_travel_from_matches_and_centroids():
    scene = lesson3.make_scene()
    m, inliers, disp = lesson3.travel_from_matches(scene)
    assert inliers.sum() >= 20
    assert abs(np.median(disp[:, 0]) - lesson3.BELT_SHIFT_PX) < 0.5
    assert abs(np.median(disp[:, 1])) < 1.0
    naive = lesson3.travel_from_centroids(scene)
    predicted = lesson3.travel_from_centroids(scene, float(np.median(disp[:, 0])))
    off = lambda d: int((np.abs(d[:, 0] - lesson3.BELT_SHIFT_PX) > 5).sum())  # noqa: E731
    assert off(naive) == 2 and off(predicted) == 0  # the two touching pairs
    assert abs(np.median(predicted[:, 0]) - lesson3.BELT_SHIFT_PX) < 0.5
    point, travel = lesson3.marker_click_sensitivity(scene, m, inliers, trials=50)
    assert travel < 0.5 < point  # a click error moves positions far more than it moves travel
