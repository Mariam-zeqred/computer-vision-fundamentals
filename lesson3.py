"""Lesson 3 - Features and matching, homography.

Runs every piece of code shown in lesson3-features-matching-homography.md, prints
the numbers the lesson quotes, and writes the figures to figures/lesson3/.

    python lesson3.py            # uses data/coffee.png and the synthetic belt

Sections 1-4 warp data/coffee.png with a homography we choose, so every keypoint,
match and estimated homography can be scored against a known truth. Section 5
looks at lesson 2's synthetic belt through an oblique camera in two frames a
moment apart; the belt travel between them is known, so the measurement can be
scored too. Nothing here is a photograph of a real belt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

import lesson1  # noqa: E402
import lesson2  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures" / "lesson3"
DATA = ROOT / "data"

# A match or a repeated keypoint counts as correct if it lands within this many
# pixels of where the true homography says it should.
TOL_PX = 3.0


# --------------------------------------------------------------------------- #
# 0. A second view of the coffee image, made with a homography we know
# --------------------------------------------------------------------------- #
def project(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to N points given as an (N, 2) array."""
    pts = np.asarray(points, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H.astype(np.float64)).reshape(-1, 2)


def known_homography(shape: tuple[int, int]) -> np.ndarray:
    """Rotate 20 degrees about the centre, shrink to 0.85, shift by (30, 10) px and
    tilt slightly (the perspective row). The numbers are arbitrary; what matters is
    that we know H exactly and can score everything against it."""
    h, w = shape
    centre = np.array([w / 2, h / 2])
    theta, scale = np.deg2rad(20), 0.85
    rot = scale * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    affine = np.eye(3)
    affine[:2, :2] = rot
    affine[:2, 2] = centre - rot @ centre + np.array([30, 10])
    tilt = np.eye(3)
    tilt[2, 0], tilt[2, 1] = 1.5e-4, -1.0e-4
    to_centre, back = np.eye(3), np.eye(3)
    to_centre[:2, 2], back[:2, 2] = -centre, centre
    H = back @ tilt @ to_centre @ affine
    return H / H[2, 2]


def second_view(gray: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warp with H, then dim the lamp: contrast x0.8 and +20 offset, so the
    descriptors also face a brightness change, as a second photograph would."""
    warped = cv2.warpPerspective(gray, H, (gray.shape[1], gray.shape[0]))
    return np.clip(warped.astype(np.float32) * 0.8 + 20, 0, 255).astype(np.uint8)


def frame_corners(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]])


# --------------------------------------------------------------------------- #
# 1. Corners: the structure tensor
# --------------------------------------------------------------------------- #
def structure_tensor(gray: np.ndarray, sigma: float = 2.0) -> tuple[np.ndarray, ...]:
    """Per-pixel Gaussian-weighted sums of Ix*Ix, Ix*Iy, Iy*Iy over a window.
    These three images are the entries of the 2x2 matrix M at every pixel."""
    gx, gy, _ = lesson2.sobel(gray)
    blur = lambda a: cv2.GaussianBlur(a, (0, 0), sigma)  # noqa: E731
    return blur(gx * gx), blur(gx * gy), blur(gy * gy)


def tensor_eigenvalues(gray: np.ndarray, row: int, col: int, sigma: float = 2.0) -> np.ndarray:
    """The two eigenvalues of M at one pixel, largest first."""
    sxx, sxy, syy = structure_tensor(gray, sigma)
    M = np.array([[sxx[row, col], sxy[row, col]], [sxy[row, col], syy[row, col]]])
    return np.sort(np.linalg.eigvalsh(M))[::-1]


def harris_response(gray: np.ndarray, k: float = 0.04, sigma: float = 2.0) -> np.ndarray:
    """det(M) - k * trace(M)^2: large positive at corners, negative on edges,
    near zero on flat regions. Same formula as cv2.cornerHarris."""
    sxx, sxy, syy = structure_tensor(gray, sigma)
    det = sxx * syy - sxy * sxy
    trace = sxx + syy
    return det - k * trace * trace


def shi_tomasi_corners(gray: np.ndarray, max_corners: int = 5000, min_distance: int = 10):
    """Shi-Tomasi: keep pixels whose smaller eigenvalue is large, strongest first,
    at least `min_distance` apart."""
    pts = cv2.goodFeaturesToTrack(gray, max_corners, 0.01, min_distance, blockSize=5)
    return pts.reshape(-1, 2)


# --------------------------------------------------------------------------- #
# 2. Keypoints and descriptors
# --------------------------------------------------------------------------- #
@dataclass
class Features:
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray

    @property
    def points(self) -> np.ndarray:
        return np.float32([k.pt for k in self.keypoints]).reshape(-1, 2)


DETECTORS = {
    # name: (constructor, distance for its descriptors)
    "ORB": (lambda: cv2.ORB_create(nfeatures=1000), cv2.NORM_HAMMING),
    "SIFT": (lambda: cv2.SIFT_create(), cv2.NORM_L2),
}


def detect(name: str, gray: np.ndarray) -> Features:
    make, _ = DETECTORS[name]
    kps, desc = make().detectAndCompute(gray, None)
    return Features(list(kps), desc)


def repeatability(a: Features, b: Features, H: np.ndarray, shape: tuple[int, int]) -> float:
    """Share of A's keypoints that, projected by the true H into B's frame and
    landing inside it, have a keypoint of B within TOL_PX."""
    h, w = shape
    proj = project(a.points, H)
    visible = (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)
    d = np.linalg.norm(proj[visible][:, None, :] - b.points[None, :, :], axis=2).min(axis=1)
    return float((d < TOL_PX).mean())


# --------------------------------------------------------------------------- #
# 3. Matching descriptors
# --------------------------------------------------------------------------- #
@dataclass
class Matches:
    src: np.ndarray  # (N, 2) points in image A
    dst: np.ndarray  # (N, 2) points in image B
    distance: np.ndarray  # descriptor distance of each pair

    def __len__(self) -> int:
        return len(self.src)

    def errors(self, H_true: np.ndarray) -> np.ndarray:
        """Reprojection error of each pair under the true homography."""
        return np.linalg.norm(project(self.src, H_true) - self.dst, axis=1)

    def correct(self, H_true: np.ndarray) -> np.ndarray:
        return self.errors(H_true) < TOL_PX


def _pack(a: Features, b: Features, pairs: list[cv2.DMatch]) -> Matches:
    return Matches(
        a.points[[m.queryIdx for m in pairs]],
        b.points[[m.trainIdx for m in pairs]],
        np.float32([m.distance for m in pairs]),
    )


def match_nearest(name: str, a: Features, b: Features) -> Matches:
    """Every descriptor in A paired with its nearest descriptor in B."""
    return _pack(a, b, cv2.BFMatcher(DETECTORS[name][1]).match(a.descriptors, b.descriptors))


def match_ratio(name: str, a: Features, b: Features, ratio: float = 0.75) -> Matches:
    """Lowe's ratio test: keep a pair only if the nearest descriptor is clearly
    nearer than the second nearest, so ambiguous (repeated-texture) points drop out."""
    knn = cv2.BFMatcher(DETECTORS[name][1]).knnMatch(a.descriptors, b.descriptors, k=2)
    return _pack(a, b, [m for m, n in knn if m.distance < ratio * n.distance])


def match_crosscheck(name: str, a: Features, b: Features) -> Matches:
    """Keep a pair only if A's nearest in B has A as its nearest in return."""
    bf = cv2.BFMatcher(DETECTORS[name][1], crossCheck=True)
    return _pack(a, b, bf.match(a.descriptors, b.descriptors))


# --------------------------------------------------------------------------- #
# 4. Estimating the homography
# --------------------------------------------------------------------------- #
def homography_from_four(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Exactly four correspondences fix the eight unknowns of H (the DLT)."""
    return cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))


def fit_homography(m: Matches, ransac: bool, threshold: float = TOL_PX):
    """Least squares over every pair (ransac=False) or RANSAC with a re-fit on
    the inliers (ransac=True). Returns (H normalised to H[2,2] = 1, mask of the
    pairs within `threshold` of the fitted H: the inliers RANSAC kept, or, for
    least squares, simply which pairs the fit happens to agree with)."""
    method = cv2.RANSAC if ransac else 0
    H, mask = cv2.findHomography(m.src, m.dst, method, threshold)
    within = np.ones(len(m), bool) if mask is None else mask.ravel().astype(bool)
    return H / H[2, 2], within


def transfer_error(H_est: np.ndarray, H_true: np.ndarray, points: np.ndarray) -> np.ndarray:
    """How far H_est sends each point from where H_true sends it."""
    return np.linalg.norm(project(points, H_est) - project(points, H_true), axis=1)


def four_point_trials(
    m: Matches, H_true: np.ndarray, shape: tuple[int, int], trials: int = 200, seed: int = 0
) -> np.ndarray:
    """Fit H from four *correct* matches chosen at random, many times, and return
    the worst corner-transfer error of each trial: how much four exact points
    can still disagree with each other."""
    rng = np.random.default_rng(seed)
    good = np.flatnonzero(m.correct(H_true))
    corners = frame_corners(shape)
    worst = []
    for _ in range(trials):
        idx = rng.choice(good, 4, replace=False)
        H = homography_from_four(m.src[idx], m.dst[idx])
        worst.append(transfer_error(H, H_true, corners).max())
    return np.array(worst)


# --------------------------------------------------------------------------- #
# 5. Worked example: how far did the belt move, in millimetres?
# --------------------------------------------------------------------------- #
BELT_SHAPE = (480, 640)  # lesson 2's top-down frame: 1 px = 1 mm of belt
BELT_SHIFT_PX = 37  # how far the belt moves between the two frames (the truth)
# Where the four corner markers of the 640 x 480 mm belt section appear in the
# oblique camera image. In a real setup you click these once, or detect printed
# markers; here the script places them.
MARKERS_IMAGE = np.float32([[150, 70], [520, 50], [610, 450], [40, 430]])


@dataclass
class BeltScene:
    frames: list[np.ndarray]  # two oblique RGB frames
    truth: list[lesson2.Belt]  # the top-down frames they were made from
    H_cam: np.ndarray  # belt mm -> image px
    H_rect: np.ndarray  # image px -> belt mm


def make_scene() -> BeltScene:
    """Two frames of lesson 2's belt seen from an oblique camera, BELT_SHIFT_PX
    apart in belt travel, with fresh sensor noise in the second frame."""
    t0 = lesson2.make_belt()
    t1 = lesson2.make_belt(shift=BELT_SHIFT_PX, noise_seed=11)
    H_cam = cv2.getPerspectiveTransform(frame_corners(BELT_SHAPE), MARKERS_IMAGE)
    h, w = BELT_SHAPE
    frames = [cv2.warpPerspective(b.rgb, H_cam, (w, h), borderValue=(40, 40, 45)) for b in (t0, t1)]
    return BeltScene(frames, [t0, t1], H_cam, np.linalg.inv(H_cam))


def rectify(frame: np.ndarray, H_rect: np.ndarray) -> np.ndarray:
    """Look straight down at the belt: warp the camera image into belt coordinates."""
    h, w = BELT_SHAPE
    return cv2.warpPerspective(frame, H_rect, (w, h))


def egg_centroids(rgb_topdown: np.ndarray) -> np.ndarray:
    """Lesson 2's counter, returning the (x, y) of each distance-transform peak."""
    res = lesson2.count_eggs(rgb_topdown)
    _, _, _, cents = cv2.connectedComponentsWithStats(res.peaks.astype(np.uint8), connectivity=8)
    return cents[1:]


def travel_from_matches(
    scene: BeltScene, name: str = "ORB"
) -> tuple[Matches, np.ndarray, np.ndarray]:
    """Match features between the two oblique frames, map both ends of every
    inlier match to belt millimetres and read the displacement.
    Returns (ratio-tested matches, RANSAC inlier mask, per-inlier displacement in mm)."""
    g0, g1 = (cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in scene.frames)
    m = match_ratio(name, detect(name, g0), detect(name, g1))
    _, inliers = fit_homography(m, ransac=True)
    disp = project(m.dst[inliers], scene.H_rect) - project(m.src[inliers], scene.H_rect)
    return m, inliers, disp


def travel_from_centroids(scene: BeltScene, predicted_mm: float = 0.0) -> np.ndarray:
    """Rectify both frames, count eggs with lesson 2, pair each egg with the egg
    nearest to where it is expected in the next frame (its position plus the
    predicted travel along the belt), return the displacements in mm.
    With predicted_mm = 0 this is plain nearest-neighbour pairing."""
    c0, c1 = (egg_centroids(rectify(f, scene.H_rect)) for f in scene.frames)
    expected = c0 + np.array([predicted_mm, 0.0])
    d = np.linalg.norm(expected[:, None, :] - c1[None, :, :], axis=2)
    return c1[d.argmin(axis=1)] - c0


def marker_click_sensitivity(
    scene: BeltScene, m: Matches, inliers: np.ndarray, jitter_px: float = 2.0, trials: int = 200
) -> tuple[float, float]:
    """Exercise 4: perturb the four clicked markers by up to jitter_px, and return
    the worst shift of the belt's centre point in mm, and the worst change of the
    measured travel (median over the same inlier matches) in mm."""
    rng = np.random.default_rng(1)
    centre_img = project(np.float32([[320, 240]]), scene.H_cam)
    true_travel = float(
        np.median(
            (project(m.dst[inliers], scene.H_rect) - project(m.src[inliers], scene.H_rect))[:, 0]
        )
    )
    worst_point, worst_travel = 0.0, 0.0
    for _ in range(trials):
        noisy = MARKERS_IMAGE + rng.uniform(-jitter_px, jitter_px, MARKERS_IMAGE.shape)
        H_rect = np.linalg.inv(
            cv2.getPerspectiveTransform(frame_corners(BELT_SHAPE), np.float32(noisy))
        )
        moved = project(centre_img, H_rect)
        worst_point = max(worst_point, float(np.linalg.norm(moved - [320, 240])))
        travel = float(
            np.median((project(m.dst[inliers], H_rect) - project(m.src[inliers], H_rect))[:, 0])
        )
        worst_travel = max(worst_travel, abs(travel - true_travel))
    return worst_point, worst_travel


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_corners(gray: np.ndarray, probes: dict[str, tuple[int, int]]) -> None:
    R = harris_response(gray)
    corners = shi_tomasi_corners(gray)
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    ax[0].imshow(gray, cmap="gray")
    for label, (r, c) in probes.items():
        ax[0].plot(c, r, "o", ms=9, mfc="none", mew=2, label=label)
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].set_title("three probe pixels", fontsize=9)
    lim = np.percentile(np.abs(R), 99.5)
    ax[1].imshow(R, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[1].set_title("Harris response (red +, blue -)", fontsize=9)
    ax[2].imshow(gray, cmap="gray")
    ax[2].plot(corners[:, 0], corners[:, 1], "c.", ms=4)
    ax[2].set_title(f"Shi-Tomasi: {len(corners)} corners, 10 px apart", fontsize=9)
    for a in ax:
        a.axis("off")
    _save(fig, "01_corners.png")


def fig_keypoints(
    gray: np.ndarray, warped: np.ndarray, feats: dict[str, tuple[Features, Features]]
):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.4))
    for row, name in enumerate(feats):
        for col, (img, f) in enumerate(zip((gray, warped), feats[name], strict=True)):
            # SIFT keypoints carry a scale and an orientation worth drawing; ORB's
            # are FAST corners whose nominal size would only hide the image.
            flag = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if name == "SIFT" else 0
            canvas = cv2.drawKeypoints(img, f.keypoints, None, (0, 255, 255), flag)
            ax[row, col].imshow(canvas)
            ax[row, col].set_title(
                f"{name}, {'original' if col == 0 else 'second view'}: "
                f"{len(f.keypoints)} keypoints",
                fontsize=9,
            )
            ax[row, col].axis("off")
    _save(fig, "02_keypoints.png")


def fig_matches(gray: np.ndarray, warped: np.ndarray, m: Matches, H: np.ndarray, title: str):
    h, w = gray.shape
    canvas = np.hstack([gray, warped])
    ok = m.correct(H)
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.imshow(canvas, cmap="gray")
    for (x0, y0), (x1, y1), good in zip(m.src, m.dst, ok, strict=True):
        ax.plot([x0, x1 + w], [y0, y1], color="lime" if good else "red", lw=0.6, alpha=0.8)
    ax.set_title(f"{title}: {len(m)} matches, {ok.sum()} correct (green)", fontsize=9)
    ax.axis("off")
    _save(fig, "03_matches.png")


def fig_homography(warped: np.ndarray, H_true: np.ndarray, fits: dict[str, np.ndarray], shape):
    corners = frame_corners(shape)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.imshow(warped, cmap="gray")
    styles = {
        "true": ("lime", "-", 2.5),
        "least squares": ("red", "--", 1.5),
        "RANSAC": ("cyan", "-", 1.2),
    }
    for label, H in {"true": H_true, **fits}.items():
        q = project(corners, H)
        q = np.vstack([q, q[:1]])
        colour, ls, lw = styles[label]
        ax.plot(q[:, 0], q[:, 1], color=colour, ls=ls, lw=lw, label=label)
    ax.legend(fontsize=8)
    ax.set_title("where each H says the original frame's border lies", fontsize=9)
    ax.axis("off")
    _save(fig, "04_homography.png")


def fig_belt(scene: BeltScene, m: Matches, inliers: np.ndarray, disp: np.ndarray) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.ravel()
    ax[0].imshow(scene.frames[0])
    mk = np.vstack([MARKERS_IMAGE, MARKERS_IMAGE[:1]])
    ax[0].plot(mk[:, 0], mk[:, 1], "c-o", lw=1.2, ms=5)
    ax[0].set_title("frame t0 from the oblique camera, four markers", fontsize=9)
    ax[1].imshow(scene.frames[1])
    for (x0, y0), (x1, y1), ok in zip(m.src, m.dst, inliers, strict=True):
        ax[1].plot([x0, x1], [y0, y1], color="lime" if ok else "red", lw=1.0)
    ax[1].set_title(
        f"frame t1 with {len(m)} ORB matches drawn as motion ({inliers.sum()} inliers)", fontsize=9
    )
    rect = rectify(scene.frames[1], scene.H_rect)
    c1 = egg_centroids(rect)
    ax[2].imshow(rect)
    ax[2].plot(c1[:, 0], c1[:, 1], "c+", ms=12, mew=2)
    ax[2].set_title(f"frame t1 rectified to belt millimetres: {len(c1)} eggs counted", fontsize=9)
    ax[3].hist(disp[:, 0], bins=np.arange(30, 45, 0.5), color="tab:blue")
    ax[3].axvline(BELT_SHIFT_PX, color="lime", lw=2, label=f"true travel {BELT_SHIFT_PX} mm")
    ax[3].axvline(np.median(disp[:, 0]), color="k", ls="--", label="median of matches")
    ax[3].set_xlabel("travel along the belt per inlier match (mm)")
    ax[3].legend(fontsize=8)
    ax[3].set_title("belt travel read from the matches", fontsize=9)
    for a in ax[:3]:
        a.axis("off")
    _save(fig, "05_belt_travel.png")


# --------------------------------------------------------------------------- #
def main() -> None:
    rgb = lesson1.load_rgb(DATA / "coffee.png")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    shape = gray.shape
    H = known_homography(shape)
    warped = second_view(gray, H)
    print("== 0. a second view with a known homography")
    print("H =\n", np.round(H, 5))
    print("frame corners land at", project(frame_corners(shape), H).round(1).tolist())

    print("\n== 1. corners: eigenvalues of the structure tensor (window sigma 2)")
    strongest = shi_tomasi_corners(gray)[0]
    probes = {
        "flat (table)": (
            lesson1.PATCHES["wood (top left)"][0].start + 25,
            lesson1.PATCHES["wood (top left)"][1].start + 40,
        ),
        "edge (saucer rim)": (300, 359),
        "corner (strongest)": (int(round(strongest[1])), int(round(strongest[0]))),
    }
    R = harris_response(gray)
    print(f"{'pixel':20s} {'row,col':>9s} {'lambda_1':>10s} {'lambda_2':>10s} {'Harris R':>12s}")
    for label, (r, c) in probes.items():
        l1, l2 = tensor_eigenvalues(gray, r, c)
        print(f"{label:20s} {r:4d},{c:3d} {l1:10,.0f} {l2:10,.0f} {R[r, c]:12,.0f}")
    print(f"pixels with R above 1 % of its maximum: {(R > 0.01 * R.max()).sum()}")
    print(
        "Shi-Tomasi corners (quality 0.01 of the strongest, 10 px apart): "
        f"{len(shi_tomasi_corners(gray))}"
    )

    print("\n== 2. keypoints and descriptors")
    feats = {name: (detect(name, gray), detect(name, warped)) for name in DETECTORS}
    for name, (a, b) in feats.items():
        print(
            f"{name}: {len(a.keypoints)} keypoints in the original, {len(b.keypoints)} in the "
            f"second view; descriptor {a.descriptors.shape[1]} x {a.descriptors.dtype}; "
            f"repeatability within {TOL_PX:.0f} px: {repeatability(a, b, H, shape):.1%}"
        )
    geometric_only = cv2.warpPerspective(gray, H, (shape[1], shape[0]))
    for name in DETECTORS:
        b = detect(name, geometric_only)
        print(
            f"  {name} repeatability if only the geometry changed (no dimming): "
            f"{repeatability(feats[name][0], b, H, shape):.1%} with {len(b.keypoints)} keypoints"
        )

    print("\n== 3. matching")
    print(f"{'detector / matcher':30s} {'matches':>8s} {'correct':>8s} {'precision':>10s}")
    table: dict[tuple[str, str], Matches] = {}
    for name, (a, b) in feats.items():
        for label, m in (
            ("nearest neighbour", match_nearest(name, a, b)),
            ("ratio test 0.75", match_ratio(name, a, b)),
            ("cross-check", match_crosscheck(name, a, b)),
        ):
            table[name, label] = m
            ok = m.correct(H)
            print(f"{name + ' / ' + label:30s} {len(m):8d} {ok.sum():8d} {ok.mean():10.1%}")
    for name in DETECTORS:
        m = table[name, "nearest neighbour"]
        ok = m.correct(H)
        unit = "bits" if name == "ORB" else "L2"
        print(
            f"  {name} nearest-neighbour distance, median: correct pairs "
            f"{np.median(m.distance[ok]):.0f} {unit}, wrong pairs "
            f"{np.median(m.distance[~ok]):.0f} {unit}"
        )

    print("\n== 4. estimating H from the ratio-tested matches")
    corners = frame_corners(shape)
    fits: dict[str, np.ndarray] = {}
    for name in DETECTORS:
        m = table[name, "ratio test 0.75"]
        worst = four_point_trials(m, H, shape)
        print(
            f"{name}: four correct matches at random, 200 trials: worst corner error "
            f"median {np.median(worst):.1f} px, 90th percentile {np.percentile(worst, 90):.1f} px"
        )
        for label, ransac in (("least squares", False), ("RANSAC", True)):
            H_est, within = fit_homography(m, ransac)
            err = transfer_error(H_est, H, corners)
            used = (
                f"{within.sum()} inliers kept of {len(m)}"
                if ransac
                else f"all {len(m)} pairs used, {within.sum()} within {TOL_PX:.0f} px afterwards"
            )
            print(
                f"  {label:14s}: {used}; corner error mean {err.mean():.2f} px, "
                f"max {err.max():.2f} px"
            )
            if name == "ORB":
                fits[label] = H_est

    print("\n== 5. worked example: belt travel from an oblique camera")
    scene = make_scene()
    print("H_cam (belt mm -> image px) =\n", np.round(scene.H_cam, 4))
    for i, f in enumerate(scene.frames):
        direct = lesson2.count_eggs(f).n_eggs
        rect = lesson2.count_eggs(rectify(f, scene.H_rect)).n_eggs
        print(
            f"frame t{i}: lesson-2 counter on the oblique image {direct}, "
            f"on the rectified image {rect}, truth {scene.truth[i].n_eggs}"
        )
    widths = []
    for (r, c), (_, _, a, _, _) in zip(scene.truth[0].centres, lesson2.EGGS, strict=True):
        ends = project(np.float32([[c - a, r], [c + a, r]]), scene.H_cam)
        widths.append(float(np.linalg.norm(ends[1] - ends[0])))
    mm_per_px = []
    for y in (0, 480):  # far and near edge of the belt section
        p = project(np.float32([[320, y]]), scene.H_cam)
        q = project(p + [1.0, 0.0], scene.H_rect) - project(p, scene.H_rect)
        mm_per_px.append(float(np.linalg.norm(q)))
    print(
        f"egg length (60 mm) seen by the oblique camera: {min(widths):.0f} to "
        f"{max(widths):.0f} px; "
        f"one image pixel is {mm_per_px[0]:.1f} mm of belt at the far edge, "
        f"{mm_per_px[1]:.1f} mm at the near edge"
    )
    m, inliers, disp = travel_from_matches(scene)
    print(
        f"inlier travel along the belt ranges from {disp[:, 0].min():.1f} "
        f"to {disp[:, 0].max():.1f} mm"
    )
    print(
        f"ORB between the frames: {len(m)} ratio-tested matches, {inliers.sum()} RANSAC inliers; "
        f"travel in belt mm, median along {np.median(disp[:, 0]):.2f}, "
        f"across {np.median(disp[:, 1]):.2f}; "
        f"truth {BELT_SHIFT_PX}, 0"
    )
    H_frames, _ = fit_homography(m, ransac=True)
    in_belt = scene.H_rect @ H_frames @ scene.H_cam
    in_belt /= in_belt[2, 2]
    print("the same motion as a homography, expressed in belt coordinates:\n", np.round(in_belt, 3))
    for label, predicted in (
        ("nearest egg", 0.0),
        ("nearest to predicted", float(np.median(disp[:, 0]))),
    ):
        cd = travel_from_centroids(scene, predicted)
        wrong = int((np.abs(cd[:, 0] - BELT_SHIFT_PX) > 5).sum())
        print(
            f"egg centroids paired by {label:20s}: {len(cd)} eggs, travel median along "
            f"{np.median(cd[:, 0]):.2f}, across {np.median(cd[:, 1]):.2f}; "
            f"{wrong} pairs more than 5 mm off the truth; "
            f"spread of the rest {np.ptp(cd[np.abs(cd[:, 0] - BELT_SHIFT_PX) <= 5, 0]):.2f} mm"
        )
    worst_point, worst_travel = marker_click_sensitivity(scene, m, inliers)
    print(
        f"exercise 4: markers clicked up to 2 px off move the belt's centre point by up to "
        f"{worst_point:.1f} mm but change the measured travel by at most {worst_travel:.2f} mm"
    )
    for i, f in enumerate(scene.frames):
        out = DATA / f"belt_oblique_t{i}.png"
        cv2.imwrite(str(out), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    print("oblique frames written to data/belt_oblique_t0.png and t1.png")

    fig_corners(gray, probes)
    fig_keypoints(gray, warped, feats)
    fig_matches(gray, warped, table["ORB", "ratio test 0.75"], H, "ORB, ratio test 0.75")
    fig_homography(warped, H, fits, shape)
    fig_belt(scene, m, inliers, disp)
    print("\nfigures written to", FIG_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
