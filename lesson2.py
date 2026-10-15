"""Lesson 2 - Filtering, edges, morphology.

Runs every piece of code shown in lesson2-filtering-edges-morphology.md, prints
the numbers the lesson quotes, and writes the figures to figures/lesson2/.

    python lesson2.py            # uses data/coffee.png and the synthetic belt

The conveyor-belt image is synthetic. make_belt() draws it from a fixed seed, so
the egg count is known exactly and the pipeline can be scored, not just looked at.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

import lesson1  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures" / "lesson2"
BELT_PNG = ROOT / "data" / "belt_synthetic.png"


# --------------------------------------------------------------------------- #
# 0. Synthetic conveyor belt with a known number of eggs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Belt:
    """A rendered belt frame, the noise-free version, and the truth behind it."""

    rgb: np.ndarray  # what the camera would give you (noise, specks, uneven light)
    clean: np.ndarray  # the same frame before noise, used to score the filters
    centres: list[tuple[int, int]]  # (row, col) of every egg

    @property
    def n_eggs(self) -> int:
        return len(self.centres)


# Egg centres (row, col) and half-axes (a, b) in pixels, laid out by hand so that
# two pairs touch (eggs 4+5 and 9+10) and the rest are separate.
EGGS: list[tuple[int, int, int, int, int]] = [
    # row, col, half-axis along, half-axis across, angle in degrees
    (90, 80, 30, 22, 15),
    (100, 230, 29, 21, -20),
    (80, 400, 31, 22, 70),
    (110, 560, 30, 21, 0),
    (240, 120, 30, 22, 10),
    (248, 170, 29, 21, -5),  # touches the one above
    (230, 330, 30, 22, 80),
    (250, 480, 31, 22, 35),
    (380, 90, 30, 21, -30),
    (390, 300, 30, 22, 0),
    (392, 352, 29, 21, 5),  # touches the one above
    (370, 540, 31, 22, 60),
]


def make_belt(seed: int = 7, shape: tuple[int, int] = (480, 640)) -> Belt:
    """Render a dark rubber belt with 12 cream-coloured eggs, lit more strongly on
    the right than on the left, with dirt specks, sensor noise and dead pixels.

    Everything here is invented; the point is a frame whose answer we know.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    # Reflectance: belt 0.45 with a faint woven texture, eggs 0.92 with shading.
    refl = np.full((h, w), 0.45, np.float32)
    refl += 0.02 * np.sin(np.arange(w) / 3.0)[None, :]  # weave along the belt
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    centres: list[tuple[int, int]] = []
    for r, c, a, b, deg in EGGS:
        t = np.deg2rad(deg)
        u = (xx - c) * np.cos(t) + (yy - r) * np.sin(t)
        v = -(xx - c) * np.sin(t) + (yy - r) * np.cos(t)
        inside = (u / a) ** 2 + (v / b) ** 2 <= 1.0
        # Shade each egg: brightest a little up-left of centre, darker at the rim.
        shade = 0.92 - 0.25 * ((u / a) ** 2 + (v / b) ** 2) - 0.05 * (u + v) / (a + b)
        refl[inside] = shade[inside]
        centres.append((r, c))
    # Dirt specks: 40 small bright discs of radius 1-2 px (3-5 px across), the kind opening removes.
    for _ in range(40):
        r, c = rng.integers(0, h), rng.integers(0, w)
        cv2.circle(refl, (int(c), int(r)), int(rng.integers(1, 3)), 0.85, -1)
    # Illumination: a lamp on the right. Multiplicative, as real light is.
    light = np.linspace(0.30, 1.0, w, dtype=np.float32)[None, :]
    clean = np.clip(refl * light * 255.0, 0, 255)
    # Camera: Gaussian read noise, then 1 % dead or hot pixels.
    noisy = clean + rng.normal(0.0, 9.0, clean.shape).astype(np.float32)
    salt = rng.random(clean.shape)
    noisy[salt < 0.005] = 0.0
    noisy[salt > 0.995] = 255.0
    gray = np.clip(noisy, 0, 255).astype(np.uint8)
    clean8 = clean.astype(np.uint8)
    # Give it a colour: cream eggs on a green-grey belt, so it looks like a photo.
    rgb = _tint(gray)
    return Belt(rgb=rgb, clean=clean8, centres=centres)


def _tint(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32) / 255.0
    r = np.clip(g * 255 * 1.00, 0, 255)
    gg = np.clip(g * 255 * 0.97, 0, 255)
    b = np.clip(g * 255 * 0.80, 0, 255)
    return np.dstack([r, gg, b]).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 1. Cleaning the lesson-1 mask
# --------------------------------------------------------------------------- #
def ellipse(size: int) -> np.ndarray:
    """A round structuring element. Round elements keep round things round."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def count_components(mask: np.ndarray, min_area: int = 1) -> tuple[int, np.ndarray, np.ndarray]:
    """Label the connected regions of a binary mask. Returns (count, labels, areas),
    with the background excluded from the count and from areas."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = areas >= min_area
    return int(keep.sum()), labels, areas[keep]


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def clean_saucer_mask(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Open to drop the wood-grain streaks, close to fill the holes, keep the one
    region we want. Each stage is returned so the figure can show them."""
    m = mask.astype(np.uint8)
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, ellipse(7))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, ellipse(21))
    return {
        "raw": mask.astype(bool),
        "opened": opened.astype(bool),
        "closed": closed.astype(bool),
        "largest": largest_component(closed),
    }


# --------------------------------------------------------------------------- #
# 2. Filtering: a kernel slides over the image
# --------------------------------------------------------------------------- #
def box_kernel(k: int) -> np.ndarray:
    return np.ones((k, k), np.float32) / (k * k)


def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """cv2.filter2D computes a correlation (no kernel flip): the same as a
    convolution for symmetric kernels such as the box, and the sign-flipped
    convolution for the antisymmetric Sobel kernel. Output in float32 so
    negative values (from derivative kernels) survive."""
    return cv2.filter2D(img.astype(np.float32), cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)


def mean_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


FILTERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "no filter": lambda g: g,
    "box 5x5": lambda g: cv2.blur(g, (5, 5)),
    "Gaussian 5x5, sigma 1": lambda g: cv2.GaussianBlur(g, (5, 5), 1.0),
    "median 5x5": lambda g: cv2.medianBlur(g, 5),
    "median 3x3 then Gaussian": lambda g: cv2.GaussianBlur(cv2.medianBlur(g, 3), (5, 5), 1.0),
}


def denoise_table(noisy: np.ndarray, clean: np.ndarray) -> list[tuple[str, float, float]]:
    """How close each filter gets a noisy frame back to the noise-free one:
    over the whole frame, and only on the egg outlines (where blur costs most)."""
    _, _, mag = sobel(clean)
    on_edge = mag > 150
    rows = []
    for name, f in FILTERS.items():
        out = f(noisy)
        rows.append(
            (name, mean_abs_error(out, clean), mean_abs_error(out[on_edge], clean[on_edge]))
        )
    return rows


# --------------------------------------------------------------------------- #
# 3. Edges: derivatives of a blurred image
# --------------------------------------------------------------------------- #
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
SOBEL_Y = SOBEL_X.T.copy()


def sobel(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Horizontal derivative, vertical derivative, gradient magnitude."""
    gx = convolve(gray, SOBEL_X)
    gy = convolve(gray, SOBEL_Y)
    return gx, gy, np.hypot(gx, gy)


def canny(gray: np.ndarray, low: int, high: int, blur_sigma: float = 0.0) -> np.ndarray:
    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    return cv2.Canny(gray, low, high) > 0


# --------------------------------------------------------------------------- #
# 4. Morphology on a toy shape
# --------------------------------------------------------------------------- #
def toy_shape() -> np.ndarray:
    """Two touching discs (60x90), a small hole in one, and three specks."""
    m = np.zeros((60, 90), np.uint8)
    cv2.circle(m, (28, 30), 18, 1, -1)
    cv2.circle(m, (62, 30), 16, 1, -1)
    cv2.line(m, (40, 30), (50, 30), 1, 1)  # make sure the two discs are joined
    cv2.circle(m, (28, 30), 2, 0, -1)  # hole in the left disc
    for x, y in [(5, 5), (84, 52), (80, 6)]:
        cv2.circle(m, (x, y), 1, 1, -1)  # specks
    return m


def morphology_stages(m: np.ndarray, k: int = 5) -> dict[str, np.ndarray]:
    se = ellipse(k)
    return {
        "input": m,
        "erode": cv2.erode(m, se),
        "dilate": cv2.dilate(m, se),
        "open": cv2.morphologyEx(m, cv2.MORPH_OPEN, se),
        "close": cv2.morphologyEx(m, cv2.MORPH_CLOSE, se),
    }


# --------------------------------------------------------------------------- #
# 5. Worked example: count the eggs on the belt
# --------------------------------------------------------------------------- #
@dataclass
class EggCount:
    gray: np.ndarray
    denoised: np.ndarray
    tophat: np.ndarray
    global_mask: np.ndarray
    global_threshold: int
    tophat_mask: np.ndarray
    tophat_threshold: int
    cleaned: np.ndarray
    labels: np.ndarray
    areas: np.ndarray
    n_blobs: int
    dist: np.ndarray
    peaks: np.ndarray
    n_eggs: int


def otsu(gray: np.ndarray) -> tuple[int, np.ndarray]:
    t, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return int(t), m > 0


def white_tophat(gray: np.ndarray, size: int) -> np.ndarray:
    """image minus its opening: keeps only bright things smaller than `size`,
    which flattens a slowly varying background such as uneven lighting."""
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, ellipse(size))


def count_eggs(
    rgb: np.ndarray, egg_diameter: int = 60, min_area: int = 400, peak_frac: float = 0.6
) -> EggCount:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    denoised = cv2.GaussianBlur(cv2.medianBlur(gray, 3), (5, 5), 1.0)
    g_t, g_mask = otsu(denoised)
    tophat = white_tophat(denoised, int(egg_diameter * 1.5) | 1)
    t_t, t_mask = otsu(tophat)
    cleaned = cv2.morphologyEx(t_mask.astype(np.uint8), cv2.MORPH_OPEN, ellipse(7))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, ellipse(7))
    n_blobs, labels, areas = count_components(cleaned, min_area)
    # Split touching eggs: the distance transform peaks once per egg.
    dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    peaks = (dist >= peak_frac * dist.max()).astype(np.uint8)
    n_eggs, _, _ = count_components(peaks)
    return EggCount(
        gray,
        denoised,
        tophat,
        g_mask,
        g_t,
        t_mask,
        t_t,
        cleaned.astype(bool),
        labels,
        areas,
        n_blobs,
        dist,
        peaks.astype(bool),
        n_eggs,
    )


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _show_row(images: list[tuple[str, np.ndarray]], name: str, cmap: str | None = "gray") -> None:
    fig, ax = plt.subplots(1, len(images), figsize=(3.5 * len(images), 3))
    for a, (title, img) in zip(ax, images, strict=True):
        a.imshow(img, cmap=None if img.ndim == 3 else cmap)
        a.set_title(title, fontsize=9)
        a.axis("off")
    _save(fig, name)


def fig_saucer(rgb: np.ndarray, stages: dict[str, np.ndarray]) -> None:
    _show_row(
        [
            ("lesson 1 mask", stages["raw"]),
            ("open, 7 px", stages["opened"]),
            ("then close, 21 px", stages["closed"]),
            ("largest component", stages["largest"]),
            ("applied to the image", rgb * stages["largest"][..., None]),
        ],
        "01_saucer_cleanup.png",
    )


def fig_filters(belt: Belt) -> None:
    gray = cv2.cvtColor(belt.rgb, cv2.COLOR_RGB2GRAY)
    crop = (slice(200, 300), slice(80, 230))
    _show_row(
        [
            ("noisy frame (crop)", gray[crop]),
            ("box 5x5", cv2.blur(gray, (5, 5))[crop]),
            ("Gaussian 5x5", cv2.GaussianBlur(gray, (5, 5), 1.0)[crop]),
            ("median 5x5", cv2.medianBlur(gray, 5)[crop]),
            ("noise-free", belt.clean[crop]),
        ],
        "02_filters.png",
    )


def fig_edges(gray: np.ndarray) -> None:
    gx, gy, mag = sobel(gray)
    fig, ax = plt.subplots(1, 5, figsize=(17.5, 3))
    ax[0].imshow(gx, cmap="gray", vmin=-300, vmax=300)
    ax[0].set_title("Sobel x (horizontal change)", fontsize=9)
    ax[1].imshow(gy, cmap="gray", vmin=-300, vmax=300)
    ax[1].set_title("Sobel y (vertical change)", fontsize=9)
    ax[2].imshow(mag, cmap="gray", vmin=0, vmax=400)
    ax[2].set_title("gradient magnitude", fontsize=9)
    ax[3].imshow(canny(gray, 50, 150), cmap="gray")
    ax[3].set_title("Canny 50/150, no blur", fontsize=9)
    ax[4].imshow(canny(gray, 50, 150, 2.0), cmap="gray")
    ax[4].set_title("Canny 50/150, blur sigma 2", fontsize=9)
    for a in ax:
        a.axis("off")
    _save(fig, "03_edges.png")


def fig_morphology(stages: dict[str, np.ndarray]) -> None:
    _show_row([(k, v) for k, v in stages.items()], "04_morphology.png")


def fig_eggs(belt: Belt, res: EggCount) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
    ax = ax.ravel()
    ax[0].imshow(belt.rgb)
    ax[0].set_title(f"synthetic belt, {belt.n_eggs} eggs", fontsize=9)
    ax[1].imshow(res.global_mask, cmap="gray")
    ax[1].set_title(f"global Otsu threshold ({res.global_threshold})", fontsize=9)
    ax[2].imshow(res.tophat, cmap="gray")
    ax[2].set_title("white top-hat: image minus opening", fontsize=9)
    ax[3].imshow(res.cleaned, cmap="gray")
    ax[3].set_title(
        f"Otsu on top-hat ({res.tophat_threshold}), open+close: {res.n_blobs} blobs", fontsize=9
    )
    ax[4].imshow(res.dist, cmap="magma")
    ax[4].set_title("distance transform", fontsize=9)
    ax[5].imshow(belt.rgb)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(
        res.peaks.astype(np.uint8), connectivity=8
    )
    for cx, cy in cents[1:]:
        ax[5].plot(cx, cy, "c+", ms=12, mew=2)
    ax[5].set_title(f"one peak per egg: {res.n_eggs} counted", fontsize=9)
    for a in ax:
        a.axis("off")
    _save(fig, "05_egg_count.png")


# --------------------------------------------------------------------------- #
def main() -> None:
    rgb = lesson1.load_rgb(ROOT / "data" / "coffee.png")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    print("== 1. cleaning the lesson-1 saucer mask")
    raw = lesson1.saucer_mask_hue(rgb, 8)
    stages = clean_saucer_mask(raw)
    for name, m in stages.items():
        n, _, areas = count_components(m)
        print(
            f"{name:8s}: {m.mean():5.1%} of pixels, {n:4d} connected regions, "
            f"largest {int(areas.max()) if len(areas) else 0:,} px"
        )
    n_small = count_components(stages["raw"])[2]
    print(f"  of the raw regions, {int((n_small < 50).sum())} are smaller than 50 px")
    for name, (r, c) in lesson1.PATCHES.items():
        print(f"  cleaned mask inside {name:20s}: {stages['largest'][r, c].mean():6.1%}")

    print("\n== 2. filtering")
    print(
        "box 5x5 kernel sums to", float(box_kernel(5).sum()), "; one entry =", box_kernel(5)[0, 0]
    )
    k1 = cv2.getGaussianKernel(5, 1.0).ravel()
    print("Gaussian 5-tap, sigma 1:", np.round(k1, 3), "sums to", round(float(k1.sum()), 3))
    print("multiplies per pixel, 15x15 kernel: 2-D", 15 * 15, " separable", 15 + 15)
    belt = make_belt()
    cv2.imwrite(str(BELT_PNG), cv2.cvtColor(belt.rgb, cv2.COLOR_RGB2BGR))
    bgray = cv2.cvtColor(belt.rgb, cv2.COLOR_RGB2GRAY)
    print("belt frame:", lesson1.describe(bgray), "written to", BELT_PNG.relative_to(ROOT))
    print(f"{'filter':26s} {'all pixels':>10s} {'egg outlines':>12s}   (mean |filtered - clean|)")
    for name, err, edge_err in denoise_table(bgray, belt.clean):
        print(f"{name:26s} {err:10.2f} {edge_err:12.2f}")

    print("\n== 3. edges")
    gx, gy, mag = sobel(gray)
    print(
        "Sobel x range",
        round(float(gx.min())),
        "to",
        round(float(gx.max())),
        "; magnitude max",
        round(float(mag.max())),
    )
    row = 300
    col = int(np.argmax(np.abs(gx[row, 20:-20]))) + 20
    print(f"strongest horizontal change on row {row} is at column {col}:")
    print("  gray    ", gray[row, col - 3 : col + 4])
    print("  Sobel x ", np.round(gx[row, col - 3 : col + 4]).astype(int))
    for sigma in (0.0, 1.0, 2.0):
        for lo, hi in ((50, 150), (100, 200)):
            e = canny(gray, lo, hi, sigma)
            print(f"Canny {lo}/{hi}, blur sigma {sigma}: {e.mean():5.2%} of pixels are edges")

    print("\n== 4. morphology on a toy shape (5 px round element)")
    toy = toy_shape()
    tstages = morphology_stages(toy)
    for name, m in tstages.items():
        n, _, _ = count_components(m)
        holes = count_components(1 - m)[0] - 1  # background is one region
        print(f"{name:7s}: {int(m.sum()):4d} px on, {n} regions, {holes} hole(s)")

    for label, first, second in [
        ("open then close", cv2.MORPH_OPEN, cv2.MORPH_CLOSE),
        ("close then open", cv2.MORPH_CLOSE, cv2.MORPH_OPEN),
    ]:
        both = cv2.morphologyEx(cv2.morphologyEx(toy, first, ellipse(5)), second, ellipse(5))
        print(
            f"  exercise 2: {label} -> {int(both.sum())} px on, "
            f"{count_components(both)[0]} region(s), {count_components(1 - both)[0] - 1} hole(s)"
        )
    big = cv2.morphologyEx(toy, cv2.MORPH_OPEN, ellipse(45))
    print("  exercise 3: opening the toy with a 45 px element leaves", int(big.sum()), "pixels on")

    print("\n== 5. worked example: count the eggs")
    res = count_eggs(belt.rgb)
    print("true number of eggs:", belt.n_eggs)
    print(
        f"belt brightness left vs right (clean, row 30): {belt.clean[30, 5]} vs "
        f"{belt.clean[30, -5]}; egg brightness left vs right: {belt.clean[90, 80]} vs "
        f"{belt.clean[110, 560]}"
    )
    gn, _, gareas = count_components(res.global_mask, 400)
    print(
        f"global Otsu threshold {res.global_threshold}: {gn} blobs over 400 px, "
        f"largest {int(gareas.max()):,} px (the lit half of the belt)"
    )
    print(f"top-hat then Otsu threshold {res.tophat_threshold}: {res.n_blobs} blobs over 400 px")
    print("blob areas:", sorted(int(a) for a in res.areas))
    print("one-egg area, median:", int(np.median(res.areas)))
    print(f"distance transform max {res.dist.max():.1f} px; peaks at >= 60 %:", res.n_eggs, "eggs")

    fig_saucer(rgb, stages)
    fig_filters(belt)
    fig_edges(gray)
    fig_morphology(tstages)
    fig_eggs(belt, res)
    print("\nfigures written to", FIG_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
