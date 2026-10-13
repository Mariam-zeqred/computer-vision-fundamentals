"""Lesson 1 - Images as arrays, colour spaces, histograms.

Runs every piece of code shown in lesson1-images-as-arrays.md, prints the
numbers the lesson quotes, and writes the figures to figures/lesson1/.

    python lesson1.py            # uses data/coffee.png
    python lesson1.py my.jpg     # any image; the numbers will differ
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures" / "lesson1"

# Named patches (row slice, column slice) used to compare the saucer with the
# table. Chosen by eye on data/coffee.png; see section 5 of the lesson.
PATCHES: dict[str, tuple[slice, slice]] = {
    "saucer (left)": (slice(280, 330), slice(90, 150)),
    "saucer (bottom)": (slice(330, 360), slice(230, 330)),
    "wood (top left)": (slice(10, 60), slice(10, 90)),
    "wood (bottom left)": (slice(330, 390), slice(10, 80)),
    "wood (right)": (slice(200, 300), slice(520, 590)),
    "crema": (slice(120, 180), slice(260, 340)),
}


# --------------------------------------------------------------------------- #
# 1. Images are arrays
# --------------------------------------------------------------------------- #
def load_rgb(path: Path | str) -> np.ndarray:
    """Read an image file as an RGB uint8 array of shape (height, width, 3).

    OpenCV decodes to BGR channel order; almost everything else (matplotlib,
    PIL, PyTorch) expects RGB, so convert once at the boundary and forget it.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def describe(img: np.ndarray) -> str:
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    return (
        f"shape={img.shape} dtype={img.dtype} -> {h} rows x {w} columns x "
        f"{channels} channel(s) = {img.nbytes:,} bytes"
    )


# --------------------------------------------------------------------------- #
# 2. uint8 arithmetic: wrap-around versus saturation
# --------------------------------------------------------------------------- #
def brighten_wrong(img: np.ndarray, delta: int) -> np.ndarray:
    """What beginners write. uint8 + uint8 wraps around: 200 + 100 = 44."""
    return img + np.uint8(delta)


def brighten_right(img: np.ndarray, delta: int) -> np.ndarray:
    """Do the maths in a wider type, then clip back into 0..255."""
    out = img.astype(np.int16) + delta
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 3. Colour spaces
# --------------------------------------------------------------------------- #
def to_gray_manual(rgb: np.ndarray) -> np.ndarray:
    """Luma: a weighted sum, not the mean. Weights follow ITU-R BT.601,
    which is what cv2.COLOR_RGB2GRAY uses."""
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    return np.round(0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


def to_hsv(rgb: np.ndarray) -> np.ndarray:
    """OpenCV's HSV for uint8 input: H in 0..179 (degrees / 2), S and V in 0..255."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


# --------------------------------------------------------------------------- #
# 4. Histograms
# --------------------------------------------------------------------------- #
def histogram(channel: np.ndarray, bins: int = 256) -> np.ndarray:
    """Count how many pixels take each value. np.bincount is the whole story
    for uint8 data; cv2.calcHist does the same with more ceremony."""
    return np.bincount(channel.ravel(), minlength=bins)[:bins]


def hue_valley(hue: np.ndarray, lo: int = 3, hi: int = 12) -> int:
    """Find the least-populated hue between two peaks, after a 3-bin smoothing
    so a single empty bin does not win."""
    counts = histogram(hue, 180).astype(np.float64)
    smooth = np.convolve(counts, np.ones(3) / 3, mode="same")
    return lo + int(np.argmin(smooth[lo:hi]))


def percentile_stretch(channel: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Map the [low, high] percentiles of a channel onto [0, 255], clipping the tails."""
    lo, hi = np.percentile(channel, [low, high])
    out = (channel.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1.0))
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 5. Worked example: separate the saucer from the table
# --------------------------------------------------------------------------- #
def saucer_mask_rgb(rgb: np.ndarray) -> np.ndarray:
    """A best effort in RGB: 'quite red, not much green'. It cannot work well,
    because lit wood and shaded saucer overlap in every channel."""
    return (rgb[..., 0] > 130) & (rgb[..., 1] < 90)


def saucer_mask_hue(rgb: np.ndarray, max_hue: int, min_sat: int = 150) -> np.ndarray:
    """Pixels whose hue is at or below the valley and that are clearly coloured."""
    hsv = to_hsv(rgb)
    return (hsv[..., 0] <= max_hue) & (hsv[..., 1] >= min_sat)


def patch_table(rgb: np.ndarray) -> list[tuple[str, float, float, float, float]]:
    """Mean and spread of the red channel and of hue inside each named patch."""
    hue = to_hsv(rgb)[..., 0]
    rows = []
    for name, (r, c) in PATCHES.items():
        red = rgb[r, c, 0]
        rows.append((name, red.mean(), red.std(), hue[r, c].mean(), hue[r, c].std()))
    return rows


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_channels(rgb: np.ndarray) -> None:
    fig, ax = plt.subplots(1, 4, figsize=(14, 3))
    ax[0].imshow(rgb)
    ax[0].set_title("RGB")
    for i, name in enumerate("RGB"):
        ax[i + 1].imshow(rgb[..., i], cmap="gray", vmin=0, vmax=255)
        ax[i + 1].set_title(f"{name} channel")
    for a in ax:
        a.axis("off")
    _save(fig, "01_channels.png")


def fig_hsv(rgb: np.ndarray) -> None:
    hsv = to_hsv(rgb)
    fig, ax = plt.subplots(1, 4, figsize=(14, 3))
    ax[0].imshow(to_gray_manual(rgb), cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("gray (BT.601 luma)")
    ax[1].imshow(hsv[..., 0], cmap="hsv", vmin=0, vmax=179)
    ax[1].set_title("H (0-179)")
    ax[2].imshow(hsv[..., 1], cmap="gray", vmin=0, vmax=255)
    ax[2].set_title("S")
    ax[3].imshow(hsv[..., 2], cmap="gray", vmin=0, vmax=255)
    ax[3].set_title("V")
    for a in ax:
        a.axis("off")
    _save(fig, "02_gray_and_hsv.png")


def fig_uint8(rgb: np.ndarray) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    ax[0].imshow(rgb)
    ax[0].set_title("original")
    ax[1].imshow(brighten_wrong(rgb, 80))
    ax[1].set_title("img + 80 on uint8 (wraps)")
    ax[2].imshow(brighten_right(rgb, 80))
    ax[2].set_title("int16 + 80, then clip")
    for a in ax:
        a.axis("off")
    _save(fig, "03_uint8_trap.png")


def fig_histograms(rgb: np.ndarray, valley: int) -> None:
    gray = to_gray_manual(rgb)
    hsv = to_hsv(rgb)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.2))
    ax[0].plot(histogram(gray), color="k")
    ax[0].set_title("gray histogram")
    ax[0].set_xlabel("value")
    for i, col in enumerate(["r", "g", "b"]):
        ax[1].plot(histogram(rgb[..., i]), color=col, alpha=0.8)
    ax[1].set_title("R, G, B histograms")
    ax[1].set_xlabel("value")
    hue_counts = histogram(hsv[..., 0], 180)[:40]
    ax[2].bar(range(40), hue_counts, color="tab:orange")
    ax[2].axvline(valley, color="k", linestyle="--", label=f"valley at H = {valley}")
    ax[2].set_title("hue histogram (first 40 of 180 bins)")
    ax[2].set_xlabel("H (OpenCV units, degrees / 2)")
    ax[2].legend()
    for a in ax:
        a.set_ylabel("pixels")
    _save(fig, "04_histograms.png")


def fig_saucer(rgb: np.ndarray, valley: int) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.2))
    ax[0].imshow(rgb)
    for name, (r, c) in PATCHES.items():
        colour = "cyan" if name.startswith("saucer") else "yellow"
        rect = plt.Rectangle(
            (c.start, r.start), c.stop - c.start, r.stop - r.start, fill=False, color=colour, lw=1.5
        )
        ax[0].add_patch(rect)
    ax[0].set_title("patches: saucer (cyan), wood / crema (yellow)")
    ax[1].imshow(saucer_mask_rgb(rgb), cmap="gray")
    ax[1].set_title("RGB rule: R > 130 and G < 90")
    ax[2].imshow(saucer_mask_hue(rgb, valley), cmap="gray")
    ax[2].set_title(f"hue rule: H <= {valley} and S >= 150")
    for a in ax:
        a.axis("off")
    _save(fig, "05_saucer_mask.png")


# --------------------------------------------------------------------------- #
def main(path: Path) -> None:
    rgb = load_rgb(path)
    print("== 1. the array")
    print(describe(rgb))
    print("top-left pixel rgb[0, 0] =", rgb[0, 0])
    print("pixel at row 150, column 300 (in the crema) =", rgb[150, 300])
    print("crop rgb[100:200, 50:150].shape =", rgb[100:200, 50:150].shape)
    print("every second pixel rgb[::2, ::2].shape =", rgb[::2, ::2].shape)
    print("mean per channel R, G, B =", np.round(rgb.reshape(-1, 3).mean(axis=0), 1))

    print("\n== 2. uint8 arithmetic")
    with np.errstate(over="ignore"):
        print("np.uint8(200) + np.uint8(100) =", np.uint8(200) + np.uint8(100))
    a, b = np.full((1, 1), 200, np.uint8), np.full((1, 1), 100, np.uint8)
    print("cv2.add(200, 100) on uint8 images =", int(cv2.add(a, b)[0, 0]), "(saturates)")
    print("int16 then clip:", brighten_right(np.uint8([200]), 100)[0])

    print("\n== 3. colour spaces")
    gray_manual = to_gray_manual(rgb)
    gray_cv = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    print("max |manual luma - cv2 gray| =", int(np.abs(gray_manual.astype(int) - gray_cv).max()))
    print(
        "mean of channels vs luma for pure green (0,255,0):",
        255 // 3,
        to_gray_manual(np.uint8([[[0, 255, 0]]]))[0, 0],
    )
    print(
        "mean of channels vs luma for pure blue  (0,0,255):",
        255 // 3,
        to_gray_manual(np.uint8([[[0, 0, 255]]]))[0, 0],
    )
    hsv = to_hsv(rgb)
    print(
        "HSV at (150, 300) crema =",
        hsv[150, 300],
        " at (300, 120) saucer =",
        hsv[300, 120],
        " at (50, 50) wood =",
        hsv[50, 50],
    )
    print(
        "white cup reflecting the saucer: rgb[200, 230] =", rgb[200, 230], " hsv =", hsv[200, 230]
    )
    print("cup inner rim, near grey: hsv[60, 300] =", hsv[60, 300])

    print("\n== 4. histograms")
    hist = histogram(gray_manual)
    print(
        "gray histogram sums to", int(hist.sum()), "= number of pixels", rgb.shape[0] * rgb.shape[1]
    )
    print("gray: share of pixels below 64 =", f"{hist[:64].sum() / hist.sum():.1%}")
    valley = hue_valley(hsv[..., 0])
    print("hue histogram valley between the two peaks at H =", valley)
    hue_counts = histogram(hsv[..., 0], 180)
    peak_lo = int(np.argmax(hue_counts[:valley]))
    peak_hi = valley + int(np.argmax(hue_counts[valley:40]))
    print("hue peaks at H =", peak_lo, "and H =", peak_hi)
    print(
        "share of pixels with H >= 40 (bins not drawn) =",
        f"{hue_counts[40:].sum() / hue_counts.sum():.1%}",
    )

    print("\n== 5. worked example: saucer vs table")
    print(f"{'patch':20s} {'R mean':>7s} {'R sd':>6s} {'H mean':>7s} {'H sd':>5s}")
    for name, rm, rs, hm, hs in patch_table(rgb):
        print(f"{name:20s} {rm:7.1f} {rs:6.1f} {hm:7.1f} {hs:5.1f}")
    for name, (r, c) in PATCHES.items():
        if name.startswith("saucer"):
            patch_hue = hsv[r, c, 0]
            wrapped = patch_hue >= 170
            print(
                f"  {name}: {int(wrapped.sum())} of {patch_hue.size} pixels have H >= 170 "
                f"(the other end of red); H sd without them = {patch_hue[~wrapped].std():.1f}"
            )
    m_rgb = saucer_mask_rgb(rgb)
    m_hue = saucer_mask_hue(rgb, valley)
    print(f"RGB rule marks {m_rgb.mean():.1%} of pixels; hue rule marks {m_hue.mean():.1%}")
    for name, (r, c) in PATCHES.items():
        print(f"  hue rule inside {name:20s}: {m_hue[r, c].mean():6.1%} of the patch")

    print("\n== exercise 5: stretch / equalise V, then threshold by hue again")
    h, s, v = cv2.split(hsv)
    for name, v2 in [
        ("percentile stretch", percentile_stretch(v)),
        ("equalizeHist", cv2.equalizeHist(v)),
    ]:
        back = cv2.cvtColor(cv2.merge([h, s, v2]), cv2.COLOR_HSV2RGB)
        new_valley = hue_valley(to_hsv(back)[..., 0])
        share = saucer_mask_hue(back, valley).mean()
        print(
            f"{name:18s} on V: valley at H = {new_valley}, "
            f"mask (H <= {valley}, S >= 150) covers {share:.1%}"
        )

    fig_channels(rgb)
    fig_hsv(rgb)
    fig_uint8(rgb)
    fig_histograms(rgb, valley)
    fig_saucer(rgb, valley)
    print("\nfigures written to", FIG_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "coffee.png")
