"""Lesson 4 - Classical detection to CNNs: what a convolution learns.

Runs every piece of code shown in lesson4-classical-detection-to-cnns.md, prints
the numbers the lesson quotes, and writes the figures to figures/lesson4/.

    python lesson4.py            # trains everything on the CPU, about a minute

Everything is trained and scored on synthetic patches drawn by make_patch_set()
(an egg or not, on a piece of belt, with the same rendering rules as lesson 2's
belt) and on lesson 2's synthetic belt frames, where the egg positions are known.
Nothing here is a photograph. The point is a detector whose every number can be
checked against a known answer, so the mechanism can be studied rather than
admired.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import torch  # noqa: E402
from torch import nn  # noqa: E402

import lesson1  # noqa: E402
import lesson2  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures" / "lesson4"
DATA = ROOT / "data"

PATCH = 64  # a window large enough to hold one 60 x 44 px egg
MATCH_PX = 15.0  # a detection this close to a true centre counts as that egg
NMS_RADIUS = 24  # lesson 2's touching pairs are 50 px apart, so peaks closer than this merge
N_TRAIN, N_VAL = 4000, 1000
KINDS = ("egg", "belt", "off-centre", "specks", "blob", "label")
NEGATIVE_MIX = {"belt": 0.15, "off-centre": 0.30, "specks": 0.15, "blob": 0.20, "label": 0.20}


# --------------------------------------------------------------------------- #
# 0. Patches with a known label
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PatchSet:
    """Grayscale 64 x 64 patches, a label (1 = an egg centred in the patch) and the
    kind of scene each one is, for the per-kind tables."""

    x: np.ndarray  # (n, 64, 64) uint8
    y: np.ndarray  # (n,) int, 1 or 0
    kind: np.ndarray  # (n,) str, one of KINDS

    def __len__(self) -> int:
        return len(self.y)


def _draw_egg(refl: np.ndarray, centre: tuple[float, float], a: float, b: float, deg: float):
    """Lesson 2's egg: an ellipse with the same shading rule as make_belt()."""
    h, w = refl.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r, c = centre
    t = np.deg2rad(deg)
    u = (xx - c) * np.cos(t) + (yy - r) * np.sin(t)
    v = -(xx - c) * np.sin(t) + (yy - r) * np.cos(t)
    inside = (u / a) ** 2 + (v / b) ** 2 <= 1.0
    shade = 0.92 - 0.25 * ((u / a) ** 2 + (v / b) ** 2) - 0.05 * (u + v) / (a + b)
    refl[inside] = shade[inside]


def _specks(refl: np.ndarray, rng: np.random.Generator, n: int) -> None:
    h, w = refl.shape
    for _ in range(n):
        r, c = rng.integers(0, h), rng.integers(0, w)
        cv2.circle(refl, (int(c), int(r)), int(rng.integers(1, 3)), 0.85, -1)


def render_patch(rng: np.random.Generator, kind: str, size: int = PATCH) -> np.ndarray:
    """One patch of belt, lit at a random level, with or without an egg, as uint8.

    The rendering rules are lesson 2's: reflectance 0.45 with a faint weave for the
    belt, a shaded ellipse of reflectance about 0.9 for an egg, bright specks of
    dirt, a multiplicative lamp, Gaussian read noise and 1 % dead or hot pixels.
    The random choices are what make the patches a data set rather than one frame.
    """
    refl = np.full((size, size), 0.45, np.float32)
    refl += 0.02 * np.sin(np.arange(size) / 3.0 + rng.uniform(0, 2 * np.pi))[None, :]
    mid = size / 2
    if kind in ("egg", "off-centre"):
        a, b = rng.uniform(26, 34), rng.uniform(19, 24)
        if kind == "egg":
            centre = (mid + rng.uniform(-6, 6), mid + rng.uniform(-6, 6))
        else:  # an egg whose centre is 26-44 px away: part of it is in view, it is not "here"
            d, phi = rng.uniform(26, 44), rng.uniform(0, 2 * np.pi)
            centre = (mid + d * np.sin(phi), mid + d * np.cos(phi))
        _draw_egg(refl, centre, a, b, rng.uniform(0, 180))
    elif kind == "blob":  # a bright round thing, smaller than an egg and flat: not an egg
        c = (int(mid + rng.uniform(-6, 6)), int(mid + rng.uniform(-6, 6)))
        cv2.circle(refl, c, int(rng.integers(10, 19)), float(rng.uniform(0.80, 0.92)), -1)
    elif kind == "label":  # a bright rectangle, egg-sized, egg-bright, with corners: not an egg
        box = (
            (mid + rng.uniform(-6, 6), mid + rng.uniform(-6, 6)),
            (rng.uniform(40, 64), rng.uniform(24, 44)),
            rng.uniform(0, 180),
        )
        cv2.fillPoly(refl, [cv2.boxPoints(box).astype(np.int32)], float(rng.uniform(0.80, 0.92)))
    if kind == "specks":
        _specks(refl, rng, int(rng.integers(3, 9)))
    elif kind != "belt" and rng.random() < 0.5:
        _specks(refl, rng, int(rng.integers(1, 4)))  # dirt lands on eggs too
    level, gradient = rng.uniform(0.30, 1.0), rng.uniform(-0.12, 0.12)
    light = level * (1.0 + gradient * (np.arange(size, dtype=np.float32) / size - 0.5))[None, :]
    clean = np.clip(refl * light * 255.0, 0, 255)
    noisy = clean + rng.normal(0.0, 9.0, clean.shape).astype(np.float32)
    salt = rng.random(clean.shape)
    noisy[salt < 0.005] = 0.0
    noisy[salt > 0.995] = 255.0
    return np.clip(noisy, 0, 255).astype(np.uint8)


def make_patch_set(n: int, seed: int) -> PatchSet:
    """n patches, half of them eggs, the other half the four kinds of not-egg in
    NEGATIVE_MIX proportions, shuffled. The same seed gives the same set."""
    rng = np.random.default_rng(seed)
    n_pos = n // 2
    kinds = ["egg"] * n_pos
    for k, share in NEGATIVE_MIX.items():
        kinds += [k] * int(round(share * (n - n_pos)))
    kinds = kinds[:n] + ["belt"] * (n - len(kinds))
    kinds = np.array(kinds)
    rng.shuffle(kinds)
    x = np.stack([render_patch(rng, k) for k in kinds])
    y = (kinds == "egg").astype(np.int64)
    return PatchSet(x, y, kinds)


def standardise(x: np.ndarray) -> np.ndarray:
    """Each patch as a flat vector with zero mean and unit standard deviation, so a
    patch under a dim lamp and the same patch under a bright one become the same
    vector. This is also what normalised cross-correlation does to each window."""
    z = x.reshape(len(x), -1).astype(np.float32)
    z -= z.mean(axis=1, keepdims=True)
    z /= np.maximum(z.std(axis=1, keepdims=True), 1e-3)
    return z


def accuracy_by_kind(scores: np.ndarray, ps: PatchSet, threshold: float) -> dict[str, float]:
    pred = scores > threshold
    out = {"all": float((pred == ps.y).mean())}
    for k in KINDS:
        m = ps.kind == k
        out[k] = float((pred[m] == ps.y[m]).mean())
    return out


# --------------------------------------------------------------------------- #
# 1. Classical detection: a template, normalised cross-correlation, non-maximum suppression
# --------------------------------------------------------------------------- #
def mean_template(ps: PatchSet) -> np.ndarray:
    """The average of every egg patch: the classical hand-made detector."""
    return ps.x[ps.y == 1].astype(np.float32).mean(axis=0)


def ncc_scores(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation of each patch with the template, in [-1, 1].
    The same number cv2.matchTemplate(..., TM_CCOEFF_NORMED) gives for that window."""
    zt = standardise(template[None])[0]
    return standardise(x) @ zt / zt.size


def ncc_map(gray: np.ndarray, template: np.ndarray) -> np.ndarray:
    """NCC at every window position of a frame; entry (i, j) is the window whose
    top-left corner is (i, j), so its centre is (i + 32, j + 32)."""
    return cv2.matchTemplate(
        gray.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED
    )


def best_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """The cut between two consecutive sorted scores that classifies most patches
    correctly, and that accuracy. Chosen on training patches, never on the test."""
    order = np.argsort(scores)
    s, t = scores[order], y[order]
    # Predict 1 above the cut: correct = positives above + negatives at or below.
    pos_above = t.sum() - np.cumsum(t)
    neg_below = np.cumsum(1 - t)
    correct = pos_above + neg_below
    i = int(np.argmax(correct))
    cut = float((s[i] + s[i + 1]) / 2) if i + 1 < len(s) else float(s[i])
    return cut, float(correct[i] / len(t))


def detect_peaks(
    score_map: np.ndarray, threshold: float, stride: int = 1, offset: int = PATCH // 2
) -> np.ndarray:
    """Non-maximum suppression: keep positions above the threshold that are the
    maximum within NMS_RADIUS, strongest first. Returns (k, 3) rows of x, y, score
    in frame pixels."""
    k = 2 * (NMS_RADIUS // stride) + 1
    local_max = cv2.dilate(score_map, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ii, jj = np.nonzero((score_map >= local_max) & (score_map > threshold))
    cand = sorted(zip(ii, jj, score_map[ii, jj], strict=True), key=lambda t: -t[2])
    kept: list[tuple[float, float, float]] = []
    for i, j, s in cand:
        x, y = j * stride + offset, i * stride + offset
        if all((x - kx) ** 2 + (y - ky) ** 2 > NMS_RADIUS**2 for kx, ky, _ in kept):
            kept.append((float(x), float(y), float(s)))
    return np.array(kept, np.float32).reshape(-1, 3)


@dataclass(frozen=True)
class Score:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)


def score_detections(dets: np.ndarray, centres: list[tuple[int, int]], tol: float = MATCH_PX):
    """Greedy matching, strongest detection first: each true centre can be claimed
    once, by a detection within `tol` pixels of it."""
    truth = np.array([(c, r) for r, c in centres], np.float32).reshape(-1, 2)
    free = np.ones(len(truth), bool)
    tp = 0
    for x, y, _ in dets:
        if not free.any():
            break
        d = np.hypot(truth[:, 0] - x, truth[:, 1] - y)
        d[~free] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            free[j] = False
            tp += 1
    return Score(tp, len(dets) - tp, len(truth) - tp)


# --------------------------------------------------------------------------- #
# 2. What "learning a kernel" means, in nine numbers
# --------------------------------------------------------------------------- #
def correlate_valid(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Correlation (what cv2.filter2D and every CNN library compute), keeping only
    the output pixels whose whole window is inside the image."""
    m = k.shape[0] // 2
    return cv2.filter2D(x, -1, k.astype(np.float32))[m:-m, m:-m]


def kernel_gradient(x: np.ndarray, err: np.ndarray, size: int) -> np.ndarray:
    """d(mean squared error)/dk: for each kernel entry, the mean of the error times
    the input pixel that entry multiplied. It is itself a correlation of x with err."""
    h, w = err.shape
    grad = np.empty((size, size), np.float32)
    for u in range(size):
        for v in range(size):
            grad[u, v] = 2.0 * np.mean(err * x[u : u + h, v : v + w])
    return grad


def kernel_hessian(x: np.ndarray, size: int = 3) -> np.ndarray:
    """The 9 x 9 matrix of second derivatives of the loss: twice the covariance of
    the nine pixels of a window, averaged over all windows. Its largest eigenvalue
    bounds the step size, its smallest says how slowly the worst direction learns."""
    h, w = x.shape[0] - 2 * (size // 2), x.shape[1] - 2 * (size // 2)
    A = np.stack([x[u : u + h, v : v + w].ravel() for u in range(size) for v in range(size)], 1)
    A = A.astype(np.float64)
    return 2.0 * A.T @ A / A.shape[0]


def learn_kernel(
    x: np.ndarray, target: np.ndarray, size: int = 3, steps: int = 300, lr: float | None = None
) -> tuple[np.ndarray, list[float]]:
    """Gradient descent on the nine entries of a 3 x 3 kernel so that
    correlate(x, k) matches the target image. The default step is the inverse of
    the largest curvature, the largest step that cannot diverge."""
    if lr is None:
        lr = 1.0 / float(np.linalg.eigvalsh(kernel_hessian(x, size)).max())
    k = np.zeros((size, size), np.float32)
    losses = []
    for _ in range(steps):
        err = correlate_valid(x, k) - target
        losses.append(float(np.mean(err**2)))
        k -= lr * kernel_gradient(x, err, size)
    return k, losses


def solve_kernel(x: np.ndarray, target: np.ndarray, size: int = 3) -> np.ndarray:
    """The same problem as a least-squares solve: nine unknowns, one equation per
    output pixel. Possible because the model is linear in the kernel."""
    h, w = target.shape
    cols = [x[u : u + h, v : v + w].ravel() for u in range(size) for v in range(size)]
    A = np.stack(cols, axis=1)
    k, *_ = np.linalg.lstsq(A, target.ravel(), rcond=None)
    return k.reshape(size, size).astype(np.float32)


# --------------------------------------------------------------------------- #
# 3. One learned filter: logistic regression on the patches
# --------------------------------------------------------------------------- #
def train_linear(
    ps: PatchSet, steps: int = 400, lr: float = 2.0, l2: float = 1e-3
) -> tuple[np.ndarray, float, list[float]]:
    """A single 64 x 64 weight image w and a bias b, so that sigmoid(w . z + b) is the
    probability that the standardised patch z is an egg. Full-batch gradient descent
    on the cross-entropy. Returns w as an image, b, and the loss curve."""
    Z = standardise(ps.x) / PATCH  # scaled so that a step size of order 1 is safe
    y = ps.y.astype(np.float32)
    w = np.zeros(Z.shape[1], np.float32)
    b = 0.0
    losses = []
    for _ in range(steps):
        logit = Z @ w + b
        p = 1.0 / (1.0 + np.exp(-logit))
        losses.append(
            float(np.mean(np.logaddexp(0, -logit) * y + np.logaddexp(0, logit) * (1 - y)))
        )
        g = p - y
        w -= lr * (Z.T @ g / len(y) + l2 * w)
        b -= lr * float(g.mean())
    return (w / PATCH).reshape(PATCH, PATCH), b, losses


def linear_scores(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """The logit of each patch under the learned filter."""
    return standardise(x) @ w.ravel() + b


def linear_map(gray: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """The same logit at every window of a frame, computed with three correlations:
    w against the frame, and a box against the frame and its square for each
    window's mean and standard deviation. Same layout as ncc_map."""
    f = gray.astype(np.float32)
    ones = np.ones((PATCH, PATCH), np.float32)
    n = float(PATCH * PATCH)
    mu = cv2.matchTemplate(f, ones, cv2.TM_CCORR) / n
    var = cv2.matchTemplate(f * f, ones, cv2.TM_CCORR) / n - mu * mu
    sd = np.sqrt(np.maximum(var, 1e-6))
    corr = cv2.matchTemplate(f, w.astype(np.float32), cv2.TM_CCORR)
    return (corr - mu * w.sum()) / sd + b


# --------------------------------------------------------------------------- #
# 4. A small convolutional network
# --------------------------------------------------------------------------- #
class EggNet(nn.Module):
    """conv 5x5 (1 -> 8), ReLU, pool 2, conv 5x5 (8 -> 16), ReLU, pool 2, conv 13x13 (16 -> 1).

    No padding anywhere, so on a 64 x 64 patch the output is one number (the logit)
    and on a whole frame it is a map of logits at a stride of 4 pixels, each one
    exactly the logit the network would give the 64 x 64 window centred there.
    The last layer is the "fully connected" classifier written as a convolution.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.head = nn.Conv2d(16, 1, 13)
        self.pool = nn.MaxPool2d(2)

    def features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = self.pool(torch.relu(self.conv1(x)))
        f2 = self.pool(torch.relu(self.conv2(f1)))
        return f1, f2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, f2 = self.features(x)
        return self.head(f2)


CNN_STRIDE = 4  # two pools of 2
CNN_SEED = 0


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).float().unsqueeze(1) / 255.0


def train_cnn(
    train: PatchSet, val: PatchSet, epochs: int = 16, batch: int = 64, seed: int = CNN_SEED
) -> tuple[EggNet, list[tuple[float, float]]]:
    """Adam on the binary cross-entropy, a fixed seed, no augmentation. Returns the
    network and (training loss, validation accuracy) per epoch."""
    torch.manual_seed(seed)
    net = EggNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    xt, yt = _to_tensor(train.x), torch.from_numpy(train.y).float()
    history = []
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        order = torch.randperm(len(train), generator=g)
        total = 0.0
        for start in range(0, len(train), batch):
            idx = order[start : start + batch]
            opt.zero_grad()
            loss = loss_fn(net(xt[idx]).flatten(), yt[idx])
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(idx)
        acc = float(((cnn_scores(net, val.x) > 0) == val.y).mean())
        history.append((total / len(train), acc))
    return net, history


@torch.no_grad()
def cnn_scores(net: EggNet, x: np.ndarray) -> np.ndarray:
    net.eval()
    return torch.cat(
        [net(_to_tensor(x[i : i + 256])).flatten() for i in range(0, len(x), 256)]
    ).numpy()


@torch.no_grad()
def cnn_map(net: EggNet, gray: np.ndarray) -> np.ndarray:
    """Run the whole frame through the network once: a logit map at stride 4 whose
    entry (i, j) is the logit of the window with top-left corner (4 i, 4 j)."""
    net.eval()
    return net(_to_tensor(gray[None]))[0, 0].numpy()


@torch.no_grad()
def feature_maps(net: EggNet, patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    net.eval()
    f1, f2 = net.features(_to_tensor(patch[None]))
    return f1[0].numpy(), f2[0].numpy()


def first_layer_filters(net: EggNet) -> np.ndarray:
    return net.conv1.weight.detach().numpy()[:, 0]


def n_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


# --------------------------------------------------------------------------- #
# 5. Detection on whole frames
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Detector:
    name: str
    threshold: float
    run: object  # gray frame -> (score map, stride)


def make_detectors(template, ncc_thr, w, b, net) -> list[Detector]:
    return [
        Detector("template NCC", ncc_thr, lambda g: (ncc_map(g, template), 1)),
        Detector("learned filter", 0.0, lambda g: (linear_map(g, w, b), 1)),
        Detector("CNN", 0.0, lambda g: (cnn_map(net, g), CNN_STRIDE)),
    ]


def detect(det: Detector, gray: np.ndarray) -> np.ndarray:
    score_map, stride = det.run(gray)
    return detect_peaks(score_map, det.threshold, stride)


# Things on the belt that are not eggs, placed by hand in the gaps between lesson 2's
# twelve eggs: three bright labels (centre row, col, size, angle) and two flat blobs.
LABELS = [((165, 420), (56, 36), 20.0), ((320, 230), (48, 30), 0.0), ((445, 430), (60, 40), 60.0)]
BLOBS = [((160, 60), 15), ((320, 570), 12)]


def distractors(refl: np.ndarray, rng: np.random.Generator) -> None:
    """Draw LABELS and BLOBS on a belt's reflectance map (a make_belt decorate hook)."""
    for (r, c), (w, h), deg in LABELS:
        pts = cv2.boxPoints(((float(c), float(r)), (float(w), float(h)), deg)).astype(np.int32)
        cv2.fillPoly(refl, [pts], 0.88)
    for (r, c), radius in BLOBS:
        cv2.circle(refl, (c, r), radius, 0.86, -1)


def frames_for_scoring() -> list[tuple[str, lesson2.Belt]]:
    """Lesson 2's committed belt, lesson 3's second frame (shifted 37 px, fresh noise)
    and a third with different dirt and noise plus the labels and blobs. The eggs are
    the same twelve in every frame."""
    return [
        ("t0 (lesson 2)", lesson2.make_belt()),
        ("t1 (lesson 3)", lesson2.make_belt(shift=37, noise_seed=11)),
        ("t2 (labels)", lesson2.make_belt(seed=21, shift=40, noise_seed=22, decorate=distractors)),
    ]


def gray_of(belt: lesson2.Belt) -> np.ndarray:
    return cv2.cvtColor(belt.rgb, cv2.COLOR_RGB2GRAY)


def rescaled(belt: lesson2.Belt, scale: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """The frame seen by a camera `scale` times further away: smaller eggs."""
    g = cv2.resize(gray_of(belt), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    centres = [(int(round(r * scale)), int(round(c * scale))) for r, c in belt.centres]
    return g, centres


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_patches(ps: PatchSet) -> None:
    fig, ax = plt.subplots(2, len(KINDS), figsize=(2.2 * len(KINDS), 4.6))
    for col, kind in enumerate(KINDS):
        idx = np.nonzero(ps.kind == kind)[0][:2]
        for row, i in enumerate(idx):
            ax[row, col].imshow(ps.x[i], cmap="gray", vmin=0, vmax=255)
            ax[row, col].set_title(f"{kind} (label {ps.y[i]})", fontsize=9)
            ax[row, col].axis("off")
    _save(fig, "01_patches.png")


def fig_learned_kernel(target: np.ndarray, runs: dict[str, tuple[np.ndarray, list[float]]]) -> None:
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
    ax[0].imshow(target[100:250, 250:450], cmap="gray")
    ax[0].set_title("target: Sobel-x of the coffee image (crop)", fontsize=9)
    ax[0].axis("off")
    for a, (name, (k, losses)) in zip(ax[1:3], runs.items(), strict=True):
        a.imshow(k, cmap="RdBu_r", vmin=-2, vmax=2)
        for (u, v), val in np.ndenumerate(k):
            a.text(v, u, f"{val:.3f}", ha="center", va="center", fontsize=9)
        a.set_title(f"learned on {name}, {len(losses)} steps", fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
        ax[3].semilogy(losses, label=name)
    ax[3].set_xlabel("gradient step")
    ax[3].set_ylabel("mean squared error")
    ax[3].legend(fontsize=8)
    ax[3].set_title("loss", fontsize=9)
    _save(fig, "02_learned_kernel.png")


def fig_linear(template: np.ndarray, w: np.ndarray, gray: np.ndarray, maps: dict[str, np.ndarray]):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.2))
    ax = ax.ravel()
    ax[0].imshow(template, cmap="gray")
    ax[0].set_title("mean egg patch (the template)", fontsize=9)
    lim = float(np.abs(w).max())
    ax[1].imshow(w, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[1].set_title("learned filter w (red +, blue -)", fontsize=9)
    for a, (name, m) in zip(ax[2:], maps.items(), strict=True):
        a.imshow(m, cmap="magma")
        a.set_title(f"{name}: score at every window of frame t0", fontsize=9)
    for a in ax:
        a.axis("off")
    _save(fig, "03_learned_filter.png")


def fig_cnn(net: EggNet, patch: np.ndarray) -> None:
    filters = first_layer_filters(net)
    f1, _ = feature_maps(net, patch)
    fig, ax = plt.subplots(3, 8, figsize=(13, 5.4))
    hand = [
        ("Sobel x", lesson2.SOBEL_X),
        ("Sobel y", lesson2.SOBEL_Y),
        ("Gaussian", cv2.getGaussianKernel(5, 1.0) @ cv2.getGaussianKernel(5, 1.0).T),
        ("box", lesson2.box_kernel(5)),
    ]
    for j in range(8):
        if j < len(hand):
            name, k = hand[j]
            lim = float(np.abs(k).max())
            ax[0, j].imshow(k, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax[0, j].set_title(f"hand-made: {name}", fontsize=8)
        else:
            ax[0, j].set_visible(False)
        lim = float(np.abs(filters[j]).max())
        ax[1, j].imshow(filters[j], cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax[1, j].set_title(f"learned {j}: sum {filters[j].sum():+.2f}", fontsize=8)
        ax[2, j].imshow(f1[j], cmap="gray")
        ax[2, j].set_title(f"response {j} on an egg", fontsize=8)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    _save(fig, "04_cnn_filters.png")


def fig_detection(
    frames: list[tuple[str, np.ndarray, list[tuple[int, int]], dict[str, np.ndarray]]],
) -> None:
    fig, ax = plt.subplots(len(frames), 3, figsize=(13, 3.4 * len(frames)))
    ax = np.atleast_2d(ax)
    for row, (title, gray, centres, dets) in enumerate(frames):
        for col, (name, d) in enumerate(dets.items()):
            a = ax[row, col]
            a.imshow(gray, cmap="gray", vmin=0, vmax=255)
            for r, c in centres:
                a.plot(c, r, "c+", ms=10, mew=1.5)
            if len(d):
                a.plot(d[:, 0], d[:, 1], "o", mfc="none", mec="lime", ms=18, mew=1.5)
            s = score_detections(d, centres)
            a.set_title(f"{title}, {name}: {s.tp} found, {s.fp} false, {s.fn} missed", fontsize=9)
            a.axis("off")
    _save(fig, "05_detection.png")


# --------------------------------------------------------------------------- #
def main() -> None:
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    print("== 0. patches with a known label")
    train, val = make_patch_set(N_TRAIN, seed=1), make_patch_set(N_VAL, seed=2)
    counts = {k: int((train.kind == k).sum()) for k in KINDS}
    print(f"training set: {len(train)} patches of {PATCH} x {PATCH}, by kind {counts}")
    print(f"validation set: {len(val)} patches, {int(val.y.sum())} eggs")
    fig_patches(train)

    print("\n== 1. classical detection: the mean egg as a template")
    template = mean_template(train)
    ncc_thr, ncc_train_acc = best_threshold(ncc_scores(train.x, template), train.y)
    ncc_val = accuracy_by_kind(ncc_scores(val.x, template), val, ncc_thr)
    print(
        f"NCC threshold chosen on the training patches: {ncc_thr:.3f} "
        f"(training accuracy {ncc_train_acc:.1%}); validation accuracy {ncc_val['all']:.1%}"
    )

    print("\n== 2. learning a 3 x 3 kernel by gradient descent")
    rgb = lesson1.load_rgb(DATA / "coffee.png")
    coffee = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    coffee -= coffee.mean()  # centred, as every network's input is
    noise = np.random.default_rng(0).random(coffee.shape, dtype=np.float32) - 0.5
    runs = {}
    for name, x, steps in (("the coffee image", coffee, 300), ("white noise", noise, 20)):
        target = correlate_valid(x, lesson2.SOBEL_X)
        ev = np.linalg.eigvalsh(kernel_hessian(x))
        learned, losses = learn_kernel(x, target, steps=steps)
        runs[name] = (learned, losses)
        print(
            f"{name}: curvature from {ev.min():.2e} to {ev.max():.3f} "
            f"(ratio {ev.max() / ev.min():.0f}), "
            f"step size {1 / ev.max():.2f}; loss at step 0 {losses[0]:.4f}, "
            f"at step {steps} {losses[-1]:.1e}; largest difference from Sobel-x "
            f"{np.abs(learned - lesson2.SOBEL_X).max():.4f}"
        )
        print("  learned kernel:", np.round(learned, 3).tolist())
        off = np.abs(solve_kernel(x, target) - lesson2.SOBEL_X).max()
        print(f"  least-squares solve: largest difference {off:.1e}")
    long, _ = learn_kernel(coffee, correlate_valid(coffee, lesson2.SOBEL_X), steps=1000)
    print(f"the coffee image after 1000 steps: {np.abs(long - lesson2.SOBEL_X).max():.4f}")
    fig_learned_kernel(correlate_valid(coffee, lesson2.SOBEL_X), runs)

    print("\n== 3. one learned filter: logistic regression on standardised patches")
    w, b, lin_losses = train_linear(train)
    print(f"cross-entropy after 400 steps: {lin_losses[-1]:.4f}; bias {b:.2f}")
    lin_train_acc = float(((linear_scores(train.x, w, b) > 0) == train.y).mean())
    lin_val = accuracy_by_kind(linear_scores(val.x, w, b), val, 0.0)
    print(f"training accuracy {lin_train_acc:.1%}, validation accuracy {lin_val['all']:.1%}")
    zt = standardise(template[None])[0].reshape(PATCH, PATCH)
    cos = float((w * zt).sum() / (np.linalg.norm(w) * np.linalg.norm(zt)))
    print(f"cosine between the learned filter and the standardised template: {cos:.2f}")
    print(f"filter weight at the centre {w[32, 32]:+.4f}, on the ring 30 px out {w[32, 2]:+.4f}")

    print("\n== 4. a small CNN")
    net, history = train_cnn(train, val)
    print(f"parameters: {n_parameters(net)} (the learned filter has {PATCH * PATCH + 1})")
    for e, (loss, acc) in enumerate(history, 1):
        print(f"epoch {e}: training loss {loss:.4f}, validation accuracy {acc:.1%}")
    cnn_val = accuracy_by_kind(cnn_scores(net, val.x), val, 0.0)
    filters = first_layer_filters(net)
    sums = filters.sum(axis=(1, 2))
    ratio = np.abs(sums) / np.abs(filters).sum(axis=(1, 2))
    print("first-layer filters, sum of entries:", [round(float(v), 2) for v in sums])
    print(
        "  |sum| / sum|entries| (0 = differentiates, 1 = averages):",
        [round(float(v), 2) for v in ratio],
    )
    egg_idx = int(np.nonzero(val.kind == "egg")[0][0])
    fig_cnn(net, val.x[egg_idx])

    print("\n   validation accuracy by kind of patch")
    print(f"{'detector':16s}" + "".join(f"{k:>12s}" for k in ("all", *KINDS)))
    for name, acc in (("template NCC", ncc_val), ("learned filter", lin_val), ("CNN", cnn_val)):
        print(f"{name:16s}" + "".join(f"{acc[k]:12.1%}" for k in ("all", *KINDS)))

    print("\n== 5. the patch classifiers as detectors on whole frames")
    dets = make_detectors(template, ncc_thr, w, b, net)
    g0 = gray_of(lesson2.make_belt())
    m = cnn_map(net, g0)
    i, j = 30, 40
    direct = cnn_scores(
        net,
        g0[None, CNN_STRIDE * i : CNN_STRIDE * i + PATCH, CNN_STRIDE * j : CNN_STRIDE * j + PATCH],
    )[0]
    print(
        f"CNN logit map on frame t0: {m.shape[0]} x {m.shape[1]} at stride {CNN_STRIDE}; "
        f"entry ({i}, {j}) = {m[i, j]:.4f}, the patch there scored alone = {direct:.4f}"
    )
    fig_linear(
        template,
        w,
        g0,
        {"template NCC": ncc_map(g0, template), "learned filter": linear_map(g0, w, b)},
    )
    rows = []
    print(f"{'frame':16s} {'detector':16s} {'found':>6s} {'false':>6s} {'missed':>7s}")
    for title, belt in frames_for_scoring():
        g = gray_of(belt)
        found = {}
        for d in dets:
            found[d.name] = detect(d, g)
            s = score_detections(found[d.name], belt.centres)
            print(f"{title:16s} {d.name:16s} {s.tp:6d} {s.fp:6d} {s.fn:7d}")
        rows.append((title, g, belt.centres, found))
    print("\n   frame t0 seen from further away (smaller eggs), then resized back up")
    scale_rows = []
    belt = frames_for_scoring()[0][1]
    for scale in (0.8, 0.6, 0.5):
        g, centres = rescaled(belt, scale)
        up = cv2.resize(g, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_LINEAR)
        up_centres = [(int(round(r / scale)), int(round(c / scale))) for r, c in centres]
        found = {}
        for d in dets:
            found[d.name] = detect(d, g)
            s = score_detections(found[d.name], centres)
            s_up = score_detections(detect(d, up), up_centres)
            print(
                f"{'t0 x' + str(scale):16s} {d.name:16s} {s.tp:6d} {s.fp:6d} {s.fn:7d}"
                f"   resized back: {s_up.tp} found, {s_up.fp} false, {s_up.fn} missed"
            )
        if scale == 0.5:
            scale_rows.append((f"t0 x{scale}", g, centres, found))
    fig_detection(rows + scale_rows)
    print(f"\nfigures written to {FIG_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
