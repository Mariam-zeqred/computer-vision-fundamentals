"""Lesson 5 - Training a small classifier properly: splits, augmentation, metrics,
confusion matrix.

Runs every piece of code shown in lesson5-training-a-small-classifier.md, prints
the numbers the lesson quotes, and writes the figures to figures/lesson5/.

    python lesson5.py            # downloads Fashion-MNIST (30 MB) once, then trains on the CPU

The data set is Fashion-MNIST (Zalando Research, MIT licence): 70 000 real 28 x 28
grayscale photographs of clothing in ten classes, 60 000 for training and 10 000
held out as the test set by its authors. The four files are fetched from the
dataset's GitHub repository into data/fashion-mnist/ and checked against their
SHA-256 hashes before anything reads them. Nothing in this lesson is synthetic.

The protocol is the point, not the network: one split made once, a validation set
that chooses and a test set that is opened exactly once, augmentation that is
checked rather than assumed, several seeds, and a confusion matrix instead of one
accuracy.
"""

from __future__ import annotations

import gzip
import hashlib
import struct
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import torch  # noqa: E402
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures" / "lesson5"
DATA_DIR = ROOT / "data" / "fashion-mnist"

BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
# file name -> SHA-256 of the gzip as published (the MD5s in the dataset README agree)
FILES = {
    "train-images-idx3-ubyte.gz": (
        "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84"
    ),
    "train-labels-idx1-ubyte.gz": (
        "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845"
    ),
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}
CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
K = len(CLASSES)
VAL_FRACTION = 0.1
SPLIT_SEED = 0
SEEDS = (0, 1, 2)
SIZES = (1_000, 10_000)
BATCH_BY_SIZE = {1_000: 32, 10_000: 128}
EPOCHS_BY_SIZE = {1_000: 40, 10_000: 8}  # 1 280 and 632 gradient steps
RECIPES = ("none", "shift")  # the two recipes scored with three seeds at both sizes
FINAL_EPOCHS = 10  # the budget affordable on all 54 000 images (about 90 s on 4 CPU cores)


# --------------------------------------------------------------------------- #
# 0. Data: fetch, verify, parse, split
# --------------------------------------------------------------------------- #
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_file(name: str, root: Path = DATA_DIR, expected: str | None = None) -> Path:
    """Download one Fashion-MNIST file if it is missing, then check its hash. A file
    with the wrong hash is never read: the error says to delete it and retry."""
    expected = FILES[name] if expected is None else expected
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(BASE_URL + name, tmp)
        tmp.replace(path)
    got = sha256_of(path)
    if got != expected:
        raise ValueError(f"{path} has SHA-256 {got}, expected {expected}; delete it and re-run")
    return path


def read_idx(path: Path) -> np.ndarray:
    """Parse the IDX format MNIST-style sets use: a 4-byte magic number whose last byte
    is the number of dimensions, one big-endian uint32 per dimension, then the
    uint8 payload. Works on the gzipped file directly."""
    with gzip.open(path, "rb") as f:
        magic = f.read(4)
        if magic[:3] != b"\x00\x00\x08":
            raise ValueError(f"{path}: not an IDX file of uint8 ({magic!r})")
        ndim = magic[3]
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{path}: header promises {shape}, payload has {data.size} bytes")
    return data.reshape(shape)


@dataclass(frozen=True)
class Split:
    """Images as (n, 28, 28) uint8 and labels as (n,) int64."""

    x: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    def take(self, idx: np.ndarray) -> Split:
        return Split(self.x[idx], self.y[idx])


def load_fashion_mnist(root: Path = DATA_DIR) -> tuple[Split, Split]:
    """The authors' 60 000 training images and their 10 000 test images, as two Splits."""
    x_tr = read_idx(ensure_file("train-images-idx3-ubyte.gz", root))
    y_tr = read_idx(ensure_file("train-labels-idx1-ubyte.gz", root)).astype(np.int64)
    x_te = read_idx(ensure_file("t10k-images-idx3-ubyte.gz", root))
    y_te = read_idx(ensure_file("t10k-labels-idx1-ubyte.gz", root)).astype(np.int64)
    return Split(x_tr, y_tr), Split(x_te, y_te)


def class_counts(y: np.ndarray, k: int = K) -> np.ndarray:
    return np.bincount(y, minlength=k)


def stratified_split(y: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices of a validation set holding `fraction` of *each class*, and the rest.
    Shuffled within class from the seed, so the same seed gives the same split."""
    rng = np.random.default_rng(seed)
    val: list[np.ndarray] = []
    for c in np.unique(y):
        idx = np.nonzero(y == c)[0]
        rng.shuffle(idx)
        val.append(idx[: int(round(fraction * len(idx)))])
    val_idx = np.sort(np.concatenate(val))
    mask = np.ones(len(y), bool)
    mask[val_idx] = False
    return np.nonzero(mask)[0], val_idx


def subset(split: Split, n: int, seed: int) -> Split:
    """n images with the class balance of the whole: a stratified sample."""
    _, idx = stratified_split(split.y, n / len(split), seed)
    return split.take(idx)


def facing_check(split: Split) -> dict[str, float]:
    """For each class, the share of images closer (squared distance) to the class mean
    than to the class mean mirrored left-right. Near 100 % means every image of the
    class faces the same way, so a mirrored copy is a view the data set never holds."""
    out = {}
    x = split.x.astype(np.float32)
    for c in range(K):
        xc = x[split.y == c]
        mean = xc.mean(axis=0)
        d_same = ((xc - mean) ** 2).sum(axis=(1, 2))
        d_mirror = ((xc - mean[:, ::-1]) ** 2).sum(axis=(1, 2))
        out[CLASSES[c]] = float((d_same < d_mirror).mean())
    return out


def centroid_spread(split: Split) -> dict[str, float]:
    """Per class, the standard deviation in pixels of the images' horizontal centre
    of mass: how much positional variation the data set holds for a shift to imitate."""
    x = split.x.astype(np.float32)
    cols = np.arange(x.shape[2], dtype=np.float32)
    centroid = (x.sum(axis=1) * cols).sum(axis=1) / x.sum(axis=(1, 2))
    return {CLASSES[c]: float(centroid[split.y == c].std()) for c in range(K)}


# --------------------------------------------------------------------------- #
# 1. Two models and one training loop
# --------------------------------------------------------------------------- #
class LinearNet(nn.Module):
    """Multinomial logistic regression: ten 28 x 28 templates and ten biases."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(28 * 28, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class SmallNet(nn.Module):
    """Lesson 4's shape with a ten-way head: conv 3x3 (1 -> 16), pool, conv 3x3
    (16 -> 32), pool, linear. 20 490 parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 7 * 7, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))  # 28 -> 14
        x = self.pool(torch.relu(self.conv2(x)))  # 14 -> 7
        return self.fc(x.flatten(1))


def n_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """uint8 images -> float in [0, 1] with a channel axis: (n, 1, 28, 28)."""
    return torch.from_numpy(np.array(x, copy=True)).float().unsqueeze(1) / 255.0


Augment = Callable[[torch.Tensor, torch.Generator], torch.Tensor]


def shift(xb: torch.Tensor, g: torch.Generator, max_px: int = 2) -> torch.Tensor:
    """Each image moved by an independent random offset of up to max_px pixels in
    each direction, the uncovered border filled with black (the images' background)."""
    n, _, h, w = xb.shape
    padded = torch.nn.functional.pad(xb, (max_px,) * 4)
    dr = torch.randint(0, 2 * max_px + 1, (n,), generator=g)
    dc = torch.randint(0, 2 * max_px + 1, (n,), generator=g)
    return torch.stack(
        [padded[i, :, r : r + h, c : c + w] for i, (r, c) in enumerate(zip(dr, dc, strict=True))]
    )


def hflip(xb: torch.Tensor, g: torch.Generator, p: float = 0.5) -> torch.Tensor:
    """Mirror each image left-right with probability p."""
    flip = torch.rand(len(xb), generator=g) < p
    out = xb.clone()
    out[flip] = torch.flip(xb[flip], dims=[3])
    return out


def shift_and_flip(xb: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    return hflip(shift(xb, g), g)


AUGMENTATIONS: dict[str, Augment | None] = {
    "none": None,
    "shift": shift,
    "shift+flip": shift_and_flip,
}


@dataclass
class History:
    train_loss: list[float]
    val_loss: list[float]
    val_acc: list[float]

    @property
    def best_epoch(self) -> int:
        """1-based index of the epoch with the highest validation accuracy."""
        return int(np.argmax(self.val_acc)) + 1

    @property
    def min_loss_epoch(self) -> int:
        """1-based index of the epoch with the lowest validation loss."""
        return int(np.argmin(self.val_loss)) + 1


@torch.no_grad()
def predict_logits(net: nn.Module, x: np.ndarray, batch: int = 1000) -> torch.Tensor:
    net.eval()
    return torch.cat([net(to_tensor(x[i : i + batch])) for i in range(0, len(x), batch)])


def predict(net: nn.Module, x: np.ndarray) -> np.ndarray:
    return predict_logits(net, x).argmax(1).numpy()


def evaluate(net: nn.Module, split: Split) -> tuple[float, float]:
    """(mean cross-entropy, accuracy) on a split."""
    logits = predict_logits(net, split.x)
    y = torch.from_numpy(split.y)
    loss = nn.functional.cross_entropy(logits, y).item()
    return loss, float((logits.argmax(1) == y).float().mean())


def train(
    make_net: Callable[[], nn.Module],
    tr: Split,
    val: Split,
    *,
    epochs: int,
    seed: int,
    augment: Augment | None = None,
    batch: int = 128,
    lr: float = 1e-3,
) -> tuple[nn.Module, History]:
    """Adam on the cross-entropy. The validation set is scored after every epoch and
    the weights of the best epoch are the ones returned: the validation set chooses
    the stopping point, which is one reason it may not also be the test set."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xt, yt = to_tensor(tr.x), torch.from_numpy(tr.y)
    hist = History([], [], [])
    best_state, best_acc = None, -1.0
    for _ in range(epochs):
        net.train()
        order = torch.randperm(len(tr), generator=g)
        total = 0.0
        for start in range(0, len(tr), batch):
            idx = order[start : start + batch]
            xb = xt[idx] if augment is None else augment(xt[idx], g)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(net(xb), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        vl, va = evaluate(net, val)
        hist.train_loss.append(total / len(tr))
        hist.val_loss.append(vl)
        hist.val_acc.append(va)
        if va > best_acc:
            best_acc = va
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, hist


# --------------------------------------------------------------------------- #
# 2. Metrics: the confusion matrix and what is read off it
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int = K) -> np.ndarray:
    """cm[i, j] = number of images whose true class is i and predicted class is j.
    Rows are the truth, columns the prediction; the diagonal is what went right."""
    cm = np.zeros((k, k), np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: np.ndarray  # per class: of the images called c, how many were c
    recall: np.ndarray  # per class: of the images that are c, how many were called c
    f1: np.ndarray

    @property
    def macro_f1(self) -> float:
        return float(self.f1.mean())


def metrics_from_confusion(cm: np.ndarray) -> Metrics:
    tp = np.diag(cm).astype(np.float64)
    called = cm.sum(axis=0)  # column sums: predicted as c
    actual = cm.sum(axis=1)  # row sums: truly c
    precision = tp / np.maximum(called, 1)
    recall = tp / np.maximum(actual, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return Metrics(float(tp.sum() / cm.sum()), precision, recall, f1)


def most_confused(cm: np.ndarray, n: int = 3) -> list[tuple[int, int, int]]:
    """The n largest off-diagonal entries as (true, predicted, count)."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off.ravel())[::-1][:n]
    return [(int(i // cm.shape[1]), int(i % cm.shape[1]), int(off.ravel()[i])) for i in flat]


def spread(values: list[float]) -> str:
    """mean (min to max) over seeds, as a percentage string."""
    v = np.asarray(values)
    return f"{v.mean():.1%} ({v.min():.1%} to {v.max():.1%})"


# --------------------------------------------------------------------------- #
# 3. The mistake that makes you look good: augment, then split
# --------------------------------------------------------------------------- #
def augment_then_split(
    pool: Split, copies: int, seed: int
) -> tuple[Split, Split, np.ndarray, np.ndarray]:
    """Make `copies` shifted versions of every image, pool them with the originals,
    then split the pool at random 90/10. Returns the two halves and, for the
    diagnosis, each validation image's source index and the set of source indices
    whose copies landed in the training half."""
    g = torch.Generator().manual_seed(seed)
    xt = to_tensor(pool.x)
    xs = [pool.x] + [(shift(xt, g) * 255).round().byte().squeeze(1).numpy() for _ in range(copies)]
    x_all = np.concatenate(xs)
    y_all = np.tile(pool.y, copies + 1)
    source = np.tile(np.arange(len(pool)), copies + 1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x_all))
    n_val = len(x_all) // 10
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    return (
        Split(x_all[tr_idx], y_all[tr_idx]),
        Split(x_all[val_idx], y_all[val_idx]),
        source[val_idx],
        np.unique(source[tr_idx]),
    )


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_samples(split: Split, per_class: int = 8) -> None:
    fig, ax = plt.subplots(K, per_class, figsize=(per_class * 0.9, K * 0.95))
    for c in range(K):
        idx = np.nonzero(split.y == c)[0][:per_class]
        for j, i in enumerate(idx):
            ax[c, j].imshow(split.x[i], cmap="gray", vmin=0, vmax=255)
            ax[c, j].set_xticks([])
            ax[c, j].set_yticks([])
        ax[c, 0].set_ylabel(f"{c} {CLASSES[c]}", rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle("Fashion-MNIST: the first eight training images of each class", fontsize=10)
    _save(fig, "01_samples.png")


def fig_curves(runs: dict[tuple[int, str], History]) -> None:
    fig, ax = plt.subplots(2, len(SIZES), figsize=(5 * len(SIZES), 6.5), sharex="col")
    colours = {"none": "C3", "shift": "C0", "shift+flip": "C2"}
    for col, n in enumerate(SIZES):
        for name, colour in colours.items():
            if (n, name) not in runs:
                continue
            h = runs[(n, name)]
            e = np.arange(1, len(h.train_loss) + 1)
            ax[0, col].plot(e, h.train_loss, colour, ls="--", label=f"{name}: training")
            ax[0, col].plot(e, h.val_loss, colour, label=f"{name}: validation")
            ax[1, col].plot(e, 100 * np.asarray(h.val_acc), colour, label=name)
        batch = BATCH_BY_SIZE[n]
        ax[0, col].set_title(
            f"{n:,}".replace(",", " ") + f" training images, batch {batch}, seed 0"
        )
        ax[0, col].set_ylabel("cross-entropy")
        ax[1, col].set_ylabel("validation accuracy, %")
        ax[1, col].set_xlabel("epoch")
        ax[0, col].legend(fontsize=7)
        ax[1, col].legend(fontsize=7)
        ax[0, col].grid(alpha=0.3)
        ax[1, col].grid(alpha=0.3)
    _save(fig, "02_learning_curves.png")


def fig_augmentation(split: Split) -> None:
    g = torch.Generator().manual_seed(3)
    picks = [int(np.nonzero(split.y == c)[0][0]) for c in (6, 7)]  # a shirt and a sneaker
    fig, ax = plt.subplots(2, 7, figsize=(10, 3.4))
    for row, i in enumerate(picks):
        x = to_tensor(split.x[i : i + 1])
        views = [x[0, 0]] + [shift(x, g)[0, 0] for _ in range(3)] + [torch.flip(x, [3])[0, 0]]
        views += [torch.flip(x, [2])[0, 0], torch.rot90(x, 1, (2, 3))[0, 0]]
        titles = ["original", "shift", "shift", "shift", "h-flip", "v-flip", "rot 90"]
        for j, (v, t) in enumerate(zip(views, titles, strict=True)):
            ax[row, j].imshow(v.numpy(), cmap="gray", vmin=0, vmax=1)
            ax[row, j].set_xticks([])
            ax[row, j].set_yticks([])
            if row == 0:
                ax[row, j].set_title(t, fontsize=9, color="C3" if j >= 5 else "black")
        ax[row, 0].set_ylabel(CLASSES[split.y[i]], fontsize=9)
    fig.suptitle(
        "Label-preserving views (used) and views the data set never contains (not used)",
        fontsize=10,
    )
    _save(fig, "03_augmentation.png")


def fig_seeds(table: dict[tuple[int, str], list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    xs, labels = [], []
    pos = 0
    for n in SIZES:
        for name in RECIPES:
            accs = 100 * np.asarray(table[(n, name)])
            ax.scatter([pos] * len(accs), accs, s=28, zorder=3)
            ax.hlines(accs.mean(), pos - 0.3, pos + 0.3, color="k", lw=1)
            xs.append(pos)
            labels.append(f"{n // 1000}k images\n{name}")
            pos += 1
        pos += 0.6
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("validation accuracy, % (best epoch)")
    ax.set_title("Three seeds per recipe; the bar is the mean", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "04_seeds.png")


def fig_confusion(cm: np.ndarray, test: Split, pred: np.ndarray) -> None:
    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.15, 1))
    ax = fig.add_subplot(gs[0])
    shown = cm.astype(np.float64)
    np.fill_diagonal(shown, 0)  # colour the errors only; the diagonal would drown them
    ax.imshow(shown, cmap="Reds")
    for i in range(K):
        for j in range(K):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if shown[i, j] > shown.max() / 2 else "black",
                fontweight="bold" if i == j else "normal",
            )
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(CLASSES, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Test set confusion matrix (colour: off-diagonal only)", fontsize=10)

    pairs = most_confused(cm, 3)
    inner = gs[1].subgridspec(len(pairs), 6, hspace=0.5)
    for r, (t, p, count) in enumerate(pairs):
        idx = np.nonzero((test.y == t) & (pred == p))[0][:6]
        for c, i in enumerate(idx):
            a = fig.add_subplot(inner[r, c])
            a.imshow(test.x[i], cmap="gray", vmin=0, vmax=255)
            a.set_xticks([])
            a.set_yticks([])
            if c == 0:
                a.set_title(f"{CLASSES[t]} called {CLASSES[p]} ({count})", fontsize=8, loc="left")
    _save(fig, "05_confusion.png")


# --------------------------------------------------------------------------- #
def main() -> None:
    t0 = time.time()
    print("== 0. data: fetch, verify, split")
    full, test = load_fashion_mnist()
    print(f"downloaded and verified {len(FILES)} files in {DATA_DIR.relative_to(ROOT)}/")
    print(f"training images {full.x.shape} {full.x.dtype}, test images {test.x.shape}")
    print("class counts in the 60 000:", class_counts(full.y).tolist())
    tr_idx, val_idx = stratified_split(full.y, VAL_FRACTION, SPLIT_SEED)
    tr, val = full.take(tr_idx), full.take(val_idx)
    print(
        f"split once with seed {SPLIT_SEED}: "
        f"{len(tr)} train, {len(val)} validation, {len(test)} test"
    )
    print("validation counts per class:", class_counts(val.y).tolist())
    rng = np.random.default_rng(SPLIT_SEED)
    random_val = class_counts(full.y[rng.permutation(len(full))[: len(val) // 10]])
    print(
        f"a random (not stratified) sample of 600 would hold "
        f"{random_val.min()} to {random_val.max()} per class"
    )
    fig_samples(tr)

    print("\n== 1. a baseline, then the learning curves at two data sizes")
    print(f"majority class: {class_counts(val.y).max() / len(val):.1%}")
    print(f"LinearNet {n_parameters(LinearNet())} parameters, SmallNet {n_parameters(SmallNet())}")
    sub1k = subset(tr, SIZES[0], seed=0)
    _, lin_hist = train(
        LinearNet,
        sub1k,
        val,
        epochs=EPOCHS_BY_SIZE[SIZES[0]],
        seed=0,
        batch=BATCH_BY_SIZE[SIZES[0]],
    )
    print(
        f"LinearNet on {len(sub1k)} images: best validation accuracy {max(lin_hist.val_acc):.1%} "
        f"at epoch {lin_hist.best_epoch}"
    )
    runs: dict[tuple[int, str], History] = {}
    table: dict[tuple[int, str], list[float]] = {}
    subsets = {n: subset(tr, n, seed=0) for n in SIZES}
    for n in SIZES:
        names = tuple(AUGMENTATIONS) if n == SIZES[0] else RECIPES
        for name in names:
            seeds = SEEDS if name in RECIPES else SEEDS[:1]
            for seed in seeds:
                t1 = time.time()
                _, hist = train(
                    SmallNet,
                    subsets[n],
                    val,
                    epochs=EPOCHS_BY_SIZE[n],
                    seed=seed,
                    augment=AUGMENTATIONS[name],
                    batch=BATCH_BY_SIZE[n],
                )
                table.setdefault((n, name), []).append(max(hist.val_acc))
                if seed == 0:
                    runs[(n, name)] = hist
                    print(
                        f"  SmallNet, {n:>6} images, {name:10s} seed {seed}: "
                        f"best {max(hist.val_acc):.1%} "
                        f"at epoch {hist.best_epoch}/{len(hist.val_acc)}, "
                        f"last {hist.val_acc[-1]:.1%}; validation loss lowest "
                        f"{min(hist.val_loss):.3f} at epoch {hist.min_loss_epoch}, "
                        f"last {hist.val_loss[-1]:.3f}; "
                        f"training loss {hist.train_loss[-1]:.3f} ({time.time() - t1:.0f} s)"
                    )
    fig_curves(runs)
    print("\n   what the images allow: are the views an augmentation makes views the set holds?")
    print(
        "horizontal centre of mass, one standard deviation in px per class: "
        + ", ".join(f"{k} {v:.2f}" for k, v in centroid_spread(tr).items())
    )
    facing = facing_check(tr)
    print(
        "closer to the class mean than to its mirror image: "
        + ", ".join(f"{k} {v:.1%}" for k, v in facing.items())
    )
    fig_augmentation(tr)

    print("\n== 2. several seeds: best validation accuracy, mean (min to max) over seeds", SEEDS)
    for n in SIZES:
        for name in RECIPES:
            print(f"  {n:>6} images, {name:6s}: {spread(table[(n, name)])}")
    fig_seeds(table)

    print("\n== 3. the leak: augment first, split second")
    leak_tr, leak_val, src, in_train = augment_then_split(sub1k, copies=3, seed=0)
    overlap = np.isin(src, in_train).mean()
    print(
        f"{len(sub1k)} images x 4 views = {len(leak_tr) + len(leak_val)}, split 90/10 at random: "
        f"{overlap:.1%} of the validation images have a sibling in the training half"
    )
    leak_net, leak_hist = train(SmallNet, leak_tr, leak_val, epochs=15, seed=0, batch=32)
    _, honest_acc = evaluate(leak_net, val)
    print(
        f"the same network scored on the leaked validation half: {max(leak_hist.val_acc):.1%}; "
        f"on the clean validation set: {honest_acc:.1%} "
        f"(honest runs on the same 1 000 images: none {spread(table[(SIZES[0], 'none')])}, "
        f"shift {spread(table[(SIZES[0], 'shift')])})"
    )

    print("\n== 4. worked example: the chosen recipe, all the training data, the test set once")
    t1 = time.time()
    net, hist = train(SmallNet, tr, val, epochs=FINAL_EPOCHS, seed=0, augment=None, batch=128)
    print(
        f"SmallNet, no augmentation, {len(tr)} images, {FINAL_EPOCHS} epochs "
        f"({time.time() - t1:.0f} s): validation accuracy per epoch "
        f"{[round(100 * a, 1) for a in hist.val_acc]}"
    )
    print(f"kept epoch {hist.best_epoch} with validation accuracy {max(hist.val_acc):.1%}")
    pred = predict(net, test.x)
    cm = confusion_matrix(test.y, pred)
    m = metrics_from_confusion(cm)
    print(f"TEST accuracy {m.accuracy:.1%} (opened once), macro F1 {m.macro_f1:.3f}")
    print(f"{'class':14s} {'precision':>9s} {'recall':>7s} {'F1':>6s} {'n':>5s}")
    for c in range(K):
        print(
            f"{CLASSES[c]:14s} {m.precision[c]:9.1%} {m.recall[c]:7.1%} {m.f1[c]:6.3f} "
            f"{cm[c].sum():5d}"
        )
    print("most confused (true -> predicted, count):")
    for t, p, count in most_confused(cm, 5):
        print(f"  {CLASSES[t]:12s} -> {CLASSES[p]:12s} {count:4d}")
    upper = {0, 2, 4, 6}  # T-shirt/top, Pullover, Coat, Shirt
    inside = sum(cm[i, j] for i in upper for j in upper if i != j)
    print(f"errors among the four upper-body classes: {inside} of {cm.sum() - np.trace(cm)}")
    fig_confusion(cm, test, pred)
    print(f"\nfigures written to {FIG_DIR.relative_to(ROOT)}/ in {time.time() - t0:.0f} s total")


if __name__ == "__main__":
    main()
