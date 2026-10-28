"""Lesson 6 - Object detection with a pre-trained model: fifty images, mAP, the edge.

Runs every piece of code shown in lesson6-object-detection.md, prints the numbers
the lesson quotes, and writes the figures to figures/lesson6/.

    python lesson6.py            # downloads yolov8n.pt (6 MB) once, then trains on the CPU

The detector is Ultralytics YOLOv8n with its COCO weights, downloaded from the
Ultralytics GitHub release and checked against the SHA-256 recorded below. It is
fine-tuned on eighty synthetic belt frames rendered by make_scene() with lesson 2's
rules (50 train, 10 validation, 20 test, split by frame), compared with the same
network trained from scratch at the same budget, scored with an average precision
written out here in NumPy and checked against the library's, run on the frames of
lessons 2-4, and exported to ONNX for a latency table. Nothing here is a
photograph except data/coffee.png, which the COCO model sees in section 0.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")  # must be set before ultralytics is imported

import cv2  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

import lesson2  # noqa: E402
import lesson3  # noqa: E402
import lesson4  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures" / "lesson6"
DATA_DIR = ROOT / "data" / "eggs-synthetic"  # gitignored; rewritten on every run
DOWNLOADS = ROOT / "data" / "downloads"  # gitignored; the COCO weights land here
RUNS_DIR = DOWNLOADS / "lesson6-runs"  # gitignored; training logs, weights, ONNX files

WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
WEIGHTS_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
COCO_CONF = 0.25  # the library's default display threshold, used for the out-of-the-box look

SPLITS = {"train": (50, 1_000), "val": (10, 2_000), "test": (20, 3_000)}  # frames, first seed
FRAME_SHAPE = (480, 640)
IMGSZ = 640  # the long side the detector is trained and scored at
EPOCHS = 20
BATCH = 8
EDGE_SIZES = (640, 320, 160)  # ONNX exports at three input sizes
IOU_NMS = 0.7  # the library's default for non-maximum suppression
SCORE_FLOOR = 0.001  # keep nearly everything when scoring: average precision ranks by score
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)  # candidate operating points


# --------------------------------------------------------------------------- #
# 0. The COCO weights, fetched once and verified
# --------------------------------------------------------------------------- #
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_weights(url: str = WEIGHTS_URL, root: Path = DOWNLOADS, expected: str | None = None):
    """Download `url` into `root` once and refuse a file whose SHA-256 is not `expected`."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / url.rsplit("/", 1)[1]
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    expected = WEIGHTS_SHA256 if expected is None else expected
    got = sha256_of(path)
    if got != expected:
        raise ValueError(f"{path.name}: SHA-256 {got} does not match the recorded {expected}")
    return path


# --------------------------------------------------------------------------- #
# 1. Fifty frames with boxes: a synthetic belt where the labels are free
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scene:
    """A rendered belt frame and the box around every egg on it."""

    rgb: np.ndarray
    boxes: np.ndarray  # (n, 4) rows of x1, y1, x2, y2 in pixels

    @property
    def n_eggs(self) -> int:
        return len(self.boxes)


def ellipse_box(r: float, c: float, a: float, b: float, deg: float) -> np.ndarray:
    """The axis-aligned box around an ellipse with half-axes a, b rotated by `deg`."""
    t = np.deg2rad(deg)
    hw = np.sqrt((a * np.cos(t)) ** 2 + (b * np.sin(t)) ** 2)
    hh = np.sqrt((a * np.sin(t)) ** 2 + (b * np.cos(t)) ** 2)
    return np.array([c - hw, r - hh, c + hw, r + hh], np.float32)


def make_scene(seed: int, shape: tuple[int, int] = FRAME_SHAPE) -> Scene:
    """Lesson 2's belt with 2-12 eggs at random places, sizes and angles, up to three
    labels and two blobs (lesson 4's distractors) kept clear of the eggs, dirt, a
    lamp of random strength and direction, read noise and dead pixels.

    Every frame has a different answer, which is what makes eighty of them a data
    set rather than one belt. Nothing here is a photograph.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    refl = np.full((h, w), 0.45, np.float32)
    refl += 0.02 * np.sin(np.arange(w) / 3.0 + rng.uniform(0, 2 * np.pi))[None, :]
    eggs: list[tuple[float, float, float, float, float]] = []
    n_eggs = int(rng.integers(2, 13))
    for _ in range(500):  # rejection sampling: centres at least 48 px apart, so pairs may touch
        if len(eggs) == n_eggs:
            break
        r, c = rng.uniform(40, h - 40), rng.uniform(40, w - 40)
        if all(np.hypot(r - r0, c - c0) >= 48 for r0, c0, *_ in eggs):
            eggs.append((r, c, rng.uniform(26, 34), rng.uniform(19, 24), rng.uniform(0, 180)))
    for r, c, a, b, deg in eggs:
        lesson4._draw_egg(refl, (r, c), a, b, deg)

    def clear_of_eggs(r: float, c: float, margin: float) -> bool:
        return all(np.hypot(r - r0, c - c0) >= margin for r0, c0, *_ in eggs)

    for _ in range(int(rng.integers(0, 4))):  # labels: bright rectangles with corners
        r, c = rng.uniform(30, h - 30), rng.uniform(30, w - 30)
        if clear_of_eggs(r, c, 75):
            box = ((c, r), (rng.uniform(40, 64), rng.uniform(24, 44)), rng.uniform(0, 180))
            pts = cv2.boxPoints(box).astype(np.int32)
            cv2.fillPoly(refl, [pts], float(rng.uniform(0.80, 0.92)))
    for _ in range(int(rng.integers(0, 3))):  # blobs: bright discs, smaller than an egg
        r, c = rng.uniform(30, h - 30), rng.uniform(30, w - 30)
        if clear_of_eggs(r, c, 60):
            cv2.circle(refl, (int(c), int(r)), int(rng.integers(10, 19)), 0.86, -1)
    lesson4._specks(refl, rng, 40)  # dirt lands on eggs too
    lo, hi = rng.uniform(0.30, 1.0, 2)  # the lamp may be on either side, at any strength
    light = np.linspace(lo, hi, w, dtype=np.float32)[None, :]
    clean = np.clip(refl * light * 255.0, 0, 255)
    noisy = clean + rng.normal(0.0, 9.0, clean.shape).astype(np.float32)
    salt = rng.random(clean.shape)
    noisy[salt < 0.005] = 0.0
    noisy[salt > 0.995] = 255.0
    gray = np.clip(noisy, 0, 255).astype(np.uint8)
    boxes = np.array([ellipse_box(*e) for e in eggs], np.float32).reshape(-1, 4)
    return Scene(rgb=lesson2._tint(gray), boxes=boxes)


def scenes_of(split: str) -> list[Scene]:
    n, seed0 = SPLITS[split]
    return [make_scene(seed0 + i) for i in range(n)]


def write_labels(path: Path, boxes: np.ndarray, shape: tuple[int, int]) -> None:
    """One line per box in the YOLO text format: class, centre x, centre y, width,
    height, all four as fractions of the image size."""
    h, w = shape
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def read_labels(path: Path, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    rows = []
    for line in path.read_text().splitlines():
        _, cx, cy, bw, bh = (float(v) for v in line.split())
        rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, np.float32).reshape(-1, 4)


def write_dataset(root: Path = DATA_DIR) -> Path:
    """Render every split to disk in the layout the library reads and return the
    data.yaml path. The frames are regenerated from their seeds on every run."""
    for split, (n, seed0) in SPLITS.items():
        img_dir, lab_dir = root / "images" / split, root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            scene = make_scene(seed0 + i)
            cv2.imwrite(
                str(img_dir / f"{split}_{i:03d}.png"), cv2.cvtColor(scene.rgb, cv2.COLOR_RGB2BGR)
            )
            write_labels(lab_dir / f"{split}_{i:03d}.txt", scene.boxes, scene.rgb.shape[:2])
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: egg\n"
    )
    return yaml_path


# --------------------------------------------------------------------------- #
# 2. Boxes, overlap and average precision, written out
# --------------------------------------------------------------------------- #
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over union of every box in `a` (n, 4) with every box in `b` (m, 4)."""
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match_detections(
    dets: list[np.ndarray], truths: list[np.ndarray], iou_thr: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Rank every detection in the set by score and decide, in that order, whether
    it is a true positive: the best-overlapping unclaimed truth in its own image
    must overlap by at least `iou_thr`. Returns (scores, is_tp) in rank order and
    the number of truths."""
    scores, hits = [], []
    n_truth = 0
    rows = []
    for i, d in enumerate(dets):
        for row in np.asarray(d, np.float32).reshape(-1, 5):
            rows.append((row[4], i, row[:4]))
    rows.sort(key=lambda t: -t[0])
    claimed = [np.zeros(len(np.asarray(t).reshape(-1, 4)), bool) for t in truths]
    n_truth = sum(len(c) for c in claimed)
    for score, i, box in rows:
        tp = False
        if len(claimed[i]):
            ious = iou_matrix(box, truths[i])[0]
            ious[claimed[i]] = -1.0
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr:
                claimed[i][j] = True
                tp = True
        scores.append(score)
        hits.append(tp)
    return np.array(scores, np.float32), np.array(hits, bool), n_truth


def precision_recall(is_tp: np.ndarray, n_truth: int) -> tuple[np.ndarray, np.ndarray]:
    """Precision and recall after each detection in rank order."""
    tp = np.cumsum(is_tp)
    fp = np.cumsum(~is_tp)
    return tp / np.maximum(tp + fp, 1), tp / max(n_truth, 1)


def average_precision(precision: np.ndarray, recall: np.ndarray, points: int = 101) -> float:
    """COCO's definition: the mean, over `points` evenly spaced recall levels from 0
    to 1, of the best precision reached at that recall or any higher recall."""
    ap = 0.0
    for r in np.linspace(0, 1, points):
        above = precision[recall >= r]
        ap += above.max() if len(above) else 0.0
    return ap / points


def ap_at(dets: list[np.ndarray], truths: list[np.ndarray], iou_thr: float) -> float:
    scores, hits, n_truth = match_detections(dets, truths, iou_thr)
    p, r = precision_recall(hits, n_truth)
    return average_precision(p, r)


def map_50_95(dets: list[np.ndarray], truths: list[np.ndarray]) -> float:
    return float(np.mean([ap_at(dets, truths, t) for t in np.arange(0.5, 0.96, 0.05)]))


def counts_at(dets: list[np.ndarray], truths: list[np.ndarray], thr: float, iou_thr=0.5):
    """Precision, recall and mean absolute count error when only detections scoring
    at least `thr` are kept: the numbers an operating point is chosen on."""
    kept = [d[d[:, 4] >= thr] for d in dets]
    _, hits, n_truth = match_detections(kept, truths, iou_thr)
    tp = int(hits.sum())
    fp = len(hits) - tp
    count_err = float(np.mean([abs(len(k) - len(t)) for k, t in zip(kept, truths, strict=True)]))
    return tp / max(tp + fp, 1), tp / max(n_truth, 1), count_err


def choose_threshold(table: list[tuple[float, float, float, float]]) -> float:
    """Rows of (threshold, precision, recall, count error) on the validation frames.
    Keep the rows with the lowest count error, among them the highest precision,
    among them the highest recall, and return the middle of that plateau: an
    operating point at the edge of a plateau fails as soon as the scores shift."""
    best = min(row[3] for row in table)
    rows = [row for row in table if row[3] == best]
    rows = [row for row in rows if row[1] == max(r[1] for r in rows)]
    rows = [row for row in rows if row[2] == max(r[2] for r in rows)]
    return rows[(len(rows) - 1) // 2][0]


# --------------------------------------------------------------------------- #
# 3. Training and predicting through the library
# --------------------------------------------------------------------------- #
def _yolo(source):
    from ultralytics import YOLO, settings
    from ultralytics.utils import LOGGER

    settings.update({"sync": False})  # no analytics from the sandbox
    LOGGER.setLevel(logging.WARNING)  # the lesson prints its own numbers
    return YOLO(str(source))


def train_detector(name: str, source: Path | str, data_yaml: Path, pretrained: bool):
    """Fine-tune (or train from scratch) with the library's default recipe on the
    CPU; the validation set picks the kept epoch. Returns best.pt and the per-epoch log."""
    model = _yolo(source)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device="cpu",
        workers=0,
        seed=0,
        deterministic=True,
        pretrained=pretrained,
        project=str(RUNS_DIR),
        name=name,
        exist_ok=True,
        plots=False,
        verbose=False,
        amp=False,
    )
    run = RUNS_DIR / name
    return run / "weights" / "best.pt", read_results(run / "results.csv")


def read_results(path: Path) -> list[dict[str, float]]:
    with open(path, newline="") as f:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    return rows


def predict_boxes(model, frames: list[np.ndarray], imgsz: int = IMGSZ, conf=SCORE_FLOOR):
    """Run the library's pipeline (letterbox, network, NMS) and return, per frame,
    (k, 5) rows of x1, y1, x2, y2, score in frame pixels."""
    bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
    out = []
    for r in model.predict(bgr, imgsz=imgsz, conf=conf, iou=IOU_NMS, device="cpu", verbose=False):
        xyxy = r.boxes.xyxy.cpu().numpy().reshape(-1, 4)
        score = r.boxes.conf.cpu().numpy().reshape(-1, 1)
        out.append(np.hstack([xyxy, score]).astype(np.float32))
    return out


def coco_detections(
    model, rgb: np.ndarray, conf: float = COCO_CONF
) -> list[tuple[str, float, np.ndarray]]:
    r = model.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), conf=conf, device="cpu", verbose=False)[
        0
    ]
    names = r.names
    return [
        (names[int(c)], float(s), b)
        for c, s, b in zip(
            r.boxes.cls.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            r.boxes.xyxy.cpu().numpy(),
            strict=True,
        )
    ]


def n_parameters(model) -> int:
    return sum(p.numel() for p in model.model.parameters())


# --------------------------------------------------------------------------- #
# 4. The edge: ONNX, and the pipeline the library hides, written out
# --------------------------------------------------------------------------- #
def letterbox(rgb: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize so the longer side is `size`, then pad with grey to size x size, centred.
    Returns the square image, the scale and the (x, y) padding of the top-left corner."""
    h, w = rgb.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[py : py + nh, px : px + nw] = resized
    return canvas, scale, (px, py)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = IOU_NMS) -> np.ndarray:
    """Greedy non-maximum suppression: keep the strongest box, drop every box that
    overlaps it by more than `iou_thr`, repeat. Returns the kept indices, strongest first."""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        ious = iou_matrix(boxes[i], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.array(keep, int)


def decode(raw: np.ndarray, scale: float, pad: tuple[int, int], conf: float = SCORE_FLOOR):
    """Turn the network's raw (1, 4 + classes, anchors) output into (k, 5) rows of
    x1, y1, x2, y2, score in the original frame's pixels, after NMS."""
    pred = raw[0].T  # one row per anchor: cx, cy, w, h, then one score per class
    scores = pred[:, 4:].max(axis=1)
    keep = scores >= conf
    pred, scores = pred[keep], scores[keep]
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes /= scale
    kept = nms(boxes, scores)
    return np.hstack([boxes[kept], scores[kept, None]]).astype(np.float32).reshape(-1, 5)


def onnx_session(path: Path, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def onnx_detect(session, rgb: np.ndarray, size: int) -> np.ndarray:
    square, scale, pad = letterbox(rgb, size)
    x = square.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    raw = session.run(None, {session.get_inputs()[0].name: x})[0]
    return decode(raw, scale, pad)


def latency_ms(session, size: int, repeats: int = 10) -> float:
    """Median wall time of one forward pass on a random frame, after three warm-ups."""
    x = np.random.default_rng(0).random((1, 3, size, size), np.float32)
    name = session.get_inputs()[0].name
    for _ in range(3):
        session.run(None, {name: x})
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        session.run(None, {name: x})
        times.append((time.perf_counter() - t) * 1e3)
    return float(np.median(times))


# --------------------------------------------------------------------------- #
# 5. The frames of lessons 2-4, and counting
# --------------------------------------------------------------------------- #
def centres_of(dets: np.ndarray) -> np.ndarray:
    """Box centres as (k, 3) rows of x, y, score, the shape lesson 4's scorer reads."""
    d = np.asarray(dets, np.float32).reshape(-1, 5)
    return np.stack([(d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2, d[:, 4]], axis=1)


def earlier_frames() -> list[tuple[str, np.ndarray, list[tuple[int, int]]]]:
    """Lesson 4's three scoring frames, the first at half size, and lesson 3's two
    oblique views, each with its true egg centres (row, col)."""
    frames = []
    for name, belt in lesson4.frames_for_scoring():
        frames.append((name, belt.rgb, belt.centres))
    t0 = lesson2.make_belt()
    half = cv2.resize(t0.rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    frames.append(("t0 at half size", half, [(r // 2, c // 2) for r, c in t0.centres]))
    scene = lesson3.make_scene()
    for i, (frame, truth) in enumerate(zip(scene.frames, scene.truth, strict=True)):
        pts = lesson3.project(np.array([(c, r) for r, c in truth.centres], np.float32), scene.H_cam)
        frames.append(
            (f"oblique t{i} (lesson 3)", frame, [(int(round(y)), int(round(x))) for x, y in pts])
        )
    return frames


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _draw(
    ax, rgb: np.ndarray, boxes: np.ndarray, labels: list[str] | None, colour: str, title: str
):
    ax.imshow(rgb)
    for k, (x1, y1, x2, y2) in enumerate(np.asarray(boxes).reshape(-1, 4)):
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=colour, lw=1.2))
        if labels is not None:
            ax.text(x1 + 1, y1 + 1, labels[k], color=colour, fontsize=6, va="top")
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def fig_pretrained(coffee: np.ndarray, coffee_dets, belt: np.ndarray, belt_dets) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, img, dets, title in (
        (ax[0], coffee, coffee_dets, "COCO weights on data/coffee.png"),
        (ax[1], belt, belt_dets, "COCO weights on lesson 2's belt"),
    ):
        _draw(
            a,
            img,
            np.array([b for _, _, b in dets]).reshape(-1, 4),
            [f"{n} {s:.2f}" for n, s, _ in dets],
            "yellow",
            f"{title}: {len(dets)} boxes at score >= {COCO_CONF}",
        )
    _save(fig, "01_pretrained.png")


def fig_dataset(scenes: list[Scene]) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for a, s, i in zip(ax.ravel(), scenes[:4], range(4), strict=True):
        _draw(
            a,
            s.rgb,
            s.boxes,
            None,
            "lime",
            f"train_{i:03d}: {s.n_eggs} eggs, boxes from the ellipses",
        )
    fig.suptitle("Four of the fifty training frames with their free labels", fontsize=10)
    _save(fig, "02_dataset.png")


def fig_finetune(histories: dict[str, list[dict[str, float]]]) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, rows in histories.items():
        ep = [r["epoch"] for r in rows]
        ax[0].plot(ep, [100 * r["metrics/mAP50(B)"] for r in rows], marker="o", ms=3, label=name)
        ax[1].plot(ep, [r["train/box_loss"] for r in rows], marker="o", ms=3, label=name)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("validation AP at IoU 0.5 (%)")
    ax[0].set_ylim(0, 102)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("training box loss")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(
        "The same network, from COCO weights and from random weights, same budget", fontsize=10
    )
    _save(fig, "03_finetune.png")


def fig_map(
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    ap_by_iou: dict[str, list[float]],
    val_table: list[tuple[float, float, float, float]],
    chosen: float,
) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for name, (p, r, ap) in curves.items():
        ax[0].step(r, p, where="post", label=f"{name}: AP {100 * ap:.1f} %")
    ax[0].set_xlabel("recall")
    ax[0].set_ylabel("precision")
    ax[0].set_xlim(0, 1.02)
    ax[0].set_ylim(0, 1.02)
    ax[0].set_title("Precision-recall on the 20 test frames, IoU 0.5", fontsize=9)
    ax[0].legend(fontsize=8)
    thr = np.arange(0.5, 0.96, 0.05)
    for name, aps in ap_by_iou.items():
        ax[1].plot(thr, [100 * a for a in aps], marker="o", ms=3, label=name)
    ax[1].set_xlabel("IoU required for a match")
    ax[1].set_ylabel("AP (%)")
    ax[1].set_ylim(0, 102)
    ax[1].set_title("How much box tightness the AP asks for", fontsize=9)
    ax[1].legend(fontsize=8)
    t = [row[0] for row in val_table]
    ax[2].plot(t, [100 * row[1] for row in val_table], label="precision")
    ax[2].plot(t, [100 * row[2] for row in val_table], label="recall")
    ax[2].plot(t, [10 * row[3] for row in val_table], label="count error x 10")
    ax[2].axvline(chosen, color="k", ls="--", lw=0.8, label=f"chosen {chosen:.2f}")
    ax[2].set_xlabel("score threshold")
    ax[2].set_title("Operating point, chosen on the 10 validation frames", fontsize=9)
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    _save(fig, "04_map.png")


def fig_frames(rows: list[tuple[str, np.ndarray, np.ndarray, lesson4.Score]]) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
    for a, (name, rgb, dets, sc) in zip(ax.ravel(), rows, strict=True):
        _draw(
            a,
            rgb,
            dets[:, :4],
            [f"{s:.2f}" for s in dets[:, 4]],
            "lime",
            f"{name}\n{len(dets)} boxes: {sc.tp} of {len(dets) - sc.fp + sc.fn} eggs, "
            f"{sc.fp} false, {sc.fn} missed",
        )
    fig.suptitle(
        "The fine-tuned detector on the earlier lessons' frames, at the chosen threshold",
        fontsize=10,
    )
    _save(fig, "05_frames.png")


# --------------------------------------------------------------------------- #
def main() -> None:
    t_start = time.time()
    torch_threads = os.cpu_count() or 1
    print(f"== 0. the pre-trained detector out of the box ({torch_threads} CPU cores)")
    weights = ensure_weights()
    print(f"{weights.relative_to(ROOT)}: {weights.stat().st_size / 1e6:.1f} MB, SHA-256 verified")
    coco = _yolo(weights)
    from ultralytics.utils.torch_utils import get_flops

    gflops = get_flops(coco.model, imgsz=IMGSZ)
    print(
        f"YOLOv8n: {n_parameters(coco):,} parameters, {gflops:.1f} GFLOPs at {IMGSZ} px, "
        f"{len(coco.names)} COCO classes"
    )
    coffee = cv2.cvtColor(cv2.imread(str(ROOT / "data" / "coffee.png")), cv2.COLOR_BGR2RGB)
    coffee_dets = coco_detections(coco, coffee)
    print(
        f"coffee.png at score >= {COCO_CONF}: "
        + ", ".join(f"{n} {s:.2f}" for n, s, _ in coffee_dets)
    )
    belt0 = lesson2.make_belt()
    belt_dets = coco_detections(coco, belt0.rgb)
    low = coco_detections(coco, belt0.rgb, conf=0.05)
    print(
        f"lesson 2's belt at score >= {COCO_CONF}: {len(belt_dets)} boxes"
        + (": " + ", ".join(f"{n} {s:.2f}" for n, s, _ in belt_dets) if belt_dets else "")
    )
    print(
        f"lesson 2's belt at score >= 0.05: {len(low)} boxes"
        + (": " + ", ".join(f"{n} {s:.2f}" for n, s, _ in low) if low else "")
    )
    fig_pretrained(coffee, coffee_dets, belt0.rgb, belt_dets)

    print("\n== 1. fifty frames with boxes")
    data_yaml = write_dataset()
    splits = {s: scenes_of(s) for s in SPLITS}
    for s, scenes in splits.items():
        n = [sc.n_eggs for sc in scenes]
        print(f"{s}: {len(scenes)} frames, {sum(n)} eggs ({min(n)} to {max(n)} per frame)")
    wh = np.concatenate([sc.boxes[:, 2:] - sc.boxes[:, :2] for sc in splits["train"]])
    print(
        f"training boxes: width {wh[:, 0].min():.0f}-{wh[:, 0].max():.0f} px, "
        f"height {wh[:, 1].min():.0f}-{wh[:, 1].max():.0f} px, "
        f"in {FRAME_SHAPE[1]} x {FRAME_SHAPE[0]} frames"
    )
    back = read_labels(DATA_DIR / "labels" / "train" / "train_000.txt", FRAME_SHAPE)
    print(f"label round trip: max error {np.abs(back - splits['train'][0].boxes).max():.3f} px")
    fig_dataset(splits["train"])

    print(f"\n== 2. fine-tuning from COCO weights against training from scratch ({EPOCHS} epochs)")
    histories = {}
    best = {}
    for name, source, pretrained in (
        ("fine-tuned", weights, True),
        ("from scratch", "yolov8n.yaml", False),
    ):
        t = time.time()
        best[name], histories[name] = train_detector(
            name.replace(" ", "-"), source, data_yaml, pretrained
        )
        m50 = [100 * r["metrics/mAP50(B)"] for r in histories[name]]
        print(
            f"{name}: {time.time() - t:.0f} s; validation AP50 per epoch: "
            + " ".join(f"{v:.0f}" for v in m50)
            + f"; best {max(m50):.1f} % at epoch {int(np.argmax(m50)) + 1}"
        )
    fig_finetune(histories)

    print("\n== 3. average precision, by hand and by the library, on the 20 test frames")
    truths = [sc.boxes for sc in splits["test"]]
    frames = [sc.rgb for sc in splits["test"]]
    curves, ap_by_iou = {}, {}
    models = {name: _yolo(path) for name, path in best.items()}
    dets = {name: predict_boxes(m, frames) for name, m in models.items()}
    for name in models:
        scores, hits, n_truth = match_detections(dets[name], truths, 0.5)
        p, r = precision_recall(hits, n_truth)
        ap50 = average_precision(p, r)
        curves[name] = (p, r, ap50)
        ap_by_iou[name] = [ap_at(dets[name], truths, t) for t in np.arange(0.5, 0.96, 0.05)]
        lib = models[name].val(
            data=str(data_yaml),
            split="test",
            imgsz=IMGSZ,
            batch=BATCH,
            device="cpu",
            plots=False,
            verbose=False,
            project=str(RUNS_DIR),
            name=f"val-{name}",
            exist_ok=True,
            conf=SCORE_FLOOR,
            iou=IOU_NMS,
        )
        print(
            f"{name}: {len(scores)} detections kept at score >= {SCORE_FLOOR} for {n_truth} eggs; "
            f"AP50 by hand {100 * ap50:.1f} %, library {100 * lib.box.map50:.1f} %; "
            f"AP50-95 by hand {100 * np.mean(ap_by_iou[name]):.1f} %, "
            f"library {100 * lib.box.map:.1f} %"
        )
        print("  AP at IoU 0.50 .. 0.95: " + " ".join(f"{100 * a:.0f}" for a in ap_by_iou[name]))

    print("\n== 4. an operating point, chosen on the 10 validation frames")
    ft = models["fine-tuned"]
    val_dets = predict_boxes(ft, [sc.rgb for sc in splits["val"]])
    val_truth = [sc.boxes for sc in splits["val"]]
    val_table = [(float(t), *counts_at(val_dets, val_truth, float(t))) for t in THRESHOLDS]
    for t, p, r, e in val_table:
        if abs(t * 100 % 25) < 1e-6 or t in (0.05, 0.95):
            print(
                f"  threshold {t:.2f}: precision {100 * p:.1f} %, recall {100 * r:.1f} %, "
                f"count error {e:.2f}"
            )
    chosen = choose_threshold(val_table)
    print(
        f"chosen threshold {chosen:.2f} (middle of the thresholds with the lowest count error, "
        "then the highest precision and recall)"
    )
    p, r, e = counts_at(dets["fine-tuned"], truths, chosen)
    print(
        f"on the test frames at {chosen:.2f}: precision {100 * p:.1f} %, recall {100 * r:.1f} %, "
        f"mean count error {e:.2f} eggs per frame"
    )
    fig_map(curves, ap_by_iou, val_table, chosen)

    print("\n== 5. worked example: the frames of lessons 2-4, then the edge")
    rows = []
    for name, rgb, centres in earlier_frames():
        d = predict_boxes(ft, [rgb])[0]
        d = d[d[:, 4] >= chosen]
        sc = lesson4.score_detections(centres_of(d), centres)
        rows.append((name, rgb, d, sc))
        print(f"  {name}: {len(d)} boxes; {sc.tp} of {len(centres)} eggs, {sc.fp} false alarms")
    fig_frames(rows)

    print("\nONNX export and the hand-written pipeline (letterbox, forward, decode, NMS):")
    many = f"ms, {torch_threads} threads"
    print(f"  {'input':>6} {'file':>8} {'AP50 test':>10} {'ms, 1 thread':>13} {many:>14}")
    for size in EDGE_SIZES:
        onnx_path = Path(
            ft.export(format="onnx", imgsz=size, simplify=False, dynamic=False, verbose=False)
        )
        sess4 = onnx_session(onnx_path, torch_threads)
        sess1 = onnx_session(onnx_path, 1)
        edge_dets = [onnx_detect(sess4, f, size) for f in frames]
        ap = ap_at(edge_dets, truths, 0.5)
        print(
            f"  {size:>6} {onnx_path.stat().st_size / 1e6:>6.1f} MB {100 * ap:>9.1f} % "
            f"{latency_ms(sess1, size):>13.0f} {latency_ms(sess4, size):>14.0f}"
        )
    print(f"\ndone in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
