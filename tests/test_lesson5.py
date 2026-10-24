import gzip
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lesson5  # noqa: E402


@pytest.fixture(scope="module")
def data():
    """The real Fashion-MNIST files (downloaded once into data/fashion-mnist/)."""
    return lesson5.load_fashion_mnist()


def _write_idx(path: Path, arr: np.ndarray) -> None:
    header = b"\x00\x00\x08" + bytes([arr.ndim]) + struct.pack(">" + "I" * arr.ndim, *arr.shape)
    with gzip.open(path, "wb") as f:
        f.write(header + arr.astype(np.uint8).tobytes())


def test_read_idx_round_trip(tmp_path):
    arr = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    _write_idx(tmp_path / "x.gz", arr)
    assert np.array_equal(lesson5.read_idx(tmp_path / "x.gz"), arr)
    labels = np.array([3, 1, 4], np.uint8)
    _write_idx(tmp_path / "y.gz", labels)
    assert np.array_equal(lesson5.read_idx(tmp_path / "y.gz"), labels)
    with gzip.open(tmp_path / "bad.gz", "wb") as f:
        f.write(b"\x00\x00\x08\x01" + struct.pack(">I", 5) + b"\x01\x02")  # promises 5, has 2
    with pytest.raises(ValueError, match="payload"):
        lesson5.read_idx(tmp_path / "bad.gz")


def test_ensure_file_refuses_a_wrong_hash(tmp_path):
    name = "t10k-labels-idx1-ubyte.gz"
    (tmp_path / name).write_bytes(b"not the data set")
    with pytest.raises(ValueError, match="SHA-256"):
        lesson5.ensure_file(name, root=tmp_path)
    # and never reads it: the file is left for the user to delete
    assert (tmp_path / name).read_bytes() == b"not the data set"


def test_stratified_split_is_disjoint_balanced_and_repeatable():
    y = np.repeat(np.arange(10), [100, 50, 60, 70, 80, 90, 100, 110, 120, 20])
    tr, val = lesson5.stratified_split(y, 0.1, seed=4)
    assert len(np.intersect1d(tr, val)) == 0 and len(tr) + len(val) == len(y)
    assert np.array_equal(np.sort(np.concatenate([tr, val])), np.arange(len(y)))
    assert lesson5.class_counts(y[val]).tolist() == [10, 5, 6, 7, 8, 9, 10, 11, 12, 2]
    tr2, val2 = lesson5.stratified_split(y, 0.1, seed=4)
    assert np.array_equal(val, val2) and np.array_equal(tr, tr2)
    _, val3 = lesson5.stratified_split(y, 0.1, seed=5)
    assert not np.array_equal(val, val3)


def test_confusion_matrix_and_metrics_by_hand():
    y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    p = np.array([0, 0, 1, 1, 0, 2, 2, 0, 1])
    cm = lesson5.confusion_matrix(y, p, k=3)
    assert cm.tolist() == [[2, 1, 0], [1, 1, 0], [1, 1, 2]]
    m = lesson5.metrics_from_confusion(cm)
    assert m.accuracy == pytest.approx(5 / 9)
    assert m.precision.tolist() == pytest.approx([2 / 4, 1 / 3, 1.0])
    assert m.recall.tolist() == pytest.approx([2 / 3, 1 / 2, 2 / 4])
    f1_0 = 2 * (2 / 4) * (2 / 3) / (2 / 4 + 2 / 3)
    assert m.f1[0] == pytest.approx(f1_0) and m.f1[2] == pytest.approx(2 / 3)
    assert m.macro_f1 == pytest.approx(m.f1.mean())
    assert lesson5.most_confused(cm, 2)[0][2] == 1 and len(lesson5.most_confused(cm, 2)) == 2
    # a class never predicted gets precision 0, not a division error
    cm2 = lesson5.confusion_matrix(np.array([0, 1]), np.array([0, 0]), k=2)
    assert lesson5.metrics_from_confusion(cm2).precision.tolist() == [0.5, 0.0]


def test_shift_keeps_pixels_and_flip_mirrors():
    g = torch.Generator().manual_seed(0)
    x = torch.zeros(16, 1, 28, 28)
    x[:, :, 6:22, 6:22] = torch.rand(16, 1, 16, 16, generator=g)  # a 6 px black border
    s = lesson5.shift(x, g)
    assert s.shape == x.shape and s.dtype == x.dtype
    assert torch.allclose(s.sum(dim=(1, 2, 3)), x.sum(dim=(1, 2, 3)))  # nothing fell off
    assert not torch.equal(s, x)
    moved = [(s[i] != x[i]).any().item() for i in range(16)]
    assert sum(moved) >= 12  # offset (2, 2) is the identity, 1 case in 25
    f = lesson5.hflip(x, g, p=1.0)
    assert torch.equal(f, torch.flip(x, dims=[3]))
    assert torch.equal(lesson5.hflip(x, g, p=0.0), x)


def test_models_have_the_quoted_sizes():
    assert lesson5.n_parameters(lesson5.LinearNet()) == 28 * 28 * 10 + 10
    assert lesson5.n_parameters(lesson5.SmallNet()) == 20_490
    out = lesson5.SmallNet()(torch.zeros(2, 1, 28, 28))
    assert out.shape == (2, 10)


def test_real_data_has_the_published_shape(data):
    full, test = data
    assert full.x.shape == (60_000, 28, 28) and full.x.dtype == np.uint8
    assert test.x.shape == (10_000, 28, 28)
    assert lesson5.class_counts(full.y).tolist() == [6_000] * 10
    assert lesson5.class_counts(test.y).tolist() == [1_000] * 10
    tr_idx, val_idx = lesson5.stratified_split(full.y, lesson5.VAL_FRACTION, lesson5.SPLIT_SEED)
    assert len(val_idx) == 6_000 and lesson5.class_counts(full.y[val_idx]).tolist() == [600] * 10


def test_training_keeps_the_best_epoch_and_beats_the_baseline(data):
    """A short run on 2 000 images (5 epochs of 63 batches): the returned weights are
    the best epoch's, and the accuracy is well above the 10 % majority baseline (the
    lesson's 1 000-image runs reach 83 % after 40 epochs)."""
    full, _ = data
    tr_idx, val_idx = lesson5.stratified_split(full.y, lesson5.VAL_FRACTION, lesson5.SPLIT_SEED)
    tr, val = full.take(tr_idx), full.take(val_idx)
    sub = lesson5.subset(tr, 2_000, seed=0)
    assert lesson5.class_counts(sub.y).tolist() == [200] * 10
    net, hist = lesson5.train(lesson5.SmallNet, sub, val, epochs=5, seed=0, batch=32)
    assert len(hist.val_acc) == 5 and hist.train_loss[-1] < hist.train_loss[0]
    _, acc = lesson5.evaluate(net, val)
    assert acc == pytest.approx(max(hist.val_acc), abs=1e-6)
    assert 0.72 < acc < 0.90
    pred = lesson5.predict(net, val.x)
    cm = lesson5.confusion_matrix(val.y, pred)
    assert cm.sum() == len(val) and np.trace(cm) == round(acc * len(val))


def test_augment_then_split_leaks(data):
    full, _ = data
    sub = lesson5.subset(full, 500, seed=0)
    tr, val, src, in_train = lesson5.augment_then_split(sub, copies=3, seed=0)
    assert len(tr) + len(val) == 2_000 and len(val) == 200
    assert np.isin(src, in_train).mean() > 0.95
    assert tr.x.dtype == np.uint8 and tr.x.shape[1:] == (28, 28)
