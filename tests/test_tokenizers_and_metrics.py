"""Tokenizer round-trips, bits-per-byte maths, n-gram normalisation and calibration."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from morrowglass.data.corpus import load_split
from morrowglass.data.loader import TokenSplit, eval_windows, token_byte_lengths
from morrowglass.eval.metrics import bootstrap_ci, calibration_report, evaluate_lm
from morrowglass.eval.ngram import KneserNey, pack, uniform_eval, unigram_eval
from morrowglass.model.transformer import ModelConfig, MorrowglassLM
from morrowglass.tokenizer.tokenizers import BPETokenizer, ByteTokenizer

TEXTS = ["To be, or not to be", "naïve café — ünïcödé ✓ 🎭", "", "line\n\n\ttabs  spaces", "x" * 300]


@pytest.fixture(scope="module")
def bpe() -> BPETokenizer:
    return BPETokenizer.train(load_split("shakespeare-sample", "train"), 300)


@pytest.mark.parametrize("text", TEXTS)
def test_byte_roundtrip(text: str) -> None:
    t = ByteTokenizer()
    assert t.decode(t.encode(text)) == text


@pytest.mark.parametrize("text", TEXTS)
def test_bpe_roundtrip_and_byte_lengths(bpe: BPETokenizer, text: str) -> None:
    ids = bpe.encode(text)
    assert bpe.decode(ids) == text
    assert sum(len(bpe.token_bytes(i)) for i in ids) == len(text.encode())


def test_bpe_compresses_and_is_deterministic(bpe: BPETokenizer) -> None:
    text = load_split("shakespeare-sample", "val")[:5000]
    assert len(bpe.encode(text)) < 0.8 * len(text.encode())  # only 44 merges at vocab 300
    again = BPETokenizer.train(load_split("shakespeare-sample", "train"), 300)
    assert again.fingerprint() == bpe.fingerprint()


def test_eval_windows_cover_every_target_once() -> None:
    for n, T in [(10, 4), (33, 32), (2, 8), (100, 7)]:
        wins = eval_windows(n, T)
        covered = [s + 1 + j for s, L in wins for j in range(L)]
        assert covered == list(range(1, n))


def test_bpb_equals_loss_over_ln2_for_bytes() -> None:
    torch.manual_seed(0)
    model = MorrowglassLM(ModelConfig(vocab_size=256, block_size=16, n_layer=1, n_head=2, d_model=16))
    ids = np.frombuffer(b"the quick brown fox jumps over the lazy dog " * 4, dtype=np.uint8).astype(np.int64)
    split = TokenSplit(ids=ids, byte_len=token_byte_lengths(ByteTokenizer()), n_bytes=len(ids), sha256="x")
    r = evaluate_lm(model, split, "cpu", ci=True, calibration=True)
    assert r["bits_per_byte"] == pytest.approx(r["loss_nats_per_token"] / math.log(2), rel=1e-9)
    assert r["tokens_predicted"] == len(ids) - 1
    lo, hi = r["bits_per_byte_ci95"]
    assert lo <= r["bits_per_byte"] <= hi
    assert 0 <= r["calibration"]["ece"] <= 1


def test_uniform_baseline_is_exactly_log2_vocab() -> None:
    ids = np.arange(100) % 256
    r = uniform_eval(ids, np.ones(256, dtype=np.int64), 256)
    assert r["bits_per_byte"] == pytest.approx(8.0)


@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_kneser_ney_distribution_sums_to_one(order: int) -> None:
    ids = np.frombuffer(load_split("shakespeare-sample", "train")[:20000].encode(), dtype=np.uint8).astype(np.int64)
    kn = KneserNey(order, 256, 1).fit(ids)
    for hist in [b"", b"th", b"the ", b"\x00\x01\x02\x03", b"zqx"]:
        d = kn.distribution(hist[-(order - 1):] if order > 1 else b"")
        assert d.sum() == pytest.approx(1.0, abs=1e-9)
        assert (d > 0).all()


def test_kneser_ney_two_byte_packing() -> None:
    ids = np.array([300, 2, 300, 2, 300, 7, 1023])
    kn = KneserNey(2, 1024, 2).fit(ids)
    p = kn.distribution(pack(np.array([300]), 2))
    assert p.sum() == pytest.approx(1.0)
    assert p[2] > p[7] > p[5]


def test_ngram_beats_unigram_beats_uniform_on_held_out() -> None:
    tr = np.frombuffer(load_split("shakespeare-sample", "train").encode(), dtype=np.uint8).astype(np.int64)
    va = np.frombuffer(load_split("shakespeare-sample", "val")[:20000].encode(), dtype=np.uint8).astype(np.int64)
    bl = np.ones(256, dtype=np.int64)
    u = uniform_eval(va, bl, 256)["bits_per_byte"]
    g = unigram_eval(tr, va, bl, 256)["bits_per_byte"]
    k = KneserNey(4, 256, 1).fit(tr).evaluate(va, bl)["bits_per_byte"]
    assert k < g < u


def test_bootstrap_ci_brackets_ratio() -> None:
    rng = np.random.default_rng(0)
    nll = rng.uniform(1, 2, size=200)
    den = np.ones(200)
    lo, hi = bootstrap_ci(nll, den)
    assert lo < nll.mean() < hi and hi - lo < 0.2


def test_calibration_perfect_and_overconfident() -> None:
    rng = np.random.default_rng(1)
    conf = rng.uniform(0, 1, 50000)
    correct = rng.uniform(0, 1, 50000) < conf
    assert calibration_report(conf, correct)["ece"] < 0.02
    over = calibration_report(np.full(1000, 0.9), np.zeros(1000, dtype=bool))
    assert over["ece"] == pytest.approx(0.9)
