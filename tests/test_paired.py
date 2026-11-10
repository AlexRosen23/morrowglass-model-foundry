"""Paired comparison: per-token bits match the project's evaluators, and byte alignment is tokenizer-independent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import tiny_cfg
from morrowglass.data.loader import encode_split
from morrowglass.eval.metrics import evaluate_lm
from morrowglass.eval.ngram import KneserNey
from morrowglass.eval.paired import (
    paired_bootstrap,
    paired_report_md,
    per_group,
    play_starts,
    token_bits_kn,
    token_bits_transformer,
)
from morrowglass.tokenizer.tokenizers import get_tokenizer
from morrowglass.train.trainer import load_model_from_checkpoint, train_from_config


def test_transformer_bits_reproduce_evaluate_lm(tmp_path: Path) -> None:
    s = train_from_config(tiny_cfg(max_steps=10, experiment="unit"), quiet=True, runs_dir=tmp_path / "r",
                          ckpt_root=tmp_path / "c")
    model, _ = load_model_from_checkpoint(tmp_path / "c" / s["run_id"] / "best.pt")
    va = encode_split("shakespeare-sample", "val", get_tokenizer("byte", "shakespeare-sample"))
    bits = token_bits_transformer(model, va)
    assert np.isnan(bits[0]) and not np.isnan(bits[1:]).any()
    bpb = np.nansum(bits) / va.byte_len[va.ids][1:].sum()
    assert bpb == pytest.approx(evaluate_lm(model, va, "cpu", ci=False)["bits_per_byte"], rel=1e-6)


def test_kn_bits_reproduce_kn_evaluate() -> None:
    tok = get_tokenizer("byte", "shakespeare-sample")
    tr, va = (encode_split("shakespeare-sample", s, tok) for s in ("train", "val"))
    kn = KneserNey(3, tok.vocab_size, 1).fit(tr.ids)
    ids = va.ids[:5000]
    bits = token_bits_kn(kn, ids)
    assert np.nansum(bits) / va.byte_len[ids][1:].sum() == pytest.approx(
        kn.evaluate(ids, va.byte_len)["bits_per_byte"], rel=1e-9)


def test_byte_blocks_line_up_across_tokenizers() -> None:
    """Every tokenizer puts the same bytes in each block, up to one token straddling a boundary."""
    byte = encode_split("shakespeare-sample", "val", get_tokenizer("byte", "shakespeare-sample"))
    bpe = encode_split("shakespeare-sample", "val", get_tokenizer("bpe1024", "shakespeare-sample"))
    assert byte.n_bytes == bpe.n_bytes
    block = 4096
    groups = np.arange(byte.n_bytes) // block
    n = int(groups[-1]) + 1
    ones_byte, ones_bpe = np.ones(len(byte.ids)), np.ones(len(bpe.ids))
    _, b1 = per_group(ones_byte, byte, groups, n)
    _, b2 = per_group(ones_bpe, bpe, groups, n)
    assert b1.sum() == b2.sum() == byte.n_bytes
    assert np.abs(b1 - b2).max() <= int(bpe.byte_len.max())


def test_paired_bootstrap_signs() -> None:
    rng = np.random.default_rng(0)
    nbytes = np.full(60, 8192.0)
    hard = rng.uniform(1.8, 2.4, 60) * nbytes  # shared difficulty, much larger than the model gap
    same = paired_bootstrap(hard, nbytes, hard, nbytes)
    assert same["difference"] == 0 and same["ci95"][0] <= 0 <= same["ci95"][1]
    better = paired_bootstrap(hard - 0.01 * nbytes, nbytes, hard, nbytes)
    assert better["ci95"][1] < 0 and better["blocks_where_a_is_lower"] == 60
    assert better["difference"] == pytest.approx(-0.01)


def test_play_starts_are_byte_offsets() -> None:
    text = "ÉPILOGUE\nACT ONE\nhello\nACT TWO\nbye"
    assert play_starts(text, ["ACT TWO", "ACT ONE"]) == [("ACT ONE", 10), ("ACT TWO", 24)]
    with pytest.raises(ValueError):
        play_starts(text, ["ACT THREE"])


def test_report_renders_from_the_record() -> None:
    from morrowglass.repro import REPO_ROOT, read_json

    p = REPO_ROOT / "reports" / "paired" / "summary.json"
    if not p.exists():
        pytest.skip("paired study not measured in this checkout")
    s = read_json(p)
    md = paired_report_md(s)
    flag = s["runs"][s["primary_run"]]["paired"][str(s["protocol"]["primary_block_bytes"])]
    assert f"{flag['ci95'][1]:.4f}" in md and "Post-hoc" in md
    assert (REPO_ROOT / "docs" / "PAIRED_RESULTS.md").read_text() == md
