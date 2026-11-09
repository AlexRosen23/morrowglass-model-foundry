"""Generation determinism, sampling constraints and attention extraction."""

from __future__ import annotations

import pytest
import torch

from morrowglass.inference.generate import MAX_NEW_TOKENS, _filtered_probs, attention_maps, generate
from morrowglass.model.transformer import ModelConfig, MorrowglassLM
from morrowglass.tokenizer.tokenizers import ByteTokenizer


@pytest.fixture(scope="module")
def model() -> MorrowglassLM:
    torch.manual_seed(0)
    return MorrowglassLM(ModelConfig(vocab_size=256, block_size=16, n_layer=1, n_head=2, d_model=16)).eval()


def test_generation_deterministic_under_seed(model: MorrowglassLM) -> None:
    tok = ByteTokenizer()
    a = generate(model, tok, "ROMEO:", max_new_tokens=30, seed=7)
    b = generate(model, tok, "ROMEO:", max_new_tokens=30, seed=7)
    c = generate(model, tok, "ROMEO:", max_new_tokens=30, seed=8)
    assert a["tokens"] == b["tokens"]
    assert a["tokens"] != c["tokens"]


def test_long_prompt_is_cropped_to_context(model: MorrowglassLM) -> None:
    out = generate(model, ByteTokenizer(), "x" * 100, max_new_tokens=5, seed=0)
    assert len(out["tokens"]) == 5


def test_top_k_and_greedy() -> None:
    logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
    p = _filtered_probs(logits, 1.0, top_k=2)
    assert (p[[0, 3]] == 0).all() and p.sum() == pytest.approx(1.0)
    g = _filtered_probs(logits, 0.0, None)
    assert g.tolist() == [0.0, 1.0, 0.0, 0.0]


def test_token_records_are_labelled_uncalibrated(model: MorrowglassLM) -> None:
    out = generate(model, ByteTokenizer(), "a", max_new_tokens=3, top_k=5)
    assert "not calibrated" in out["note"]
    for t in out["tokens"]:
        assert 0 < t["prob"] <= 1 and 0 < t["raw_prob"] <= 1 and len(t["top"]) == 5


def test_max_new_tokens_cap(model: MorrowglassLM) -> None:
    with pytest.raises(ValueError):
        generate(model, ByteTokenizer(), "a", max_new_tokens=MAX_NEW_TOKENS + 1)


def test_attention_maps_shape_and_caveat(model: MorrowglassLM) -> None:
    r = attention_maps(model, ByteTokenizer(), "hello there", max_tokens=8)
    assert len(r["tokens"]) == 8 and r["n_layer"] == 1
    assert len(r["attention"][0]) == 2 and len(r["attention"][0][0]) == 8
    assert "not a causal explanation" in r["caveat"]
