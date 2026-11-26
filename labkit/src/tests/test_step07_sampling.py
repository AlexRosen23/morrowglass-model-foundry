"""Step 7: temperature, top-k and the generation loop (without the cache)."""

from __future__ import annotations

import pytest
import torch

from scratchlm.data import sample_batch
from scratchlm.sample import generate, next_token_probs
from scratchlm.train import TrainConfig, make_optimizer, train_step
from tests.conftest import pattern_ids, tiny_model


def test_step07_temperature_zero_is_greedy() -> None:
    p = next_token_probs(torch.tensor([0.1, 3.0, -1.0, 2.9]), temperature=0)
    assert p.tolist() == [0.0, 1.0, 0.0, 0.0]


def test_step07_temperature_sharpens_or_flattens() -> None:
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    assert torch.allclose(next_token_probs(logits, 1.0), torch.softmax(logits, -1))
    assert torch.allclose(next_token_probs(logits, 0.5), torch.softmax(logits * 2, -1))

    def entropy(p: torch.Tensor) -> float:
        return float(-(p * p.clamp_min(1e-12).log()).sum())

    assert entropy(next_token_probs(logits, 0.5)) < entropy(next_token_probs(logits, 1.0)) < entropy(
        next_token_probs(logits, 2.0)
    )


def test_step07_top_k_keeps_only_the_k_best() -> None:
    logits = torch.tensor([0.5, 4.0, 1.0, 3.0, -2.0, 2.0])
    p = next_token_probs(logits, 1.0, top_k=3)
    assert torch.isclose(p.sum(), torch.tensor(1.0))
    assert (p > 0).nonzero().flatten().tolist() == [1, 3, 5]
    assert torch.allclose(p[[1, 3, 5]], torch.softmax(torch.tensor([4.0, 3.0, 2.0]), -1))
    assert torch.allclose(next_token_probs(logits, 1.0, top_k=None), torch.softmax(logits, -1))
    assert torch.allclose(next_token_probs(logits, 1.0, top_k=6), torch.softmax(logits, -1))


def test_step07_generation_is_reproducible_and_has_the_right_length() -> None:
    m = tiny_model(seed=5)
    a = generate(m, "ab", max_new_tokens=12, temperature=1.0, top_k=None, seed=3, use_cache=False)
    b = generate(m, "ab", max_new_tokens=12, temperature=1.0, top_k=None, seed=3, use_cache=False)
    c = generate(m, "ab", max_new_tokens=12, temperature=1.0, top_k=None, seed=4, use_cache=False)
    assert a == b and a != c
    assert len(a.encode("utf-8", errors="surrogatepass")) >= 1
    longer = generate(m, "", max_new_tokens=40, temperature=0, seed=0, use_cache=False)  # past block_size 16
    assert isinstance(longer, str) and len(longer) > 0


@pytest.mark.parametrize("temperature", [0.0, 0.5])
def test_step07_a_trained_model_continues_the_pattern(temperature: float) -> None:
    m = tiny_model(seed=0)
    opt = make_optimizer(m, TrainConfig(lr=3e-3))
    gen = torch.Generator().manual_seed(0)
    for _ in range(150):  # one to two seconds on a laptop CPU
        x, y = sample_batch(pattern_ids(), 16, 16, gen)
        train_step(m, opt, x, y)
    out = generate(m, "the cat sat on ", max_new_tokens=9, temperature=temperature, top_k=5, seed=0,
                   use_cache=False)
    assert out == "the mat. "
