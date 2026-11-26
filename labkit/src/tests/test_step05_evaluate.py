"""Step 5: windows, loss, perplexity and bits per byte on held-out text."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn

from scratchlm.evaluate import eval_windows, evaluate
from tests.conftest import pattern_ids, tiny_model


class Uniform(nn.Module):
    """A 'model' that knows nothing: equal logits for all 256 bytes."""

    def __init__(self, block_size: int = 8) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(block_size=block_size)

    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=torch.zeros(*x.shape, 256))


def test_step05_windows_predict_every_token_exactly_once() -> None:
    for n, block in [(2, 8), (17, 4), (100, 16), (33, 32), (32, 32)]:
        wins = eval_windows(n, block)
        assert all(1 <= length <= block for _, length in wins)
        predicted = [s + 1 + i for s, length in wins for i in range(length)]
        assert predicted == list(range(1, n))
    assert eval_windows(17, 4) == [(0, 4), (4, 4), (8, 4), (12, 4)]
    assert eval_windows(1, 4) == []


def test_step05_max_windows_is_an_even_subset_with_both_ends() -> None:
    full = eval_windows(1001, 10)
    sub = eval_windows(1001, 10, max_windows=5)
    assert len(full) == 100 and len(sub) == 5
    assert sub[0] == full[0] and sub[-1] == full[-1]
    assert all(w in full for w in sub) and sub == sorted(sub)
    assert eval_windows(1001, 10, max_windows=500) == full


def test_step05_a_model_that_knows_nothing_scores_8_bits_per_byte() -> None:
    ev = evaluate(Uniform(), torch.arange(50) % 256)
    assert math.isclose(ev["loss"], math.log(256), rel_tol=1e-6)
    assert math.isclose(ev["perplexity"], 256.0, rel_tol=1e-5)
    assert math.isclose(ev["bits_per_byte"], 8.0, rel_tol=1e-6)
    assert ev["tokens"] == 49


def test_step05_matches_a_direct_computation_and_keeps_the_mode() -> None:
    m = tiny_model(seed=4)
    ids = pattern_ids()[:40]  # windows of 16, 16 and 7
    m.train()
    ev = evaluate(m, ids, batch_size=2)
    assert m.training
    with torch.no_grad():
        m.eval()
        nll = 0.0
        for s, n in [(0, 16), (16, 16), (32, 7)]:
            logp = torch.log_softmax(m(ids[s : s + n][None]).logits[0], -1)
            nll -= float(logp.gather(-1, ids[s + 1 : s + n + 1, None]).sum())
    assert ev["tokens"] == 39
    assert math.isclose(ev["loss"], nll / 39, rel_tol=1e-5)
    assert math.isclose(ev["bits_per_byte"], ev["loss"] / math.log(2), rel_tol=1e-9)
