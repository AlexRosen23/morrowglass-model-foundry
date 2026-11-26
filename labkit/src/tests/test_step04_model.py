"""Step 4: the full language model."""

from __future__ import annotations

import math

import pytest
import torch

from scratchlm.model import (
    LanguageModel,
    ModelConfig,
    analytic_param_count,
    count_parameters,
    sinusoidal_table,
)
from tests.conftest import tiny_model


def test_step04_logits_shape_and_a_near_uniform_start() -> None:
    m = tiny_model()
    x = torch.randint(0, 256, (4, 16), generator=torch.Generator().manual_seed(0))
    y = torch.randint(0, 256, (4, 16), generator=torch.Generator().manual_seed(9))
    out = m(x, targets=y)
    assert out.logits.shape == (4, 16, 256)
    assert out.loss is not None and out.loss.dim() == 0
    # Small random weights: every byte about equally likely, so the loss is close to ln(256) = 5.545.
    assert abs(out.loss.item() - math.log(256)) < 0.1
    assert m(x).loss is None


def test_step04_loss_is_cross_entropy_and_ignores_minus_100() -> None:
    m = tiny_model(seed=1)
    x = torch.randint(0, 256, (2, 10), generator=torch.Generator().manual_seed(1))
    y = torch.randint(0, 256, (2, 10), generator=torch.Generator().manual_seed(2))
    out = m(x, targets=y)
    logp = torch.log_softmax(out.logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(out.loss, -logp.mean(), atol=1e-5)
    y2 = y.clone()
    y2[:, :5] = -100
    assert torch.allclose(m(x, targets=y2).loss, -logp[:, 5:].mean(), atol=1e-5)


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(tie_weights=False),
        dict(pos_encoding="sinusoidal"),
        dict(n_layer=3, n_head=4, d_model=48, d_ff=100, block_size=20),
    ],
)
def test_step04_parameter_count_matches_the_formula(kw) -> None:
    cfg = ModelConfig(**{**dict(block_size=16, n_layer=2, n_head=2, d_model=32), **kw})
    assert analytic_param_count(cfg) == count_parameters(LanguageModel(cfg))


def test_step04_sinusoidal_table() -> None:
    t = sinusoidal_table(50, 8)
    assert t.shape == (50, 8)
    assert torch.allclose(t[0, 0::2], torch.zeros(4)) and torch.allclose(t[0, 1::2], torch.ones(4))
    assert math.isclose(float(t[3, 0]), math.sin(3.0), abs_tol=1e-6)
    assert math.isclose(float(t[3, 3]), math.cos(3.0 / 10000 ** (2 / 8)), abs_tol=1e-6)
    assert torch.all(t.abs() <= 1.0)


def test_step04_positions_learned_and_sinusoidal() -> None:
    learned = tiny_model()
    assert torch.equal(learned.positions(4, 2), learned.pos_emb.weight[2:6])
    sin = tiny_model(pos_encoding="sinusoidal")
    assert torch.allclose(sin.positions(16), sinusoidal_table(16, 32))
    assert torch.allclose(sin.positions(3, 5), sinusoidal_table(16, 32)[5:8])
    with pytest.raises(ValueError):
        learned.positions(10, 7)  # 7..16 does not fit in block_size 16


def test_step04_position_changes_the_prediction_and_context_is_limited() -> None:
    m = tiny_model(seed=2)
    with torch.no_grad():
        m.pos_emb.weight.normal_(0, 1.0)
    x = torch.full((1, 6), ord("a"))
    logits = m(x).logits[0]
    assert not torch.allclose(logits[0], logits[5], atol=1e-4)  # same token, different position
    with pytest.raises(ValueError):
        m(torch.zeros(1, 17, dtype=torch.long))  # longer than block_size
