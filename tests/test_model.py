"""Architecture correctness: shapes, parameter count, causal masking, attention maths."""

from __future__ import annotations

import pytest
import torch

from morrowglass.model.transformer import (
    ModelConfig,
    MorrowglassLM,
    analytic_param_count,
    blueprint,
    compute_budget,
    count_parameters,
)

TINY = ModelConfig(vocab_size=64, block_size=32, n_layer=2, n_head=4, d_model=32)


@pytest.mark.parametrize("pos", ["learned", "sinusoidal", "none"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("tie", [True, False])
def test_param_count_matches_analytic(pos: str, bias: bool, tie: bool) -> None:
    cfg = ModelConfig(vocab_size=50, block_size=16, n_layer=3, n_head=2, d_model=24,
                      pos_encoding=pos, bias=bias, tie_weights=tie)  # type: ignore[arg-type]
    model = MorrowglassLM(cfg)
    assert count_parameters(model) == analytic_param_count(cfg)
    bp_total = sum(r["params"] for r in blueprint(cfg))
    assert bp_total == analytic_param_count(cfg)


def test_shapes_and_loss() -> None:
    torch.manual_seed(0)
    model = MorrowglassLM(TINY)
    idx = torch.randint(0, 64, (3, 20))
    targets = torch.randint(0, 64, (3, 20))  # independent of inputs (tied weights favour copying)
    out = model(idx, targets=targets, return_attn=True)
    assert out.logits.shape == (3, 20, 64)
    assert out.loss is not None and out.loss.ndim == 0
    assert len(out.attentions) == 2 and out.attentions[0].shape == (3, 4, 20, 20)
    # Fresh model ≈ uniform prediction: loss near ln(V).
    assert abs(out.loss.item() - torch.log(torch.tensor(64.0)).item()) < 0.3


def test_rejects_too_long_sequences() -> None:
    model = MorrowglassLM(TINY)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 33, dtype=torch.long))


@pytest.mark.parametrize("pos", ["learned", "sinusoidal"])
def test_causal_no_future_leakage(pos: str) -> None:
    """Changing tokens after position t must not change logits at positions <= t."""
    torch.manual_seed(1)
    cfg = ModelConfig(vocab_size=64, block_size=32, n_layer=3, n_head=4, d_model=32, pos_encoding=pos)  # type: ignore[arg-type]
    model = MorrowglassLM(cfg).eval()
    a = torch.randint(0, 64, (2, 24))
    for t in (0, 5, 17, 22):
        b = a.clone()
        b[:, t + 1 :] = torch.randint(0, 64, b[:, t + 1 :].shape)
        for return_attn in (False, True):  # both the fused and the explicit attention path
            la = model(a, return_attn=return_attn).logits
            lb = model(b, return_attn=return_attn).logits
            torch.testing.assert_close(la[:, : t + 1], lb[:, : t + 1], rtol=0, atol=1e-5)


def test_non_causal_model_fails_leakage_check() -> None:
    """The leakage test must be able to fail: with the mask removed, the past sees the future."""
    torch.manual_seed(2)
    cfg = ModelConfig(vocab_size=64, block_size=32, n_layer=2, n_head=4, d_model=32, causal=False)
    model = MorrowglassLM(cfg).eval()
    a = torch.randint(0, 64, (1, 16))
    b = a.clone()
    b[:, 10:] = (b[:, 10:] + 1) % 64
    diff = (model(a).logits[:, :5] - model(b).logits[:, :5]).abs().max().item()
    assert diff > 1e-4


def test_fused_and_explicit_attention_agree() -> None:
    torch.manual_seed(3)
    model = MorrowglassLM(TINY).eval()
    idx = torch.randint(0, 64, (2, 30))
    torch.testing.assert_close(model(idx).logits, model(idx, return_attn=True).logits, rtol=1e-4, atol=1e-5)


def test_attention_rows_are_causal_distributions() -> None:
    torch.manual_seed(4)
    model = MorrowglassLM(TINY).eval()
    att = model(torch.randint(0, 64, (1, 12)), return_attn=True).attentions
    for a in att:
        torch.testing.assert_close(a.sum(-1), torch.ones_like(a.sum(-1)))
        upper = torch.triu(torch.ones(12, 12, dtype=torch.bool), diagonal=1)
        assert a[..., upper].abs().max().item() == 0.0


def test_outputs_finite_and_position_offsets() -> None:
    torch.manual_seed(5)
    model = MorrowglassLM(TINY).eval()
    idx = torch.randint(0, 64, (2, 10))
    out0 = model(idx).logits
    assert torch.isfinite(out0).all()
    # Per-example tensor offsets of zero equal the default positions.
    out_t = model(idx, pos_offset=torch.zeros(2, dtype=torch.long)).logits
    torch.testing.assert_close(out0, out_t)
    assert not torch.allclose(out0, model(idx, pos_offset=5).logits)


def test_compute_budget_is_consistent() -> None:
    b = compute_budget(TINY, tokens=1000)
    assert b["params_total"] == analytic_param_count(TINY)
    assert b["approx_train_flops"] == pytest.approx(b["approx_flops_per_token"] * 1000)
    assert b["approx_flops_per_token"] > 6 * b["params_non_embedding"]


def test_invalid_config() -> None:
    with pytest.raises(ValueError):
        ModelConfig(d_model=30, n_head=4)
