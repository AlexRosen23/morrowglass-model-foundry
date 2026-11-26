"""Step 2: causal self-attention."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from scratchlm.attention import CausalSelfAttention, attention, causal_mask, merge_heads, split_heads


def test_step02_causal_mask_square_and_rectangular() -> None:
    m = causal_mask(4, 4)
    assert m.dtype == torch.bool and m.shape == (4, 4)
    assert torch.equal(m, torch.tril(torch.ones(4, 4, dtype=torch.bool)))
    # 2 new queries at the end of 5 keys: they sit at positions 3 and 4.
    r = causal_mask(2, 5)
    assert r.tolist() == [[True, True, True, True, False], [True, True, True, True, True]]
    assert causal_mask(1, 7).all()
    with pytest.raises(ValueError):
        causal_mask(5, 2)


def test_step02_attention_matches_pytorch() -> None:
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 6, 8) for _ in range(3))
    y, w = attention(q, k, v, causal_mask(6, 6))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert y.shape == (2, 3, 6, 8) and w.shape == (2, 3, 6, 6)
    assert torch.allclose(y, ref, atol=1e-5)
    y_full, _ = attention(q, k, v, None)
    assert torch.allclose(y_full, F.scaled_dot_product_attention(q, k, v), atol=1e-5)


def test_step02_weights_are_a_distribution_over_the_past_only() -> None:
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 2, 5, 4) for _ in range(3))
    _, w = attention(q, k, v, causal_mask(5, 5))
    assert torch.allclose(w.sum(-1), torch.ones(1, 2, 5), atol=1e-6)
    assert torch.all(w[..., ~causal_mask(5, 5)] == 0)
    assert torch.allclose(w[..., 0, 0], torch.ones(1, 2))  # the first token can only look at itself


def test_step02_scores_are_scaled_by_sqrt_head_dim() -> None:
    q = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])  # (1, T_q=1, D=4)
    k = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]])  # two keys
    v = torch.eye(2).unsqueeze(0)
    _, w = attention(q, k, v)
    # score 4 / sqrt(4) = 2 against 0: softmax gives e^2 / (e^2 + 1)
    assert math.isclose(float(w[0, 0, 0]), math.exp(2) / (math.exp(2) + 1), rel_tol=1e-5)


def test_step02_split_and_merge_heads() -> None:
    x = torch.arange(2 * 3 * 8, dtype=torch.float32).view(2, 3, 8)
    h = split_heads(x, 4)
    assert h.shape == (2, 4, 3, 2)
    assert torch.equal(h[0, 1, 2], x[0, 2, 2:4])  # head 1 of position 2 = channels 2..3
    assert torch.equal(merge_heads(h), x)


def test_step02_future_tokens_do_not_change_the_past() -> None:
    torch.manual_seed(2)
    attn = CausalSelfAttention(d_model=16, n_head=4)
    x = torch.randn(2, 10, 16)
    y, w, present = attn(x, return_weights=True)
    assert y.shape == x.shape and w is not None and w.shape == (2, 4, 10, 10) and present is None
    x2 = x.clone()
    x2[:, 6:] = torch.randn(2, 4, 16)
    y2, _, _ = attn(x2)
    assert torch.allclose(y[:, :6], y2[:, :6], atol=1e-6)
    assert not torch.allclose(y[:, 6:], y2[:, 6:])
