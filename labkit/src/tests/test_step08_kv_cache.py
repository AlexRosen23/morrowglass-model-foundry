"""Step 8: the key/value cache gives exactly the same results as recomputing, with less work."""

from __future__ import annotations

import torch

from scratchlm.attention import CausalSelfAttention
from scratchlm.sample import Decoder, generate
from tests.conftest import tiny_model

ATOL = 1e-5


@torch.no_grad()
def test_step08_attention_with_past_equals_full_attention() -> None:
    torch.manual_seed(0)
    attn = CausalSelfAttention(16, 4)
    x = torch.randn(2, 9, 16)
    full, _, _ = attn(x)
    first, _, past = attn(x[:, :5], use_cache=True)
    assert past is not None and past[0].shape == past[1].shape == (2, 4, 5, 4)
    rest, _, present = attn(x[:, 5:], past=past, use_cache=True)
    assert present is not None and present[0].shape == (2, 4, 9, 4)
    assert torch.allclose(torch.cat([first, rest], 1), full, atol=ATOL)


@torch.no_grad()
def test_step08_model_token_by_token_equals_one_full_pass() -> None:
    for pos in ("learned", "sinusoidal"):
        m = tiny_model(pos_encoding=pos).eval()
        x = torch.randint(0, 256, (2, 14), generator=torch.Generator().manual_seed(1))
        full = m(x).logits
        out = m(x[:, :4], use_cache=True)
        assert out.past_kv is not None and len(out.past_kv) == m.cfg.n_layer
        parts = [out.logits]
        for t in range(4, 14):
            out = m(x[:, t : t + 1], past_kv=out.past_kv, use_cache=True)
            parts.append(out.logits)
        assert torch.allclose(torch.cat(parts, 1), full, atol=ATOL)
        assert m(x[:, :3]).past_kv is None


@torch.no_grad()
def test_step08_decoder_logits_match_across_the_block_boundary() -> None:
    m = tiny_model(seed=2).eval()  # block_size 16
    ids = torch.randint(0, 256, (24,), generator=torch.Generator().manual_seed(3)).tolist()
    cached, plain = Decoder(m, use_cache=True), Decoder(m, use_cache=False)
    for n in range(3, 25):
        assert torch.allclose(cached.next_logits(ids[:n]), plain.next_logits(ids[:n]), atol=ATOL)
    assert cached.stats["prefill"] == 1
    assert cached.stats["cached_step"] == 13  # lengths 4 .. 16 reuse the cache
    assert cached.stats["full_recompute"] == 8  # lengths 17 .. 24 slide the window


@torch.no_grad()
def test_step08_generation_with_and_without_the_cache_is_identical() -> None:
    m = tiny_model(seed=4)
    for temperature in (0.0, 1.0):
        kw = dict(max_new_tokens=30, temperature=temperature, top_k=20, seed=7)
        assert generate(m, "To be", use_cache=True, **kw) == generate(m, "To be", use_cache=False, **kw)
