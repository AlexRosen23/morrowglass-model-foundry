"""KV cache: cached and uncached inference give the same logits (CPU, tight tolerance)."""

from __future__ import annotations

import pytest
import torch

from morrowglass.inference.generate import CachedDecoder, generate
from morrowglass.model.transformer import ModelConfig, MorrowglassLM
from morrowglass.tokenizer.tokenizers import ByteTokenizer

ATOL = 1e-5


def _model(pos: str = "learned", block: int = 24) -> MorrowglassLM:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=256, block_size=block, n_layer=2, n_head=2, d_model=32, pos_encoding=pos)  # type: ignore[arg-type]
    return MorrowglassLM(cfg).eval()


@pytest.mark.parametrize("pos", ["learned", "sinusoidal", "none"])
@torch.no_grad()
def test_incremental_forward_matches_full_forward(pos: str) -> None:
    m = _model(pos)
    x = torch.randint(0, 256, (2, 20))
    full = m(x).logits
    out = m(x[:, :7], use_cache=True)  # prefill
    parts = [out.logits]
    past = out.past_kv
    for t in range(7, 20):  # one token at a time
        out = m(x[:, t : t + 1], past_kv=past, use_cache=True)
        past = out.past_kv
        parts.append(out.logits)
    assert torch.allclose(torch.cat(parts, 1), full, atol=ATOL, rtol=0)


@torch.no_grad()
def test_multi_token_chunk_after_cache_uses_offset_causal_mask() -> None:
    m = _model()
    x = torch.randint(0, 256, (1, 18))
    full = m(x).logits
    first = m(x[:, :5], use_cache=True)
    rest = m(x[:, 5:], past_kv=first.past_kv, use_cache=True)  # 13 new queries over 18 keys
    assert torch.allclose(torch.cat([first.logits, rest.logits], 1), full, atol=ATOL, rtol=0)
    # The explicit (attention-returning) path agrees with the fused cached path.
    rest_explicit = m(x[:, 5:], past_kv=first.past_kv, return_attn=True)
    assert torch.allclose(rest_explicit.logits, rest.logits, atol=ATOL, rtol=0)


@torch.no_grad()
def test_cache_cannot_exceed_block_size() -> None:
    m = _model(block=8)
    out = m(torch.zeros(1, 8, dtype=torch.long), use_cache=True)
    with pytest.raises(ValueError):
        m(torch.zeros(1, 1, dtype=torch.long), past_kv=out.past_kv, use_cache=True)


@torch.no_grad()
def test_greedy_decoding_logits_identical_with_and_without_cache_across_window_overflow() -> None:
    m = _model(block=24)
    ids_c = list(range(65, 75))  # 10-token prompt, then 40 new tokens: crosses block_size at step 15
    ids_u = list(ids_c)
    cached, uncached = CachedDecoder(m, use_cache=True), CachedDecoder(m, use_cache=False)
    worst = 0.0
    for _ in range(40):
        lc, lu = cached.next_logits(ids_c), uncached.next_logits(ids_u)
        worst = max(worst, float((lc - lu).abs().max()))
        nxt = int(lc.argmax())
        assert nxt == int(lu.argmax())
        ids_c.append(nxt)
        ids_u.append(nxt)
    assert worst < ATOL
    assert cached.stats["prefill"] == 1 and cached.stats["cached_step"] == 14
    assert cached.stats["full_recompute"] == 25  # once the window slides, every step recomputes
    assert uncached.stats["full_recompute"] == 40


def test_sampled_generation_identical_with_and_without_cache() -> None:
    m = _model(block=32)
    tok = ByteTokenizer()
    a = generate(m, tok, "ROMEO:", max_new_tokens=50, seed=11, use_cache=True)
    b = generate(m, tok, "ROMEO:", max_new_tokens=50, seed=11, use_cache=False)
    assert a["text"] == b["text"]
    for ta, tb in zip(a["tokens"], b["tokens"], strict=True):
        assert ta["raw_prob"] == pytest.approx(tb["raw_prob"], abs=1e-4)


@torch.no_grad()
def test_decoder_reprefills_when_the_sequence_is_not_an_extension() -> None:
    m = _model()
    d = CachedDecoder(m)
    d.next_logits([1, 2, 3])
    d.next_logits([1, 2, 3, 4])
    lo = d.next_logits([9, 9, 3, 4, 5])  # edited history: must not reuse the stale cache
    ref = m(torch.tensor([[9, 9, 3, 4, 5]])).logits[0, -1]
    assert torch.allclose(lo, ref, atol=ATOL, rtol=0)
    assert d.stats["prefill"] == 2
