"""Sampling and inspection.

Every generated token carries the model's own probability for it plus the top
alternatives and the entropy of the distribution it was drawn from. These are model
scores, not calibrated confidence: a tiny model trained on 4.5 MB of Shakespeare is often
confidently wrong (see the calibration report in each run's summary).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

import torch

from morrowglass.model.transformer import KVCache, MorrowglassLM
from morrowglass.tokenizer.tokenizers import Tokenizer

MAX_NEW_TOKENS = 400


@dataclass
class GenToken:
    id: int
    text: str
    prob: float  # model probability of the sampled token, after temperature/top-k
    raw_prob: float  # probability under the untempered model distribution
    entropy_bits: float  # entropy of the untempered next-token distribution
    top: list[tuple[str, float]]  # top-5 alternatives under the untempered distribution


class CachedDecoder:
    """Next-token logits for a sequence that grows by one token per call.

    With ``use_cache=True`` the first call runs the whole prompt once (prefill) and keeps every
    layer's keys and values; each later call feeds only the newest token. The model uses learned
    absolute positions, so once the sequence is longer than ``block_size`` the window slides and every
    position changes: from then on each step recomputes the last ``block_size`` tokens without a cache,
    exactly as the uncached path does. ``tests/test_kv_cache.py`` checks both paths give the same
    logits at every step, across that boundary.
    """

    def __init__(self, model: MorrowglassLM, device: str = "cpu", use_cache: bool = True) -> None:
        self.model = model
        self.device = device
        self.use_cache = use_cache
        self._past: KVCache | None = None
        self._ids: list[int] = []
        self.stats = {"prefill": 0, "cached_step": 0, "full_recompute": 0}

    def reset(self) -> None:
        self._past, self._ids = None, []

    @torch.no_grad()
    def next_logits(self, ids: list[int]) -> torch.Tensor:
        """Logits (V,) for the token after ``ids``, on the model's device."""
        T = self.model.cfg.block_size
        if not ids:
            raise ValueError("need at least one token of context")
        if not self.use_cache or len(ids) > T:
            self.reset()
            self.stats["full_recompute"] += 1
            return self.model(torch.tensor([ids[-T:]], device=self.device)).logits[0, -1]
        if self._past is not None and len(ids) == len(self._ids) + 1 and ids[:-1] == self._ids:
            out = self.model(torch.tensor([ids[-1:]], device=self.device), past_kv=self._past, use_cache=True)
            self.stats["cached_step"] += 1
        else:
            out = self.model(torch.tensor([ids], device=self.device), use_cache=True)
            self.stats["prefill"] += 1
        self._past, self._ids = out.past_kv, list(ids)
        return out.logits[0, -1]


def _filtered_probs(logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
    if temperature <= 0:
        p = torch.zeros_like(logits)
        p[logits.argmax()] = 1.0
        return p
    scaled = logits / temperature
    if top_k is not None and 0 < top_k < scaled.numel():
        kth = torch.topk(scaled, top_k).values[-1]
        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
    return torch.softmax(scaled, dim=-1)


@torch.no_grad()
def stream(
    model: MorrowglassLM,
    tok: Tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
    seed: int = 0,
    device: str = "cpu",
    use_cache: bool = True,
) -> Iterator[GenToken]:
    """Yield tokens one at a time. Deterministic for a fixed seed on CPU (with or without the KV cache)."""
    if max_new_tokens > MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens capped at {MAX_NEW_TOKENS}")
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    ids = tok.encode(prompt) or tok.encode("\n")
    dec = CachedDecoder(model, device, use_cache)  # crops to the last block_size tokens when longer
    for _ in range(max_new_tokens):
        logits = dec.next_logits(ids).float().cpu()
        raw = torch.softmax(logits, -1)
        p = _filtered_probs(logits, temperature, top_k)
        nxt = int(torch.multinomial(p, 1, generator=gen)) if temperature > 0 else int(p.argmax())
        ent = float(-(raw * torch.log2(raw.clamp_min(1e-12))).sum())
        tv, ti = torch.topk(raw, 5)
        top = [(tok.decode([int(i)]), round(float(v), 4)) for v, i in zip(tv, ti, strict=True)]
        ids.append(nxt)
        yield GenToken(nxt, tok.decode([nxt]), round(float(p[nxt]), 4), round(float(raw[nxt]), 4),
                       round(ent, 3), top)


def generate(model: MorrowglassLM, tok: Tokenizer, prompt: str, **kw: Any) -> dict[str, Any]:
    toks = list(stream(model, tok, prompt, **kw))
    ids = [t.id for t in toks]
    mean_lp = sum(math.log2(max(t.raw_prob, 1e-12)) for t in toks) / max(1, len(toks))
    return {
        "prompt": prompt,
        "text": tok.decode(ids),
        "tokens": [asdict(t) for t in toks],
        "mean_bits_per_token_self": round(-mean_lp, 3),
        "note": "Probabilities are the model's own scores and are not calibrated confidence.",
    }


@torch.no_grad()
def attention_maps(model: MorrowglassLM, tok: Tokenizer, text: str, max_tokens: int = 64,
                   device: str = "cpu") -> dict[str, Any]:
    ids = tok.encode(text)[:max_tokens]
    if not ids:
        raise ValueError("empty input")
    out = model(torch.tensor([ids], device=device), return_attn=True)
    maps = [a[0].cpu().numpy().round(4).tolist() for a in out.attentions]  # per layer: (H, T, T)
    return {
        "tokens": [tok.decode([i]) for i in ids],
        "n_layer": len(maps),
        "n_head": model.cfg.n_head,
        "attention": maps,
        "caveat": ("Attention weights show where each position reads from in one layer. They are not a "
                   "causal explanation of the prediction: values, MLPs and later layers also matter."),
    }
