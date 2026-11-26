"""Step 7: generating text, and step 8: doing it with a key/value cache.

Generation is a loop: read the logits for the next token, turn them into probabilities (temperature,
top-k), draw one token, append it, repeat. Without a cache every step runs the model over the whole
context again. With the cache (step 8) the keys and values of earlier positions are kept, and each
step only processes the newest token.
"""

from __future__ import annotations

import torch
from torch import Tensor

from scratchlm.attention import LayerCache
from scratchlm.model import LanguageModel
from scratchlm.tokenizer import ByteTokenizer


def next_token_probs(logits: Tensor, temperature: float = 1.0, top_k: int | None = None) -> Tensor:
    """Probabilities (V,) to draw the next token from, given the model's logits (V,).

    - temperature <= 0: greedy, probability 1 on the arg-max token (the first one on a tie);
    - otherwise softmax(logits / temperature);
    - with top_k (0 < top_k < V), only the k most likely tokens keep a non-zero probability.
    """
    #lab:begin 7 temperature and top-k
    if temperature <= 0:
        p = torch.zeros_like(logits, dtype=torch.float32)
        p[int(logits.argmax())] = 1.0
        return p
    scaled = logits.float() / temperature
    if top_k is not None and 0 < top_k < scaled.numel():
        kth = torch.topk(scaled, top_k).values[-1]
        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
    return torch.softmax(scaled, dim=-1)
    #lab:stub
    #> # TODO(step 7): greedy for temperature <= 0; else divide the logits by the temperature, set every
    #> # logit below the k-th largest to -inf when top_k is set, and softmax.
    #> raise NotImplementedError("TODO(step 7): next_token_probs")
    #lab:end


class Decoder:
    """Next-token logits for a sequence that grows by one token per call.

    Without the cache, every call runs the model over the last ``block_size`` tokens. With
    ``use_cache=True`` (step 8), the first call runs the whole context once and keeps every layer's
    keys and values ("prefill"); each later call feeds only the newest token. Learned positions stop at
    block_size, so once the sequence is longer than that the window slides, every position changes and
    the cache is useless: from then on the decoder falls back to the uncached path.
    """

    def __init__(self, model: LanguageModel, use_cache: bool = False) -> None:
        self.model = model
        self.use_cache = use_cache
        self._past: list[LayerCache] | None = None
        self._ids: list[int] = []
        self.stats = {"prefill": 0, "cached_step": 0, "full_recompute": 0}

    def reset(self) -> None:
        self._past, self._ids = None, []

    @torch.no_grad()
    def next_logits(self, ids: list[int]) -> Tensor:
        """Logits (V,) for the token that follows ``ids``."""
        if not ids:
            raise ValueError("need at least one token of context")
        T = self.model.cfg.block_size
        device = self.model.tok_emb.weight.device
        if not self.use_cache or len(ids) > T:
            self.reset()
            self.stats["full_recompute"] += 1
            return self.model(torch.tensor([ids[-T:]], device=device)).logits[0, -1]
        #lab:begin 8 prefill once, then one token per step
        if self._past is not None and len(ids) == len(self._ids) + 1 and ids[:-1] == self._ids:
            out = self.model(torch.tensor([ids[-1:]], device=device), past_kv=self._past, use_cache=True)
            self.stats["cached_step"] += 1
        else:
            out = self.model(torch.tensor([ids], device=device), use_cache=True)
            self.stats["prefill"] += 1
        self._past, self._ids = out.past_kv, list(ids)
        return out.logits[0, -1]
        #lab:stub
        #> # TODO(step 8): if the cache holds exactly ids[:-1], run only the newest token with past_kv=self._past
        #> # and count a "cached_step"; otherwise run all of ids with use_cache=True and count a "prefill".
        #> # Either way keep out.past_kv and a copy of ids for the next call, and return the last logits.
        #> raise NotImplementedError("TODO(step 8): Decoder.next_logits with the cache")
        #lab:end


@torch.no_grad()
def generate(
    model: LanguageModel,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
    seed: int = 0,
    use_cache: bool = False,
    tok: ByteTokenizer | None = None,
) -> str:
    """Continue ``prompt`` by ``max_new_tokens`` tokens and return only the new text.

    Draw each token with ``torch.multinomial`` and a ``torch.Generator`` seeded with ``seed`` (greedy
    when temperature <= 0), so the same seed gives the same text. An empty prompt starts from "\\n".
    """
    #lab:begin 7 the sampling loop
    tok = tok or ByteTokenizer()
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    ids = tok.encode(prompt) or tok.encode("\n")
    start = len(ids)
    dec = Decoder(model, use_cache=use_cache)
    for _ in range(max_new_tokens):
        p = next_token_probs(dec.next_logits(ids).cpu(), temperature, top_k)
        nxt = int(torch.multinomial(p, 1, generator=gen)) if temperature > 0 else int(p.argmax())
        ids.append(nxt)
    return tok.decode(ids[start:])
    #lab:stub
    #> # TODO(step 7): encode the prompt (or "\n"), make a Decoder(model, use_cache), and repeat: logits ->
    #> # next_token_probs -> torch.multinomial(p, 1, generator=gen) (or argmax when greedy) -> append.
    #> # Decode only the new tokens.
    #> raise NotImplementedError("TODO(step 7): generate")
    #lab:end
