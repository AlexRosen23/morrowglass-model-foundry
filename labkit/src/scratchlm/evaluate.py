"""Step 5: measuring a language model on held-out text.

- cross-entropy (loss), in nats per token: the mean of -ln p(next token | context)
- perplexity = exp(loss): "as unsure as a uniform choice among this many tokens"
- bits per byte = total -log2 p / bytes predicted. With the byte tokenizer one token is one byte, so it
  is loss / ln 2. A model that knows nothing (uniform over 256 bytes) scores exactly 8.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

LN2 = math.log(2.0)


def eval_windows(n_tokens: int, block_size: int, max_windows: int | None = None) -> list[tuple[int, int]]:
    """Non-overlapping ``(start, length)`` windows over a split of ``n_tokens`` tokens.

    Window (s, n) feeds tokens s .. s+n-1 and predicts tokens s+1 .. s+n. Together the windows predict
    every token after the first exactly once; each length is at most ``block_size`` (the last one may be
    shorter). With ``max_windows``, return an evenly spaced subset of at most that many windows, always
    including the first and the last (for quick checks during training).
    """
    #lab:begin 5 windows that predict every token once
    wins: list[tuple[int, int]] = []
    s = 0
    while s < n_tokens - 1:
        length = min(block_size, n_tokens - 1 - s)
        wins.append((s, length))
        s += length
    if max_windows is not None and len(wins) > max_windows:
        if max_windows == 1:
            return wins[:1]
        step = (len(wins) - 1) / (max_windows - 1)
        wins = [wins[round(i * step)] for i in range(max_windows)]
    return wins
    #lab:stub
    #> # TODO(step 5): walk from s = 0, taking windows of min(block_size, n_tokens - 1 - s) tokens, until
    #> # every token after the first is predicted. Then, if max_windows is set and smaller than the number
    #> # of windows, keep max_windows of them, evenly spaced, including the first and the last.
    #> raise NotImplementedError("TODO(step 5): eval_windows")
    #lab:end


@torch.no_grad()
def evaluate(model: nn.Module, ids: Tensor, batch_size: int = 16, max_windows: int | None = None) -> dict[str, Any]:
    """Score ``model`` on the 1-D token tensor ``ids``.

    ``model(x).logits`` must be (B, T, V) and ``model.cfg.block_size`` the context length. Returns a dict
    with ``loss`` (nats per token), ``perplexity``, ``bits_per_byte`` and ``tokens`` (number of tokens
    predicted). Leaves the model in the train/eval mode it was in.
    """
    #lab:begin 5 total negative log-likelihood over the windows
    was_training = model.training
    model.eval()
    wins = eval_windows(len(ids), model.cfg.block_size, max_windows)
    total_nll, total_tokens = 0.0, 0
    for length in sorted({n for _, n in wins}, reverse=True):  # batch windows of equal length together
        group = [s for s, n in wins if n == length]
        for i in range(0, len(group), batch_size):
            starts = torch.tensor(group[i : i + batch_size])
            offsets = torch.arange(length)
            x = ids[starts[:, None] + offsets]
            y = ids[starts[:, None] + offsets + 1]
            logits = model(x).logits.float()
            logp = torch.log_softmax(logits, dim=-1).gather(-1, y.unsqueeze(-1))
            total_nll -= float(logp.sum())
            total_tokens += y.numel()
    model.train(was_training)
    loss = total_nll / max(total_tokens, 1)
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 50.0)),
        "bits_per_byte": loss / LN2,
        "tokens": total_tokens,
    }
    #lab:stub
    #> # TODO(step 5): put the model in eval mode; for every window (s, n) feed ids[s:s+n] and add up
    #> # -log_softmax(logits) at the true next tokens ids[s+1:s+n+1]; divide by the number of tokens
    #> # predicted. Batch windows of the same length together. Restore the model's mode at the end.
    #> raise NotImplementedError("TODO(step 5): evaluate")
    #lab:end
