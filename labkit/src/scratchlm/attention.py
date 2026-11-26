"""Step 2: causal self-attention (and, in step 8, its key/value cache).

Shapes: B = batch, T = sequence length, C = model width (d_model), H = heads, D = C // H.

    x                 (B, T, C)
    q, k, v           (B, H, T, D)       one projection of x, split into H heads of width D
    scores = q k^T/√D (B, H, T_q, T_k)   how strongly each query position matches each key position
    mask              (T_q, T_k)          True where a query may look at a key (only itself and the past)
    weights           (B, H, T_q, T_k)   softmax over keys: every row sums to 1
    y = weights @ v   (B, H, T_q, D)     a weighted average of the values, merged back to (B, T, C)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# Per layer: (keys, values), each (B, H, T_past, D). Made by forward(..., use_cache=True) in step 8.
LayerCache = tuple[Tensor, Tensor]


def causal_mask(t_q: int, t_k: int, device: torch.device | str | None = None) -> Tensor:
    """Boolean (t_q, t_k) mask, True where query i may attend to key j.

    The queries are the *last* t_q positions of the t_k keys, so query i sits at absolute position
    ``t_k - t_q + i`` and may see keys 0 .. t_k - t_q + i. When t_q == t_k this is the familiar
    lower-triangular matrix. (t_q < t_k happens with the KV cache in step 8.) Raises ValueError if
    t_q > t_k.
    """
    #lab:begin 2 the causal mask, square or rectangular
    offset = t_k - t_q
    if offset < 0:
        raise ValueError(f"more queries ({t_q}) than keys ({t_k})")
    q_pos = torch.arange(t_q, device=device)[:, None] + offset
    k_pos = torch.arange(t_k, device=device)[None, :]
    return k_pos <= q_pos
    #lab:stub
    #> # TODO(step 2): query i is at absolute position (t_k - t_q + i). It may see key j when j is not
    #> # after it. Build that comparison with two torch.arange vectors and broadcasting.
    #> raise NotImplementedError("TODO(step 2): causal_mask")
    #lab:end


def attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Scaled dot-product attention written out by hand.

    q: (..., T_q, D), k and v: (..., T_k, D), mask: (T_q, T_k) bool or None.
    Returns ``(y, weights)`` with y: (..., T_q, D) and weights: (..., T_q, T_k).
    """
    #lab:begin 2 scores, mask, softmax, weighted sum
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights
    #lab:stub
    #> # TODO(step 2): scores = q @ k^T / sqrt(D); where mask is False set the score to -inf; softmax over
    #> # the last (key) dimension; return (weights @ v, weights).
    #> raise NotImplementedError("TODO(step 2): attention")
    #lab:end


def split_heads(x: Tensor, n_head: int) -> Tensor:
    """(B, T, C) -> (B, H, T, C // H)."""
    #lab:begin 2 split the width into heads
    B, T, C = x.shape
    return x.view(B, T, n_head, C // n_head).transpose(1, 2)
    #lab:stub
    #> # TODO(step 2): view (B, T, C) as (B, T, H, D), then swap the T and H dimensions.
    #> raise NotImplementedError("TODO(step 2): split_heads")
    #lab:end


def merge_heads(y: Tensor) -> Tensor:
    """(B, H, T, D) -> (B, T, H * D). The inverse of split_heads."""
    #lab:begin 2 merge the heads back
    B, H, T, D = y.shape
    return y.transpose(1, 2).contiguous().view(B, T, H * D)
    #lab:stub
    #> # TODO(step 2): swap H and T back, then flatten the last two dimensions. transpose() leaves the
    #> # tensor non-contiguous, so call .contiguous() before .view() (or use .reshape()).
    #> raise NotImplementedError("TODO(step 2): merge_heads")
    #lab:end


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention: one qkv projection, H heads, one output projection."""

    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        if d_model % n_head:
            raise ValueError(f"d_model={d_model} is not divisible by n_head={n_head}")
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(
        self, x: Tensor, past: LayerCache | None = None, use_cache: bool = False, return_weights: bool = False
    ) -> tuple[Tensor, Tensor | None, LayerCache | None]:
        """Returns ``(y, weights or None, present cache or None)``; y has the shape of x.

        With ``past`` (keys and values of earlier positions, from step 8), ``x`` holds only the new
        positions. With ``use_cache``, the keys and values of all positions so far are returned.
        """
        C = x.size(-1)
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (split_heads(t, self.n_head) for t in (q, k, v))
        #lab:begin 8 reuse cached keys and values
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v) if use_cache else None
        #lab:stub
        #> # TODO(step 8): when `past` is given, put its keys and values in front of the new ones (along the
        #> # time dimension, dim=2). When `use_cache` is set, return the full (k, v) as `present`.
        #> if past is not None or use_cache:
        #>     raise NotImplementedError("TODO(step 8): KV cache in CausalSelfAttention.forward")
        #> present = None
        #lab:end
        mask = causal_mask(q.size(2), k.size(2), x.device)
        y, weights = attention(q, k, v, mask)
        return self.proj(merge_heads(y)), (weights if return_weights else None), present
