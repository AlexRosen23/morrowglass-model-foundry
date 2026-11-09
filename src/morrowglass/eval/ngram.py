"""Count-based baselines: uniform, unigram (frequency), and interpolated Kneser-Ney n-grams.

Interpolated Kneser-Ney (Chen & Goodman 1998, single discount D per order):

    P_n(w | h) = max(c(h w) - D, 0) / c(h)  +  D * N1+(h •) / c(h) * P_{n-1}(w | h')

where h' drops the oldest token of h. Lower orders use *continuation* counts
N1+(• h' w) in place of raw counts, and the recursion bottoms out in a uniform
distribution over the vocabulary, so every token has non-zero probability. Contexts
never seen in training fall back to the lower order entirely.

Sequences are packed into ``bytes`` with a fixed width per token (1 byte for byte-level,
2 for BPE ids < 65536) so that n-gram keys are cheap byte slices.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from morrowglass.eval.metrics import LN2, bootstrap_ci


def pack(ids: np.ndarray, width: int) -> bytes:
    dtype = {1: np.uint8, 2: ">u2"}[width]
    if int(ids.max(initial=0)) >= 256**width:
        raise ValueError("token id does not fit packing width")
    return ids.astype(dtype).tobytes()


@dataclass
class _Order:
    num: dict[bytes, float] = field(default_factory=dict)  # count of (h, w)  (raw or continuation)
    den: dict[bytes, float] = field(default_factory=dict)  # sum over w of num(h, w)
    types: dict[bytes, int] = field(default_factory=dict)  # N1+(h •)
    discount: float = 0.75


class KneserNey:
    def __init__(self, order: int, vocab_size: int, width: int) -> None:
        if order < 1:
            raise ValueError("order >= 1")
        self.n = order
        self.V = vocab_size
        self.w = width
        self.orders: list[_Order] = []  # index k -> n-grams of length k+1

    def fit(self, ids: np.ndarray) -> KneserNey:
        data = pack(ids, self.w)
        w, n = self.w, self.n
        L = len(data) // w
        highest: Counter[bytes] = Counter(data[i * w : (i + n) * w] for i in range(L - n + 1))
        grams: list[Counter[bytes]] = [Counter() for _ in range(n)]
        grams[n - 1] = highest
        # Continuation counts for lower orders: number of distinct left extensions.
        for k in range(n - 1, 0, -1):
            cont: Counter[bytes] = Counter()
            for g in grams[k]:
                cont[g[w:]] += 1
            grams[k - 1] = cont
        self.orders = []
        for k in range(n):
            o = _Order()
            den: defaultdict[bytes, float] = defaultdict(float)
            types: defaultdict[bytes, int] = defaultdict(int)
            n1 = n2 = 0
            for g, c in grams[k].items():
                h = g[:-w]
                den[h] += c
                types[h] += 1
                n1 += c == 1
                n2 += c == 2
            o.num = dict(grams[k])
            o.den = dict(den)
            o.types = dict(types)
            o.discount = n1 / (n1 + 2 * n2) if n1 + 2 * n2 > 0 else 0.75
            self.orders.append(o)
        return self

    def prob(self, history: bytes, token: bytes) -> float:
        """P(token | history); history holds up to n-1 packed tokens (most recent last)."""
        w = self.w
        p = 1.0 / self.V
        for k in range(self.n):
            h_len = k
            if len(history) < h_len * w:
                break
            h = history[len(history) - h_len * w :] if h_len else b""
            o = self.orders[k]
            den = o.den.get(h)
            if not den:
                continue  # unseen context: keep the lower-order estimate
            c = o.num.get(h + token, 0.0)
            D = o.discount
            p = max(c - D, 0.0) / den + D * o.types[h] / den * p
        return p

    def distribution(self, history: bytes) -> np.ndarray:
        out = np.empty(self.V)
        fmt = {1: lambda t: bytes([t]), 2: lambda t: t.to_bytes(2, "big")}[self.w]
        for t in range(self.V):
            out[t] = self.prob(history, fmt(t))
        return out

    def evaluate(self, ids: np.ndarray, byte_len: np.ndarray, window: int = 256) -> dict[str, Any]:
        """Score every token after the first. Returns totals plus per-window sums for a bootstrap CI."""
        data = pack(ids, self.w)
        w, n = self.w, self.n
        nll_w, bytes_w, tok_w = [], [], []
        acc_nll = acc_bytes = 0.0
        cnt = 0
        for i in range(1, len(ids)):
            hist = data[max(0, i - (n - 1)) * w : i * w]
            p = self.prob(hist, data[i * w : (i + 1) * w])
            acc_nll -= math.log(p)
            acc_bytes += float(byte_len[ids[i]])
            cnt += 1
            if cnt == window:
                nll_w.append(acc_nll)
                bytes_w.append(acc_bytes)
                tok_w.append(cnt)
                acc_nll = acc_bytes = 0.0
                cnt = 0
        if cnt:
            nll_w.append(acc_nll)
            bytes_w.append(acc_bytes)
            tok_w.append(cnt)
        nll_a, bytes_a = np.array(nll_w), np.array(bytes_w)
        total_tok = int(sum(tok_w))
        loss = float(nll_a.sum()) / total_tok
        lo, hi = bootstrap_ci(nll_a / LN2, bytes_a)
        return {
            "loss_nats_per_token": loss,
            "perplexity_per_token": math.exp(min(loss, 50)),
            "bits_per_byte": float(nll_a.sum()) / LN2 / float(bytes_a.sum()),
            "bits_per_byte_ci95": [lo, hi],
            "tokens_predicted": total_tok,
            "bytes_predicted": int(bytes_a.sum()),
        }

    def sample(self, prompt_ids: list[int], max_new: int, rng: np.random.Generator,
               temperature: float = 1.0) -> list[int]:
        out = list(prompt_ids)
        for _ in range(max_new):
            hist = pack(np.array(out[-(self.n - 1):] if self.n > 1 else [], dtype=np.int64), self.w)
            p = self.distribution(hist)
            if temperature != 1.0:
                p = p ** (1.0 / max(temperature, 1e-3))
            p = p / p.sum()
            out.append(int(rng.choice(self.V, p=p)))
        return out[len(prompt_ids):]


def unigram_eval(train_ids: np.ndarray, ids: np.ndarray, byte_len: np.ndarray, vocab: int) -> dict[str, Any]:
    """Frequency baseline with add-one smoothing (the 'majority-class' analogue for LM)."""
    counts = np.bincount(train_ids, minlength=vocab).astype(np.float64) + 1.0
    logp = np.log(counts / counts.sum())
    nll = -logp[ids[1:]]
    return _from_token_nll(nll, byte_len[ids[1:]])


def uniform_eval(ids: np.ndarray, byte_len: np.ndarray, vocab: int) -> dict[str, Any]:
    nll = np.full(len(ids) - 1, math.log(vocab))
    return _from_token_nll(nll, byte_len[ids[1:]])


def _from_token_nll(nll: np.ndarray, nbytes: np.ndarray, window: int = 256) -> dict[str, Any]:
    k = len(nll) // window * window
    nw = [nll[:k].reshape(-1, window).sum(1)] + ([np.array([nll[k:].sum()])] if k < len(nll) else [])
    bw = [nbytes[:k].reshape(-1, window).sum(1)] + ([np.array([nbytes[k:].sum()])] if k < len(nll) else [])
    nll_w, bytes_w = np.concatenate(nw), np.concatenate(bw).astype(np.float64)
    loss = float(nll.mean())
    lo, hi = bootstrap_ci(nll_w / LN2, bytes_w)
    return {"loss_nats_per_token": loss, "perplexity_per_token": math.exp(min(loss, 50)),
            "bits_per_byte": float(nll.sum()) / LN2 / float(nbytes.sum()), "bits_per_byte_ci95": [lo, hi],
            "tokens_predicted": len(nll), "bytes_predicted": int(nbytes.sum())}
