"""Held-out evaluation: cross-entropy, perplexity, bits-per-byte and bootstrap intervals.

Definitions (all on held-out text the model never trained on):
- cross-entropy per token (nats) = mean over predicted tokens of -ln p(token | context)
- perplexity per token = exp(cross-entropy); only comparable between models that share a tokenizer
- bits-per-byte (bpb) = total -log2 p over the split / UTF-8 bytes predicted;
  tokenizer independent, so it is the headline metric for byte-vs-BPE comparisons.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from morrowglass.data.loader import TokenSplit, eval_windows
from morrowglass.model.transformer import MorrowglassLM

LN2 = math.log(2.0)


def bootstrap_ci(
    nll_sums: np.ndarray, denom: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap over evaluation windows of a ratio statistic sum(nll)/sum(denom)."""
    rng = np.random.default_rng(seed)
    n = len(nll_sums)
    if n < 2:
        v = float(nll_sums.sum() / max(denom.sum(), 1))
        return v, v
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = nll_sums[idx].sum(1) / denom[idx].sum(1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def summarize(nll_nats: np.ndarray, n_tokens: np.ndarray, n_bytes: np.ndarray, ci: bool = True) -> dict[str, Any]:
    total_nll, total_tok, total_bytes = float(nll_nats.sum()), int(n_tokens.sum()), int(n_bytes.sum())
    loss = total_nll / max(total_tok, 1)
    out: dict[str, Any] = {
        "loss_nats_per_token": loss,
        "perplexity_per_token": math.exp(min(loss, 50.0)),
        "bits_per_byte": total_nll / LN2 / max(total_bytes, 1),
        "tokens_predicted": total_tok,
        "bytes_predicted": total_bytes,
        "bytes_per_token": total_bytes / max(total_tok, 1),
        "windows": len(nll_nats),
    }
    if ci:
        lo, hi = bootstrap_ci(nll_nats / LN2, n_bytes.astype(np.float64))
        out["bits_per_byte_ci95"] = [lo, hi]
    return out


@torch.no_grad()
def evaluate_lm(
    model: MorrowglassLM,
    split: TokenSplit,
    device: str,
    batch_size: int = 32,
    max_windows: int | None = None,
    ci: bool = True,
    calibration: bool = False,
) -> dict[str, Any]:
    """Evaluate next-token prediction over non-overlapping windows of a held-out split."""
    was_training = model.training
    model.eval()
    T = model.cfg.block_size
    wins = eval_windows(len(split), T, max_windows)
    ids = split.ids
    nll, ntok, nbytes = [], [], []
    confs: list[np.ndarray] = []
    hits: list[np.ndarray] = []
    # Group windows by length so full-length windows batch together.
    for length in sorted({w[1] for w in wins}, reverse=True):
        group = [w for w in wins if w[1] == length]
        for i in range(0, len(group), batch_size):
            chunk = group[i : i + batch_size]
            x = torch.from_numpy(np.stack([ids[s : s + length] for s, _ in chunk])).to(device)
            y_np = np.stack([ids[s + 1 : s + 1 + length] for s, _ in chunk])
            y = torch.from_numpy(y_np).to(device)
            logits = model(x).logits.float()
            logp_all = torch.log_softmax(logits, dim=-1)
            lp = logp_all.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # (b, length)
            if calibration:
                top_lp, top_id = logp_all.max(-1)
                confs.append(top_lp.exp().flatten().cpu().numpy())
                hits.append((top_id == y).flatten().cpu().numpy())
            nll.extend((-lp.sum(1)).cpu().double().numpy().tolist())
            ntok.extend([length] * len(chunk))
            nbytes.extend(split.byte_len[y_np].sum(1).tolist())
    if was_training:
        model.train()
    out = summarize(np.array(nll), np.array(ntok), np.array(nbytes), ci=ci)
    if calibration and confs:
        out["calibration"] = calibration_report(np.concatenate(confs), np.concatenate(hits))
    return out


def calibration_report(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> dict[str, Any]:
    """Top-1 reliability: does a 70%-confident prediction come true ~70% of the time?

    ECE = sum_b (n_b / N) * |accuracy_b - mean_confidence_b| over equal-width bins.
    The model's softmax probability is a model score, not a guarantee; this measures how
    far the two diverge on held-out text.
    """
    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    ece = 0.0
    N = len(conf)
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= 0) & (conf <= edges[1])
        n = int(m.sum())
        if n == 0:
            bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": 0, "conf": None, "acc": None})
            continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        ece += n / N * abs(a - c)
        bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": n, "conf": c, "acc": a})
    return {"ece": ece, "top1_accuracy": float(correct.mean()), "mean_confidence": float(conf.mean()),
            "n_tokens": N, "bins": bins}
