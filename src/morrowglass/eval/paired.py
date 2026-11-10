"""Paired comparison of two models on the same held-out bytes (registered: EVALUATIONS["paired-baseline"]).

Two models scored on the same text make correlated errors: a passage that is hard for one is usually hard
for the other. Comparing their two independent 95% intervals ignores that and is conservative. Here each
model's per-token bits are aligned to *byte* positions of the text (so byte-level and BPE models line up),
summed into contiguous byte blocks, and the difference in bits-per-byte is bootstrapped over blocks with the
same resampled indices for both models.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from morrowglass.data.loader import TokenSplit, eval_windows
from morrowglass.eval.metrics import LN2
from morrowglass.eval.ngram import KneserNey, pack
from morrowglass.model.transformer import MorrowglassLM


@torch.no_grad()
def token_bits_transformer(model: MorrowglassLM, split: TokenSplit, device: str = "cpu",
                           batch_size: int = 32) -> np.ndarray:
    """bits[i] = -log2 p(token i | its window context), on the same windows as ``evaluate_lm``; NaN at 0."""
    was_training = model.training
    model.eval()
    ids = split.ids
    out = np.full(len(ids), np.nan)
    wins = eval_windows(len(ids), model.cfg.block_size)
    for length in sorted({w[1] for w in wins}, reverse=True):
        group = [w for w in wins if w[1] == length]
        for i in range(0, len(group), batch_size):
            chunk = group[i: i + batch_size]
            x = torch.from_numpy(np.stack([ids[s: s + length] for s, _ in chunk])).to(device)
            y = torch.from_numpy(np.stack([ids[s + 1: s + 1 + length] for s, _ in chunk])).to(device)
            lp = torch.log_softmax(model(x).logits.float(), -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
            bits = (-lp / LN2).double().cpu().numpy()
            for (s, _), row in zip(chunk, bits, strict=True):
                out[s + 1: s + 1 + length] = row
    if was_training:
        model.train()
    return out


def token_bits_kn(kn: KneserNey, ids: np.ndarray) -> np.ndarray:
    """bits[i] = -log2 P_KN(token i | previous n-1 tokens), exactly as ``KneserNey.evaluate``; NaN at 0."""
    data, w, n = pack(ids, kn.w), kn.w, kn.n
    out = np.full(len(ids), np.nan)
    for i in range(1, len(ids)):
        out[i] = -math.log2(kn.prob(data[max(0, i - (n - 1)) * w: i * w], data[i * w: (i + 1) * w]))
    return out


def byte_ends(split: TokenSplit) -> np.ndarray:
    """Exclusive byte offset at which each token ends in the split's text."""
    return np.cumsum(split.byte_len[split.ids])


def per_group(bits: np.ndarray, split: TokenSplit, group_of_byte: np.ndarray, n_groups: int
              ) -> tuple[np.ndarray, np.ndarray]:
    """Sum bits and predicted bytes per group; a token belongs to the group holding its last byte."""
    ok = ~np.isnan(bits)
    g = group_of_byte[byte_ends(split)[ok] - 1]
    nbytes = split.byte_len[split.ids][ok].astype(np.float64)
    return (np.bincount(g, weights=bits[ok], minlength=n_groups),
            np.bincount(g, weights=nbytes, minlength=n_groups))


def paired_bootstrap(bits_a: np.ndarray, bytes_a: np.ndarray, bits_b: np.ndarray, bytes_b: np.ndarray,
                     n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict[str, Any]:
    """Percentile bootstrap of bpb(A) - bpb(B) over blocks, with the same resampled blocks for both models."""
    n = len(bits_a)
    point = bits_a.sum() / bytes_a.sum() - bits_b.sum() / bytes_b.sum()
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    stats = bits_a[idx].sum(1) / bytes_a[idx].sum(1) - bits_b[idx].sum(1) / bytes_b[idx].sum(1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    per_block = bits_a / np.maximum(bytes_a, 1) - bits_b / np.maximum(bytes_b, 1)
    return {"blocks": n, "difference": float(point), "ci95": [float(lo), float(hi)],
            "share_of_resamples_below_0": float((stats < 0).mean()),
            "blocks_where_a_is_lower": int((per_block < 0).sum()), "resamples": n_boot, "seed": seed}


def unpaired_ci(bits: np.ndarray, nbytes: np.ndarray, n_boot: int = 2000, seed: int = 0) -> list[float]:
    """The usual single-model interval on the same blocks (for contrast with the paired one)."""
    idx = np.random.default_rng(seed).integers(0, len(bits), size=(n_boot, len(bits)))
    s = bits[idx].sum(1) / nbytes[idx].sum(1)
    return [float(x) for x in np.quantile(s, [0.025, 0.975])]


def play_starts(text: str, titles: list[str]) -> list[tuple[str, int]]:
    """Byte offset where each work's title first appears (works are concatenated in manifest order)."""
    out = []
    for t in titles:
        i = text.find(t)
        if i < 0:
            raise ValueError(f"title {t!r} not found in the split text")
        out.append((t, len(text[:i].encode("utf-8"))))
    return sorted(out, key=lambda x: x[1])


def score_vs_baseline(run_ids: list[str] | tuple[str, ...], device: str = "cpu") -> dict[str, Any]:
    """Paired comparison of each run with the val-chosen n-gram, using the EVALUATIONS['paired-baseline'] protocol."""
    from morrowglass.data.corpus import MANIFEST_DIR, load_split
    from morrowglass.data.loader import encode_split
    from morrowglass.experiments.registry import EVALUATIONS
    from morrowglass.repro import REPO_ROOT, read_json
    from morrowglass.tokenizer.tokenizers import get_tokenizer
    from morrowglass.train.trainer import CHECKPOINT_ROOT, RUNS_DIR, load_model_from_checkpoint

    proto = EVALUATIONS["paired-baseline"].protocol
    torch.set_num_threads(2)
    text = load_split("shakespeare", "val")
    n_text = len(text.encode("utf-8"))
    titles = read_json(MANIFEST_DIR / "shakespeare.json")["files"]["val"]["works"]
    starts = play_starts(text, titles)
    play_of_byte = np.zeros(n_text, dtype=np.int64)
    for k, (_, s) in enumerate(starts):
        play_of_byte[s:] = k
    block_sizes = [int(b) for b in proto["block_bytes"]]
    groupings = {f"{b}": (np.arange(n_text) // b, math.ceil(n_text / b)) for b in block_sizes}
    groupings["play"] = (play_of_byte, len(starts))

    bl = read_json(REPO_ROOT / "reports" / "baselines-shakespeare.json")
    best = bl["best"]
    btok = get_tokenizer(best["tokenizer"], "shakespeare")
    btr, bva = encode_split("shakespeare", "train", btok), encode_split("shakespeare", "val", btok)
    kn = KneserNey(best["order"], btok.vocab_size, 1 if best["tokenizer"] == "byte" else 2).fit(btr.ids)
    kn_bits = token_bits_kn(kn, bva.ids)
    kn_groups = {k: per_group(kn_bits, bva, g, n) for k, (g, n) in groupings.items()}
    kn_bpb = float(np.nansum(kn_bits) / bva.byte_len[bva.ids][1:].sum())
    baseline = {"model": best, "val_bpb": kn_bpb, "recorded_val_bpb": bl["best_val_bpb"],
                "recorded_val_bpb_ci95": bl["best_val_bpb_ci95"],
                "matches_record": abs(kn_bpb - bl["best_val_bpb"]) < 1e-6}

    runs: dict[str, Any] = {}
    for rid in run_ids:
        summary = read_json(RUNS_DIR / rid / "summary.json")
        cfg = read_json(RUNS_DIR / rid / "config.json")["train"]
        model, state = load_model_from_checkpoint(CHECKPOINT_ROOT / rid / "best.pt")
        model.to(device)
        tok = get_tokenizer(state["tokenizer"], "shakespeare")
        va = encode_split("shakespeare", "val", tok)
        bits = token_bits_transformer(model, va, device)
        bpb = float(np.nansum(bits) / va.byte_len[va.ids][1:].sum())
        rec = summary["final"]["val"]
        row: dict[str, Any] = {
            "tokenizer": cfg["tokenizer"], "params": summary["params"], "val_bpb": bpb,
            "recorded_val_bpb": rec["bits_per_byte"], "recorded_val_bpb_ci95": rec["bits_per_byte_ci95"],
            "matches_record": abs(bpb - rec["bits_per_byte"]) < 1e-3, "paired": {}, "unpaired": {}}
        for key, (g, n) in groupings.items():
            a_bits, a_bytes = per_group(bits, va, g, n)
            b_bits, b_bytes = kn_groups[key]
            if key == "play":
                row["plays"] = [{"title": starts[k][0], "bytes": int(a_bytes[k]),
                                 "model_bpb": float(a_bits[k] / a_bytes[k]),
                                 "baseline_bpb": float(b_bits[k] / b_bytes[k]),
                                 "difference": float(a_bits[k] / a_bytes[k] - b_bits[k] / b_bytes[k])}
                                for k in range(n)]
                continue
            row["paired"][key] = paired_bootstrap(a_bits, a_bytes, b_bits, b_bytes, proto["resamples"],
                                                  proto["seed"])
            row["unpaired"][key] = {"model_ci95": unpaired_ci(a_bits, a_bytes, proto["resamples"], proto["seed"]),
                                    "baseline_ci95": unpaired_ci(b_bits, b_bytes, proto["resamples"],
                                                                 proto["seed"])}
        runs[rid] = row

    return {"protocol": proto, "validation_bytes": n_text, "plays": [{"title": t, "start_byte": s} for t, s in starts],
            "baseline": baseline, "runs": runs}


def run_paired_study(device: str = "cpu") -> dict[str, Any]:
    """Score every registered run and the best n-gram on validation; write reports/paired/summary.json."""
    from morrowglass.experiments.registry import EVALUATIONS, FLAGSHIP_RUN
    from morrowglass.repro import REPO_ROOT, git_provenance, now_iso, write_json

    ev = EVALUATIONS["paired-baseline"]
    proto = ev.protocol
    t0 = time.time()
    scored = score_vs_baseline(ev.runs, device)
    runs, baseline, n_text = scored["runs"], scored["baseline"], scored["validation_bytes"]
    starts = [(p["title"], p["start_byte"]) for p in scored["plays"]]
    flag = runs[FLAGSHIP_RUN]
    primary = flag["paired"][str(proto["primary_block_bytes"])]
    checks = {"paired_upper_below_0": primary["ci95"][1] < 0,
              "negative_in_every_play": all(p["difference"] < 0 for p in flag["plays"]),
              "scores_match_records": all(r["matches_record"] for r in runs.values()) and baseline["matches_record"]}
    if not checks["scores_match_records"]:
        verdict = "invalid (rescored totals do not match the run records)"
    else:
        verdict = "supported" if checks["paired_upper_below_0"] and checks["negative_in_every_play"] else (
            "partially supported" if checks["paired_upper_below_0"] or checks["negative_in_every_play"]
            else "not supported")
    out = {"evaluation": ev.name, "title": ev.title, "registered": ev.registered, "question": ev.question,
           "hypothesis": ev.hypothesis, "criterion": ev.criterion, "caveats": list(ev.caveats),
           "protocol": proto, "measured": now_iso(), "git": git_provenance(), "device": device,
           "seconds": round(time.time() - t0, 1), "validation_bytes": n_text,
           "plays": [{"title": t, "start_byte": s} for t, s in starts], "baseline": baseline, "runs": runs,
           "primary_run": FLAGSHIP_RUN, "checks": checks, "verdict": verdict,
           "preregistered_scratch_verdict_unchanged": "not supported (non-overlapping intervals required)"}
    write_json(REPO_ROOT / "reports" / "paired" / "summary.json", out)
    (REPO_ROOT / "docs" / "PAIRED_RESULTS.md").write_text(paired_report_md(out))
    return out


def run_seed_paired(device: str = "cpu") -> dict[str, Any]:
    """Paired comparison for every seed of the flagship recipe (registered: build('scratch-seeds'))."""
    from morrowglass.experiments.registry import FLAGSHIP_RUN, build
    from morrowglass.repro import REPO_ROOT, git_provenance, now_iso, read_json, write_json
    from morrowglass.train.trainer import RUNS_DIR

    t0 = time.time()
    ids = [FLAGSHIP_RUN, *(a.run_id for a in build("scratch-seeds").arms)]
    scored = score_vs_baseline(ids, device)
    for rid, row in scored["runs"].items():
        row["seed"] = read_json(RUNS_DIR / rid / "config.json")["train"]["seed"]
    out = {"experiment": "scratch-seeds", "measured": now_iso(), "git": git_provenance(), "device": device,
           "seconds": round(time.time() - t0, 1), **scored}
    write_json(REPO_ROOT / "reports" / "paired" / "scratch-seeds.json", out)
    return out


def paired_report_md(s: dict[str, Any]) -> str:
    """docs/PAIRED_RESULTS.md from reports/paired/summary.json (numbers only from the record)."""
    def f(x: float, d: int = 4) -> str:
        return f"{x:+.{d}f}" if x < 0 or x > 0 else f"{x:.{d}f}"

    def ci(c: list[float], d: int = 3) -> str:
        return f"[{c[0]:.{d}f}, {c[1]:.{d}f}]"

    b = s["baseline"]
    blk = str(s["protocol"]["primary_block_bytes"])
    lines = [
        "# Paired re-analysis: scratch models vs the best n-gram",
        "",
        "Generated by `morrowglass paired` from `reports/paired/summary.json`; do not edit by hand.",
        "",
        f"- **Registered:** {s['registered']}, in `experiments/registry.py` (`EVALUATIONS['paired-baseline']`), "
        "committed before the first measurement. **Post-hoc**: written after the scratch verdict was known.",
        f"- **Question:** {s['question']}",
        f"- **Hypothesis:** {s['hypothesis']}",
        f"- **Criterion:** {s['criterion']}",
        f"- **Verdict (flagship):** **{s['verdict']}**. The pre-registered scratch verdict stays "
        f"**{s['preregistered_scratch_verdict_unchanged']}**.",
        f"- **Data:** validation split only ({s['validation_bytes']:,} bytes, three plays); the locked test split "
        f"stayed closed. Measured {s['measured']} on {s['device']} at `{s['git']['commit'][:8]}`, "
        f"{s['seconds']} s.",
        f"- **Baseline:** {b['model']['model']} on {b['model']['tokenizer']}, val {b['val_bpb']:.4f} bpb "
        f"(re-scored; matches the record: {b['matches_record']}).",
        "",
        f"## Paired difference, bpb(model) - bpb(baseline), {int(blk) // 1024} KiB blocks",
        "",
        "| run | val bpb | difference | paired 95% interval | blocks where model is lower | unpaired intervals "
        "(model / baseline, same blocks) |",
        "|---|---:|---:|---|---:|---|",
    ]
    for rid, r in s["runs"].items():
        p, u = r["paired"][blk], r["unpaired"][blk]
        lines.append(f"| `{rid}` | {r['val_bpb']:.4f} | {f(p['difference'])} | {ci(p['ci95'], 4)} | "
                     f"{p['blocks_where_a_is_lower']} / {p['blocks']} | {ci(u['model_ci95'])} / "
                     f"{ci(u['baseline_ci95'])} |")
    lines += ["", "## Per play (difference in bpb)", "", "| run | " + " | ".join(
        p["title"].title() for p in next(iter(s["runs"].values()))["plays"]) + " |",
        "|---|" + "---:|" * len(s["plays"])]
    for rid, r in s["runs"].items():
        lines.append(f"| `{rid}` | " + " | ".join(f(p["difference"]) for p in r["plays"]) + " |")
    lines += ["", "## 256-byte blocks (the original per-window granularity)", "",
              "| run | paired 95% interval | unpaired model interval |", "|---|---|---|"]
    for rid, r in s["runs"].items():
        lines.append(f"| `{rid}` | {ci(r['paired']['256']['ci95'], 4)} | {ci(r['unpaired']['256']['model_ci95'])} |")
    lines += ["", "## What this shows", "",
              "- Pairing removes the variation the two models share (hard passages are hard for both). On the same "
              "blocks the unpaired intervals overlap while the paired interval of the difference does not include 0.",
              "- Intervals from 8 KiB blocks are wider than from 256-byte blocks: neighbouring windows are correlated, "
              "so the per-window intervals elsewhere in this project are somewhat optimistic about absolute bpb.",
              "", "## Caveats", ""] + [f"- {c}" for c in s["caveats"]] + [""]
    return "\n".join(lines)
