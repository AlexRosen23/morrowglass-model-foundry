"""Inference latency, throughput and memory for a saved checkpoint (measured, not estimated)."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import torch

from morrowglass.hardware import hardware_report, memory_in_use_mb, mps_usable
from morrowglass.repro import now_iso, read_json, write_json
from morrowglass.train.trainer import CHECKPOINT_ROOT, RUNS_DIR, load_model_from_checkpoint


def _sync(dev: str) -> None:
    if dev == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def bench_run(run_id: str, repeats: int = 20, threads: int = 4) -> dict[str, Any]:
    torch.set_num_threads(threads)
    ck = CHECKPOINT_ROOT / run_id / "best.pt"
    model, state = load_model_from_checkpoint(ck)
    T, V = model.cfg.block_size, model.cfg.vocab_size
    out: dict[str, Any] = {"run_id": run_id, "measured": now_iso(), "threads": threads,
                           "checkpoint_bytes": ck.stat().st_size,
                           "checkpoint_sha256": read_json(ck.with_suffix(".sha256.json"))["sha256"],
                           "params": sum(p.numel() for p in model.parameters()), "devices": {}}
    devices = ["cpu"] + (["mps"] if mps_usable() else [])
    for dev in devices:
        m = model.to(dev)
        x1 = torch.randint(0, V, (1, T), device=dev)
        xb = torch.randint(0, V, (32, T), device=dev)
        for _ in range(3):
            m(x1)
            m(xb)
        _sync(dev)
        lat = []
        for _ in range(repeats):
            t = time.perf_counter()
            m(x1)
            _sync(dev)
            lat.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        for _ in range(max(3, repeats // 4)):
            m(xb)
        _sync(dev)
        thr = 32 * T * max(3, repeats // 4) / (time.perf_counter() - t)
        # Autoregressive decoding cost per new token (no KV cache: full forward each step).
        ctx = torch.randint(0, V, (1, 16), device=dev)
        t = time.perf_counter()
        n_new = 64
        for _ in range(n_new):
            nxt = m(ctx[:, -T:]).logits[:, -1].argmax(-1, keepdim=True)
            ctx = torch.cat([ctx, nxt], 1)
        _sync(dev)
        out["devices"][dev] = {
            "forward_latency_ms_batch1_fullctx": {"median": round(statistics.median(lat), 3),
                                                  "p90": round(sorted(lat)[int(0.9 * len(lat)) - 1], 3)},
            "forward_tokens_per_s_batch32": round(thr),
            "greedy_decode_ms_per_token": round((time.perf_counter() - t) * 1000 / n_new, 3),
            "memory_mb": round(memory_in_use_mb(dev), 1),
            "memory_note": "MPS: allocated tensors; CPU: process peak RSS (includes Python + torch)",
        }
    out["hardware"] = {k: hardware_report()[k] for k in ("chip", "ram_gb", "torch", "cpu_count")}
    out["caveat"] = "Shared machine: other workloads were running; treat as indicative, not a benchmark."
    write_json(RUNS_DIR / run_id / "bench.json", out)
    _ = state
    return out


# ---------------------------------------------------------------- KV cache study (registered: EVALUATIONS["kv-cache"])

SERVING_DIR = RUNS_DIR.parent / "serving"


def _val_prompt_ids(run_id: str, n: int) -> list[int]:
    """The first ``n`` tokens of the validation split, in the run's own tokenizer (never test)."""
    from morrowglass.data.loader import encode_split
    from morrowglass.tokenizer.tokenizers import get_tokenizer

    cfg = read_json(RUNS_DIR / run_id / "config.json")["train"]
    tok = get_tokenizer(cfg["tokenizer"], cfg["dataset"])
    return [int(i) for i in encode_split(cfg["dataset"], "val", tok).ids[:n]]


@torch.no_grad()
def greedy_decode_timed(model: Any, prompt: list[int], new_tokens: int, use_cache: bool,
                        device: str) -> dict[str, Any]:
    """Greedy decoding through the same ``CachedDecoder`` the API uses, timing every step."""
    from morrowglass.inference.generate import CachedDecoder

    dec = CachedDecoder(model, device, use_cache)
    ids = list(prompt)
    steps: list[float] = []
    logits: list[torch.Tensor] = []
    t_all = time.perf_counter()
    for _ in range(new_tokens):
        t = time.perf_counter()
        lo = dec.next_logits(ids)
        nxt = int(lo.argmax())  # .item() synchronises MPS
        steps.append((time.perf_counter() - t) * 1000)
        logits.append(lo.float().cpu())
        ids.append(nxt)
    total = time.perf_counter() - t_all
    return {"tokens": ids[len(prompt):], "step_ms": steps, "total_s": total, "logits": logits, "stats": dec.stats}


def bench_kv_cache(run_id: str, threads: int = 2, devices: list[str] | None = None,
                   out_dir: Path | None = None) -> dict[str, Any]:
    from morrowglass.experiments.registry import EVALUATIONS
    from morrowglass.repro import git_provenance

    ev = EVALUATIONS["kv-cache"]
    p = ev.protocol
    torch.set_num_threads(threads)
    ck = CHECKPOINT_ROOT / run_id / "best.pt"
    model, _ = load_model_from_checkpoint(ck)
    lengths, n_new, repeats = p["prompt_tokens"], p["new_tokens"], p["repeats"]
    if max(lengths) + n_new > model.cfg.block_size:
        raise ValueError("protocol exceeds block_size")
    prompts = {L: _val_prompt_ids(run_id, L) for L in lengths}

    # 1) Equivalence on CPU (not timed): same greedy tokens, logits within tolerance.
    model = model.to("cpu")
    equivalence = []
    for L in lengths:
        a = greedy_decode_timed(model, prompts[L], n_new, True, "cpu")
        b = greedy_decode_timed(model, prompts[L], n_new, False, "cpu")
        diff = max(float((x - y).abs().max()) for x, y in zip(a["logits"], b["logits"], strict=True))
        equivalence.append({"prompt_tokens": L, "same_tokens": a["tokens"] == b["tokens"],
                            "max_abs_logit_diff": diff, "cache_stats": a["stats"]})

    # 2) Speed, per device / prompt length / cache on-off.
    devs = devices or (["cpu"] + (["mps"] if mps_usable() else []))
    rows = []
    for dev in devs:
        m = model.to(dev)
        greedy_decode_timed(m, prompts[lengths[0]], 8, True, dev)  # warm-up
        greedy_decode_timed(m, prompts[lengths[0]], 8, False, dev)
        for L in lengths:
            for use_cache in (False, True):
                runs = [greedy_decode_timed(m, prompts[L], n_new, use_cache, dev) for _ in range(repeats)]
                rows.append({
                    "device": dev, "prompt_tokens": L, "new_tokens": n_new, "kv_cache": use_cache,
                    "tokens_per_s": round(statistics.median(n_new / r["total_s"] for r in runs), 1),
                    "ttft_ms": round(statistics.median(r["step_ms"][0] for r in runs), 3),
                    "decode_ms_per_token": round(statistics.median(statistics.median(r["step_ms"][1:])
                                                                   for r in runs), 3),
                    "p90_decode_ms_per_token": round(statistics.median(
                        sorted(r["step_ms"][1:])[int(0.9 * (n_new - 1))] for r in runs), 3),
                })
        model = m.to("cpu")

    def speedup(dev: str, L: int) -> float:
        on = next(r for r in rows if r["device"] == dev and r["prompt_tokens"] == L and r["kv_cache"])
        off = next(r for r in rows if r["device"] == dev and r["prompt_tokens"] == L and not r["kv_cache"])
        return round(on["tokens_per_s"] / off["tokens_per_s"], 3)

    speedups = {dev: {str(L): speedup(dev, L) for L in lengths} for dev in devs}
    cpu = speedups["cpu"]
    checks = {
        "same_tokens": all(e["same_tokens"] for e in equivalence),
        "logits_within_1e-4": all(e["max_abs_logit_diff"] < 1e-4 for e in equivalence),
        "faster_at_every_length_cpu": all(v > 1 for v in cpu.values()),
        "speedup_grows_with_prompt_cpu": cpu[str(lengths[-1])] > cpu[str(lengths[0])],
    }
    out = {
        "evaluation": ev.name, "run_id": run_id, "measured": now_iso(), "provenance": "measured",
        "command": f"morrowglass bench-kv {run_id}", "git": git_provenance(),
        "checkpoint_sha256": read_json(ck.with_suffix(".sha256.json"))["sha256"],
        "params": sum(p.numel() for p in model.parameters()), "block_size": model.cfg.block_size,
        "protocol": p, "hypothesis": ev.hypothesis, "criterion": ev.criterion,
        "equivalence_cpu": equivalence, "rows": rows, "speedup_tokens_per_s": speedups, "checks": checks,
        "verdict": "supported" if all(checks.values()) else "not supported",
        "hardware": {k: hardware_report()[k] for k in ("chip", "ram_gb", "torch", "cpu_count")},
        "threads": threads, "caveats": list(ev.caveats),
    }
    write_json((out_dir or SERVING_DIR / "kv-cache") / f"{run_id}.json", out)
    return out
