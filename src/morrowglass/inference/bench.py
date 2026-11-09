"""Inference latency, throughput and memory for a saved checkpoint (measured, not estimated)."""

from __future__ import annotations

import statistics
import time
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
