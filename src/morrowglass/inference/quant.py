"""Post-training dynamic int8 quantization on CPU (registered study: EVALUATIONS["int8-dynamic"]).

``torch.ao.quantization.quantize_dynamic`` stores every ``nn.Linear`` weight as int8 and quantizes activations
on the fly; embeddings and LayerNorms stay fp32. It is deprecated in torch 2.15 (torchao is the successor)
but still runs, on the qnnpack engine on Apple silicon. The study measures what it costs in validation
bits/byte and what it buys in decode speed and weight size.
"""

from __future__ import annotations

import copy
import io
import statistics
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn

from morrowglass.hardware import hardware_report
from morrowglass.repro import git_provenance, now_iso, read_json, write_json
from morrowglass.train.trainer import CHECKPOINT_ROOT, RUNS_DIR, load_model_from_checkpoint

INT8_DIR = RUNS_DIR.parent / "serving" / "int8"


def quantize_int8(model: nn.Module) -> nn.Module:
    """A dynamic-int8 copy of ``model`` (CPU only). The original is left untouched."""
    engines = torch.backends.quantized.supported_engines
    if "qnnpack" in engines and torch.backends.quantized.engine != "qnnpack":
        torch.backends.quantized.engine = "qnnpack"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch.ao deprecation notices
        q = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model).cpu().eval(), {nn.Linear}, dtype=torch.qint8)
    return q


def state_dict_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return round(buf.tell() / 2**20, 3)


@torch.no_grad()
def bench_int8(run_id: str, threads: int = 2, out_dir: Path | None = None) -> dict[str, Any]:
    from morrowglass.data.loader import encode_split
    from morrowglass.eval.metrics import evaluate_lm
    from morrowglass.experiments.registry import EVALUATIONS
    from morrowglass.inference.bench import _val_prompt_ids, greedy_decode_timed
    from morrowglass.tokenizer.tokenizers import get_tokenizer

    ev = EVALUATIONS["int8-dynamic"]
    torch.set_num_threads(threads)
    ck = CHECKPOINT_ROOT / run_id / "best.pt"
    fp32, _ = load_model_from_checkpoint(ck)
    fp32 = fp32.cpu().eval()
    int8 = quantize_int8(fp32)
    cfg = read_json(RUNS_DIR / run_id / "config.json")["train"]
    val = encode_split(cfg["dataset"], "val", get_tokenizer(cfg["tokenizer"], cfg["dataset"]))  # never test
    prompt = _val_prompt_ids(run_id, 64)
    res: dict[str, Any] = {}
    tokens: dict[str, list[int]] = {}
    for name, m in (("fp32", fp32), ("int8", int8)):
        ev_val = evaluate_lm(m, val, "cpu", ci=True)  # type: ignore[arg-type]
        greedy_decode_timed(m, prompt, 8, True, "cpu")  # warm-up
        runs = [greedy_decode_timed(m, prompt, 64, True, "cpu") for _ in range(5)]
        tokens[name] = runs[0]["tokens"]
        res[name] = {"val_bpb": ev_val["bits_per_byte"], "val_bpb_ci95": ev_val["bits_per_byte_ci95"],
                     "tokens_per_s": round(statistics.median(64 / r["total_s"] for r in runs), 1),
                     "decode_ms_per_token": round(statistics.median(statistics.median(r["step_ms"][1:])
                                                                    for r in runs), 3),
                     "weights_mb": state_dict_mb(m)}
    delta = res["int8"]["val_bpb"] - res["fp32"]["val_bpb"]
    checks = {"abs_delta_val_bpb_below_0.01": abs(delta) < 0.01,
              "int8_faster_decode": res["int8"]["tokens_per_s"] > res["fp32"]["tokens_per_s"]}
    out = {
        "evaluation": ev.name, "run_id": run_id, "measured": now_iso(), "provenance": "measured",
        "command": f"morrowglass quant-int8 {run_id}", "git": git_provenance(),
        "checkpoint_sha256": read_json(ck.with_suffix(".sha256.json"))["sha256"],
        "protocol": ev.protocol, "hypothesis": ev.hypothesis, "criterion": ev.criterion,
        "quantized_engine": torch.backends.quantized.engine, **res, "delta_val_bpb": delta,
        "greedy_token_agreement_64": sum(a == b for a, b in zip(tokens["fp32"], tokens["int8"], strict=True)) / 64,
        "speedup_tokens_per_s": round(res["int8"]["tokens_per_s"] / res["fp32"]["tokens_per_s"], 3),
        "checks": checks, "verdict": "supported" if all(checks.values()) else "not supported",
        "hardware": {k: hardware_report()[k] for k in ("chip", "ram_gb", "torch", "cpu_count")},
        "threads": threads, "caveats": list(ev.caveats),
    }
    write_json((out_dir or INT8_DIR) / f"{run_id}.json", out)
    return out
