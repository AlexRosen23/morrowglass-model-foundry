"""Optional track: LoRA adaptation of a small pretrained Hugging Face causal LM vs the frozen model.

Status: implemented and plumbing-tested on a locally built random tiny GPT-2 (no download).
The real comparison needs a pretrained checkpoint download (distilgpt2 is ~350 MB), which needs
the owner's approval under the project rules, so ``run_lora`` refuses unless ``approved=True``.

Comparison (same split and metric as the scratch track):
    frozen pretrained model  vs  same model + LoRA adapters trained on the Shakespeare train split
    metric: bits per UTF-8 byte on a fixed, deterministic subset of validation windows
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from morrowglass.data.corpus import load_split
from morrowglass.repro import REPO_ROOT, git_provenance, now_iso, set_seed, write_json

KNOWN_DOWNLOADS_MB = {"distilgpt2": 353, "gpt2": 548}


def _require_hf() -> Any:
    try:
        import peft
        import transformers
    except ImportError as e:  # pragma: no cover - depends on extras
        raise RuntimeError("install the optional extra first: uv sync --extra hf") from e
    return transformers, peft


def lora_wrap(model: torch.nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.05,
              targets: tuple[str, ...] = ("c_attn",)) -> torch.nn.Module:
    _, peft = _require_hf()
    cfg = peft.LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(targets),
                          task_type="CAUSAL_LM", bias="none")
    return peft.get_peft_model(model, cfg)


def trainable_fraction(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


@torch.no_grad()
def eval_bpb(model: torch.nn.Module, ids: np.ndarray, text_bytes_per_token: np.ndarray, block: int,
             n_windows: int | None, device: str) -> dict[str, Any]:
    """bits/byte over non-overlapping windows (all of them when ``n_windows`` is None), with a bootstrap CI.

    Same windowing as the scratch track's evaluator: context restarts at each window and every
    token after the first is predicted once.
    """
    from morrowglass.eval.metrics import LN2, bootstrap_ci

    model.eval()
    nll_w: list[float] = []
    bytes_w: list[float] = []
    s = 0
    while s < len(ids) - 1 and (n_windows is None or len(nll_w) < n_windows):
        L = min(block, len(ids) - 1 - s)
        x = torch.from_numpy(ids[s : s + L][None]).to(device)
        y = torch.from_numpy(ids[s + 1 : s + L + 1][None]).to(device)
        logits = model(input_ids=x).logits.float()
        lp = torch.log_softmax(logits, -1).gather(-1, y[..., None]).squeeze(-1)
        nll_w.append(-float(lp.sum()))
        bytes_w.append(float(text_bytes_per_token[ids[s + 1 : s + L + 1]].sum()))
        s += L
    nll_a, b_a = np.array(nll_w), np.array(bytes_w)
    lo, hi = bootstrap_ci(nll_a / LN2, b_a)
    return {"bits_per_byte": float(nll_a.sum() / LN2 / max(b_a.sum(), 1.0)), "bits_per_byte_ci95": [lo, hi],
            "windows": len(nll_w), "bytes_predicted": int(b_a.sum())}


def token_byte_lengths(tok: Any) -> np.ndarray:
    """Exact UTF-8 byte length of every GPT-2 byte-level BPE token (via the byte<->unicode table)."""
    from morrowglass.tokenizer.tokenizers import _bytes_to_unicode

    u2b = {v: k for k, v in _bytes_to_unicode().items()}
    toks = tok.convert_ids_to_tokens(list(range(len(tok))))
    return np.array([len(bytes(u2b[c] for c in t if c in u2b)) for t in toks], dtype=np.int64)


def train_lora(model: torch.nn.Module, ids: np.ndarray, block: int, steps: int, batch: int, lr: float,
               device: str, seed: int) -> list[float]:
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    losses = []
    for _ in range(steps):
        starts = torch.randint(0, len(ids) - block - 1, (batch,), generator=gen).numpy()
        x = torch.from_numpy(np.stack([ids[s : s + block] for s in starts])).to(device)
        out = model(input_ids=x, labels=x)
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(float(out.loss.detach()))
    return losses


def run_lora(model_name: str = "distilgpt2", approved: bool = False, steps: int = 300, batch: int = 8,
             block: int = 256, lr: float = 2e-4, n_windows: int | None = None, device: str = "auto",
             seed: int = 0) -> dict[str, Any]:
    if not approved:
        mb = KNOWN_DOWNLOADS_MB.get(model_name, "unknown")
        raise PermissionError(f"LoRA track needs a pretrained download of {model_name} (~{mb} MB) from the "
                              "Hugging Face Hub. Re-run with --approved after the owner agrees.")
    from morrowglass.data.corpus import dataset_hash
    from morrowglass.hardware import hardware_report, memory_in_use_mb, resolve_device

    transformers, _ = _require_hf()
    dev = resolve_device(device)
    set_seed(seed)
    t_dl = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    tok.model_max_length = 10**9  # we window manually; silence the long-sequence warning
    base = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(dev)
    load_s = time.time() - t_dl
    train_text, val_text = load_split("shakespeare", "train"), load_split("shakespeare", "val")
    tr_ids = np.array(tok(train_text)["input_ids"], dtype=np.int64)
    va_ids = np.array(tok(val_text)["input_ids"], dtype=np.int64)
    bpt = token_byte_lengths(tok)
    if int(bpt[va_ids].sum()) != len(val_text.encode()):
        raise AssertionError("GPT-2 token byte lengths do not reconstruct the validation text")
    t0 = time.time()
    frozen = eval_bpb(base, va_ids, bpt, block, n_windows, dev)
    model = lora_wrap(base)
    trainable, total = trainable_fraction(model)
    t1 = time.time()
    losses = train_lora(model, tr_ids, block, steps, batch, lr, dev, seed)
    train_s = time.time() - t1
    adapted = eval_bpb(model, va_ids, bpt, block, n_windows, dev)
    out = {"model": model_name, "status": "completed", "split": "val (full; test untouched)",
           "dataset_hash": dataset_hash("shakespeare"),
           "frozen_val": frozen, "lora_val": adapted,
           "delta_bpb": adapted["bits_per_byte"] - frozen["bits_per_byte"],
           "trainable_params": trainable, "total_params": total,
           "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "targets": ["c_attn"]},
           "steps": steps, "batch": batch, "block": block, "lr": lr, "seed": seed,
           "train_tokens": steps * batch * block, "train_split_tokens": len(tr_ids),
           "loss_curve": losses[:: max(1, steps // 50)], "train_seconds": round(train_s, 1),
           "load_seconds": round(load_s, 1), "wall_time_s": round(time.time() - t0, 1), "device": dev,
           "memory_mb": round(memory_in_use_mb(dev), 1), "hardware": hardware_report()["chip"],
           "git": git_provenance(), "finished": now_iso(),
           "caveats": ["distilgpt2 was pretrained on OpenWebText; Shakespeare quotations may appear in its "
                       "pretraining data, so 'held-out' is weaker than for the scratch models.",
                       "Different tokenizer (GPT-2 BPE, 50,257) and context (256 GPT-2 tokens ~ 4x more bytes "
                       "than the byte models' 256); bpb keeps the comparison in the same unit.",
                       "One seed; validation only; the locked test split is untouched."]}
    write_json(REPO_ROOT / "reports" / "lora" / f"{model_name}-s{seed}.json", out)
    return out
