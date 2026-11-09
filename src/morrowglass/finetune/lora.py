"""Optional track: LoRA adaptation of a small pretrained Hugging Face causal LM vs the frozen model.

Status: implemented and plumbing-tested on a locally built random tiny GPT-2 (no download).
The real comparison needs a pretrained checkpoint download (distilgpt2 is ~350 MB), which needs
the owner's approval under the project rules, so ``run_lora`` refuses unless ``approved=True``.

Comparison (same split and metric as the scratch track):
    frozen pretrained model  vs  same model + LoRA adapters trained on the Shakespeare train split
    metric: bits per UTF-8 byte on a fixed, deterministic subset of validation windows
"""

from __future__ import annotations

import math
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
             n_windows: int, device: str) -> float:
    """bits/byte over the first ``n_windows`` non-overlapping windows (deterministic subset)."""
    model.eval()
    nll, nbytes = 0.0, 0
    for w in range(n_windows):
        s = w * block
        if s + block + 1 > len(ids):
            break
        x = torch.from_numpy(ids[s : s + block][None]).to(device)
        y = torch.from_numpy(ids[s + 1 : s + block + 1][None]).to(device)
        logits = model(input_ids=x).logits.float()
        lp = torch.log_softmax(logits, -1).gather(-1, y[..., None]).squeeze(-1)
        nll -= float(lp.sum())
        nbytes += int(text_bytes_per_token[ids[s + 1 : s + block + 1]].sum())
    return nll / math.log(2) / max(nbytes, 1)


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
             block: int = 256, lr: float = 2e-4, n_windows: int = 200, device: str = "auto",
             seed: int = 0) -> dict[str, Any]:
    if not approved:
        mb = KNOWN_DOWNLOADS_MB.get(model_name, "unknown")
        raise PermissionError(f"LoRA track needs a pretrained download of {model_name} (~{mb} MB) from the "
                              "Hugging Face Hub. Re-run with --approved after the owner agrees.")
    from morrowglass.hardware import resolve_device

    transformers, _ = _require_hf()
    dev = resolve_device(device)
    set_seed(seed)
    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    base = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(dev)
    tr_ids = np.array(tok(load_split("shakespeare", "train"))["input_ids"], dtype=np.int64)
    va_ids = np.array(tok(load_split("shakespeare", "val"))["input_ids"], dtype=np.int64)
    bpt = np.array([len(tok.convert_tokens_to_string([t]).encode()) for t in tok.convert_ids_to_tokens(
        list(range(len(tok))))], dtype=np.int64)
    t0 = time.time()
    frozen = eval_bpb(base, va_ids, bpt, block, n_windows, dev)
    model = lora_wrap(base)
    trainable, total = trainable_fraction(model)
    losses = train_lora(model, tr_ids, block, steps, batch, lr, dev, seed)
    adapted = eval_bpb(model, va_ids, bpt, block, n_windows, dev)
    out = {"model": model_name, "status": "completed", "frozen_val_bpb_subset": frozen,
           "lora_val_bpb_subset": adapted, "trainable_params": trainable, "total_params": total,
           "steps": steps, "batch": batch, "block": block, "lr": lr, "seed": seed, "n_val_windows": n_windows,
           "loss_curve": losses[:: max(1, steps // 50)], "wall_time_s": round(time.time() - t0, 1), "device": dev,
           "git": git_provenance(), "finished": now_iso(),
           "caveats": ["Pretrained on WebText; Shakespeare-like text may appear in its pretraining data.",
                       "Validation subset only (first n windows); the test split is untouched."]}
    write_json(REPO_ROOT / "reports" / "lora" / f"{model_name}-s{seed}.json", out)
    return out
