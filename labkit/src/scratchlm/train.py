"""Step 6: training. You write the learning-rate schedule, the optimizer setup and one training step;
the loop around them (``train``) is given.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from scratchlm.checkpoint import save_checkpoint
from scratchlm.data import sample_batch
from scratchlm.evaluate import evaluate
from scratchlm.model import LanguageModel, ModelConfig, count_parameters


@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 16
    lr: float = 3e-3  # peak learning rate
    min_lr_ratio: float = 0.1  # the cosine decays to lr * min_lr_ratio
    warmup_steps: int = 50
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # max gradient norm; 0 disables clipping
    eval_interval: int = 100
    eval_windows: int | None = 32  # validation windows per evaluation during training (None: all)
    seed: int = 0
    threads: int = 2  # CPU threads PyTorch may use

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Learning rate for (0-based) ``step``.

    Linear warmup: steps 0 .. warmup_steps-1 get lr * (step + 1) / warmup_steps. After that a cosine
    decay from lr (at step == warmup_steps) down to lr * min_lr_ratio (at step == cfg.steps), and that
    floor for any later step.
    """
    #lab:begin 6 linear warmup, then cosine decay to a floor
    if cfg.warmup_steps and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    floor = cfg.lr * cfg.min_lr_ratio
    return floor + (cfg.lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    #lab:stub
    #> # TODO(step 6): warmup first; then progress = (step - warmup) / (steps - warmup), clamped to [0, 1],
    #> # and lr = floor + (lr - floor) * 0.5 * (1 + cos(pi * progress)) with floor = lr * min_lr_ratio.
    #> raise NotImplementedError("TODO(step 6): lr_at")
    #lab:end


def make_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with two parameter groups: weight decay for matrices (2-D or more: Linear and Embedding
    weights) and none for vectors (biases, LayerNorm). Each parameter appears once, even when tied."""
    #lab:begin 6 AdamW with decay only on matrices
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    #lab:stub
    #> # TODO(step 6): sort the trainable parameters into `decay` (p.dim() >= 2) and `no_decay`, skipping any
    #> # you have already seen (model.parameters() already skips tied duplicates, but be explicit), and
    #> # build torch.optim.AdamW with two groups, betas=(beta1, beta2) and lr=cfg.lr.
    #> raise NotImplementedError("TODO(step 6): make_optimizer")
    #lab:end


def train_step(
    model: LanguageModel, opt: torch.optim.Optimizer, x: Tensor, y: Tensor, grad_clip: float = 1.0
) -> tuple[float, float]:
    """One optimisation step on the batch (x, y). Returns ``(loss, gradient norm before clipping)``.

    Raises FloatingPointError, without changing any weight, if the loss or the gradient norm is not
    finite (NaN or infinity).
    """
    #lab:begin 6 forward, backward, clip, update
    model.train()
    opt.zero_grad(set_to_none=True)
    out = model(x, targets=y)
    assert out.loss is not None
    if not torch.isfinite(out.loss):
        raise FloatingPointError(f"loss is {out.loss.item()}")
    out.loss.backward()
    max_norm = grad_clip if grad_clip > 0 else float("inf")
    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
    if not math.isfinite(gnorm):
        opt.zero_grad(set_to_none=True)
        raise FloatingPointError(f"gradient norm is {gnorm}")
    opt.step()
    return out.loss.item(), gnorm
    #lab:stub
    #> # TODO(step 6): train mode; clear old gradients; forward with targets; refuse a non-finite loss;
    #> # backward; clip with torch.nn.utils.clip_grad_norm_ (it returns the norm before clipping); refuse
    #> # a non-finite norm (clear the gradients first); opt.step().
    #> raise NotImplementedError("TODO(step 6): train_step")
    #lab:end


def train(
    model_cfg: ModelConfig,
    cfg: TrainConfig,
    train_ids: Tensor,
    val_ids: Tensor,
    out_dir: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> tuple[LanguageModel, dict[str, Any]]:
    """Train a new model. Returns ``(model with the best validation weights, summary)``.

    With ``out_dir``, writes ``best.pt`` (lowest validation bits per byte), ``last.pt`` and
    ``summary.json`` there. Stops early with status "diverged" if a step raises FloatingPointError.
    """
    torch.set_num_threads(max(1, cfg.threads))
    torch.manual_seed(cfg.seed)
    model = LanguageModel(model_cfg)
    opt = make_optimizer(model, cfg)
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    out = Path(out_dir) if out_dir is not None else None
    best_bpb, best_step, best_state = math.inf, 0, None
    history: list[dict[str, Any]] = []
    status = "completed"
    t0 = time.time()
    log(f"model: {count_parameters(model):,} parameters; training for {cfg.steps} steps")
    step = 0
    for step in range(1, cfg.steps + 1):
        lr = lr_at(step - 1, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = sample_batch(train_ids, model_cfg.block_size, cfg.batch_size, gen)
        try:
            loss, gnorm = train_step(model, opt, x, y, cfg.grad_clip)
        except FloatingPointError as e:
            status = "diverged"
            log(f"step {step}: stopped, {e} (try a lower learning rate)")
            break
        if step % cfg.eval_interval == 0 or step == cfg.steps:
            ev = evaluate(model, val_ids, max_windows=cfg.eval_windows)
            improved = ev["bits_per_byte"] < best_bpb
            if improved:
                best_bpb, best_step = ev["bits_per_byte"], step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if out is not None:
                    save_checkpoint(out / "best.pt", model, {"step": step, "train": cfg.to_dict()})
            history.append({"step": step, "lr": lr, "train_loss": loss, "grad_norm": gnorm,
                            "val_bits_per_byte": ev["bits_per_byte"], "elapsed_s": round(time.time() - t0, 2)})
            log(f"step {step:5d}  lr {lr:.2e}  train loss {loss:.3f}  val bits/byte {ev['bits_per_byte']:.3f}"
                f"{'  *' if improved else ''}")
    wall = time.time() - t0
    if out is not None:
        save_checkpoint(out / "last.pt", model, {"step": step, "train": cfg.to_dict()})
    if best_state is not None:
        model.load_state_dict(best_state)
    tokens = step * cfg.batch_size * model_cfg.block_size
    summary = {
        "status": status,
        "steps": step,
        "params": count_parameters(model),
        "best_step": best_step,
        "best_val_bits_per_byte": best_bpb if math.isfinite(best_bpb) else None,
        "wall_time_s": round(wall, 2),
        "tokens_per_s": round(tokens / max(wall, 1e-9)),
        "history": history,
        "model": model_cfg.to_dict(),
        "train": cfg.to_dict(),
    }
    if out is not None:
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return model, summary
