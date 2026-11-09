"""The training loop: seeded, checkpointed, integrity-checked, and honest about failure.

Features: AdamW with decoupled weight decay, linear warmup + cosine/linear/constant decay,
gradient accumulation, gradient clipping, periodic held-out evaluation, early stopping,
non-finite loss/gradient handling, wall-clock budget, cooperative cancellation (a STOP
file in the run directory), and resumable checkpoints whose sha256 is verified on load.
"""

from __future__ import annotations

import io
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from morrowglass.hardware import hardware_report, memory_in_use_mb, resolve_device
from morrowglass.model.transformer import ModelConfig, MorrowglassLM, compute_budget, count_parameters
from morrowglass.repro import (
    REPO_ROOT,
    IntegrityError,
    config_hash,
    git_provenance,
    now_iso,
    set_seed,
    sha256_bytes,
    write_json,
    write_verified,
)
from morrowglass.train.tasks import LMTask, SumTask, Task

RUNS_DIR = REPO_ROOT / "reports" / "runs"
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "checkpoints"


@dataclass
class TrainConfig:
    name: str
    experiment: str = "adhoc"
    task: str = "lm"  # "lm" | "digitsum"
    dataset: str = "shakespeare"
    tokenizer: str = "byte"
    model: dict[str, Any] = field(default_factory=dict)  # ModelConfig fields except vocab_size
    batch_size: int = 32
    grad_accum: int = 1
    max_steps: int = 1000
    lr: float = 3e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    schedule: str = "cosine"  # cosine | linear | constant
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # 0 disables clipping
    eval_interval: int = 100
    eval_windows: int = 64
    early_stopping_patience: int = 0  # evals without improvement; 0 disables
    seed: int = 0
    device: str = "auto"
    num_threads: int = 4  # stay polite on a shared CPU
    checkpoint_interval: int = 500
    log_interval: int = 20
    max_minutes: float = 0.0  # wall-clock budget; 0 = none
    nan_policy: str = "abort"  # abort | skip
    max_bad_steps: int = 5
    pos_augment: bool = False  # digitsum only
    profile: str = "custom"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def identity(self) -> dict[str, Any]:
        """Fields that define the experiment (excludes device/threads/logging cadence)."""
        skip = {"device", "num_threads", "log_interval", "checkpoint_interval", "notes", "profile", "experiment"}
        return {k: v for k, v in self.to_dict().items() if k not in skip}

    @property
    def run_id(self) -> str:
        return f"{self.name}-{config_hash(self.identity())[:8]}-s{self.seed}"


def lr_at(step: int, cfg: TrainConfig) -> float:
    if cfg.warmup_steps and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if cfg.schedule == "constant":
        return cfg.lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    floor = cfg.lr * cfg.min_lr_ratio
    if cfg.schedule == "linear":
        return floor + (cfg.lr - floor) * (1 - progress)
    return floor + (cfg.lr - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def build_task(cfg: TrainConfig) -> Task:
    if cfg.task == "lm":
        return LMTask(cfg.dataset, cfg.tokenizer, cfg.eval_windows)
    if cfg.task == "digitsum":
        return SumTask(cfg.model.get("block_size", 40), pos_augment=cfg.pos_augment)
    raise ValueError(f"unknown task {cfg.task!r}")


def build_model(cfg: TrainConfig, vocab_size: int) -> MorrowglassLM:
    return MorrowglassLM(ModelConfig.from_dict({**cfg.model, "vocab_size": vocab_size}))


def make_optimizer(model: MorrowglassLM, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    seen: set[int] = set()
    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


# ---------------------------------------------------------------- checkpoints


def save_checkpoint(path: Path, state: dict[str, Any]) -> dict[str, Any]:
    buf = io.BytesIO()
    torch.save(state, buf)
    data = buf.getvalue()
    sha = write_verified(path, data)
    meta = {"path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
            "sha256": sha, "bytes": len(data), "step": state["step"], "saved": now_iso()}
    write_json(path.with_suffix(".sha256.json"), meta)
    return meta


def load_checkpoint(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    """Load a checkpoint after verifying its sha256 (from the sidecar file if not given)."""
    data = Path(path).read_bytes()
    if expected_sha256 is None:
        side = Path(path).with_suffix(".sha256.json")
        if not side.exists():
            raise IntegrityError(f"{path}: no recorded hash (sidecar {side.name} missing)")
        expected_sha256 = json.loads(side.read_text())["sha256"]
    actual = sha256_bytes(data)
    if actual != expected_sha256:
        raise IntegrityError(f"{path}: checkpoint sha256 {actual[:12]}… != recorded {expected_sha256[:12]}…")
    state: dict[str, Any] = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    return state


def load_model_from_checkpoint(path: Path) -> tuple[MorrowglassLM, dict[str, Any]]:
    state = load_checkpoint(path)
    model = MorrowglassLM(ModelConfig.from_dict(state["model_config"]))
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


# ---------------------------------------------------------------- training


class Trainer:
    def __init__(self, cfg: TrainConfig, runs_dir: Path = RUNS_DIR, ckpt_root: Path = CHECKPOINT_ROOT,
                 quiet: bool = False) -> None:
        self.cfg = cfg
        self.run_dir = runs_dir / cfg.run_id
        self.ckpt_dir = ckpt_root / cfg.run_id
        self.quiet = quiet
        self.device = resolve_device(cfg.device)
        torch.set_num_threads(max(1, cfg.num_threads))
        set_seed(cfg.seed)
        self.task = build_task(cfg)
        set_seed(cfg.seed)  # model init independent of task-building RNG use
        self.model = build_model(cfg, self.task.tokenizer.vocab_size).to(self.device)
        self.opt = make_optimizer(self.model, cfg)
        self.gen = torch.Generator().manual_seed(cfg.seed + 1)
        self.step = 0
        self.best_metric = math.inf
        self.best_step = -1
        self.evals_since_best = 0
        self.bad_steps = 0
        self.skipped_steps = 0
        self.tokens_seen = 0

    # -- logging -------------------------------------------------------------
    def _log(self, rec: dict[str, Any]) -> None:
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _say(self, msg: str) -> None:
        if not self.quiet:
            print(f"[{self.cfg.run_id}] {msg}", file=sys.stderr, flush=True)

    # -- checkpoint state ----------------------------------------------------
    def state(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "step": self.step,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "evals_since_best": self.evals_since_best,
            "tokens_seen": self.tokens_seen,
            "skipped_steps": self.skipped_steps,
            "rng_torch": torch.get_rng_state(),
            "rng_batch": self.gen.get_state(),
            "train_config": self.cfg.to_dict(),
            "model_config": self.model.cfg.to_dict(),
            "tokenizer": self.task.tokenizer.name,
            "tokenizer_sha256": self.task.tokenizer.fingerprint(),
        }

    def restore(self, path: Path) -> None:
        s = load_checkpoint(path)
        self.model.load_state_dict(s["model"])
        self.opt.load_state_dict(s["optimizer"])
        self.step = int(s["step"])
        self.best_metric = float(s["best_metric"])
        self.best_step = int(s["best_step"])
        self.evals_since_best = int(s["evals_since_best"])
        self.tokens_seen = int(s["tokens_seen"])
        self.skipped_steps = int(s["skipped_steps"])
        torch.set_rng_state(s["rng_torch"])
        self.gen.set_state(s["rng_batch"])

    # -- main loop -----------------------------------------------------------
    def train(self, resume: Path | None = None) -> dict[str, Any]:
        cfg = self.cfg
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        stop_file = self.run_dir / "STOP"
        stop_file.unlink(missing_ok=True)
        if resume is None:
            (self.run_dir / "metrics.jsonl").unlink(missing_ok=True)
        else:
            self.restore(resume)
            self._say(f"resumed from step {self.step}")
        block = self.model.cfg.block_size
        tokens_per_step = cfg.batch_size * cfg.grad_accum * block
        meta = {
            "run_id": cfg.run_id,
            "config_hash": config_hash(cfg.identity()),
            "seed": cfg.seed,
            "git": git_provenance(),
            "command": " ".join(sys.argv),
            "device": self.device,
            "hardware": hardware_report(),
            "started": now_iso(),
            "data": self.task.provenance(),
            "params": count_parameters(self.model),
            "budget": compute_budget(self.model.cfg, tokens_per_step * cfg.max_steps),
            "resumed_from": str(resume) if resume else None,
            "status": "running",
        }
        write_json(self.run_dir / "config.json", {"train": cfg.to_dict(), "model": self.model.cfg.to_dict()})
        write_json(self.run_dir / "metadata.json", meta)
        self._say(f"device={self.device} params={meta['params']:,} steps={cfg.max_steps} "
                  f"tokens/step={tokens_per_step}")
        status = "completed"
        failure: dict[str, Any] | None = None
        t0 = time.time()
        t_last = t0
        tok_last = self.tokens_seen
        self.model.train()
        checkpoints: dict[str, Any] = {}
        while self.step < cfg.max_steps:
            if stop_file.exists():
                status = "cancelled"
                break
            if cfg.max_minutes and (time.time() - t0) / 60 > cfg.max_minutes:
                status = "time_budget"
                break
            lr = lr_at(self.step, cfg)
            for g in self.opt.param_groups:
                g["lr"] = lr
            loss_sum = 0.0
            finite = True
            for _ in range(cfg.grad_accum):
                b = self.task.batch(cfg.batch_size, block, self.gen)
                x, y = b[0].to(self.device), b[1].to(self.device)
                off = b[2].to(self.device) if len(b) > 2 else 0
                out = self.model(x, targets=y, pos_offset=off)
                assert out.loss is not None
                loss = out.loss / cfg.grad_accum
                if not torch.isfinite(loss):
                    finite = False
                    break
                loss.backward()
                loss_sum += float(loss.detach())
            clip = cfg.grad_clip if cfg.grad_clip > 0 else float("inf")
            gnorm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)) if finite else math.nan
            if not finite or not math.isfinite(gnorm):
                self.opt.zero_grad(set_to_none=True)
                self.bad_steps += 1
                self.skipped_steps += 1
                self._log({"kind": "nonfinite", "step": self.step, "lr": lr, "loss": loss_sum if finite else None,
                           "grad_norm": None})
                if cfg.nan_policy == "abort" or self.bad_steps >= cfg.max_bad_steps:
                    status = "diverged"
                    failure = {"step": self.step, "lr": lr, "reason": "non-finite loss or gradient norm",
                               "consecutive_bad_steps": self.bad_steps}
                    self._say(f"DIVERGED at step {self.step} (lr={lr:.2e})")
                    break
                self.step += 1
                continue
            self.bad_steps = 0
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.step += 1
            self.tokens_seen += tokens_per_step
            if self.step % cfg.log_interval == 0 or self.step == 1:
                if self.device == "mps":
                    torch.mps.synchronize()
                now = time.time()
                tps = (self.tokens_seen - tok_last) / max(now - t_last, 1e-9)
                t_last, tok_last = now, self.tokens_seen
                self._log({"kind": "train", "step": self.step, "tokens": self.tokens_seen, "lr": lr,
                           "train_loss": loss_sum, "grad_norm": gnorm, "tokens_per_s": round(tps, 1),
                           "mem_mb": round(memory_in_use_mb(self.device), 1), "elapsed_s": round(now - t0, 2)})
            if self.step % cfg.eval_interval == 0 or self.step == cfg.max_steps:
                ev = self.task.evaluate(self.model, self.device, full=False)
                metric = ev[self.task.primary_metric]
                improved = metric < self.best_metric - 1e-6
                if improved:
                    self.best_metric, self.best_step, self.evals_since_best = metric, self.step, 0
                    checkpoints["best"] = save_checkpoint(self.ckpt_dir / "best.pt", self.state())
                else:
                    self.evals_since_best += 1
                self._log({"kind": "eval", "step": self.step, "tokens": self.tokens_seen, **ev,
                           "best": improved, "elapsed_s": round(time.time() - t0, 2)})
                self._say(f"step {self.step:5d} lr {lr:.2e} train {loss_sum:.4f} "
                          f"{self.task.primary_metric} {metric:.4f}{' *' if improved else ''}")
                if cfg.early_stopping_patience and self.evals_since_best >= cfg.early_stopping_patience:
                    status = "early_stopped"
                    break
            if self.step % cfg.checkpoint_interval == 0:
                checkpoints["last"] = save_checkpoint(self.ckpt_dir / "last.pt", self.state())
        wall = time.time() - t0
        checkpoints["last"] = save_checkpoint(self.ckpt_dir / "last.pt", self.state())
        # Final, full held-out evaluation uses the best-on-validation checkpoint (test is touched once).
        final: dict[str, Any] = {}
        best_path = self.ckpt_dir / "best.pt"
        if "best" not in checkpoints and best_path.exists():  # best saved before a resume
            checkpoints["best"] = json.loads(best_path.with_suffix(".sha256.json").read_text())
        if "best" in checkpoints:
            best_state = load_checkpoint(self.ckpt_dir / "best.pt")
            eval_model = build_model(cfg, self.task.tokenizer.vocab_size)
            eval_model.load_state_dict(best_state["model"])
            eval_model.to(self.device)
            final = self.task.final_eval(eval_model, self.device)
        summary = {
            "run_id": cfg.run_id,
            "experiment": cfg.experiment,
            "status": status,
            "failure": failure,
            "steps_completed": self.step,
            "skipped_steps": self.skipped_steps,
            "tokens_seen": self.tokens_seen,
            "wall_time_s": round(wall, 1),
            "mean_tokens_per_s": round(self.tokens_seen / max(wall, 1e-9), 1),
            "best_step": self.best_step,
            "best_" + self.task.primary_metric: self.best_metric if math.isfinite(self.best_metric) else None,
            "final_eval_on": "best checkpoint" if final else None,
            "final": final,
            "params": count_parameters(self.model),
            "checkpoints": checkpoints,
            "finished": now_iso(),
        }
        meta.update({"status": status, "finished": summary["finished"]})
        write_json(self.run_dir / "metadata.json", meta)
        write_json(self.run_dir / "summary.json", summary)
        self._say(f"{status} in {wall:.1f}s; best {self.task.primary_metric}={summary['best_' + self.task.primary_metric]}")
        return summary


def train_from_config(cfg: TrainConfig, quiet: bool = False, resume: Path | None = None,
                      runs_dir: Path = RUNS_DIR, ckpt_root: Path = CHECKPOINT_ROOT) -> dict[str, Any]:
    return Trainer(cfg, runs_dir=runs_dir, ckpt_root=ckpt_root, quiet=quiet).train(resume=resume)
