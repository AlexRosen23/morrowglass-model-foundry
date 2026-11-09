from __future__ import annotations

from pathlib import Path

import pytest
import torch

from morrowglass.train.trainer import TrainConfig

torch.set_num_threads(2)


def tiny_cfg(**kw: object) -> TrainConfig:
    base = dict(name="t", task="lm", dataset="shakespeare-sample", tokenizer="byte",
                model={"n_layer": 1, "n_head": 2, "d_model": 32, "block_size": 32},
                batch_size=8, max_steps=20, lr=3e-3, warmup_steps=2, eval_interval=10, eval_windows=8,
                checkpoint_interval=10, log_interval=5, device="cpu", num_threads=2)
    base.update(kw)
    return TrainConfig(**base)  # type: ignore[arg-type]


@pytest.fixture
def run_dirs(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "runs", tmp_path / "ckpt"
