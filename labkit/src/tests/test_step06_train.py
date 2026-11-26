"""Step 6: learning-rate schedule, optimizer groups, the training step, and a short training run."""

from __future__ import annotations

import json
import math

import pytest
import torch

from scratchlm.checkpoint import load_checkpoint
from scratchlm.data import sample_batch
from scratchlm.model import ModelConfig, count_parameters
from scratchlm.train import TrainConfig, lr_at, make_optimizer, train, train_step
from tests.conftest import pattern_ids, tiny_model


def test_step06_warmup_then_cosine_down_to_the_floor() -> None:
    cfg = TrainConfig(steps=110, lr=1.0, warmup_steps=10, min_lr_ratio=0.1)
    assert math.isclose(lr_at(0, cfg), 0.1) and math.isclose(lr_at(4, cfg), 0.5)
    assert math.isclose(lr_at(9, cfg), 1.0) and math.isclose(lr_at(10, cfg), 1.0)
    assert math.isclose(lr_at(60, cfg), 0.55)  # half-way through the decay: half-way to the floor
    assert math.isclose(lr_at(110, cfg), 0.1) and math.isclose(lr_at(500, cfg), 0.1)
    lrs = [lr_at(s, cfg) for s in range(10, 111)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:], strict=False))
    assert math.isclose(lr_at(0, TrainConfig(warmup_steps=0, lr=2.0)), 2.0)


def test_step06_weight_decay_only_on_matrices_and_every_parameter_once() -> None:
    m = tiny_model()
    opt = make_optimizer(m, TrainConfig(weight_decay=0.2, lr=1e-3, beta1=0.8, beta2=0.9))
    assert isinstance(opt, torch.optim.AdamW)
    decay = [g for g in opt.param_groups if g["weight_decay"] > 0]
    none = [g for g in opt.param_groups if g["weight_decay"] == 0]
    assert len(decay) == 1 and len(none) == 1 and decay[0]["weight_decay"] == 0.2
    assert all(p.dim() >= 2 for p in decay[0]["params"])
    assert all(p.dim() < 2 for p in none[0]["params"])
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids))
    assert sum(p.numel() for g in opt.param_groups for p in g["params"]) == count_parameters(m)
    assert any(p is m.tok_emb.weight for p in decay[0]["params"])
    assert opt.defaults["betas"] == (0.8, 0.9)


def test_step06_training_steps_learn_a_pattern() -> None:
    m = tiny_model(seed=0)
    opt = make_optimizer(m, TrainConfig(lr=3e-3))
    ids = pattern_ids()
    gen = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(80):  # about a second on a laptop CPU
        x, y = sample_batch(ids, 16, 16, gen)
        loss, gnorm = train_step(m, opt, x, y, grad_clip=1.0)
        assert math.isfinite(gnorm) and gnorm >= 0
        losses.append(loss)
    assert losses[0] > 5.0
    assert sum(losses[-10:]) / 10 < 0.4 * losses[0]


def test_step06_a_non_finite_loss_raises_and_changes_nothing() -> None:
    m = tiny_model()
    opt = make_optimizer(m, TrainConfig())
    with torch.no_grad():
        m.ln_f.weight[0] = float("nan")
    before = {k: v.clone() for k, v in m.state_dict().items()}
    x, y = sample_batch(pattern_ids(), 16, 4, torch.Generator().manual_seed(0))
    with pytest.raises(FloatingPointError):
        train_step(m, opt, x, y)
    for k, v in m.state_dict().items():
        assert torch.equal(v.nan_to_num(), before[k].nan_to_num())


def test_step06_a_short_run_beats_the_uniform_baseline(tmp_path) -> None:
    text = pattern_ids()
    mcfg = ModelConfig(block_size=16, n_layer=1, n_head=2, d_model=32)
    tcfg = TrainConfig(steps=60, batch_size=16, warmup_steps=5, eval_interval=20, eval_windows=8, threads=2)
    model, summary = train(mcfg, tcfg, text[:800], text[800:], out_dir=tmp_path, log=lambda _: None)
    assert summary["status"] == "completed" and summary["steps"] == 60
    assert [h["step"] for h in summary["history"]] == [20, 40, 60]
    assert summary["best_val_bits_per_byte"] < 3.0  # uniform guessing is 8
    loaded, extra = load_checkpoint(tmp_path / "best.pt")
    assert extra["step"] == summary["best_step"]
    assert json.loads((tmp_path / "summary.json").read_text())["params"] == count_parameters(model)
    assert (tmp_path / "last.pt").exists()
