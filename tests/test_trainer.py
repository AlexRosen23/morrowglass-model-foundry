"""Training-loop guarantees: overfit sanity, checkpoint integrity, resume equivalence, failure handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from conftest import tiny_cfg
from morrowglass.experiments.runner import overfit_sanity
from morrowglass.repro import IntegrityError, write_verified
from morrowglass.train.trainer import (
    TrainConfig,
    Trainer,
    load_checkpoint,
    load_model_from_checkpoint,
    lr_at,
    train_from_config,
)


def test_overfit_one_batch_sanity() -> None:
    r = overfit_sanity(steps=150)
    assert r["passed"], r
    assert r["final_loss"] < r["initial_loss"]


def test_training_reduces_held_out_loss(run_dirs: tuple[Path, Path]) -> None:
    s = train_from_config(tiny_cfg(max_steps=60, eval_interval=20), quiet=True, runs_dir=run_dirs[0],
                          ckpt_root=run_dirs[1])
    evals = [json.loads(ln) for ln in (run_dirs[0] / s["run_id"] / "metrics.jsonl").read_text().splitlines()
             if '"eval"' in ln]
    assert s["status"] == "completed"
    assert evals[-1]["val_bits_per_byte"] < evals[0]["val_bits_per_byte"] < 8.0
    assert "test" not in s["final"], "test split must stay locked during training"
    for name in ("config.json", "metadata.json", "summary.json", "metrics.jsonl"):
        assert (run_dirs[0] / s["run_id"] / name).exists()
    meta = json.loads((run_dirs[0] / s["run_id"] / "metadata.json").read_text())
    for k in ("seed", "git", "config_hash", "data", "hardware"):
        assert k in meta
    assert meta["data"]["dataset_hash"]


def test_checkpoint_roundtrip_identical_logits(run_dirs: tuple[Path, Path]) -> None:
    tr = Trainer(tiny_cfg(max_steps=10), runs_dir=run_dirs[0], ckpt_root=run_dirs[1], quiet=True)
    s = tr.train()
    model, _ = load_model_from_checkpoint(run_dirs[1] / s["run_id"] / "last.pt")
    x = torch.randint(0, 256, (2, 32))
    tr.model.eval()
    torch.testing.assert_close(tr.model(x).logits, model(x).logits, rtol=0, atol=0)


def test_corrupted_checkpoint_is_rejected(run_dirs: tuple[Path, Path]) -> None:
    s = train_from_config(tiny_cfg(max_steps=10), quiet=True, runs_dir=run_dirs[0], ckpt_root=run_dirs[1])
    ck = run_dirs[1] / s["run_id"] / "last.pt"
    data = bytearray(ck.read_bytes())
    data[len(data) // 2] ^= 0x01  # a single flipped bit, as a faulty RAM write would produce
    ck.write_bytes(bytes(data))
    with pytest.raises(IntegrityError):
        load_checkpoint(ck)


def test_write_verified_roundtrip(tmp_path: Path) -> None:
    sha = write_verified(tmp_path / "a.bin", b"hello")
    assert (tmp_path / "a.bin").read_bytes() == b"hello" and len(sha) == 64
    assert not (tmp_path / "a.bin.tmp").exists()


def test_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    """Train 20 steps straight vs 10 + resume to 20: parameters must be bit-identical on CPU.

    A constant schedule makes the 10-step run's learning rates identical to the first 10
    steps of the 20-step run (a cosine schedule depends on max_steps).
    """
    cfg = tiny_cfg(max_steps=20, checkpoint_interval=10, eval_interval=100, schedule="constant", warmup_steps=0)
    a = Trainer(cfg, runs_dir=tmp_path / "a", ckpt_root=tmp_path / "ca", quiet=True)
    a.train()
    half = TrainConfig.from_dict({**cfg.to_dict(), "max_steps": 10})
    Trainer(half, runs_dir=tmp_path / "b", ckpt_root=tmp_path / "cb", quiet=True).train()
    b = Trainer(cfg, runs_dir=tmp_path / "b2", ckpt_root=tmp_path / "cb2", quiet=True)
    b.train(resume=tmp_path / "cb" / half.run_id / "last.pt")
    assert b.step == 20
    for (n, pa), (_, pb) in zip(a.model.named_parameters(), b.model.named_parameters(), strict=True):
        torch.testing.assert_close(pa, pb, rtol=0, atol=0, msg=n)


def test_divergence_is_detected_not_crashed(run_dirs: tuple[Path, Path]) -> None:
    s = train_from_config(tiny_cfg(lr=1e6, warmup_steps=0, grad_clip=0.0, schedule="constant", max_steps=50,
                                   nan_policy="skip", max_bad_steps=3),
                          quiet=True, runs_dir=run_dirs[0], ckpt_root=run_dirs[1])
    assert s["status"] == "diverged"
    assert s["failure"]["reason"].startswith("non-finite")


def test_early_stopping_triggers(run_dirs: tuple[Path, Path]) -> None:
    # lr=0 means no improvement after the first eval, so patience 2 stops at the 3rd eval.
    s = train_from_config(tiny_cfg(lr=0.0, warmup_steps=0, max_steps=200, eval_interval=5,
                                   early_stopping_patience=2), quiet=True, runs_dir=run_dirs[0], ckpt_root=run_dirs[1])
    assert s["status"] == "early_stopped" and s["steps_completed"] == 15


def test_cancellation_via_stop_file(run_dirs: tuple[Path, Path]) -> None:
    tr = Trainer(tiny_cfg(max_steps=10_000), runs_dir=run_dirs[0], ckpt_root=run_dirs[1], quiet=True)
    orig = tr.task.batch
    calls = {"n": 0}

    def batch(*a: object, **k: object):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 5:
            (tr.run_dir / "STOP").touch()
        return orig(*a, **k)  # type: ignore[arg-type]

    tr.task.batch = batch  # type: ignore[method-assign]
    s = tr.train()
    assert s["status"] == "cancelled" and s["steps_completed"] < 10


def test_lr_schedule_shapes() -> None:
    c = tiny_cfg(lr=1.0, warmup_steps=10, max_steps=110, min_lr_ratio=0.1)
    assert lr_at(0, c) == pytest.approx(0.1)
    assert lr_at(9, c) == pytest.approx(1.0)
    assert lr_at(110, c) == pytest.approx(0.1)
    assert lr_at(60, c) == pytest.approx(0.55)
    const = tiny_cfg(lr=0.5, warmup_steps=0, schedule="constant")
    assert lr_at(1234, const) == 0.5


def test_run_id_ignores_device_but_not_hyperparameters() -> None:
    a, b = tiny_cfg(device="cpu"), tiny_cfg(device="mps")
    assert a.run_id == b.run_id
    assert tiny_cfg(lr=1e-3).run_id != tiny_cfg(lr=2e-3).run_id
    assert tiny_cfg(experiment="x").run_id == tiny_cfg(experiment="y").run_id
