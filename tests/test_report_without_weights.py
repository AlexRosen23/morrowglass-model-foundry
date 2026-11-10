"""`make report` must work on a fresh clone, where no checkpoints exist (weights are not in git)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from conftest import tiny_cfg
from morrowglass.experiments.runner import LEAK_FREE_RECORD, last_position_bpb
from morrowglass.repro import REPO_ROOT, read_json
from morrowglass.train.trainer import train_from_config


def test_leak_free_measurement_is_recorded_and_reused_without_weights(tmp_path: Path) -> None:
    runs, ckpts = tmp_path / "r", tmp_path / "c"
    s = train_from_config(tiny_cfg(max_steps=10, experiment="unit"), quiet=True, runs_dir=runs, ckpt_root=ckpts)
    run_dir = runs / s["run_id"]
    first = last_position_bpb(run_dir, ckpts, n_positions=256)
    rec_path = run_dir / LEAK_FREE_RECORD
    rec = read_json(rec_path)
    assert rec["result"] == first and len(rec["checkpoint_sha256"]) == 64

    stamp = rec_path.stat().st_mtime_ns
    assert last_position_bpb(run_dir, ckpts, n_positions=256) == first  # deterministic on CPU
    assert rec_path.stat().st_mtime_ns == stamp  # same weights: the record is left alone

    shutil.rmtree(ckpts / s["run_id"])  # a fresh clone
    assert last_position_bpb(run_dir, ckpts, n_positions=256) == first
    with pytest.raises(FileNotFoundError, match="reproduce"):
        last_position_bpb(run_dir, ckpts, n_positions=512)  # a different measurement can't be served from the record


def test_committed_break_mask_runs_carry_their_leak_free_record() -> None:
    for exp in ("break-mask", "break-mask-long"):
        result = REPO_ROOT / "reports" / "experiments" / exp / "result.json"
        if not result.exists():
            pytest.skip("no committed analysis in this checkout")
        obs = read_json(result)["observations"]
        for arm in ("noncausal", "causal"):
            rid = next(r for r in read_json(result)["arms"] if f"{exp}-{arm}-" in r)
            rec = read_json(REPO_ROOT / "reports" / "runs" / rid / LEAK_FREE_RECORD)
            assert rec["result"]["bits_per_byte"] == pytest.approx(obs[f"{arm}_leak_free_bpb"], rel=1e-9)
            assert rec["result"]["positions"] == obs["leak_free_positions"]
