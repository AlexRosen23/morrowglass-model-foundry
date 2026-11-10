"""Seed follow-ups are judged by their registered criteria, from records only (no training here)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from morrowglass.experiments import analyze as an
from morrowglass.experiments.registry import FLAGSHIP_RUN, build
from morrowglass.repro import write_json


def _record(root: Path, run_id: str, seed: int, final: dict[str, Any], train: dict[str, Any],
            test: dict[str, Any] | None = None) -> None:
    d = root / run_id
    write_json(d / "summary.json", {"run_id": run_id, "status": "completed", "params": 1, "steps_completed": 1,
                                    "tokens_seen": 1, "wall_time_s": 1.0, "best_step": 1, "final": final})
    write_json(d / "config.json", {"train": {"seed": seed, "tokenizer": "byte", **train}, "model": {}})
    write_json(d / "metadata.json", {"git": {"commit": "0" * 40, "dirty": False}, "device": "cpu"})
    if test is not None:
        write_json(d / "test_eval.json", {"run_id": run_id, "results": test})


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    runs, reports = tmp_path / "runs", tmp_path / "reports"
    monkeypatch.setattr(an, "runs_dir_for", lambda profile: runs)
    monkeypatch.setattr(an, "experiments_dir_for", lambda profile: reports / "experiments")
    monkeypatch.setattr(an, "REPORTS", reports)
    return runs, reports


def _digitsum(runs: Path, far: dict[int, tuple[float, float]]) -> None:
    arms = build("digitsum-scratchpad").arms + build("digitsum-scratchpad-seeds").arms
    for a in arms:
        acc = far[a.seed][1 if a.pos_augment else 0]
        _record(runs, a.run_id, a.seed, {"val_iid": {"accuracy": 1.0}}, {"pos_augment": a.pos_augment},
                {"test_ood_near": {"accuracy": acc, "n": 2000}, "test_ood_far": {"accuracy": acc, "n": 2000}})


def test_digitsum_seeds_pending_until_ood_opened(dirs: tuple[Path, Path]) -> None:
    runs, _ = dirs
    for a in build("digitsum-scratchpad").arms + build("digitsum-scratchpad-seeds").arms:
        _record(runs, a.run_id, a.seed, {"val_iid": {"accuracy": 1.0}}, {"pos_augment": a.pos_augment})
    assert an.analyze("digitsum-scratchpad-seeds")["verdict"] == "pending"


def test_digitsum_seeds_counts_transfers_beyond_noise(dirs: tuple[Path, Path]) -> None:
    runs, _ = dirs
    # Seed 0's +0.2-point gain is noise (3 SE is about 3 points), seeds 1, 3, 4 transfer clearly.
    _digitsum(runs, {0: (0.103, 0.105), 1: (0.098, 0.887), 2: (0.10, 0.11), 3: (0.10, 0.60), 4: (0.10, 0.95)})
    r = an.analyze("digitsum-scratchpad-seeds")
    assert r["observations"]["transfer_count"] == 3 and r["verdict"] == "supported"
    assert not r["observations"]["transfer"][0]["transfers"]


def test_digitsum_seeds_minority_transfer_is_partial(dirs: tuple[Path, Path]) -> None:
    runs, _ = dirs
    _digitsum(runs, {0: (0.103, 0.105), 1: (0.098, 0.887), 2: (0.10, 0.11), 3: (0.10, 0.10), 4: (0.10, 0.12)})
    r = an.analyze("digitsum-scratchpad-seeds")
    assert r["observations"]["transfer_count"] == 1 and r["verdict"] == "partially supported"


def _scratch(runs: Path, reports: Path, vals: dict[int, float], uppers: dict[int, float]) -> None:
    ids = {0: FLAGSHIP_RUN, **{a.seed: a.run_id for a in build("scratch-seeds").arms}}
    for seed, rid in ids.items():
        _record(runs, rid, seed, {"val": {"bits_per_byte": vals[seed], "bits_per_byte_ci95": [vals[seed] - .012,
                                                                                                vals[seed] + .012]}},
                {"tokenizer": "byte"})
    write_json(reports / "baselines-shakespeare.json", {"best_val_bpb": 2.107, "best_val_bpb_ci95": [2.095, 2.120]})
    write_json(reports / "paired" / "scratch-seeds.json", {
        "protocol": {"primary_block_bytes": 8192},
        "runs": {rid: {"seed": s, "matches_record": True,
                       "paired": {"8192": {"difference": uppers[s] - .01, "ci95": [uppers[s] - .02, uppers[s]]}}}
                 for s, rid in ids.items()}})


def test_scratch_seeds_supported_when_every_seed_is_below(dirs: tuple[Path, Path]) -> None:
    runs, reports = dirs
    _scratch(runs, reports, {0: 2.086, 1: 2.090, 2: 2.081}, {0: -0.011, 1: -0.004, 2: -0.015})
    r = an.analyze("scratch-seeds")
    assert r["verdict"] == "supported" and r["observations"]["seed_range"] == pytest.approx(0.009)


def test_scratch_seeds_one_seed_not_separated(dirs: tuple[Path, Path]) -> None:
    runs, reports = dirs
    _scratch(runs, reports, {0: 2.086, 1: 2.095, 2: 2.081}, {0: -0.011, 1: 0.004, 2: -0.015})
    assert an.analyze("scratch-seeds")["verdict"] == "partially supported"  # spread ok, seed 1 crosses 0
    _scratch(runs, reports, {0: 2.086, 1: 2.104, 2: 2.081}, {0: -0.011, 1: 0.004, 2: -0.015})
    assert an.analyze("scratch-seeds")["verdict"] == "not supported"  # and the spread is 0.023


def test_scratch_seeds_pending_without_paired_scores(dirs: tuple[Path, Path]) -> None:
    runs, reports = dirs
    _scratch(runs, reports, {0: 2.086, 1: 2.090, 2: 2.081}, {0: -0.011, 1: -0.004, 2: -0.015})
    (reports / "paired" / "scratch-seeds.json").unlink()
    assert an.analyze("scratch-seeds")["verdict"] == "pending"
