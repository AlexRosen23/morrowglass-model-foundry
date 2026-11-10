"""Committed evidence cannot be silently replaced, and serving never uses stale or missing inputs."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import tiny_cfg
from morrowglass.api import app as app_module
from morrowglass.api.store import Store
from morrowglass.data.corpus import same_manifest
from morrowglass.experiments import runner
from morrowglass.experiments.registry import build
from morrowglass.repro import read_json, write_json
from morrowglass.train.trainer import train_from_config


def test_force_refuses_to_overwrite_a_committed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    arm = build("scratch", "standard").arms[0]
    write_json(tmp_path / arm.run_id / "summary.json", {"run_id": arm.run_id, "status": "completed"})
    monkeypatch.setattr(runner, "runs_dir_for", lambda profile: tmp_path)
    monkeypatch.setattr(runner, "train_from_config", lambda *a, **k: pytest.fail("must not train"))
    with pytest.raises(runner.EvidenceOverwriteError, match="morrowglass reproduce"):
        runner.run_experiment("scratch", "standard", force=True, quiet=True)


def test_reproduce_leaves_the_committed_record_untouched(tmp_path: Path) -> None:
    committed, ck = tmp_path / "runs", tmp_path / "ckpt"
    s = train_from_config(tiny_cfg(max_steps=20, experiment="unit"), quiet=True, runs_dir=committed, ckpt_root=ck)
    rid = s["run_id"]
    before = {p.name: p.read_bytes() for p in (committed / rid).iterdir()}
    out = tmp_path / "reproduce"
    r = runner.reproduce_run(rid, device="cpu", quiet=True, runs_dir=committed, out_dir=out,
                             ckpt_root=tmp_path / "repro-ckpt")
    assert {p.name: p.read_bytes() for p in (committed / rid).iterdir()} == before
    assert (out / "runs" / rid / "summary.json").exists() and read_json(out / f"{rid}.json") == r
    # Same config, same seed, on CPU: the re-run lands on the committed number.
    assert r["verdict"] == "reproduced" and abs(r["val_bpb"]["difference"]) < 1e-6


def test_reproduce_unknown_run(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        runner.reproduce_run("nope-00000000-s0", runs_dir=tmp_path)


def test_manifest_comparison_ignores_only_the_timestamp() -> None:
    a = {"created": "2026-11-09T04:01:28+00:00", "files": {"val": {"sha256": "aa"}}}
    assert same_manifest(a, {**a, "created": "2026-11-10T00:00:00+00:00"})
    assert not same_manifest(a, {**a, "files": {"val": {"sha256": "bb"}}})


@pytest.fixture
def store_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Store, Path, str]:
    runs, ck = tmp_path / "runs", tmp_path / "ckpt"
    s = train_from_config(tiny_cfg(max_steps=10, experiment="unit"), quiet=True, runs_dir=runs, ckpt_root=ck)
    st = Store(sources={"measured": (runs, ck), "smoke": (tmp_path / "none", tmp_path / "none")},
               index_path=tmp_path / "idx.sqlite", experiments_dir=tmp_path / "exp")
    monkeypatch.setattr(app_module, "store", st)
    return TestClient(app_module.app), st, ck, s["run_id"]


def test_retrained_checkpoint_is_reloaded(store_client: tuple[TestClient, Store, Path, str]) -> None:
    _, st, ck, rid = store_client
    first, _ = st.model(rid)
    assert st.model(rid)[0] is first  # cached while the file is unchanged
    p = ck / rid / "best.pt"
    os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 1_000_000_000))  # as after a re-train
    assert st.model(rid)[0] is not first
    assert sum(k[0] == rid for k in st._models) == 1  # the stale entry is dropped, not kept beside it


def test_baseline_without_corpus_is_a_clear_409(store_client: tuple[TestClient, Store, Path, str],
                                                monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client, _, _, rid = store_client
    import morrowglass.data.corpus as corpus

    monkeypatch.setattr(corpus, "SAMPLE_DIR", tmp_path / "missing")  # the tiny run trains on the sample
    body = {"run_id": rid, "prompt": "KING:", "max_new_tokens": 8}
    r = client.post("/api/baseline-generate", json=body)
    assert r.status_code == 409 and "not on this machine" in r.json()["detail"]
    g = client.post("/api/generate", json={**body, "with_baseline": True})
    assert g.status_code == 200 and g.json()["baseline"]["model"] == "baseline unavailable"


def test_runs_flag_non_causal_models(store_client: tuple[TestClient, Store, Path, str]) -> None:
    client, _, _, rid = store_client
    row = next(r for r in client.get("/api/runs").json()["runs"] if r["run_id"] == rid)
    assert row["causal"] is True


def test_reproduced_checkpoint_is_installed_and_labelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    committed, ck = tmp_path / "runs", tmp_path / "ckpt"
    rid = train_from_config(tiny_cfg(max_steps=20, experiment="unit"), quiet=True, runs_dir=committed,
                            ckpt_root=ck)["run_id"]
    original = (ck / rid / "best.pt").read_bytes()
    # A checkpoint that already exists (the original) is never replaced.
    r = runner.reproduce_run(rid, device="cpu", quiet=True, runs_dir=committed, out_dir=tmp_path / "rep",
                             ckpt_root=tmp_path / "rep-ckpt", install_into=ck)
    assert r["installed_checkpoint"] is None and (ck / rid / "best.pt").read_bytes() == original
    # A fresh clone (no weights) gets the re-trained checkpoint, with a marker the API reports.
    (ck / rid / "best.pt").unlink()
    r = runner.reproduce_run(rid, device="cpu", quiet=True, runs_dir=committed, out_dir=tmp_path / "rep2",
                             ckpt_root=tmp_path / "rep2-ckpt", install_into=ck)
    assert r["installed_checkpoint"] and read_json(ck / rid / "REPRODUCED.json")["origin"] == "reproduced"
    st = Store(sources={"measured": (committed, ck), "smoke": (tmp_path / "none", tmp_path / "none")},
               index_path=tmp_path / "idx.sqlite", experiments_dir=tmp_path / "exp")
    monkeypatch.setattr(app_module, "store", st)
    client = TestClient(app_module.app)
    g = client.post("/api/generate", json={"run_id": rid, "prompt": "KING:", "max_new_tokens": 4}).json()
    assert g["checkpoint_origin"] == "reproduced" and "re-trained" in g["label"]
    row = next(x for x in client.get("/api/runs").json()["runs"] if x["run_id"] == rid)
    assert row["checkpoint_reproduced"] is True
