"""API returns saved data with provenance labels, and generation works from a real checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import tiny_cfg
from morrowglass.api import app as app_module
from morrowglass.api.store import Store
from morrowglass.contracts import export_artifact, validate_manifest
from morrowglass.repro import read_json, sha256_file
from morrowglass.train.trainer import train_from_config


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, str]:
    root = tmp_path_factory.mktemp("api")
    runs, ck = root / "runs", root / "ckpt"
    s = train_from_config(tiny_cfg(max_steps=20, experiment="unit"), quiet=True, runs_dir=runs, ckpt_root=ck)
    return runs, ck, s["run_id"]


@pytest.fixture
def client(trained: tuple[Path, Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    runs, ck, _ = trained
    st = Store(sources={"measured": (runs, ck), "smoke": (tmp_path / "none", tmp_path / "none")},
               index_path=tmp_path / "idx.sqlite", experiments_dir=tmp_path / "exp")
    monkeypatch.setattr(app_module, "store", st)
    return TestClient(app_module.app)


def test_runs_list_has_provenance(client: TestClient, trained: tuple[Path, Path, str]) -> None:
    r = client.get("/api/runs").json()
    assert r["runs"] and r["runs"][0]["run_id"] == trained[2]
    assert r["runs"][0]["provenance"] == "measured"
    assert r["runs"][0]["has_checkpoint"] is True
    assert "summary.json" in r["source"]


def test_run_detail_and_metrics_match_files(client: TestClient, trained: tuple[Path, Path, str]) -> None:
    runs, _, rid = trained
    d = client.get(f"/api/runs/{rid}").json()
    assert d["summary"] == read_json(runs / rid / "summary.json")
    m = client.get(f"/api/runs/{rid}/metrics").json()
    lines = (runs / rid / "metrics.jsonl").read_text().splitlines()
    assert len(m["metrics"]) == len(lines)


def test_blueprint_param_total_matches_run(client: TestClient, trained: tuple[Path, Path, str]) -> None:
    bp = client.get(f"/api/runs/{trained[2]}/blueprint").json()
    assert sum(r["params"] for r in bp["layers"]) == client.get(f"/api/runs/{trained[2]}").json()["summary"]["params"]


def test_unknown_run_and_path_traversal_404(client: TestClient) -> None:
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/..%2F..%2Fetc").status_code == 404


def test_generate_and_attention(client: TestClient, trained: tuple[Path, Path, str]) -> None:
    body = {"run_id": trained[2], "prompt": "KING:", "max_new_tokens": 12, "seed": 3, "with_baseline": False}
    a = client.post("/api/generate", json=body).json()
    b = client.post("/api/generate", json=body).json()
    assert a["text"] == b["text"] and len(a["tokens"]) == 12
    assert a["provenance"] == "measured" and "not calibrated" in a["note"]
    att = client.post("/api/attention", json={"run_id": trained[2], "text": "To be or not"}).json()
    assert len(att["attention"]) == 1 and "causal explanation" in att["caveat"]


def test_generate_rejects_oversized_requests(client: TestClient, trained: tuple[Path, Path, str]) -> None:
    r = client.post("/api/generate", json={"run_id": trained[2], "prompt": "a", "max_new_tokens": 5000})
    assert r.status_code == 422


def test_compare_reports_config_diff(client: TestClient, trained: tuple[Path, Path, str]) -> None:
    r = client.get(f"/api/compare?runs={trained[2]},{trained[2]}").json()
    assert r["config_diff"] == {} and len(r["runs"]) == 2


def test_only_smoke_jobs_allowed(client: TestClient) -> None:
    assert client.post("/api/jobs?kind=standard").status_code == 400


def test_ledger_index_rebuild(tmp_path: Path, trained: tuple[Path, Path, str]) -> None:
    runs, ck, rid = trained
    st = Store(sources={"measured": (runs, ck)}, index_path=tmp_path / "i.sqlite", experiments_dir=tmp_path)
    assert st.rebuild_index() == 1
    assert st.query_index("run_id = ?", (rid,))[0]["provenance"] == "measured"


def test_export_matches_schema(trained: tuple[Path, Path, str], tmp_path: Path) -> None:
    runs, ck, rid = trained
    out = export_artifact(rid, out_root=tmp_path, runs_dir=runs, ckpt_root=ck)
    manifest = read_json(tmp_path / rid / "model_artifact.json")
    assert validate_manifest(manifest) == []
    assert sha256_file(tmp_path / rid / manifest["artifact"]["path"]) == manifest["artifact"]["sha256"] == out["sha256"]
    broken = {**manifest, "artifact": {**manifest["artifact"], "sha256": "xyz"}}
    del broken["license"]
    errs = validate_manifest(broken)
    assert any("license" in e for e in errs) and any("sha256" in e for e in errs)
