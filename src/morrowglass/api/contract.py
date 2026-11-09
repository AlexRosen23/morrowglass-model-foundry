"""The project's read-only contract for other local apps (e.g. the owner's Constellation command center).

GET /health, /capabilities, /runs, /runs/{run_id}, at the root, loopback only (the API binds 127.0.0.1).
Every answer carries ``project_id: "morrowglass"``. Runs are the measured training runs in
``reports/runs`` (recorded evidence, ``mode: "recorded"``), newest first; smoke runs are left out.
Nothing here starts, changes or trains anything.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from morrowglass import __version__
from morrowglass.api.store import Store

PROJECT_ID = "morrowglass"
SCHEMA_VERSION = 1


def _headline(r: dict[str, Any]) -> str:
    params = f"{r['params'] / 1e6:.2f}M params"
    steps = f"{r['steps']:,} steps"
    if r["task"] == "lm" and r["val_bpb"] is not None:
        return f"val {r['val_bpb']:.3f} bits/byte · {r['tokenizer']} · {params} · {steps}"
    if r["val_iid_acc"] is not None:
        return f"in-distribution accuracy {r['val_iid_acc']:.3f} · {params} · {steps}"
    return f"{params} · {steps} · no final evaluation ({r['status']})"


def contract_run(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": r["run_id"],
        "project_id": PROJECT_ID,
        "kind": "training",
        "experiment": r["experiment"],
        "status": r["status"],
        "mode": "recorded",
        "started_at": r["started"],
        "ended_at": r["finished"],
        "headline": _headline(r),
        "test_split_opened": r["test_opened"],
        "git_commit": r["git_commit"],
    }


def build_router(get_store: Any) -> APIRouter:
    router = APIRouter()

    def measured() -> list[dict[str, Any]]:
        st: Store = get_store()
        rows = [r for r in st.list_runs() if r["provenance"] == "measured"]
        return sorted(rows, key=lambda r: r["finished"] or r["started"], reverse=True)

    @router.get("/health")
    def health() -> dict[str, Any]:
        rows = measured()
        last = rows[0] if rows else None
        return {
            "ok": True, "status": "ok", "app": PROJECT_ID, "project_id": PROJECT_ID, "version": __version__,
            "mode": "recorded", "runs": len(rows),
            "last_run": {"run_id": last["run_id"], "status": last["status"]} if last else None,
        }

    @router.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "project_id": PROJECT_ID,
            "name": "Morrowglass Model Foundry",
            "version": __version__,
            "active_mode": "recorded",
            "dataset": {"name": "shakespeare (Project Gutenberg #100)", "license": "public domain", "synthetic": False},
            "implemented": [
                "decoder-only Transformer written in PyTorch (causal mask unit-tested)",
                "byte and BPE-1024 tokenizers, fitted on train only",
                "Kneser-Ney n-gram baselines, order chosen on validation",
                "bits-per-byte with bootstrap 95% intervals and calibration (ECE)",
                "pre-registered experiments with computed verdicts",
                "locked test split, every opening logged",
                "sha256-verified checkpoints, bit-exact CPU resume",
                "pretrained distilgpt2 + LoRA comparison",
                "live sampling and attention maps from saved checkpoints",
            ],
            "not_implemented": [
                "multi-seed main-track arms (one seed each; three in the optimization study)",
                "RoPE or relative positions",
                "full fine-tuning comparison for LoRA",
            ],
            "disabled_by_design": [
                "training from the web UI beyond a bounded smoke job",
                "CUDA (Apple MPS or CPU only)",
                "public deployment",
            ],
            "endpoints": ["/health", "/capabilities", "/runs", "/runs/{run_id}"],
        }

    @router.get("/runs")
    def runs(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
        rows = measured()
        return {"schema_version": SCHEMA_VERSION, "project_id": PROJECT_ID, "total": len(rows),
                "runs": [contract_run(r) for r in rows[:limit]]}

    @router.get("/runs/{run_id}")
    def run(run_id: str) -> dict[str, Any]:
        match = [r for r in measured() if r["run_id"] == run_id]
        if not match:
            raise HTTPException(404, "no such measured run")
        st: Store = get_store()
        detail = st.run(run_id)
        return {"project_id": PROJECT_ID, **contract_run(match[0]),
                "summary": detail["summary"], "config": detail["config"]}

    return router
