"""FastAPI backend for the Morrowglass workbench (port 5100).

Serves only saved/observed data. Each response that carries numbers names its source
file and whether it is a ``measured`` run or a ``smoke`` run. Training is never run
inside a request handler: the only job type is the bounded smoke run, executed in a
subprocess with progress streaming and cancellation.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from morrowglass.api.contract import build_router
from morrowglass.api.store import Store
from morrowglass.data.corpus import dataset_dir
from morrowglass.hardware import hardware_report
from morrowglass.inference.generate import attention_maps, generate
from morrowglass.model.transformer import ModelConfig, blueprint, compute_budget
from morrowglass.repro import REPO_ROOT
from morrowglass.serving.backends import HF_ALLOWED, Backend, HFBackend, MorrowglassBackend
from morrowglass.serving.openai_api import APIError, build_openai_router

torch.set_num_threads(2)
store = Store()


@asynccontextmanager
async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
    try:
        store.rebuild_index()
    except Exception as e:  # pragma: no cover - index is optional
        print(f"ledger index not rebuilt: {e}", file=sys.stderr)
    if (os.environ.get("MORROWGLASS_WARM_BASELINE", "1") == "1" and store.baselines()
            and (dataset_dir("shakespeare") / "train.txt").exists()):
        # Fit the demo's n-gram baseline in the background so the first Generate is not blocked.
        # (A fresh clone has no full corpus until `make data`; then the baseline is reported as unavailable.)
        threading.Thread(target=lambda: store.baseline_model("shakespeare", "byte"), daemon=True).start()
    yield


app = FastAPI(title="Morrowglass Model Foundry API", version="0.1.0", lifespan=lifespan)
# The read-only contract for other local apps (GET /health, /capabilities, /runs, /runs/{id}), at the root.
app.include_router(build_router(lambda: store))

# ------------------------------------------------------------------ OpenAI-compatible serving
# MORROWGLASS_DEFAULT_MODEL: model used when a request names none (needed by the harness's remote tokenizer).
# MORROWGLASS_ENABLE_HF=1: also serve the pretrained reference (hf:distilgpt2) from the local HF cache only.
_hf_backends: dict[str, Backend] = {}
_backends: OrderedDict[tuple[int, str, int | None], Backend] = OrderedDict()  # one backend (and lock) per served model
_hf_lock = threading.Lock()


def resolve_backend(model: str | None) -> Backend:
    model = model or os.environ.get("MORROWGLASS_DEFAULT_MODEL")
    if not model:
        raise APIError(400, "no model given and the server has no default (start it with --model RUN_ID)", "model")
    if model in HF_ALLOWED:
        if os.environ.get("MORROWGLASS_ENABLE_HF") != "1":
            raise APIError(404, f"{model} is disabled on this server (MORROWGLASS_ENABLE_HF=1 enables it)",
                           "model", "model_not_found")
        with _hf_lock:
            if model not in _hf_backends:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never download from a request handler
                try:
                    _hf_backends[model] = HFBackend(model)
                except OSError as e:
                    raise APIError(404, f"{model} is not in the local Hugging Face cache", "model") from e
            return _hf_backends[model]
    try:
        stamp = store.checkpoint_stamp(model)
    except KeyError as e:
        raise APIError(404, f"unknown model {model!r}", "model", "model_not_found") from e
    key = (id(store), model, stamp)  # a re-trained checkpoint gets a fresh backend
    with _hf_lock:
        if key in _backends:
            _backends.move_to_end(key)
            return _backends[key]
        for stale in [k for k in _backends if k[:2] == key[:2]]:
            del _backends[stale]
    try:
        run = store.run(model)
        if run["config"]["train"]["task"] != "lm":
            raise APIError(400, f"{model} is not a language model run", "model")
        net, tok = store.model(model)
    except KeyError as e:
        raise APIError(404, f"unknown model {model!r}", "model", "model_not_found") from e
    except FileNotFoundError as e:
        raise APIError(404, str(e), "model", "model_not_found") from e
    with _hf_lock:
        prov = run["provenance"] if run["checkpoint_origin"] != "reproduced" else \
            f"{run['provenance']} record; served weights re-trained by `morrowglass reproduce`"
        b = _backends.setdefault(key, MorrowglassBackend(model, net, tok, prov))
        while len(_backends) > 4:
            _backends.popitem(last=False)
    return b


def list_models() -> list[dict[str, Any]]:
    out = [{"id": r["run_id"], "object": "model", "created": 0, "owned_by": "morrowglass",
            "morrowglass": {"provenance": r["provenance"], "tokenizer": r["tokenizer"], "params": r["params"],
                            "val_bpb": r["val_bpb"], "instruction_tuned": False}}
           for r in store.list_runs() if r["task"] == "lm" and r["has_checkpoint"]]
    if os.environ.get("MORROWGLASS_ENABLE_HF") == "1":
        out += [{"id": m, "object": "model", "created": 0, "owned_by": "huggingface (reference, local cache)",
                 "morrowglass": {"provenance": "pretrained reference", "instruction_tuned": False}}
                for m in HF_ALLOWED]
    return out


app.include_router(build_openai_router(resolve_backend, list_models))
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5101", "http://127.0.0.1:5101"],
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])


def _404(e: Exception) -> HTTPException:
    return HTTPException(404, str(e))


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "runs": len(store.list_runs())}


@app.get("/api/hardware")
def hardware() -> dict[str, Any]:
    return hardware_report()


@app.get("/api/runs")
def runs(provenance: str | None = None, experiment: str | None = None) -> dict[str, Any]:
    rows = store.list_runs()
    if provenance:
        rows = [r for r in rows if r["provenance"] == provenance]
    if experiment:
        rows = [r for r in rows if r["experiment"] == experiment]
    return {"runs": rows, "source": "reports/runs/*/summary.json (+ reports/smoke/runs for smoke)"}


@app.get("/api/runs/{run_id}")
def run(run_id: str) -> dict[str, Any]:
    try:
        return store.run(run_id)
    except KeyError as e:
        raise _404(e) from e


@app.get("/api/runs/{run_id}/metrics")
def run_metrics(run_id: str) -> dict[str, Any]:
    try:
        kind = store.run(run_id)["provenance"]
        return {"run_id": run_id, "provenance": kind, "metrics": store.metrics(run_id),
                "source": f"reports/{'smoke/' if kind == 'smoke' else ''}runs/{run_id}/metrics.jsonl"}
    except KeyError as e:
        raise _404(e) from e


@app.get("/api/runs/{run_id}/blueprint")
def run_blueprint(run_id: str) -> dict[str, Any]:
    try:
        rec = store.run(run_id)
    except KeyError as e:
        raise _404(e) from e
    cfg = ModelConfig.from_dict(rec["config"]["model"])
    tokens = rec["summary"]["tokens_seen"]
    return {"run_id": run_id, "config": cfg.to_dict(), "layers": blueprint(cfg), "budget": compute_budget(cfg, tokens)}


@app.get("/api/compare")
def compare(runs: str) -> dict[str, Any]:
    ids = [r for r in runs.split(",") if r][:8]
    out = []
    for rid in ids:
        try:
            rec = store.run(rid)
        except KeyError as e:
            raise _404(e) from e
        evals = [m for m in store.metrics(rid) if m.get("kind") == "eval"]
        out.append({"run_id": rid, "provenance": rec["provenance"], "config": rec["config"]["train"],
                    "model": rec["config"]["model"], "evals": evals})
    keys = sorted({k for o in out for k in o["config"]} | {f"model.{k}" for o in out for k in o["model"]})

    def get(o: dict[str, Any], k: str) -> Any:
        return o["model"].get(k[6:]) if k.startswith("model.") else o["config"].get(k)

    diff = {k: [get(o, k) for o in out] for k in keys if len({json.dumps(get(o, k)) for o in out}) > 1}
    return {"runs": out, "config_diff": diff}


@app.get("/api/experiments")
def experiments() -> dict[str, Any]:
    return {"experiments": store.experiments(), "source": "reports/experiments/*/result.json"}


@app.get("/api/experiments/{name}")
def experiment(name: str) -> dict[str, Any]:
    try:
        return store.experiment(name)
    except KeyError as e:
        raise _404(e) from e


@app.get("/api/experiments/{name}/plots/{plot}")
def experiment_plot(name: str, plot: str) -> FileResponse:
    try:
        return FileResponse(store.plot_path(name, plot), media_type="image/png")
    except KeyError as e:
        raise _404(e) from e


@app.get("/api/baselines")
def baselines() -> dict[str, Any]:
    return {"baselines": store.baselines(), "source": "reports/baselines-shakespeare.json"}


@app.get("/api/dataset")
def dataset() -> dict[str, Any]:
    return {"manifest": store.dataset_manifest(), "source": "data/manifests/shakespeare.json",
            "card": "docs/DATASET_CARD.md"}


@app.get("/api/test-log")
def test_log() -> dict[str, Any]:
    return {"entries": store.test_log(), "source": "reports/test_access_log.jsonl"}


@app.get("/api/serving")
def serving() -> dict[str, Any]:
    """Measured serving/eval studies (registered in experiments/registry.py EVALUATIONS), straight from disk."""
    from morrowglass.experiments.registry import EVALUATIONS
    from morrowglass.repro import read_json

    rep = REPO_ROOT / "reports"

    def folder(sub: str) -> dict[str, Any]:
        d = rep / sub
        return {p.stem: read_json(p) for p in sorted(d.glob("*.json"))} if d.exists() else {}

    le = rep / "lm_eval" / "summary.json"
    hf = {p.parent.name: read_json(p) for p in sorted((rep / "hf_export").glob("*/morrowglass.json"))}
    return {
        "registered": {k: {"title": v.title, "hypothesis": v.hypothesis, "criterion": v.criterion,
                           "registered": v.registered} for k, v in EVALUATIONS.items()},
        "kv_cache": folder("serving/kv-cache"), "int8": folder("serving/int8"),
        "lm_eval": read_json(le) if le.exists() else None, "hf_export": hf,
        "sources": ["reports/serving/kv-cache/*.json", "reports/serving/int8/*.json", "reports/lm_eval/summary.json",
                    "reports/hf_export/*/morrowglass.json"],
    }


@app.get("/api/paired")
def paired() -> dict[str, Any]:
    """Paired re-analysis of the scratch models vs the best n-gram (EVALUATIONS['paired-baseline'])."""
    from morrowglass.repro import read_json

    p = REPO_ROOT / "reports" / "paired" / "summary.json"
    if not p.exists():
        raise HTTPException(404, "not measured yet: run `make paired`")
    return {**read_json(p), "source": "reports/paired/summary.json", "report": "docs/PAIRED_RESULTS.md"}


@app.get("/api/report", response_class=PlainTextResponse)
def report() -> str:
    p = REPO_ROOT / "docs" / "EXPERIMENT_REPORT.md"
    if not p.exists():
        raise HTTPException(404, "report not generated yet")
    return p.read_text()


@app.get("/api/report/download")
def report_download() -> FileResponse:
    p = REPO_ROOT / "reports" / "experiment_report.html"
    if not p.exists():
        raise HTTPException(404, "run `morrowglass report` first")
    return FileResponse(p, media_type="text/html", filename="morrowglass_experiment_report.html")


class GenerateRequest(BaseModel):
    run_id: str
    prompt: str = Field(max_length=500)
    max_new_tokens: int = Field(default=160, ge=1, le=400)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int | None = Field(default=40, ge=1, le=1024)
    seed: int = 0
    with_baseline: bool = False


@app.post("/api/generate")
def api_generate(req: GenerateRequest) -> dict[str, Any]:
    try:
        model, tok = store.model(req.run_id)
        rec = store.run(req.run_id)
    except KeyError as e:
        raise _404(e) from e
    except FileNotFoundError as e:
        raise HTTPException(409, str(e)) from e
    out = generate(model, tok, req.prompt, max_new_tokens=req.max_new_tokens, temperature=req.temperature,
                   top_k=req.top_k, seed=req.seed)
    out["run_id"] = req.run_id
    out["provenance"] = rec["provenance"]
    out["checkpoint_origin"] = rec["checkpoint_origin"]
    out["label"] = ("model output (sampled live from a saved checkpoint)" if rec["checkpoint_origin"] == "original"
                    else "model output (sampled live from a re-trained checkpoint, not the original run's weights)")
    cfg = rec["config"]["train"]
    if req.with_baseline and cfg["task"] == "lm":
        try:
            out["baseline"] = store.baseline_generate(cfg["dataset"], cfg["tokenizer"], req.prompt,
                                                      min(req.max_new_tokens, 200), req.seed, req.temperature)
        except FileNotFoundError as e:
            out["baseline"] = {"model": "baseline unavailable", "text": "", "error": str(e)}
    return out


@app.post("/api/baseline-generate")
def api_baseline_generate(req: GenerateRequest) -> dict[str, Any]:
    """Sample from the best-on-validation Kneser-Ney baseline for the run's tokenizer."""
    try:
        cfg = store.run(req.run_id)["config"]["train"]
    except KeyError as e:
        raise _404(e) from e
    if cfg["task"] != "lm":
        raise HTTPException(400, "baselines exist only for the language-model task")
    try:
        out = store.baseline_generate(cfg["dataset"], cfg["tokenizer"], req.prompt, min(req.max_new_tokens, 200),
                                      req.seed, req.temperature)
    except FileNotFoundError as e:
        raise HTTPException(409, str(e)) from e
    out["label"] = "baseline output (sampled live from n-gram counts on the train split)"
    return out


class AttentionRequest(BaseModel):
    run_id: str
    text: str = Field(max_length=400)


@app.post("/api/attention")
def api_attention(req: AttentionRequest) -> dict[str, Any]:
    try:
        model, tok = store.model(req.run_id)
    except KeyError as e:
        raise _404(e) from e
    except FileNotFoundError as e:
        raise HTTPException(409, str(e)) from e
    try:
        out = attention_maps(model, tok, req.text, max_tokens=48)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    out["run_id"] = req.run_id
    out["checkpoint_origin"] = store.checkpoint_origin(req.run_id)
    return out


# ------------------------------------------------------------------ bounded jobs


@dataclass
class Job:
    id: str
    kind: str
    proc: subprocess.Popen[bytes]
    run_dir: Path
    started: float = field(default_factory=time.time)
    status: str = "running"
    log: list[str] = field(default_factory=list)


JOBS: dict[str, Job] = {}
JOB_TIMEOUT_S = 180
_job_lock = threading.Lock()


def _smoke_run_dir() -> Path:
    from morrowglass.experiments.registry import build
    from morrowglass.experiments.runner import runs_dir_for

    arm = build("scratch", "smoke").arms[0]
    return runs_dir_for("smoke") / arm.run_id


def _job_view(j: Job) -> dict[str, Any]:
    rc = j.proc.poll()
    if j.status == "running" and rc is not None:
        j.status = "completed" if rc == 0 else "failed"
    if j.status == "running" and time.time() - j.started > JOB_TIMEOUT_S:
        (j.run_dir / "STOP").touch()
        j.proc.terminate()
        j.status = "timed_out"
    metrics = []
    mp = j.run_dir / "metrics.jsonl"
    if mp.exists() and mp.stat().st_mtime >= j.started - 1:
        metrics = [json.loads(ln) for ln in mp.read_text().splitlines() if ln.strip()]
    return {"id": j.id, "kind": j.kind, "status": j.status, "elapsed_s": round(time.time() - j.started, 1),
            "run_id": j.run_dir.name, "metrics": metrics[-200:], "provenance": "smoke"}


@app.post("/api/jobs")
def start_job(kind: str = "smoke") -> dict[str, Any]:
    if kind != "smoke":
        raise HTTPException(400, "only the bounded 'smoke' job can be started from the UI; "
                                 "standard/extended experiments run from the CLI")
    with _job_lock:
        if any(_job_view(j)["status"] == "running" for j in JOBS.values()):
            raise HTTPException(409, "a job is already running")
        cmd = [sys.executable, "-m", "morrowglass.cli", "run", "scratch", "--profile", "smoke", "--force", "--quiet"]
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        job = Job(uuid.uuid4().hex[:8], kind, proc, _smoke_run_dir())
        JOBS[job.id] = job
    return _job_view(job)


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict[str, Any]:
    if job_id not in JOBS:
        raise HTTPException(404, "no such job")
    return _job_view(JOBS[job_id])


@app.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str) -> dict[str, Any]:
    if job_id not in JOBS:
        raise HTTPException(404, "no such job")
    j = JOBS[job_id]
    if j.proc.poll() is None:
        j.run_dir.mkdir(parents=True, exist_ok=True)
        (j.run_dir / "STOP").touch()  # cooperative stop; trainer exits at next step
        try:
            j.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            j.proc.terminate()
        j.status = "cancelled"
    return _job_view(j)


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    if job_id not in JOBS:
        raise HTTPException(404, "no such job")

    async def gen():  # type: ignore[no-untyped-def]
        while True:
            v = _job_view(JOBS[job_id])
            yield f"data: {json.dumps(v)}\n\n"
            if v["status"] != "running":
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")
