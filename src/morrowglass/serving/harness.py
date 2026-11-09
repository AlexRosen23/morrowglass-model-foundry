"""Run EleutherAI lm-evaluation-harness against this repo's own OpenAI-compatible endpoint.

Registered study: ``EVALUATIONS["lm-eval-harness"]`` in ``experiments/registry.py``.

For each model the runner starts ``morrowglass serve --model <id>`` on a loopback test port, waits for it,
runs the pinned harness (``.venv-lmeval``, installed from ``eval/lm-eval-requirements.txt``) with
``--model local-completions`` pointed at ``/v1/completions`` and the server's own tokenizer endpoints, then
stops the server. Raw harness output lands in ``reports/lm_eval/<model>/``; ``summarize`` turns it into
``reports/lm_eval/summary.json`` with the registered verdict. Nothing here reads the test split.

    uv venv .venv-lmeval --python 3.12 && VIRTUAL_ENV=.venv-lmeval uv pip sync eval/lm-eval-requirements.txt
    uv run morrowglass lm-eval                      # all registered models (~25 min on an M5 Pro)
"""

from __future__ import annotations

import glob
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from morrowglass.repro import REPO_ROOT, git_provenance, now_iso, read_json, write_json

LM_EVAL_BIN = REPO_ROOT / ".venv-lmeval" / "bin" / "lm_eval"
OUT_ROOT = REPO_ROOT / "reports" / "lm_eval"
SMOKE_ROOT = REPO_ROOT / "artifacts" / "lm_eval_smoke"
TASKS = ["lambada_openai", "wikitext", "morrowglass_shakespeare_val"]
DEFAULT_PORT = 5102


def safe_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def _wait(url: str, timeout_s: float = 90) -> None:
    t = time.time()
    while time.time() - t < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"server did not answer {url} within {timeout_s}s")


def _prefix_token(model: str, port: int) -> int | None:
    """Newline token id for the scratch tokenizers (they have no BOS/EOS); None = harness default (HF EOS)."""
    if model.startswith("hf:"):
        return None
    req = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", data=json.dumps({"prompt": "\n"}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        toks = json.loads(r.read())["tokens"]
    if len(toks) != 1:
        raise RuntimeError(f"newline is not a single token for {model}: {toks}")
    return int(toks[0])


def harness_command(model: str, port: int, max_length: int, prefix: int | None, out_dir: Path,
                    limit: int | None) -> list[str]:
    args = [f"base_url=http://127.0.0.1:{port}/v1/completions", f"model={model}", "tokenizer_backend=remote",
            "tokenized_requests=True", f"max_length={max_length}", "num_concurrent=1", "max_retries=3",
            "timeout=120"]
    if prefix is not None:
        args.append(f"custom_prefix_token_id={prefix}")
    cmd = [str(LM_EVAL_BIN), "run", "--model", "local-completions", "--model_args", ",".join(args),
           "--tasks", ",".join(TASKS), "--include_path", "eval/tasks", "--batch_size", "1",
           "--num_fewshot", "0", "--seed", "1234", "--output_path", str(out_dir), "--log_samples"]
    if limit:
        cmd += ["--limit", str(limit)]
    return cmd


def run_model(model: str, port: int = DEFAULT_PORT, limit: int | None = None) -> dict[str, Any]:
    from morrowglass.experiments.registry import EVALUATIONS

    ev = EVALUATIONS["lm-eval-harness"]
    if model not in ev.runs:
        raise ValueError(f"{model} is not registered for {ev.name}: {ev.runs}")
    if not LM_EVAL_BIN.exists():
        raise FileNotFoundError(f"{LM_EVAL_BIN} missing; see the module docstring to install the pinned harness")
    max_length = ev.protocol["max_length"]["hf:distilgpt2" if model.startswith("hf:") else "scratch"]
    # A --limit run is a plumbing check, not evidence: it never writes into reports/.
    out_dir = (SMOKE_ROOT if limit else OUT_ROOT) / safe_name(model)
    log_dir = REPO_ROOT / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "2"}
    serve = [sys.executable, "-m", "morrowglass.cli", "serve", "--port", str(port), "--model", model]
    if model.startswith("hf:"):
        serve.append("--enable-hf")
    with open(log_dir / f"lm-eval-server-{safe_name(model)}.log", "w") as slog:
        server = subprocess.Popen(serve, cwd=REPO_ROOT, stdout=slog, stderr=subprocess.STDOUT,
                                  env={**env, "MORROWGLASS_WARM_BASELINE": "0"})
        try:
            _wait(f"http://127.0.0.1:{port}/api/health")
            prefix = _prefix_token(model, port)
            cmd = harness_command(model, port, max_length, prefix, out_dir, limit)
            t0 = time.time()
            with open(log_dir / f"lm-eval-{safe_name(model)}.log", "w") as hlog:
                rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=hlog, stderr=subprocess.STDOUT,
                                    env={**env, "OPENAI_API_KEY": ""}, check=False).returncode
            wall = time.time() - t0
        finally:
            server.terminate()
            server.wait(timeout=20)
    if rc != 0:
        raise RuntimeError(f"lm_eval exited {rc} for {model}; see artifacts/logs/lm-eval-{safe_name(model)}.log")
    samples = _stash_samples(model, out_dir) if not limit else {}
    meta = {"model": model, "finished": now_iso(), "wall_time_s": round(wall, 1), "limit": limit,
            "samples_outside_git": samples,
            "command": " ".join(shlex.quote(c) for c in cmd).replace(str(REPO_ROOT) + "/", ""),
            "server_command": " ".join(serve[1:]).replace(sys.executable, "python"), "git": git_provenance(),
            "max_length": max_length, "prefix_token_id": prefix}
    write_json(out_dir / "run_meta.json", meta)
    return meta


def _stash_samples(model: str, out_dir: Path) -> dict[str, Any]:
    """Per-sample logs are large (full documents): move them to artifacts/ (not in git) and record their
    hashes; keep a compact per-document lambada record (gzip) in reports/ for item-level analysis."""
    import gzip

    from morrowglass.repro import sha256_file

    dest = REPO_ROOT / "artifacts" / "lm_eval_samples" / safe_name(model)
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    for f in sorted(out_dir.rglob("samples_*.jsonl")):
        if f.name.startswith("samples_lambada_openai"):
            rows = []
            for ln in f.read_text().splitlines():
                d = json.loads(ln)
                ll, greedy = d["filtered_resps"][0]
                rows.append({"doc_id": d["doc_id"], "doc_hash": d["doc_hash"], "target": d["target"],
                             "loglikelihood": float(ll), "is_greedy": greedy in (True, "True"), "acc": int(d["acc"])})
            data = "".join(json.dumps(r) + "\n" for r in rows).encode()
            (out_dir / "lambada_per_doc.jsonl.gz").write_bytes(gzip.compress(data, mtime=0))
        target = dest / f.name
        f.replace(target)
        out[f.name] = {"path": str(target.relative_to(REPO_ROOT)), "sha256": sha256_file(target),
                       "bytes": target.stat().st_size}
    return out


def _latest_results(model: str) -> dict[str, Any] | None:
    files = sorted(glob.glob(str(OUT_ROOT / safe_name(model) / "**" / "results_*.json"), recursive=True))
    return read_json(files[-1]) if files else None


def summarize() -> dict[str, Any]:
    from morrowglass.experiments.registry import EVALUATIONS

    ev = EVALUATIONS["lm-eval-harness"]
    rows: dict[str, dict[str, Any]] = {}
    for model in ev.runs:
        res = _latest_results(model)
        if res is None:
            continue
        r = res["results"]
        row: dict[str, Any] = {
            "lambada_acc": r["lambada_openai"]["acc,none"],
            "lambada_acc_stderr": r["lambada_openai"].get("acc_stderr,none"),
            "lambada_perplexity": r["lambada_openai"]["perplexity,none"],
            "wikitext_bits_per_byte": r["wikitext"]["bits_per_byte,none"],
            "wikitext_word_perplexity": r["wikitext"]["word_perplexity,none"],
            "shakespeare_val_bits_per_byte": r["morrowglass_shakespeare_val"]["bits_per_byte,none"],
            "n_samples": {t: res["n-samples"][t]["effective"] for t in TASKS if t in res.get("n-samples", {})},
            "lm_eval_version": res.get("lm_eval_version"), "harness_git_hash": res.get("git_hash"),
            "meta": read_json(OUT_ROOT / safe_name(model) / "run_meta.json"),
        }
        if not model.startswith("hf:"):
            own = read_json(REPO_ROOT / "reports" / "runs" / model / "summary.json")["final"]["val"]["bits_per_byte"]
            row["own_val_bpb"] = own
            row["harness_minus_own_val_bpb"] = row["shakespeare_val_bits_per_byte"] - own
        rows[model] = row
    scratch = [m for m in rows if not m.startswith("hf:")]
    ref = rows.get("hf:distilgpt2")
    checks: dict[str, Any] = {}
    if scratch:
        checks["a_plumbing_within_0.03"] = all(abs(rows[m]["harness_minus_own_val_bpb"]) < 0.03 for m in scratch)
        lam = all(rows[m]["lambada_acc"] < 0.05 for m in scratch)
        if ref:
            lam = lam and all(rows[m]["lambada_acc"] < ref["lambada_acc"] for m in scratch)
            checks["b_wikitext_bpb_above_distilgpt2"] = all(
                rows[m]["wikitext_bits_per_byte"] > ref["wikitext_bits_per_byte"] for m in scratch)
        checks["b_lambada_below_0.05" + ("_and_distilgpt2" if ref else "_only (distilgpt2 not run)")] = lam
    complete = len(rows) == len(ev.runs) and not any(r["meta"]["limit"] for r in rows.values())
    out = {
        "evaluation": ev.name, "summarized": now_iso(), "provenance": "measured", "hypothesis": ev.hypothesis,
        "criterion": ev.criterion, "protocol": ev.protocol, "caveats": list(ev.caveats), "models": rows,
        "checks": checks,
        "verdict": ("supported" if all(checks.values()) else "not supported") if complete and checks else "pending",
        "missing": [m for m in ev.runs if m not in rows],
    }
    write_json(OUT_ROOT / "summary.json", out)
    return out
