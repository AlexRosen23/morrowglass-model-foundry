"""Portable model-artifact export (contracts/model_artifact_v1.schema.json).

A deployment system (for example a release/rollback service) can ingest an exported model
from the manifest alone: weights path + sha256, tokenizer, architecture, I/O schema,
held-out scores, the tests that guard it, licences and runtime needs. It never needs
access to this repository.
"""

from __future__ import annotations

import io
import json
import platform
from pathlib import Path
from typing import Any

import torch

from morrowglass.repro import REPO_ROOT, now_iso, read_json, sha256_file, write_json, write_verified
from morrowglass.train.trainer import CHECKPOINT_ROOT, RUNS_DIR, load_model_from_checkpoint

SCHEMA_PATH = REPO_ROOT / "contracts" / "model_artifact_v1.schema.json"
EXPORT_ROOT = REPO_ROOT / "artifacts" / "exports"

GUARD_TESTS = [
    "tests/test_model.py::test_causal_no_future_leakage",
    "tests/test_model.py::test_param_count_matches_analytic",
    "tests/test_trainer.py::test_checkpoint_roundtrip_identical_logits",
    "tests/test_trainer.py::test_corrupted_checkpoint_is_rejected",
    "tests/test_inference.py::test_generation_deterministic_under_seed",
    "tests/test_contracts.py::test_export_matches_schema",
]


def export_artifact(run_id: str, out_root: Path = EXPORT_ROOT, runs_dir: Path = RUNS_DIR,
                    ckpt_root: Path = CHECKPOINT_ROOT) -> dict[str, Any]:
    rd = runs_dir / run_id
    summary, meta, cfg = read_json(rd / "summary.json"), read_json(rd / "metadata.json"), read_json(rd / "config.json")
    model, state = load_model_from_checkpoint(ckpt_root / run_id / "best.pt")
    out_dir = out_root / run_id
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)  # weights only: loadable with weights_only=True
    sha = write_verified(out_dir / "model.pt", buf.getvalue())
    tok_name = state["tokenizer"]
    tok_file = None
    if tok_name.startswith("bpe"):
        src = REPO_ROOT / "artifacts" / "tokenizers" / cfg["train"]["dataset"] / f"{tok_name}.json"
        tok_file = out_dir / "tokenizer.json"
        write_verified(tok_file, src.read_bytes())
    val = summary["final"].get("val", {})
    test = read_json(rd / "test_eval.json") if (rd / "test_eval.json").exists() else None
    manifest = {
        "schema": "morrowglass.model_artifact",
        "schema_version": "1.0.0",
        "model_name": f"morrowglass-{cfg['train']['name']}",
        "model_version": f"{run_id}",
        "created": now_iso(),
        "producer": {"project": "morrowglass-model-foundry", "git_commit": meta["git"]["commit"],
                     "git_dirty": meta["git"]["dirty"], "run_id": run_id},
        "artifact": {"path": "model.pt", "format": "pytorch-state-dict", "sha256": sha,
                     "bytes": (out_dir / "model.pt").stat().st_size,
                     "load": "torch.load(path, weights_only=True) -> MorrowglassLM(ModelConfig(**architecture.config))"},
        "tokenizer": {"name": tok_name, "sha256": state["tokenizer_sha256"],
                      "path": "tokenizer.json" if tok_file else None,
                      "kind": "utf-8 bytes (ids 0-255)" if tok_name == "byte" else "byte-level BPE (HF tokenizers JSON)"},
        "architecture": {"family": "decoder-only causal Transformer (pre-LN, tied embeddings)",
                         "config": state["model_config"], "params": summary["params"]},
        "io": {"input": {"type": "token_ids", "dtype": "int64", "shape": ["batch", "seq<=block_size"],
                         "block_size": state["model_config"]["block_size"]},
               "output": {"type": "logits", "dtype": "float32", "shape": ["batch", "seq", "vocab_size"],
                          "vocab_size": state["model_config"]["vocab_size"],
                          "semantics": "unnormalised next-token scores; softmax gives model probabilities (uncalibrated)"}},
        "evaluation": {
            "dataset": meta["data"]["dataset"], "dataset_hash": meta["data"]["dataset_hash"],
            "primary_metric": "bits_per_byte (lower is better)",
            "val": {"bits_per_byte": val.get("bits_per_byte"), "ci95": val.get("bits_per_byte_ci95"),
                    "ece": (val.get("calibration") or {}).get("ece")},
            "test": ({"bits_per_byte": test["results"]["test"]["bits_per_byte"],
                      "ci95": test["results"]["test"]["bits_per_byte_ci95"], "evaluated": test["evaluated"]}
                     if test and "test" in test["results"] else None),
            "test_access_log": "reports/test_access_log.jsonl",
        },
        "tests": GUARD_TESTS,
        "license": {"code": "MIT", "weights": "MIT (trained by this project)",
                    "training_data": "Public domain (Project Gutenberg eBook #100, boilerplate removed)"},
        "runtime": {"python": ">=3.12", "torch": torch.__version__, "devices": ["cpu", "mps"],
                    "approx_memory_mb": round(summary["params"] * 4 / 2**20 * 3 + 50),
                    "built_on": platform.platform()},
        "limitations": ["Tiny model (~2M params) trained on ~4.5 MB of Shakespeare; not a general assistant.",
                        "Outputs are pastiche text with frequent nonsense; probabilities are not calibrated confidence.",
                        "No safety tuning; do not deploy for user-facing generation."],
    }
    write_json(out_dir / "model_artifact.json", manifest)
    mpath = out_dir / "model_artifact.json"
    return {"manifest": str(mpath.relative_to(REPO_ROOT)) if mpath.is_relative_to(REPO_ROOT) else str(mpath),
            "sha256": sha,
            "manifest_sha256": sha256_file(out_dir / "model_artifact.json")}


def validate_manifest(manifest: dict[str, Any], schema: dict[str, Any] | None = None) -> list[str]:
    """Minimal JSON-Schema subset validator (required/type/const/enum, nested), no extra dependency."""
    schema = schema or json.loads(SCHEMA_PATH.read_text())
    errors: list[str] = []

    def check(obj: Any, sch: dict[str, Any], path: str) -> None:
        types = sch.get("type")
        if types:
            tl = types if isinstance(types, list) else [types]
            pymap: dict[str, Any] = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float),
                     "boolean": bool, "null": type(None)}
            if not any(isinstance(obj, pymap[t]) and not (t in ("integer", "number") and isinstance(obj, bool))
                       for t in tl):
                errors.append(f"{path}: expected {tl}, got {type(obj).__name__}")
                return
        if "const" in sch and obj != sch["const"]:
            errors.append(f"{path}: expected const {sch['const']!r}")
        if "enum" in sch and obj not in sch["enum"]:
            errors.append(f"{path}: {obj!r} not in {sch['enum']}")
        if "pattern" in sch and isinstance(obj, str):
            import re

            if not re.fullmatch(sch["pattern"], obj):
                errors.append(f"{path}: {obj!r} does not match {sch['pattern']}")
        if isinstance(obj, dict):
            for req in sch.get("required", []):
                if req not in obj:
                    errors.append(f"{path}: missing required '{req}'")
            for k, sub in sch.get("properties", {}).items():
                if k in obj:
                    check(obj[k], sub, f"{path}.{k}")
        if isinstance(obj, list) and "items" in sch:
            for i, it in enumerate(obj):
                check(it, sch["items"], f"{path}[{i}]")

    check(manifest, schema, "$")
    return errors
