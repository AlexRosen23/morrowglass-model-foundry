"""Read-only access to the experiment ledger (JSON files are the source of truth).

SQLite (``artifacts/ledger.sqlite``) is a derived index rebuilt from ``reports/runs`` so the
UI can list and filter quickly; deleting it loses nothing.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from morrowglass.data.corpus import MANIFEST_DIR
from morrowglass.model.transformer import MorrowglassLM
from morrowglass.repro import REPO_ROOT, read_json
from morrowglass.tokenizer.tokenizers import Tokenizer, get_tokenizer

REPORTS = REPO_ROOT / "reports"
SOURCES = {
    "measured": (REPORTS / "runs", REPO_ROOT / "artifacts" / "checkpoints"),
    "smoke": (REPORTS / "smoke" / "runs", REPORTS / "smoke" / "checkpoints"),
}
INDEX_PATH = REPO_ROOT / "artifacts" / "ledger.sqlite"


def rel(p: Path) -> str:
    """Repo-relative path when possible (tests use temporary directories)."""
    return str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


class Store:
    def __init__(self, sources: dict[str, tuple[Path, Path]] | None = None, index_path: Path = INDEX_PATH,
                 experiments_dir: Path = REPORTS / "experiments") -> None:
        self.sources = sources or SOURCES
        self.index_path = index_path
        self.experiments_dir = experiments_dir
        # Keyed by (run_id, checkpoint mtime): a re-trained checkpoint (e.g. a UI smoke job) is reloaded.
        self._models: OrderedDict[tuple[str, int], tuple[MorrowglassLM, Tokenizer]] = OrderedDict()
        self._baselines: dict[tuple[str, str], Any] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- runs
    def _locate(self, run_id: str) -> tuple[str, Path, Path]:
        if "/" in run_id or ".." in run_id:
            raise KeyError(run_id)
        for kind, (rdir, cdir) in self.sources.items():
            if (rdir / run_id / "summary.json").exists():
                return kind, rdir / run_id, cdir / run_id
        raise KeyError(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        out = []
        for kind, (rdir, cdir) in self.sources.items():
            if not rdir.exists():
                continue
            for d in sorted(rdir.iterdir()):
                if not (d / "summary.json").exists():
                    continue
                s, m, c = read_json(d / "summary.json"), read_json(d / "metadata.json"), read_json(d / "config.json")
                val = s["final"].get("val", {})
                out.append({
                    "run_id": s["run_id"], "provenance": kind, "experiment": c["train"]["experiment"],
                    "name": c["train"]["name"], "status": s["status"], "task": c["train"]["task"],
                    "tokenizer": c["train"]["tokenizer"], "dataset": c["train"]["dataset"], "seed": c["train"]["seed"],
                    "params": s["params"], "steps": s["steps_completed"], "tokens_seen": s["tokens_seen"],
                    "wall_time_s": s["wall_time_s"], "val_bpb": val.get("bits_per_byte"),
                    "val_iid_acc": s["final"].get("val_iid", {}).get("accuracy"),
                    "git_commit": m["git"]["commit"], "git_dirty": m["git"]["dirty"], "device": m["device"],
                    "started": m["started"], "finished": s.get("finished"),
                    "has_checkpoint": (cdir / d.name / "best.pt").exists(),
                    "checkpoint_reproduced": (cdir / d.name / "REPRODUCED.json").exists(),
                    # Without the causal mask the windowed val metric is leaky (the model can read the answer).
                    "causal": bool(c.get("model", {}).get("causal", True)),
                    "test_opened": (d / "test_eval.json").exists(),
                })
        return sorted(out, key=lambda r: r["started"])

    def run(self, run_id: str) -> dict[str, Any]:
        kind, rd, cd = self._locate(run_id)
        rec: dict[str, Any] = {"provenance": kind, "source_dir": rel(rd)}
        for name in ("summary", "metadata", "config", "test_eval", "bench"):
            p = rd / f"{name}.json"
            rec[name] = read_json(p) if p.exists() else None
        rec["has_checkpoint"] = (cd / "best.pt").exists()
        rec["checkpoint_origin"] = self.checkpoint_origin(run_id) if rec["has_checkpoint"] else None
        return rec

    def checkpoint_origin(self, run_id: str) -> str:
        """'original' for the checkpoint the run produced, 'reproduced' for one re-trained by `reproduce`."""
        _, _, cd = self._locate(run_id)
        return "reproduced" if (cd / "REPRODUCED.json").exists() else "original"

    def metrics(self, run_id: str) -> list[dict[str, Any]]:
        _, rd, _ = self._locate(run_id)
        return _jsonl(rd / "metrics.jsonl")

    def rebuild_index(self) -> int:
        runs = self.list_runs()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".tmp")
        tmp.unlink(missing_ok=True)
        con = sqlite3.connect(tmp)
        cols = ["run_id", "provenance", "experiment", "name", "status", "task", "tokenizer", "dataset", "seed",
                "params", "steps", "tokens_seen", "wall_time_s", "val_bpb", "val_iid_acc", "git_commit",
                "git_dirty", "device", "started", "finished", "has_checkpoint", "checkpoint_reproduced", "causal",
                "test_opened"]
        con.execute(f"CREATE TABLE runs ({', '.join(cols)}, PRIMARY KEY(run_id, provenance))")
        con.executemany(f"INSERT INTO runs VALUES ({', '.join('?' * len(cols))})",
                        [tuple(r[c] for c in cols) for r in runs])
        con.commit()
        con.close()
        tmp.replace(self.index_path)
        return len(runs)

    def query_index(self, where: str = "1=1", params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        con = sqlite3.connect(self.index_path)
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(f"SELECT * FROM runs WHERE {where} ORDER BY started", params)]
        con.close()
        return rows

    # ---------------------------------------------------------------- experiments / findings
    def experiments(self) -> list[dict[str, Any]]:
        if not self.experiments_dir.exists():
            return []
        return [read_json(p) for p in sorted(self.experiments_dir.glob("*/result.json"))]

    def experiment(self, name: str) -> dict[str, Any]:
        p = self.experiments_dir / name / "result.json"
        if "/" in name or not p.exists():
            raise KeyError(name)
        return read_json(p)

    def plot_path(self, name: str, plot: str) -> Path:
        p = self.experiments_dir / name / plot
        if "/" in name or "/" in plot or not p.exists() or p.suffix != ".png":
            raise KeyError(plot)
        return p

    def baselines(self) -> dict[str, Any] | None:
        p = REPORTS / "baselines-shakespeare.json"
        return read_json(p) if p.exists() else None

    def test_log(self) -> list[dict[str, Any]]:
        return _jsonl(REPORTS / "test_access_log.jsonl")

    def dataset_manifest(self) -> dict[str, Any] | None:
        p = MANIFEST_DIR / "shakespeare.json"
        return read_json(p) if p.exists() else None

    # ---------------------------------------------------------------- models
    def model(self, run_id: str) -> tuple[MorrowglassLM, Tokenizer]:
        from morrowglass.train.trainer import load_model_from_checkpoint

        with self._lock:
            _, rd, cd = self._locate(run_id)
            ck = cd / "best.pt"
            if not ck.exists():
                raise FileNotFoundError(f"no local checkpoint for {run_id} (checkpoints are not in git; "
                                        "re-run the experiment to recreate it)")
            key = (run_id, ck.stat().st_mtime_ns)
            if key in self._models:
                self._models.move_to_end(key)
                return self._models[key]
            for stale in [k for k in self._models if k[0] == run_id]:
                del self._models[stale]
            model, state = load_model_from_checkpoint(ck)
            cfg = read_json(rd / "config.json")["train"]
            tok = get_tokenizer(state["tokenizer"], cfg["dataset"] if cfg["task"] == "lm" else "shakespeare-sample")
            if tok.fingerprint() != state["tokenizer_sha256"]:
                raise RuntimeError(f"tokenizer for {run_id} does not match the one it was trained with")
            self._models[key] = (model, tok)
            while len(self._models) > 3:
                self._models.popitem(last=False)
            return model, tok

    def checkpoint_stamp(self, run_id: str) -> int | None:
        """mtime of the run's best checkpoint (None when absent); changes when the run is re-trained."""
        _, _, cd = self._locate(run_id)
        ck = cd / "best.pt"
        return ck.stat().st_mtime_ns if ck.exists() else None

    def baseline_model(self, dataset: str, tokenizer: str) -> Any:
        """Best-on-validation Kneser-Ney order for this tokenizer, fitted on train (cached)."""
        from morrowglass.data.corpus import dataset_dir
        from morrowglass.data.loader import encode_split
        from morrowglass.eval.ngram import KneserNey

        if not (dataset_dir(dataset) / "train.txt").exists():
            hint = "run `make data` (downloads the 5.6 MB Project Gutenberg file)" if dataset == "shakespeare" else \
                "the sample should be in git"
            raise FileNotFoundError(f"the {dataset} train split is not on this machine, so the n-gram baseline "
                                    f"cannot be fitted: {hint}")

        key = (dataset, tokenizer)
        with self._lock:
            if key in self._baselines:
                return self._baselines[key]
        order = 5 if tokenizer == "byte" else 3
        bl = self.baselines() if dataset == "shakespeare" else None
        if bl:
            cands = [r for r in bl["results"] if r["tokenizer"] == tokenizer and r["model"].startswith("kneser")]
            if cands:
                order = min(cands, key=lambda r: r["val"]["bits_per_byte"])["order"]
        tok = get_tokenizer(tokenizer, dataset)
        tr = encode_split(dataset, "train", tok)
        kn = KneserNey(order, tok.vocab_size, 1 if tokenizer == "byte" else 2).fit(tr.ids)
        with self._lock:
            self._baselines[key] = (kn, tok, order)
        return self._baselines[key]

    def baseline_generate(self, dataset: str, tokenizer: str, prompt: str, max_new: int, seed: int,
                          temperature: float) -> dict[str, Any]:
        kn, tok, order = self.baseline_model(dataset, tokenizer)
        ids = kn.sample(tok.encode(prompt), max_new, np.random.default_rng(seed), temperature)
        return {"model": f"kneser-ney-{order} ({tokenizer})", "text": tok.decode(ids), "order": order}
