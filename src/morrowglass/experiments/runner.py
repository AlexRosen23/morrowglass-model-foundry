"""Run experiments, baselines, the overfit sanity check, and the locked final test."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from morrowglass.data.corpus import dataset_hash
from morrowglass.data.loader import encode_split
from morrowglass.eval.ngram import KneserNey, uniform_eval, unigram_eval
from morrowglass.experiments.registry import Experiment, build
from morrowglass.hardware import hardware_report, resolve_device
from morrowglass.model.transformer import ModelConfig, MorrowglassLM, compute_budget
from morrowglass.repro import REPO_ROOT, git_provenance, now_iso, read_json, set_seed, sha256_file, write_json
from morrowglass.tokenizer.tokenizers import get_tokenizer
from morrowglass.train.trainer import (
    CHECKPOINT_ROOT,
    RUNS_DIR,
    TrainConfig,
    build_task,
    load_checkpoint,
    load_model_from_checkpoint,
    train_from_config,
)

REPORTS = REPO_ROOT / "reports"
SMOKE_DIR = REPORTS / "smoke"
TEST_LOG = REPORTS / "test_access_log.jsonl"
# Re-runs of committed runs land here (both git-ignored), never on top of the evidence in reports/runs.
REPRODUCE_DIR = REPORTS / "reproduce"
REPRODUCE_CKPT = REPO_ROOT / "artifacts" / "reproduce"


class EvidenceOverwriteError(PermissionError):
    """Raised when a re-run would replace a committed, measured run record."""


def runs_dir_for(profile: str) -> Path:
    return SMOKE_DIR / "runs" if profile == "smoke" else RUNS_DIR


def experiments_dir_for(profile: str) -> Path:
    return SMOKE_DIR / "experiments" if profile == "smoke" else REPORTS / "experiments"


# ------------------------------------------------------------------ sanity check


def overfit_sanity(steps: int = 200, device: str = "cpu", seed: int = 0) -> dict[str, Any]:
    """Memorise one fixed batch. A correct model + optimizer must drive the loss near zero."""
    set_seed(seed)
    cfg = ModelConfig(vocab_size=256, block_size=32, n_layer=2, n_head=2, d_model=64)
    model = MorrowglassLM(cfg).to(device)
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 256, (4, 32), generator=g).to(device)
    y = torch.randint(0, 256, (4, 32), generator=g).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    losses = []
    for _ in range(steps):
        out = model(x, targets=y)
        assert out.loss is not None
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        losses.append(float(out.loss.detach()))
    passed = losses[-1] < 0.1 * losses[0] and losses[-1] < 0.5
    return {"check": "overfit-one-batch", "steps": steps, "initial_loss": losses[0], "final_loss": losses[-1],
            "loss_curve": losses[:: max(1, steps // 50)], "passed": passed, "seed": seed,
            "criterion": "final loss < 10% of initial and < 0.5 nats on a fixed random batch"}


# ------------------------------------------------------------------ baselines


def run_baselines(dataset: str, out_dir: Path) -> dict[str, Any]:
    """Uniform, unigram and Kneser-Ney n-grams for byte and BPE-1024 tokenizers, scored on val."""
    t0 = time.time()
    results: list[dict[str, Any]] = []
    plan = [("byte", 1, [2, 3, 4, 5, 6, 7]), ("bpe1024", 2, [2, 3, 4])]
    for spec, width, orders in plan:
        tok = get_tokenizer(spec, dataset)
        tr, va = encode_split(dataset, "train", tok), encode_split(dataset, "val", tok)
        results.append({"tokenizer": spec, "model": "uniform", "order": 0,
                        "val": uniform_eval(va.ids, va.byte_len, tok.vocab_size)})
        results.append({"tokenizer": spec, "model": "unigram(add-1)", "order": 1,
                        "val": unigram_eval(tr.ids, va.ids, va.byte_len, tok.vocab_size)})
        for n in orders:
            t1 = time.time()
            kn = KneserNey(n, tok.vocab_size, width).fit(tr.ids)
            r = kn.evaluate(va.ids, va.byte_len)
            results.append({"tokenizer": spec, "model": f"kneser-ney-{n}", "order": n, "val": r,
                            "discounts": [o.discount for o in kn.orders],
                            "n_contexts_highest": len(kn.orders[-1].den), "seconds": round(time.time() - t1, 1)})
    best = min(results, key=lambda r: r["val"]["bits_per_byte"])
    out = {"dataset": dataset, "dataset_hash": dataset_hash(dataset), "selection": "lowest val bits-per-byte",
           "best": {k: best[k] for k in ("tokenizer", "model", "order")}, "best_val_bpb": best["val"]["bits_per_byte"],
           "best_val_bpb_ci95": best["val"]["bits_per_byte_ci95"], "results": results,
           "git": git_provenance(), "created": now_iso(), "seconds": round(time.time() - t0, 1),
           "test_locked": True}
    write_json(out_dir / f"baselines-{dataset}.json", out)
    return out


# ------------------------------------------------------------------ estimates


def estimate(exp: Experiment, device: str = "auto", probe_steps: int = 8) -> dict[str, Any]:
    """Measure a few real steps per distinct model shape, then extrapolate time/memory/disk."""
    dev = resolve_device(device)
    rows = []
    speed_cache: dict[str, float] = {}
    for arm in exp.arms:
        task_vocab = 256 if arm.tokenizer == "byte" else int(arm.tokenizer[3:])
        mcfg = ModelConfig.from_dict({**arm.model, "vocab_size": task_vocab})
        key = json.dumps([mcfg.to_dict(), arm.batch_size])
        tokens_per_step = arm.batch_size * arm.grad_accum * mcfg.block_size
        if key not in speed_cache:
            model = MorrowglassLM(mcfg).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            x = torch.randint(0, task_vocab, (arm.batch_size, mcfg.block_size), device=dev)
            for i in range(probe_steps + 2):
                if i == 2:
                    if dev == "mps":
                        torch.mps.synchronize()
                    t = time.time()
                out = model(x, targets=x)
                assert out.loss is not None
                opt.zero_grad()
                out.loss.backward()
                opt.step()
            if dev == "mps":
                torch.mps.synchronize()
            speed_cache[key] = probe_steps * tokens_per_step / (time.time() - t)
        tps = speed_cache[key]
        b = compute_budget(mcfg, tokens_per_step * arm.max_steps)
        act_mb = (arm.batch_size * mcfg.block_size * mcfg.d_model * mcfg.n_layer * 20
                  + arm.batch_size * mcfg.n_head * mcfg.block_size ** 2 * mcfg.n_layer * 2) * 4 / 2**20
        rows.append({"run_id": arm.run_id, "params": b["params_total"], "steps": arm.max_steps,
                     "tokens": b["train_tokens"], "measured_tokens_per_s": round(tps),
                     "est_minutes": round(b["train_tokens"] / tps / 60 * 1.15, 1),  # +15% for evals
                     "est_peak_mem_mb": round(b["weights_mb_fp32"] * 4 + act_mb),
                     "disk_mb": round(b["checkpoint_mb_with_adam"] * 2, 1)})
    return {"experiment": exp.name, "device": dev, "arms": rows,
            "total_minutes": round(sum(r["est_minutes"] for r in rows), 1),
            "total_disk_mb": round(sum(r["disk_mb"] for r in rows), 1),
            "max_peak_mem_mb": max(r["est_peak_mem_mb"] for r in rows),
            "method": f"{probe_steps} timed steps per model shape on {dev} right now; shared-machine load varies"}


# ------------------------------------------------------------------ experiments


def run_experiment(name: str, profile: str = "standard", approved: bool = False, only: str | None = None,
                   device: str = "auto", force: bool = False, quiet: bool = False,
                   overwrite_evidence: bool = False) -> list[dict[str, Any]]:
    exp = build(name, profile)
    runs_dir = runs_dir_for(profile)
    if force and profile != "smoke" and not overwrite_evidence:
        # The run id is a hash of the config, so --force would re-train into the committed run's own directory,
        # replace its summary/metrics, and leave any locked-test result attached to different weights.
        existing = [a.run_id for a in exp.arms if (not only or only in a.name)
                    and (runs_dir / a.run_id / "summary.json").exists()]
        if existing:
            raise EvidenceOverwriteError(
                f"--force would overwrite committed run record(s) {', '.join(existing)}. "
                "Use `morrowglass reproduce RUN_ID` to re-train into reports/reproduce/ and compare, "
                "or pass --overwrite-evidence if replacing the evidence is really intended.")
    if profile == "extended" and not approved:
        est = estimate(exp, device)
        raise PermissionError(
            f"extended profile needs explicit approval: ~{est['total_minutes']} min, "
            f"~{est['max_peak_mem_mb']} MB peak memory, ~{est['total_disk_mb']} MB of checkpoints. "
            "Re-run with --approved once the owner agrees.")
    sanity = overfit_sanity()
    if not sanity["passed"]:
        raise RuntimeError(f"overfit sanity check failed: {sanity}")
    ckpt_root = CHECKPOINT_ROOT if profile != "smoke" else SMOKE_DIR / "checkpoints"
    summaries = []
    for arm in exp.arms:
        if only and only not in arm.name:
            continue
        arm.device = device
        if profile == "smoke":
            arm.device = "cpu"  # smoke must work anywhere, including CI
        done = runs_dir / arm.run_id / "summary.json"
        if done.exists() and not force:
            s = read_json(done)
            if s.get("status") in ("completed", "early_stopped", "diverged"):
                if not quiet:
                    print(f"skip {arm.run_id}: already {s['status']}")
                summaries.append(s)
                continue
        summaries.append(train_from_config(arm, quiet=quiet, runs_dir=runs_dir, ckpt_root=ckpt_root))
    write_json(experiments_dir_for(profile) / name / "sanity.json", sanity)
    return summaries


def reproduce_run(run_id: str, device: str = "auto", quiet: bool = False, runs_dir: Path = RUNS_DIR,
                  out_dir: Path = REPRODUCE_DIR, ckpt_root: Path = REPRODUCE_CKPT,
                  install_into: Path | None = CHECKPOINT_ROOT) -> dict[str, Any]:
    """Re-train a committed run from its saved config into ``out_dir`` and compare with the committed record.

    The committed run directory is only read. The verdict is whether the reproduced validation bits-per-byte
    falls inside the committed run's 95% bootstrap interval (MPS kernels are not bit-deterministic).

    Weights are not in git, so a fresh clone cannot sample the committed models. When the reproduction passes
    and ``install_into`` has no checkpoint for this run, the re-trained best checkpoint is copied there with a
    ``REPRODUCED.json`` marker; the API then labels everything served from it as re-trained weights. An
    existing (original) checkpoint is never replaced.
    """
    committed_dir = runs_dir / run_id
    if not (committed_dir / "summary.json").exists():
        raise KeyError(f"no committed run {run_id!r} in {committed_dir.parent}")
    cfg = TrainConfig.from_dict(read_json(committed_dir / "config.json")["train"])
    if cfg.run_id != run_id:
        raise RuntimeError(f"saved config hashes to {cfg.run_id}, not {run_id}: the config schema has changed")
    cfg.device = device
    sanity = overfit_sanity()
    if not sanity["passed"]:
        raise RuntimeError(f"overfit sanity check failed: {sanity}")
    summary = train_from_config(cfg, quiet=quiet, runs_dir=out_dir / "runs", ckpt_root=ckpt_root)
    old = read_json(committed_dir / "summary.json")
    report: dict[str, Any] = {
        "run_id": run_id, "reproduced": now_iso(), "git": git_provenance(), "device": resolve_device(device),
        "committed": {"source": str(committed_dir.relative_to(REPO_ROOT)) if committed_dir.is_relative_to(REPO_ROOT)
                      else str(committed_dir), "status": old["status"], "best_step": old.get("best_step")},
        "reproduced_run": {"source": str(out_dir / "runs" / run_id), "status": summary["status"],
                           "best_step": summary.get("best_step"), "wall_time_s": summary["wall_time_s"]},
    }
    if cfg.task == "lm":
        a, b = old["final"]["val"], summary["final"]["val"]
        lo, hi = a["bits_per_byte_ci95"]
        report["val_bpb"] = {"committed": a["bits_per_byte"], "committed_ci95": [lo, hi],
                             "reproduced": b["bits_per_byte"], "reproduced_ci95": b.get("bits_per_byte_ci95"),
                             "difference": b["bits_per_byte"] - a["bits_per_byte"]}
        report["verdict"] = "reproduced" if lo <= b["bits_per_byte"] <= hi else "not reproduced"
        report["criterion"] = "reproduced val bpb inside the committed run's 95% bootstrap interval"
    else:
        a_acc, b_acc = old["final"]["val_iid"]["accuracy"], summary["final"]["val_iid"]["accuracy"]
        report["val_iid_accuracy"] = {"committed": a_acc, "reproduced": b_acc, "difference": b_acc - a_acc}
        report["verdict"] = "reproduced" if abs(b_acc - a_acc) <= 0.02 else "not reproduced"
        report["criterion"] = "|reproduced - committed| val_iid accuracy <= 0.02"
    report["installed_checkpoint"] = None
    target = install_into / run_id if install_into is not None else None
    if target is not None and report["verdict"] == "reproduced" and not (target / "best.pt").exists():
        import shutil

        target.mkdir(parents=True, exist_ok=True)
        for name in ("best.pt", "best.sha256.json"):
            shutil.copyfile(ckpt_root / run_id / name, target / name)
        load_checkpoint(target / "best.pt")  # verify the copy against its recorded hash
        marker = {"run_id": run_id, "origin": "reproduced", "reproduced": report["reproduced"],
                  "device": report["device"], "git": report["git"]["commit"],
                  "note": "Re-trained from the committed config; not the original checkpoint. Its metrics are in "
                          f"reports/reproduce/{run_id}.json; the committed record describes the original run.",
                  **({"val_bpb": report["val_bpb"]} if "val_bpb" in report else {})}
        write_json(target / "REPRODUCED.json", marker)
        report["installed_checkpoint"] = str(target)
    write_json(out_dir / f"{run_id}.json", report)
    return report


# ------------------------------------------------------------------ leak-free scoring


LEAK_FREE_RECORD = "leak_free_eval.json"


def last_position_bpb(run_dir: Path, ckpt_root: Path = CHECKPOINT_ROOT, n_positions: int = 2048,
                      seed: int = 0, split: str = "val") -> dict[str, Any]:
    """Score only the final position of each window: no model can see the future this way.

    Weights are not in git, so the measurement is also kept next to the run record
    (``leak_free_eval.json``, with the checkpoint's sha256). With the checkpoint present it is
    re-measured, and the record is (re)written only when it is missing or the weights changed;
    without the checkpoint, as on a fresh clone, the recorded value is used.
    """
    cfg = TrainConfig.from_dict(read_json(run_dir / "config.json")["train"])
    ckpt = ckpt_root / cfg.run_id / "best.pt"
    rec_path = run_dir / LEAK_FREE_RECORD
    rec = read_json(rec_path) if rec_path.exists() else None
    same = rec is not None and (rec["result"]["positions"], rec["result"]["seed"], rec["result"]["split"]) == (
        n_positions, seed, split)
    if not ckpt.exists():
        if same:
            return dict(rec["result"])  # type: ignore[index]
        raise FileNotFoundError(f"{ckpt} is missing and {rec_path} has no matching measurement; train the run "
                                f"or `morrowglass reproduce {cfg.run_id}` first")
    sha = sha256_file(ckpt)
    result = _last_position_bpb(cfg, ckpt, n_positions, seed, split)
    if not same or rec["checkpoint_sha256"] != sha:  # type: ignore[index]
        write_json(rec_path, {"result": result, "checkpoint_sha256": sha, "measured": now_iso(),
                              "git": git_provenance()})
    return result


@torch.no_grad()
def _last_position_bpb(cfg: TrainConfig, ckpt: Path, n_positions: int, seed: int, split: str) -> dict[str, Any]:
    model, _ = load_model_from_checkpoint(ckpt)
    task = build_task(cfg)
    ids = task.splits[split].ids  # type: ignore[attr-defined]
    blen = task.splits[split].byte_len  # type: ignore[attr-defined]
    T = model.cfg.block_size
    rng = np.random.default_rng(seed)
    ends = rng.integers(T, len(ids) - 1, size=n_positions)
    nll = 0.0
    nbytes = 0
    for i in range(0, n_positions, 256):
        e = ends[i : i + 256]
        x = torch.from_numpy(np.stack([ids[j - T + 1 : j + 1] for j in e]))
        y = torch.from_numpy(ids[e + 1])
        lp = torch.log_softmax(model(x).logits[:, -1].float(), -1).gather(-1, y[:, None]).squeeze(-1)
        nll -= float(lp.sum())
        nbytes += int(blen[ids[e + 1]].sum())
    return {"run_id": cfg.run_id, "split": split, "positions": n_positions, "seed": seed,
            "bits_per_byte": nll / math.log(2) / nbytes}


# ------------------------------------------------------------------ locked test


def final_test(run_ids: list[str], reason: str, runs_dir: Path = RUNS_DIR, ckpt_root: Path = CHECKPOINT_ROOT,
               device: str = "cpu", include_baseline: bool = True) -> dict[str, Any]:
    """Evaluate chosen runs on the locked test split. Every access is appended to the test log."""
    if not reason.strip():
        raise ValueError("a reason is required to open the test set")
    results: dict[str, Any] = {}
    dev = resolve_device(device)
    for rid in run_ids:
        rd = runs_dir / rid
        cfg = TrainConfig.from_dict(read_json(rd / "config.json")["train"])
        model, _ = load_model_from_checkpoint(ckpt_root / rid / "best.pt")
        model.to(dev)
        task = build_task(cfg)
        r = task.test_eval(model, dev)
        write_json(rd / "test_eval.json", {"run_id": rid, "reason": reason, "evaluated": now_iso(),
                                           "checkpoint": "best (selected on validation)", "results": r})
        results[rid] = r
    baseline: dict[str, Any] | None = None
    if include_baseline:
        bl_path = REPORTS / "baselines-shakespeare.json"
        if bl_path.exists():
            bl = read_json(bl_path)
            best = bl["best"]
            tok = get_tokenizer(best["tokenizer"], "shakespeare")
            tr, te = encode_split("shakespeare", "train", tok), encode_split("shakespeare", "test", tok)
            kn = KneserNey(best["order"], tok.vocab_size, 1 if best["tokenizer"] == "byte" else 2).fit(tr.ids)
            baseline = {"model": best, "test": kn.evaluate(te.ids, te.byte_len)}
            write_json(REPORTS / "baselines-shakespeare-test.json",
                       {"reason": reason, "evaluated": now_iso(), **baseline})
    entry = {"time": now_iso(), "runs": run_ids, "baseline": baseline is not None, "reason": reason,
             "git": git_provenance()["commit"], "hardware": hardware_report()["chip"]}
    TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"runs": results, "baseline": baseline, "log_entry": entry}
