"""Turn saved run records into verdicts, comparison tables and plots.

Nothing here invents numbers: every value is read from ``reports/runs/<id>/*.json``.
If a run is missing, the experiment's verdict is ``pending`` and says which runs are absent.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from morrowglass.experiments.registry import ALL, Experiment, build
from morrowglass.experiments.runner import (
    CHECKPOINT_ROOT,
    REPORTS,
    SMOKE_DIR,
    experiments_dir_for,
    last_position_bpb,
    runs_dir_for,
)
from morrowglass.repro import REPO_ROOT, git_provenance, now_iso, read_json, write_json


def load_run(runs_dir: Path, run_id: str) -> dict[str, Any] | None:
    d = runs_dir / run_id
    if not (d / "summary.json").exists():
        return None
    metrics = []
    if (d / "metrics.jsonl").exists():
        metrics = [json.loads(ln) for ln in (d / "metrics.jsonl").read_text().splitlines() if ln.strip()]
    rec = {"summary": read_json(d / "summary.json"), "metadata": read_json(d / "metadata.json"),
           "config": read_json(d / "config.json"), "metrics": metrics}
    if (d / "test_eval.json").exists():
        rec["test_eval"] = read_json(d / "test_eval.json")
    return rec


def _val(rec: dict[str, Any]) -> dict[str, Any]:
    return dict(rec["summary"]["final"].get("val", {}))


def _evals(rec: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in rec["metrics"] if m.get("kind") == "eval"]


def _trains(rec: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in rec["metrics"] if m.get("kind") == "train"]


def _arm_row(rec: dict[str, Any]) -> dict[str, Any]:
    s = rec["summary"]
    v = _val(rec)
    tr = _trains(rec)
    return {
        "run_id": s["run_id"], "status": s["status"], "params": s["params"], "steps": s["steps_completed"],
        "tokens_seen": s["tokens_seen"], "wall_time_s": s["wall_time_s"], "best_step": s["best_step"],
        "val_bpb": v.get("bits_per_byte"), "val_bpb_ci95": v.get("bits_per_byte_ci95"),
        "val_loss": v.get("loss_nats_per_token"), "val_ppl_token": v.get("perplexity_per_token"),
        "ece": (v.get("calibration") or {}).get("ece"),
        "train_subset_bpb": s["final"].get("train_subset", {}).get("bits_per_byte"),
        "last_train_loss": tr[-1]["train_loss"] if tr else None,
        "tokenizer": rec["config"]["train"]["tokenizer"], "seed": rec["config"]["train"]["seed"],
        "git_commit": rec["metadata"]["git"]["commit"], "git_dirty": rec["metadata"]["git"]["dirty"],
        "device": rec["metadata"]["device"],
    }


def _plot(path: Path, series: list[dict[str, Any]], title: str, xlabel: str, ylabel: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    for s in series:
        ax.plot(s["x"], s["y"], label=s["label"], linestyle=s.get("style", "-"), linewidth=1.6)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path.name


def _curve(rec: dict[str, Any], key: str, label: str, style: str = "-") -> dict[str, Any]:
    ev = [e for e in _evals(rec) if e.get(key) is not None]
    return {"x": [e["tokens"] for e in ev], "y": [e[key] for e in ev], "label": label, "style": style}


def ci_separated(lower_better: list[float] | None, other: list[float] | None) -> bool | None:
    if not lower_better or not other:
        return None
    return lower_better[1] < other[0]


def analyze(name: str, profile: str = "standard") -> dict[str, Any]:
    exp: Experiment = build(name, profile)
    runs_dir = runs_dir_for(profile)
    # Extended-profile analyses are separate findings ("<name>-extended") so they never overwrite the standard ones.
    exp_id = f"{name}-extended" if profile == "extended" else name
    out_dir = experiments_dir_for(profile) / exp_id
    recs = {a.run_id: load_run(runs_dir, a.run_id) for a in exp.arms}
    missing = [rid for rid, r in recs.items() if r is None]
    result: dict[str, Any] = {
        "experiment": exp_id, "title": exp.title + (" (extended profile)" if profile == "extended" else ""),
        "track": exp.track, "profile": profile,
        "hypothesis": exp.hypothesis, "criterion": exp.criterion, "caveats": exp.caveats,
        "arms": [a.run_id for a in exp.arms], "missing_runs": missing, "analyzed": now_iso(),
        "analysis_git": git_provenance()["commit"], "plots": [], "observations": {},
    }
    if missing:
        result["verdict"] = "pending"
        result["verdict_detail"] = f"{len(missing)} run(s) not executed yet"
        write_json(out_dir / "result.json", result)
        return result
    ok = {k: v for k, v in recs.items() if v is not None}
    rows = {rid: _arm_row(r) for rid, r in ok.items()}
    result["table"] = list(rows.values())
    by_name = {r["summary"]["run_id"].rsplit("-", 2)[0]: r for r in ok.values()}
    verdict, detail = "inconclusive", ""
    obs: dict[str, Any] = result["observations"]

    baseline_path = (SMOKE_DIR if profile == "smoke" else REPORTS) / (
        "baselines-shakespeare-sample.json" if profile == "smoke" else "baselines-shakespeare.json")
    baseline = read_json(baseline_path) if baseline_path.exists() else None

    if name == "scratch":
        rec = next(iter(ok.values()))
        v = _val(rec)
        result["plots"].append(_plot(out_dir / "curve.png", [_curve(rec, "val_bits_per_byte", "val bpb (subset)"),
                                     _curve(rec, "train_eval_bits_per_byte", "train bpb (subset)", "--")],
                                     "Scratch Transformer: bits per byte", "training tokens", "bits per byte"))
        if baseline is None:
            verdict, detail = "pending", "baselines not computed (run `morrowglass baselines`)"
        else:
            obs["baseline_best"] = baseline["best"]
            obs["baseline_val_bpb"] = baseline["best_val_bpb"]
            obs["baseline_val_bpb_ci95"] = baseline["best_val_bpb_ci95"]
            obs["transformer_val_bpb"] = v["bits_per_byte"]
            obs["transformer_val_bpb_ci95"] = v["bits_per_byte_ci95"]
            sep = ci_separated(v["bits_per_byte_ci95"], baseline["best_val_bpb_ci95"])
            verdict = "supported" if sep else "not supported"
            detail = (f"Transformer {v['bits_per_byte']:.3f} bpb vs best n-gram "
                      f"({baseline['best']['model']}, {baseline['best']['tokenizer']}) {baseline['best_val_bpb']:.3f}")
            obs["calibration"] = v.get("calibration")
    elif name == "tokenization":
        b, p = by_name["lm-byte-small"], by_name["lm-bpe1024-small"]
        vb, vp = _val(b), _val(p)
        result["plots"].append(_plot(out_dir / "curves.png", [_curve(b, "val_bits_per_byte", "byte"),
                                     _curve(p, "val_bits_per_byte", "bpe1024")],
                                     "Tokenization at equal tokens/step", "training tokens", "val bits per byte"))
        obs.update({"byte_val_bpb": vb["bits_per_byte"], "bpe_val_bpb": vp["bits_per_byte"],
                    "bpe_bytes_per_token": vp["bytes_per_token"],
                    "bytes_seen_byte": b["summary"]["tokens_seen"],
                    "bytes_seen_bpe_approx": round(p["summary"]["tokens_seen"] * vp["bytes_per_token"]),
                    "params_byte": b["summary"]["params"], "params_bpe": p["summary"]["params"]})
        sep = ci_separated(vp["bits_per_byte_ci95"], vb["bits_per_byte_ci95"])
        rev = ci_separated(vb["bits_per_byte_ci95"], vp["bits_per_byte_ci95"])
        verdict = "supported" if sep else ("contradicted" if rev else "inconclusive")
        detail = f"BPE {vp['bits_per_byte']:.3f} vs byte {vb['bits_per_byte']:.3f} val bpb"
    elif name == "capacity":
        full = {s: by_name[f"lm-byte-{s}"] for s in ("tiny", "small", "medium")}
        result["plots"].append(_plot(out_dir / "full_data.png",
                                     [_curve(r, "val_bits_per_byte", f"{s} val") for s, r in full.items()]
                                     + [_curve(r, "train_eval_bits_per_byte", f"{s} train", "--") for s, r in full.items()],
                                     "Capacity on full corpus (equal tokens)", "training tokens", "bits per byte"))
        scarce = {s: by_name[f"scarce-byte-{s}"] for s in ("tiny", "medium")}
        result["plots"].append(_plot(out_dir / "scarce_data.png",
                                     [_curve(r, "val_loss", f"{s} val") for s, r in scarce.items()]
                                     + [_curve(r, "train_eval_loss", f"{s} train", "--") for s, r in scarce.items()],
                                     "Capacity on 240 KB sample", "training tokens", "loss (nats/token)"))
        bpb = {s: _val(r)["bits_per_byte"] for s, r in full.items()}
        obs["full_val_bpb"] = bpb
        ordered = bpb["medium"] < bpb["small"] < bpb["tiny"]
        med = _evals(scarce["medium"])
        best = min(med, key=lambda e: e["val_loss"])
        last = med[-1]
        gap = last["val_loss"] - best["val_loss"]
        train_fell = last["train_eval_loss"] < best["train_eval_loss"]
        obs["scarce_medium"] = {"best_val_loss": best["val_loss"], "best_step": best["step"],
                                "final_val_loss": last["val_loss"], "rise_after_best": gap,
                                "train_loss_at_best": best["train_eval_loss"], "final_train_loss": last["train_eval_loss"]}
        t_ev = _evals(scarce["tiny"])
        obs["scarce_tiny"] = {"best_val_loss": min(e["val_loss"] for e in t_ev), "final_val_loss": t_ev[-1]["val_loss"]}
        overfit = gap > 0.05 and train_fell
        verdict = "supported" if ordered and overfit else ("partially supported" if ordered or overfit else "not supported")
        detail = (f"full-data bpb tiny {bpb['tiny']:.3f} / small {bpb['small']:.3f} / medium {bpb['medium']:.3f}; "
                  f"scarce medium val loss rose {gap:+.3f} nats after step {best['step']}")
    elif name == "optimization":
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in ok.values():
            groups.setdefault(r["summary"]["run_id"].rsplit("-", 2)[0], []).append(r)
        stats = {}
        for g, rs in groups.items():
            vals = [_val(r)["bits_per_byte"] for r in rs]
            stats[g] = {"seeds": [r["config"]["train"]["seed"] for r in rs], "val_bpb": vals,
                        "mean": statistics.fmean(vals), "min": min(vals), "max": max(vals),
                        "stdev": statistics.stdev(vals) if len(vals) > 1 else None}
        obs["by_config"] = stats
        series = []
        for g, rs in groups.items():
            for r in rs:
                c = _curve(r, "val_bits_per_byte", f"{g} s{r['config']['train']['seed']}")
                series.append(c)
        result["plots"].append(_plot(out_dir / "seeds.png", series, "Learning rate / schedule, 3 seeds",
                                     "training tokens", "val bits per byte"))
        a, b_, c_ = stats["opt-lr3e-3-cos"], stats["opt-lr1e-3-cos"], stats["opt-lr3e-3-const-nowarm"]
        order_ok = a["mean"] < b_["mean"] and a["mean"] < c_["mean"]
        sep = a["max"] < b_["min"] and a["max"] < c_["min"]
        verdict = "supported" if order_ok and sep else ("partially supported" if order_ok else "not supported")
        detail = "; ".join(f"{g}: mean {s['mean']:.4f} [{s['min']:.4f}, {s['max']:.4f}]" for g, s in stats.items())
    elif name in ("digitsum", "digitsum-scratchpad"):
        res = {}
        for r in ok.values():
            key = f"{r['summary']['run_id'].rsplit('-', 2)[0]}-s{r['config']['train']['seed']}"
            fin = r["summary"]["final"]
            te = r.get("test_eval", {}).get("results")
            res[key] = {"train_acc": fin["train"]["accuracy"], "val_iid_acc": fin["val_iid"]["accuracy"],
                        "test_ood_near_acc": te["test_ood_near"]["accuracy"] if te else None,
                        "test_ood_far_acc": te["test_ood_far"]["accuracy"] if te else None}
        obs["accuracy"] = res
        result["plots"].append(_plot(out_dir / "val_iid.png",
                                     [{"x": [e["step"] for e in _evals(r)], "y": [e["val_iid_accuracy"] for e in _evals(r)],
                                       "label": f"{r['summary']['run_id'].rsplit('-', 2)[0]} s{r['config']['train']['seed']}"}
                                      for r in ok.values()],
                                     "Digit-sum: in-distribution validation accuracy", "step", "accuracy"))
        if any(v["test_ood_far_acc"] is None for v in res.values()):
            verdict, detail = "pending", "OOD splits are locked; run `morrowglass final-test` on these runs"
        else:
            plain = [v for k, v in res.items() if k.rsplit("-s", 1)[0].endswith("plain")]
            aug = {k.replace("posaug", "plain"): v for k, v in res.items() if "posaug" in k}
            plain_ok = all(v["val_iid_acc"] > 0.95 and v["test_ood_far_acc"] < 0.5 for v in plain)
            aug_better = all(aug[k]["test_ood_near_acc"] > res[k]["test_ood_near_acc"]
                             and aug[k]["test_ood_far_acc"] > res[k]["test_ood_far_acc"] for k in aug)
            verdict = "supported" if plain_ok and aug_better else (
                "partially supported" if plain_ok or aug_better else "not supported")
            detail = "; ".join(f"{k}: iid {v['val_iid_acc']:.3f} near {v['test_ood_near_acc']:.3f} "
                               f"far {v['test_ood_far_acc']:.3f}" for k, v in sorted(res.items()))
    elif name == "break-lr":
        br, f1, f2 = by_name["break-lr-broken"], by_name["break-lr-fix1-warmup-clip"], by_name["break-lr-fix2-lr3e-3"]

        def best_bpb(r: dict[str, Any]) -> float | None:
            ev = [e["val_bits_per_byte"] for e in _evals(r)]
            return min(ev) if ev else None

        obs.update({k: {"status": r["summary"]["status"], "failure": r["summary"]["failure"],
                        "best_val_bpb": best_bpb(r), "skipped_steps": r["summary"]["skipped_steps"],
                        "max_grad_norm": max((t["grad_norm"] for t in _trains(r) if t["grad_norm"] is not None),
                                             default=None)}
                    for k, r in (("broken", br), ("fix1_warmup_clip", f1), ("fix2_lr3e-3", f2))})
        result["plots"].append(_plot(out_dir / "train_loss.png",
                                     [{"x": [t["step"] for t in _trains(r)], "y": [t["train_loss"] for t in _trains(r)],
                                       "label": lab} for lab, r in (("lr 0.1, no warmup/clip", br),
                                                                    ("lr 0.1 + warmup + clip", f1),
                                                                    ("lr 3e-3 + warmup + clip", f2))],
                                     "Break the Model: learning rate", "step", "train loss (nats)"))
        b_best = obs["broken"]["best_val_bpb"]
        broken_bad = obs["broken"]["status"] == "diverged" or (b_best is not None and b_best > 3.0)
        f2_best = obs["fix2_lr3e-3"]["best_val_bpb"]
        fixed = f2_best is not None and (b_best is None or f2_best < b_best - 0.5)
        verdict = "supported" if broken_bad and fixed else "not supported"
        def _r(x: float | None) -> str:
            return "n/a" if x is None else f"{x:.3f}"

        detail = (f"broken: {obs['broken']['status']}, best val bpb {_r(b_best)}; warmup+clip at lr 0.1: "
                  f"{_r(obs['fix1_warmup_clip']['best_val_bpb'])}; lr 3e-3: {_r(f2_best)}")
    elif name in ("break-mask", "break-mask-long"):
        nc, c = by_name[f"{name}-noncausal"], by_name[f"{name}-causal"]
        ckroot = CHECKPOINT_ROOT if profile != "smoke" else SMOKE_DIR / "checkpoints"
        lf_nc = last_position_bpb(runs_dir / nc["summary"]["run_id"], ckroot)
        lf_c = last_position_bpb(runs_dir / c["summary"]["run_id"], ckroot)
        obs.update({"noncausal_last_train_loss": _trains(nc)[-1]["train_loss"],
                    "causal_last_train_loss": _trains(c)[-1]["train_loss"],
                    "noncausal_windowed_val_bpb_leaky": _val(nc)["bits_per_byte"],
                    "causal_windowed_val_bpb": _val(c)["bits_per_byte"],
                    "noncausal_leak_free_bpb": lf_nc["bits_per_byte"], "causal_leak_free_bpb": lf_c["bits_per_byte"],
                    "leak_free_positions": lf_c["positions"]})
        result["plots"].append(_plot(out_dir / "train_loss.png",
                                     [{"x": [t["step"] for t in _trains(r)], "y": [t["train_loss"] for t in _trains(r)],
                                       "label": lab} for lab, r in (("no causal mask", nc), ("causal mask", c))],
                                     "Break the Model: causal mask removed", "step", "train loss (nats)"))
        supported = (obs["noncausal_last_train_loss"] < obs["causal_last_train_loss"]
                     and lf_nc["bits_per_byte"] > lf_c["bits_per_byte"])
        verdict = "supported" if supported else "not supported"
        detail = (f"train loss non-causal {obs['noncausal_last_train_loss']:.3f} vs causal "
                  f"{obs['causal_last_train_loss']:.3f}; leak-free bpb {lf_nc['bits_per_byte']:.3f} vs "
                  f"{lf_c['bits_per_byte']:.3f}")
    elif name == "break-overfit":
        br, fx = by_name["scarce-byte-medium"], by_name["scarce-byte-medium-dropout-es"]
        result["plots"].append(_plot(out_dir / "curves.png",
                                     [_curve(br, "val_loss", "no regularisation: val"),
                                      _curve(br, "train_eval_loss", "no regularisation: train", "--"),
                                      _curve(fx, "val_loss", "dropout 0.2 + early stop: val"),
                                      _curve(fx, "train_eval_loss", "dropout 0.2 + early stop: train", "--")],
                                     "Break the Model: memorising 240 KB", "training tokens", "loss (nats/token)"))
        bb = min(e["val_loss"] for e in _evals(br))
        fb = min(e["val_loss"] for e in _evals(fx))
        obs.update({"broken_best_val_loss": bb, "broken_final_val_loss": _evals(br)[-1]["val_loss"],
                    "fix_best_val_loss": fb, "fix_status": fx["summary"]["status"],
                    "fix_steps": fx["summary"]["steps_completed"], "broken_steps": br["summary"]["steps_completed"]})
        verdict = "supported" if fb < bb and fx["summary"]["status"] == "early_stopped" else (
            "partially supported" if fb < bb else "not supported")
        detail = (f"best val loss {bb:.3f} -> {fb:.3f}; fix status {fx['summary']['status']} at step "
                  f"{fx['summary']['steps_completed']}")
    result["verdict"] = verdict
    result["verdict_detail"] = detail
    note = out_dir / "note.md"
    result["researcher_note"] = note.read_text() if note.exists() else None
    write_json(out_dir / "result.json", result)
    return result


LORA_HYPOTHESIS = (
    "Written 2026-11-08 while the approved run was in progress, before any result was seen: LoRA adapters "
    "(r=8, alpha 16, on the attention c_attn projections; 300 steps x 8 x 256 GPT-2 tokens, lr 2e-4) lower "
    "distilgpt2's bits-per-byte on the held-out Shakespeare validation plays relative to the frozen model.")
LORA_CRITERION = "LoRA val bpb upper 95% CI < frozen val bpb lower 95% CI (full validation split)."


def analyze_lora(model_name: str = "distilgpt2", seed: int = 0) -> dict[str, Any]:
    src = REPORTS / "lora" / f"{model_name}-s{seed}.json"
    out_dir = REPORTS / "experiments" / "lora"
    result: dict[str, Any] = {
        "experiment": "lora", "title": f"Pretrained adaptation: {model_name} + LoRA vs frozen", "track": "pretrained",
        "hypothesis": LORA_HYPOTHESIS, "criterion": LORA_CRITERION,
        "caveats": ["distilgpt2 was pretrained on OpenWebText, which may contain Shakespeare: weaker held-out guarantee.",
                    "GPT-2 tokenizer and 256-token context (~4x the bytes of the byte models' context).",
                    "One seed; validation only; the locked test split is untouched."],
        "arms": [str(src.relative_to(REPO_ROOT))], "missing_runs": [], "analyzed": now_iso(),
        "analysis_git": git_provenance()["commit"], "plots": [], "observations": {},
    }
    if not src.exists():
        result.update({"verdict": "pending", "verdict_detail": "needs owner approval for the model download",
                       "missing_runs": [str(src.relative_to(REPO_ROOT))]})
    else:
        r = read_json(src)
        fz, lo = r["frozen_val"], r["lora_val"]
        result["observations"] = {k: r[k] for k in ("frozen_val", "lora_val", "delta_bpb", "trainable_params",
                                                     "total_params", "steps", "train_tokens", "train_seconds",
                                                     "device", "memory_mb")}
        sc = REPORTS / "runs" / "lm-bpe1024-small-eb29b8fb-s0" / "summary.json"
        if sc.exists():
            result["observations"]["scratch_best_val_bpb"] = read_json(sc)["final"]["val"]["bits_per_byte"]
        result["plots"].append(_plot(out_dir / "lora_loss.png",
                                     [{"x": list(range(0, r["steps"], max(1, r["steps"] // 50)))[: len(r["loss_curve"])],
                                       "y": [v / 1.0 for v in r["loss_curve"]], "label": "LoRA train loss (nats/GPT-2 token)"}],
                                     f"{model_name} + LoRA training loss", "step", "loss"))
        sep = ci_separated(lo["bits_per_byte_ci95"], fz["bits_per_byte_ci95"])
        result["verdict"] = "supported" if sep else ("contradicted" if ci_separated(fz["bits_per_byte_ci95"], lo["bits_per_byte_ci95"]) else "inconclusive")
        result["verdict_detail"] = (f"frozen {fz['bits_per_byte']:.3f} {[round(x, 3) for x in fz['bits_per_byte_ci95']]} -> "
                                    f"LoRA {lo['bits_per_byte']:.3f} {[round(x, 3) for x in lo['bits_per_byte_ci95']]} val bpb; "
                                    f"{r['trainable_params']:,} of {r['total_params']:,} params trained")
    note = out_dir / "note.md"
    result["researcher_note"] = note.read_text() if note.exists() else None
    write_json(out_dir / "result.json", result)
    return result


def analyze_all(which: str = "all", profile: str = "standard") -> dict[str, Any]:
    if which == "lora":
        r = analyze_lora()
        return {"lora": {"verdict": r["verdict"], "detail": r.get("verdict_detail")}}
    names = ALL if which == "all" else [which]
    out = {}
    if which == "all" and profile == "standard":
        lr_ = analyze_lora()
        out["lora"] = {"verdict": lr_["verdict"], "detail": lr_.get("verdict_detail")}
    for n in names:
        r = analyze(n, profile)
        out[n] = {"verdict": r["verdict"], "detail": r.get("verdict_detail")}
    return out
