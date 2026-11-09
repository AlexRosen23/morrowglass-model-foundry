"""Command-line entry point: every experiment is reproducible without the UI.

    morrowglass hw                         # hardware actually detected
    morrowglass data prepare [--fetch]     # build splits + manifest + offline sample
    morrowglass sanity                     # one-batch overfit check
    morrowglass baselines [--dataset D]    # uniform / unigram / Kneser-Ney (val only)
    morrowglass estimate EXP [--profile P] # measured time / memory / disk before running
    morrowglass run EXP [--profile P]      # train every arm of a pre-registered experiment
    morrowglass analyze [EXP|all]          # verdicts, plots and result.json
    morrowglass final-test --runs A,B --reason "..."  # open the locked test split (logged)
    morrowglass bench RUN_ID               # inference latency / throughput / memory
    morrowglass bench-kv RUN_ID            # KV cache on/off: equivalence + decode speed (registered study)
    morrowglass generate RUN_ID "prompt"   # sample from a checkpoint
    morrowglass export-artifact RUN_ID     # contracts/model_artifact_v1 manifest
    morrowglass export-hf RUN_ID           # Hugging Face Hub-compatible folder (local save only)
    morrowglass lm-eval [--models A,B]     # lm-evaluation-harness through /v1/completions
    morrowglass quant-int8 RUN_ID          # dynamic int8 on CPU: val bpb + decode speed
    morrowglass lora [--approved]          # optional LoRA track (blocked without approval)
    morrowglass ledger                     # rebuild the SQLite index of runs
    morrowglass serve [--port 5100] [--model RUN_ID]  # workbench API + OpenAI-compatible /v1 (loopback only)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from typing import Any

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="morrowglass", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hw")
    d = sub.add_parser("data")
    d.add_argument("action", choices=["prepare"])
    d.add_argument("--fetch", action="store_true", help="download pg100.txt (5.6 MB) if missing")
    sub.add_parser("sanity")
    b = sub.add_parser("baselines")
    b.add_argument("--dataset", default="shakespeare")
    for name in ("estimate", "run"):
        p = sub.add_parser(name)
        p.add_argument("experiment")
        p.add_argument("--profile", default="standard", choices=["smoke", "standard", "extended"])
        p.add_argument("--device", default="auto")
        if name == "run":
            p.add_argument("--approved", action="store_true", help="owner approved an extended run")
            p.add_argument("--only", default=None, help="substring filter on arm names")
            p.add_argument("--force", action="store_true", help="re-train even if the run record exists")
            p.add_argument("--overwrite-evidence", action="store_true",
                           help="with --force on standard/extended: really replace the committed run record")
            p.add_argument("--quiet", action="store_true")
    rp = sub.add_parser("reproduce", help="re-train a committed run into reports/reproduce/ and compare")
    rp.add_argument("run_id")
    rp.add_argument("--device", default="auto")
    rp.add_argument("--quiet", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("experiment", nargs="?", default="all")
    a.add_argument("--profile", default="standard")
    ft = sub.add_parser("final-test")
    ft.add_argument("--runs", required=True)
    ft.add_argument("--reason", required=True)
    bn = sub.add_parser("bench")
    bn.add_argument("run_id")
    bk = sub.add_parser("bench-kv", help="KV cache equivalence + decode speed (EVALUATIONS['kv-cache'])")
    bk.add_argument("run_id")
    bk.add_argument("--devices", default=None, help="comma list, default cpu + mps when usable")
    g = sub.add_parser("generate")
    g.add_argument("run_id")
    g.add_argument("prompt")
    g.add_argument("--max-new", type=int, default=200)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=40)
    g.add_argument("--seed", type=int, default=0)
    ex = sub.add_parser("export-artifact")
    ex.add_argument("run_id")
    qi = sub.add_parser("quant-int8", help="dynamic int8 on CPU vs fp32 (EVALUATIONS['int8-dynamic'])")
    qi.add_argument("run_id")
    eh = sub.add_parser("export-hf", help="Hub-compatible folder (safetensors + config + model card); local only")
    eh.add_argument("run_id")
    lo = sub.add_parser("lora", help="optional pretrained + LoRA track (needs --approved: model download)")
    lo.add_argument("--model", default="distilgpt2")
    lo.add_argument("--steps", type=int, default=300)
    lo.add_argument("--approved", action="store_true")
    le = sub.add_parser("lm-eval", help="lm-evaluation-harness via /v1/completions (EVALUATIONS['lm-eval-harness'])")
    le.add_argument("--models", default=None, help="comma list (default: every registered model)")
    le.add_argument("--limit", type=int, default=None, help="docs per task; plumbing check only (artifacts/)")
    le.add_argument("--port", type=int, default=5102)
    le.add_argument("--summarize-only", action="store_true")
    sub.add_parser("serving-report", help="docs/SERVING_RESULTS.md from the saved serving/eval records")
    sub.add_parser("ledger")
    sub.add_parser("report")
    sv = sub.add_parser("serve")
    sv.add_argument("--port", type=int, default=5100)
    sv.add_argument("--host", default="127.0.0.1", help="loopback only: 127.0.0.1, ::1 or localhost")
    sv.add_argument("--model", default=None, help="default model for /v1/* and /tokenize (a run id or hf:distilgpt2)")
    sv.add_argument("--enable-hf", action="store_true",
                    help="also serve hf:distilgpt2 from the local Hugging Face cache (never downloads)")
    args = ap.parse_args(argv)

    if args.cmd == "hw":
        from morrowglass.hardware import hardware_report

        _print(hardware_report())
    elif args.cmd == "data":
        from morrowglass.data import corpus

        if not corpus.RAW_PATH.exists():
            if not args.fetch:
                print(f"{corpus.RAW_PATH} missing; re-run with --fetch to download {corpus.SOURCE_URL} (5.6 MB)")
                return 2
            corpus.fetch_raw()
        m = corpus.prepare()
        corpus.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        committed = corpus.MANIFEST_DIR / "shakespeare.json"
        same = committed.exists() and corpus.same_manifest(json.loads(committed.read_text()), m)
        if same:
            print(f"splits match the committed manifest ({committed.name}); left unchanged")
        else:
            if committed.exists():
                print(f"WARNING: rebuilt splits differ from the committed {committed.name}; it has been replaced. "
                      "Run records made with the old splits are no longer comparable.", file=sys.stderr)
            shutil.copyfile(corpus.PROCESSED_DIR / "manifest.json", committed)
        corpus.make_sample()
        _print({k: {"bytes": v["bytes"], "works": len(v["works"])} for k, v in m["files"].items()})
    elif args.cmd == "sanity":
        from morrowglass.experiments.runner import overfit_sanity

        r = overfit_sanity()
        _print({k: v for k, v in r.items() if k != "loss_curve"})
        return 0 if r["passed"] else 1
    elif args.cmd == "baselines":
        from morrowglass.experiments.runner import REPORTS, SMOKE_DIR, run_baselines

        out = run_baselines(args.dataset, SMOKE_DIR if args.dataset.endswith("sample") else REPORTS)
        for r in out["results"]:
            print(f"{r['tokenizer']:8s} {r['model']:16s} val bpb {r['val']['bits_per_byte']:.4f}")
        print(f"best: {out['best']} -> {out['best_val_bpb']:.4f}")
    elif args.cmd == "estimate":
        from morrowglass.experiments.registry import build
        from morrowglass.experiments.runner import estimate

        _print(estimate(build(args.experiment, args.profile), args.device))
    elif args.cmd == "run":
        from morrowglass.experiments.runner import run_experiment

        try:
            sums = run_experiment(args.experiment, args.profile, args.approved, args.only, args.device,
                                  args.force, args.quiet, args.overwrite_evidence)
        except PermissionError as e:
            print(f"BLOCKED: {e}", file=sys.stderr)
            return 3
        for s in sums:
            print(f"{s['run_id']}: {s['status']} ({s['wall_time_s']}s)")
    elif args.cmd == "reproduce":
        from morrowglass.experiments.runner import reproduce_run

        r = reproduce_run(args.run_id, args.device, args.quiet)
        _print({k: r[k] for k in r if k in ("run_id", "val_bpb", "val_iid_accuracy", "verdict", "criterion")})
        print(f"report: reports/reproduce/{args.run_id}.json (the committed record was not modified)")
    elif args.cmd == "analyze":
        from morrowglass.experiments.analyze import analyze_all

        _print(analyze_all(args.experiment, args.profile))
    elif args.cmd == "final-test":
        from morrowglass.experiments.runner import final_test

        _print(final_test([r for r in args.runs.split(",") if r], args.reason))
    elif args.cmd == "bench":
        from morrowglass.inference.bench import bench_run

        _print(bench_run(args.run_id))
    elif args.cmd == "bench-kv":
        from morrowglass.inference.bench import bench_kv_cache

        r = bench_kv_cache(args.run_id, devices=args.devices.split(",") if args.devices else None)
        _print({k: r[k] for k in ("run_id", "speedup_tokens_per_s", "checks", "verdict")})
    elif args.cmd == "generate":
        from morrowglass.api.store import Store

        model, tok = Store().model(args.run_id)
        from morrowglass.inference.generate import generate

        r = generate(model, tok, args.prompt, max_new_tokens=args.max_new, temperature=args.temperature,
                     top_k=args.top_k, seed=args.seed)
        print(args.prompt + r["text"])
    elif args.cmd == "export-artifact":
        from morrowglass.contracts import export_artifact

        _print(export_artifact(args.run_id))
    elif args.cmd == "quant-int8":
        from morrowglass.inference.quant import bench_int8

        r = bench_int8(args.run_id)
        _print({k: r[k] for k in ("run_id", "fp32", "int8", "delta_val_bpb", "speedup_tokens_per_s", "checks",
                                  "verdict")})
    elif args.cmd == "export-hf":
        from morrowglass.hub import export_hf

        info = export_hf(args.run_id)
        _print({k: info[k] for k in ("run_id", "files", "roundtrip_max_abs_logit_diff",
                                     "publish_command_after_owner_approval")})
        print("NOT published. Publishing needs the owner's approval; the command above is what they would run.")
    elif args.cmd == "lora":
        from morrowglass.finetune.lora import run_lora

        try:
            _print(run_lora(args.model, approved=args.approved, steps=args.steps))
        except PermissionError as e:
            print(f"BLOCKED: {e}", file=sys.stderr)
            return 3
    elif args.cmd == "lm-eval":
        from morrowglass.experiments.registry import EVALUATIONS
        from morrowglass.serving.harness import run_model, summarize

        if not args.summarize_only:
            models = args.models.split(",") if args.models else list(EVALUATIONS["lm-eval-harness"].runs)
            for model_id in models:
                print(f"{model_id}: running lm-eval through 127.0.0.1:{args.port} ...", flush=True)
                _print(run_model(model_id, args.port, args.limit))
        if not args.limit:
            s = summarize()
            _print({k: s[k] for k in ("checks", "verdict", "missing")})
    elif args.cmd == "serving-report":
        from morrowglass.serving.report import build_serving_report

        print(build_serving_report())
    elif args.cmd == "ledger":
        from morrowglass.api.store import Store

        n = Store().rebuild_index()
        print(f"indexed {n} runs")
    elif args.cmd == "report":
        from morrowglass.experiments.report import build_report

        print(build_report())
    elif args.cmd == "serve":
        import os

        import uvicorn

        if args.host not in LOOPBACK_HOSTS:
            print(f"refusing to bind {args.host!r}: this API is loopback-only ({', '.join(sorted(LOOPBACK_HOSTS))})",
                  file=sys.stderr)
            return 2
        if args.model:
            os.environ["MORROWGLASS_DEFAULT_MODEL"] = args.model
        if args.enable_hf:
            os.environ["MORROWGLASS_ENABLE_HF"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
        uvicorn.run("morrowglass.api.app:app", host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
