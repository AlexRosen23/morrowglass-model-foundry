# Morrowglass Model Foundry

A reproducible miniature language-model laboratory: a decoder-only Transformer written in plain
PyTorch, trained from scratch on public-domain Shakespeare, measured against strong count-based
baselines, and studied with pre-registered ablations. Every number in the workbench and the docs
is read from a saved run record.

> CI: `.github/workflows/ci.yml` runs on GitHub Actions for pull requests and pushes to `main`
> (enabled with the owner's approval on 2026-11-08). `make ci` runs the same checks locally.

## Results at a glance

These are standard-profile runs on an Apple M5 Pro (MPS), with full run ids in [docs/RESULTS.md](docs/RESULTS.md).
Bits per UTF-8 byte (bpb) on held-out plays; lower is better. Intervals are 95% bootstrap CIs over evaluation windows.

| model | params | val bpb | test bpb (opened once, logged) |
|---|---|---|---|
| Kneser-Ney 4-gram on BPE-1024 tokens (best baseline, chosen on val) | — | 2.107 [2.095, 2.120] | 2.191 [2.176, 2.206] |
| Transformer, byte-level, small (pre-registered flagship) | 1.87M | 2.086 [2.074, 2.098] | 2.171 [2.158, 2.183] |
| same flagship, extended budget (10k steps, owner-approved) | 1.87M | 1.993 [1.981, 2.006] | 2.089 [2.075, 2.103] |
| Transformer, byte-level, medium | 4.85M | 2.037 [2.025, 2.048] | 2.141 [2.128, 2.154] |
| Transformer, BPE-1024, small (best scratch model on validation) | 2.02M | **2.007** [1.993, 2.021] | **2.081** [2.064, 2.095] |
| distilgpt2, pretrained, frozen | 82.1M | 1.941 [1.927, 1.955] | not opened |
| distilgpt2 + LoRA (r=8, 147K trained params, 300 steps) | 82.1M | **1.760** [1.748, 1.773] | not opened |

The pretrained rows use a different tokenizer and context, and their pretraining data (OpenWebText) may
contain Shakespeare. bpb keeps the units comparable, but the held-out guarantee is weaker.

| experiment (pre-registered) | verdict |
|---|---|
| scratch: flagship beats best n-gram with non-overlapping CIs | **not supported** at the standard budget (intervals overlap); **supported** at the extended budget (1.993 vs 2.107; reported as a separate finding) |
| tokenization: BPE-1024 beats byte at equal steps | supported (not apples-to-apples in bytes seen; stated up front) |
| capacity: bigger is better on 4.5 MB; medium overfits 240 KB | supported (val loss +1.26 nats after its minimum) |
| optimization: warmup + cosine beats the alternatives across 3 seeds | **not supported**: constant lr matched it; lr 1e-3 clearly worse |
| digit-sum mod 10 under length shift | **not supported**: stuck at chance (loss = ln 10); pipeline control test passes |
| digit-sum scratchpad follow-up (registered after that failure) | 100% in distribution for all 4 runs; plain positions at chance on 8-16 terms; position augmentation transferred for 1 of 2 seeds (88.8% on 12-16 terms) |
| break-lr / break-overfit | supported: failure reproduced, fix measured |
| break-mask: no causal mask → training-loss collapse within 400 steps | **not supported** at 400 steps |
| LoRA: adapters lower distilgpt2's val bpb vs frozen (non-overlapping CIs) | supported: 1.941 → 1.760 |
| break-mask-long follow-up (2,000 steps) | supported: leaky val 0.0135 bpb looks perfect; leak-free scoring shows 3.38 vs 2.57 for the causal model |

Negative results stay in the report; see [docs/FAILURE_ANALYSIS.md](docs/FAILURE_ANALYSIS.md).

## Quick start

```bash
make setup     # uv sync (Python 3.12, torch 2.15.0) + npm ci
make smoke     # ~1 min, offline sample, CPU: sanity check + tiny experiments
make demo      # builds the UI; API on http://127.0.0.1:5100, workbench on http://127.0.0.1:5101
```

The demo shows the full measured ledger from `reports/` straight away. Checkpoints are not in git (size),
so live generation and attention maps use whichever checkpoints exist locally: the smoke checkpoint
after `make smoke`, or every standard run after `make data && make standard`.

## Reproduce the evidence

```bash
make data                                # downloads Project Gutenberg #100 once (5.6 MB), builds splits
uv run morrowglass estimate capacity     # measured time / memory / disk before running anything
make standard                            # every standard experiment, ~45-60 min on Apple silicon (idempotent)
make report                              # verdicts, plots, docs/RESULTS.md, HTML report, resume_evidence/metrics.json
```

Single pieces:

```bash
uv run morrowglass sanity
uv run morrowglass baselines
uv run morrowglass run scratch --profile standard
uv run morrowglass final-test --runs <run_id> --reason "<why>"
uv run morrowglass bench <run_id>
uv run morrowglass export-artifact <run_id>
```

The `extended` profile and the pretrained-LoRA track (`uv run morrowglass lora --approved`, which downloads
distilgpt2, ~350 MB) refuse to run without `--approved`. Run without it, they print their cost first.

## What's inside

```
src/morrowglass/
  model/transformer.py     decoder-only Transformer in native PyTorch (+ shape walkthrough, params, FLOPs)
  data/                    Gutenberg pipeline (split by play), synthetic digit-sum, batching
  tokenizer/               byte + byte-level BPE (HF tokenizers, fitted on train only)
  train/                   trainer: seeds, AdamW, warmup/cosine, accumulation, clipping, early stopping,
                           non-finite handling, sha256-verified checkpoints, resume
  eval/                    bits/byte, bootstrap CIs, calibration (ECE), Kneser-Ney baselines
  experiments/             pre-registered registry → runner → analysis → report; locked final test
  inference/               sampling with per-token scores, attention maps, latency benchmark
  finetune/lora.py         pretrained distilgpt2 + LoRA vs frozen (executed with owner approval)
  api/                     FastAPI (127.0.0.1:5100) + derived SQLite ledger index
  contracts.py             portable model-artifact manifest (contracts/model_artifact_v1.schema.json)
web/                       Vite + React + TypeScript workbench (127.0.0.1:5101)
reports/runs/<run_id>/     config, metadata (commit, seed, hashes, hardware), metrics.jsonl, summary
reports/experiments/<exp>/ result.json (verdict), plots, researcher note
```

The workbench has three panels:
- **Model blueprint**: layer shapes and parameter counts.
- **Evidence**: overlaid curves with exact config diffs, held-out metrics, reliability diagram, findings, Break the Model, live generation next to the n-gram baseline, attention heads.
- **Ledger**: seed, commit, config hash, the test-access log.

Badges mark each item as `MEASURED`, `SMOKE` or `LIVE SAMPLE`. Keyboard: `j/k`, `space`, `1-6`, `t`, `?`.

## Guarantees backed by tests (`make test`: 87 passing, see `reports/test_summary.json`)

- The causal mask is enforced: changing future tokens never changes earlier logits. A mask-less model fails the same test.
- The analytic parameter count equals the real one. The fused and explicit attention paths agree.
- A single fixed batch can be overfit, and that check passes before any experiment.
- Tokenizer round-trips are lossless. The bits-per-byte maths is checked. Kneser-Ney distributions sum to 1. ECE behaves on known cases.
- Splits are disjoint by work. Synthetic expressions never cross splits. Training never touches test.
- Checkpoint save → load gives identical logits. A one-bit corruption is rejected. Resume equals uninterrupted training bit-for-bit on CPU.
- Divergence is detected rather than crashing. Early stopping and cancellation work.
- Generation is deterministic under a seed. API responses equal the files on disk. Model-artifact exports validate against the schema.

## Hardware and ports

- Developed on an Apple M5 Pro (18 cores, 48 GB) with PyTorch MPS. `device=auto` uses MPS only if a real op succeeds, otherwise CPU. There is no CUDA path.
- Standard runs take 15 s to 11 min each on this shared machine.
- Ports: API **5100**, web **5101**. Both were free on 2026-11-08; override with `API_PORT` / `WEB_PORT`.

## Open gaps (not done, on purpose or for lack of approval)

- **LoRA**: one seed, validation only. Full fine-tuning and other ranks have not been compared.
- **Extended profile**: run once for the flagship (owner-approved). Other experiments were not re-run at that budget.
- **Seeds**: one seed per main-track arm, three in the optimization study.
- **Reproducibility on MPS**: runs are not bit-reproducible. CPU resume is.
- **Scope**: validation and test are three plays each, from a single author.
- **Constellation**: not attached; independent verification comes first.

## Docs

- Research framing:
  - [Research question](docs/RESEARCH_QUESTION.md)
  - [Dataset card](docs/DATASET_CARD.md)
  - [Model card](docs/MODEL_CARD.md)
  - [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- Results:
  - [Results](docs/RESULTS.md) (generated)
  - [Experiment report](docs/EXPERIMENT_REPORT.md) (generated)
  - [Ablations](docs/ABLATIONS.md)
  - [Failure analysis](docs/FAILURE_ANALYSIS.md)
- Design:
  - [Architecture](docs/ARCHITECTURE.md)
  - [Engineering decisions](docs/ENGINEERING_DECISIONS.md)
  - [Model-artifact contract](docs/INTEGRATION_MODEL_ARTIFACT.md)
- Career materials:
  - [Interview defence](docs/INTERVIEW_DEFENSE.md)
  - [Recruiter demo (90 s)](docs/RECRUITER_DEMO.md)
  - [Recruiter case study](docs/RECRUITER_CASE_STUDY.md)
  - [Resume bullets](docs/RESUME_BULLETS.md)
- [Contributing](CONTRIBUTING.md)

## License

Code: MIT. Training text: public domain (Project Gutenberg eBook #100, boilerplate removed).
