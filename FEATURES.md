# Morrowglass Model Foundry: features

Morrowglass is a small language-model laboratory that runs on a laptop. It has a decoder-only Transformer written in plain PyTorch, a Shakespeare data pipeline split by play, strong count-based baselines, a set of pre-registered experiments, and a serving layer (KV cache, OpenAI-compatible `/v1` endpoints, the EleutherAI lm-evaluation-harness, a Hugging Face Hub-format export). It is for engineers who want to see how a small model behaves when data and compute are tiny, and how to measure and serve it without fooling themselves. Its best feature is that every number it shows is read from a saved record that names its commit, seed, data hash and hardware.

**This is the source-of-truth features list.** Every PR that adds, changes, removes or switches off something a person can see or use updates this file in the same PR (see "Keeping this file true").

- **Repo:** AlexRosen23/morrowglass-model-foundry (public) · **Folder:** `~/Desktop/Claude Projects/Morrowglass Model Foundry`
- **Status:** Working. All 14 research experiments and all 4 serving/eval studies have run and are analysed. Nothing is deployed; everything runs on 127.0.0.1. · **Version:** none tagged · **`main`:** `42c0a4f` (2026-11-15)
- **Last updated:** 2026-11-15, written from the code, README, docs and tests.

**Status words** (a feature with no status word is live):

| Word | Means |
| --- | --- |
| *Live* | on `main` and working (everything runs locally; there is no deployment) |
| *Off by default* | built, behind a flag or an approval gate |
| *Waiting on owner* | built, blocked on an owner action |
| *Planned* | in the README's "next steps", not built |
| *Not built* | left out on purpose |

## At a glance

| | |
| --- | --- |
| Stack | Python 3.12, PyTorch 2.15 (MPS or CPU), Hugging Face `tokenizers`, FastAPI, SQLite (derived index only), safetensors, lm-eval 0.4.13 (separate venv), Vite + React + TypeScript, `uv`, `pytest`, `ruff`, `mypy` |
| Ports | API 127.0.0.1:5100, workbench 127.0.0.1:5101, lm-eval harness endpoint 5102 |
| Python source | 6,772 lines in `src/morrowglass/` (37 files) |
| Web source | 1,314 lines in `web/src/` (7 files) |
| Tests | 114 test functions in 17 files (139 collected by pytest per the README, because some are parametrised) |
| Experiments | 12 registered experiments (14 verdicts with the two seed follow-ups) and 4 registered serving/eval studies |
| Committed evidence | 40 run records in `reports/runs/`, 14 experiment verdicts, 17 verified claims in `resume_evidence/metrics.json` |
| Best result | BPE-1024 small model: 2.007 bits per byte on validation, against 2.107 for the best n-gram baseline |
| PRs merged | 12 (#1 to #12), 2026-11-09 to 2026-11-11 |
| CI | GitHub Actions, Linux CPU runners: lint, types, tests, smoke, web build |
| Data | Project Gutenberg #100 (public domain); a 320 KB offline sample is committed |
| Paid services / API keys | None |

## Contents

1. [What it is and who it's for](#1-what-it-is-and-who-its-for)
2. [How to run it](#2-how-to-run-it)
3. [Architecture at a glance](#3-architecture-at-a-glance)
4. [Features](#4-features)
5. [Integrations](#5-integrations)
6. [Data and privacy](#6-data-and-privacy)
7. [Tests and evidence](#7-tests-and-evidence)
8. [Known limits and gaps](#8-known-limits-and-gaps)
9. [Change log](#9-change-log)
10. [Keeping this file true](#keeping-this-file-true)

## 1. What it is and who it's for

- **The question it answers.** On a fixed public-domain corpus and a laptop-scale budget, how much does a small Transformer written from scratch improve on a strong count-based language model? Which choices (tokenization, capacity, optimisation) move the result by more than run-to-run noise? (`docs/RESEARCH_QUESTION.md`)
- **The second question.** Does a model built this way hold up behind production interfaces: an OpenAI-compatible API, a standard eval harness, a Hub-format artifact and the usual latency tricks?
- **Who it is for.** Engineers and reviewers who want to see honest measurement of a small model, and anyone who wants a worked example of pre-registered experiments.
- **What you can do with it:**
  - reproduce the experiments from the command line;
  - browse 40 recorded runs and 14 experiment verdicts in a browser workbench;
  - sample from a checkpoint next to an n-gram baseline;
  - inspect attention heads;
  - watch a model being broken on purpose and then fixed;
  - call a trained checkpoint with any OpenAI client.
- **What it is not.** It is not a production serving stack (single process, batch 1, loopback only), not a CUDA project, and not a general language model. The models are 0.27M to 4.85M parameters trained on Shakespeare.

## 2. How to run it

**Prerequisites**
- `uv` 0.12 or newer (it installs Python 3.12), Node 24 or newer, free ports 5100 and 5101.
- macOS on Apple silicon (MPS), or any CPU machine. CUDA is not used.
- No API keys and no paid services. Setup needs the network only for packages.

**Commands**

| Command | What it does |
| --- | --- |
| `make setup` | `uv sync --frozen` plus `npm ci` in `web/` |
| `make data` | Downloads Gutenberg #100 once (5.6 MB), builds the splits, manifest and offline sample |
| `make smoke` | About 1 minute, offline, CPU: sanity check, sample baselines, 3 tiny experiments. Ends with `smoke OK` |
| `make demo` | Builds the UI, then serves the API on 5100 and the workbench on 5101 (`scripts/demo.sh`) |
| `make api` / `make web` | The API alone, or the Vite dev server alone |
| `make test` / `make test-fast` | Full pytest suite (writes `reports/test_summary.json`) / model, data and metric tests only |
| `make lint` / `make typecheck` / `make web-check` | ruff / mypy / TypeScript typecheck plus production build |
| `make ci` | Everything the GitHub Actions workflow runs |
| `make standard` | Every standard-profile experiment (about 45 to 60 minutes on Apple silicon, idempotent) |
| `make analyze` / `make report` | Verdicts and plots from saved runs / `docs/RESULTS.md`, HTML report, `resume_evidence/metrics.json` |
| `make serve-openai` | API plus `/v1` with the flagship as the default model (needs its local checkpoint) |
| `make bench-kv`, `make quant-int8` | The registered KV-cache and int8 studies |
| `make lm-eval-setup`, `make lm-eval` | Pinned harness venv, then the harness through `/v1/completions` (about 2.5 hours for all models) |
| `make export-hf` | Hub-compatible folders, local only |
| `make serving-report` | Regenerates `docs/SERVING_RESULTS.md` |
| `make paired` | Paired block bootstrap against the best n-gram (CPU, about 3 minutes) |
| `make reproduce-scratch` | Re-trains the flagship into `reports/reproduce/` and compares it with the record (5 to 8 minutes) |
| `make clean` / `make distclean` | Remove caches and smoke output / also delete local checkpoints and processed data (5-second abort window) |

**Environment variables** (names only; none is a secret)
- `API_PORT`, `WEB_PORT`: move the API and web preview. `demo.sh` refuses to start on a busy port.
- `MORROWGLASS_WARM_BASELINE=0`: skip fitting the n-gram baseline when the API starts.
- `MORROWGLASS_DEFAULT_MODEL`: default model for `/v1` and `/tokenize` (same as `serve --model`).
- `MORROWGLASS_ENABLE_HF=1`: also serve `hf:distilgpt2` from the local Hugging Face cache only (same as `serve --enable-hf`).

**On a fresh clone** the ledger, curves, findings and Serving tables work from committed records. Generate, Attention and `/v1` need a local checkpoint; `make reproduce-scratch` makes one (labelled as re-trained). Without it those tabs fall back to a labelled smoke checkpoint.

## 3. Architecture at a glance

Data flows in one direction: corpus, tokenizer, trainer, checkpoint, record, analysis, API, workbench.

| Area | Folder | Role |
| --- | --- | --- |
| Model | `src/morrowglass/model/` | `MorrowglassLM`, parameter count, FLOPs estimate, blueprint for the UI |
| Data | `src/morrowglass/data/` | Gutenberg pipeline split by play, synthetic digit-sum, token cache and batching |
| Tokenizers | `src/morrowglass/tokenizer/` | Byte and byte-level BPE behind one protocol |
| Training | `src/morrowglass/train/` | `trainer.py`; `tasks.py` (language-model and digit-sum tasks; test split locked) |
| Evaluation | `src/morrowglass/eval/` | `metrics.py` (bpb, bootstrap CIs, ECE), `ngram.py` (uniform, unigram, Kneser-Ney), `paired.py` |
| Experiments | `src/morrowglass/experiments/` | `registry.py` then `runner.py` then `analyze.py` then `report.py` |
| Inference | `src/morrowglass/inference/` | `generate.py` (cached decoder, sampling, attention maps), `bench.py`, `quant.py` |
| Serving | `src/morrowglass/serving/` | `openai_api.py`, `backends.py`, `harness.py`, `report.py` |
| API | `src/morrowglass/api/` | `app.py` (FastAPI), `store.py` (SQLite index), `contract.py` (read-only contract) |
| Other | `hub.py`, `contracts.py`, `finetune/lora.py`, `cli.py`, `repro.py`, `hardware.py` | Hub export, artifact manifest, LoRA track, CLI, verified writes, device detection |
| Workbench | `web/src/` | `App.tsx`, `panels.tsx`, `charts.tsx` (hand-written SVG), `paired.tsx`, `api.ts` |
| Evidence | `reports/`, `resume_evidence/`, `data/manifests/` | Run records, verdicts, serving records, hashes |
| Docs | `docs/` | 20 documents plus `docs/media/` |

**Records.** Each run writes `reports/runs/<run_id>/` with config, `metadata.json` (commit, command, seed, dataset and tokenizer hashes, hardware), `metrics.jsonl` and `summary.json`. The run id is the name plus a config hash plus the seed. Checkpoints go to `artifacts/` (not in git) with a `.sha256.json` beside them. `artifacts/ledger.sqlite` is a derived index rebuilt from the JSON.

## 4. Features

### Model and tokenizers

- **MorrowglassLM**: a pre-LayerNorm GPT-style decoder in native PyTorch. Position options are learned, sinusoidal or none; embeddings are tied; both a fused attention path and an explicit one exist, and a test proves they agree. `src/morrowglass/model/transformer.py`
- **Model sizes**: nano (1 layer, d_model 32, context 64), tiny (2 layers, 96), small (4 layers, 4 heads, 192, 1.87M parameters), medium (6 layers, 8 heads, 256, 4.85M). Context is 256 except nano. `experiments/registry.py`
- **Parameter count and FLOPs estimate**: the analytic parameter count equals the actual count (tested).
- **KV cache**: the model supports cached decoding. Past 256 tokens the cache cannot slide (learned positions) and falls back to exact recomputation.
- **Byte tokenizer** (vocabulary 256) and **byte-level BPE with 1,024 tokens**, fitted on the training split only. Both round-trip losslessly (tested). `tokenizer/tokenizers.py`

### Data

- **Shakespeare pipeline**: downloads Gutenberg #100, strips the boilerplate, splits by whole work (38 train, 3 validation, 3 test plays; seed 1729; poems always in train) and writes a manifest with sha256 hashes. Line overlap is measured and recorded (3 of 7,696 validation lines and 1 of 6,651 test lines also appear in train). `data/corpus.py`, `data/manifests/shakespeare.json`
- **Offline sample**: a 320 KB sample in git (`data/sample/shakespeare/`) used by smoke and CI.
- **Token cache and batching**: `data/loader.py`.
- **Digit-sum generator**: synthetic arithmetic data with length-shift splits (train short, test long) and a scratchpad variant that writes running sums. Expressions never cross splits (tested). `data/synthetic.py`

### Baselines and metrics

- **Uniform, unigram and interpolated Kneser-Ney n-grams**: byte orders 2 to 7 and BPE orders 2 to 4, fitted on train, order chosen on validation. KN distributions sum to 1 (tested). `eval/ngram.py`
- **One unit for every model: bits per UTF-8 byte**, with a 95% bootstrap interval over 256-token windows. Byte, BPE, n-gram and pretrained models all use it. `eval/metrics.py`
- **Calibration**: top-1 expected calibration error and a reliability diagram.
- **Paired comparison**: both models are scored on the same validation bytes, mapped onto contiguous 8 KiB blocks, and the difference is bootstrapped with the same resampled blocks. This is labelled post hoc. `eval/paired.py`, `docs/PAIRED_RESULTS.md`

### Training

- **Trainer**: AdamW, warmup plus cosine decay, gradient accumulation, clipping, early stopping, non-finite detection that stops a diverged run without crashing, a STOP-file cancel and a time budget. `train/trainer.py`
- **Verified checkpoints**: written with fsync and a read-back sha256 check; loaded with `weights_only=True`; a one-bit corruption is rejected (tested). `repro.py`
- **Resume**: bit-identical on CPU (tested); MPS runs are not bit-reproducible.
- **Position augmentation** for the digit-sum track (rolls each example within its block). `train/tasks.py`
- **Profiles**: `smoke`, `standard`, `extended`. `extended` and the LoRA track refuse to run without `--approved` and print time, memory and disk cost first. *(Off by default)*

### Pre-registered experiments

Each experiment's hypothesis and pass/fail criterion are written in `experiments/registry.py` before it runs; the verdict is computed by code in `experiments/analyze.py`.

- **12 registered experiments**: `scratch`, `tokenization`, `capacity`, `optimization`, `digitsum`, `digitsum-scratchpad`, `break-lr`, `break-mask`, `break-mask-long`, `break-overfit`, `scratch-seeds`, `digitsum-scratchpad-seeds`.
- **4 registered serving/eval studies** (`EVALUATIONS`): `kv-cache`, `lm-eval-harness`, `int8-dynamic`, `paired-baseline`.
- **Verdicts** are in `reports/experiments/<name>/result.json` with plots and a researcher note (`note.md`). Five registered hypotheses failed (four research, one serving) and are reported as prominently as the successes: `docs/FAILURE_ANALYSIS.md`.
- **Run control**: `run` skips runs whose record already exists; `--force` refuses to overwrite a committed record; `--overwrite-evidence` is a separate, explicit flag. `estimate <exp>` prints time, memory and disk first.
- **Reproduce**: `morrowglass reproduce <run_id>` re-trains a committed run into `reports/reproduce/` and compares it with the record. `tests/test_evidence_integrity.py` checks that the committed directory stays byte-identical.
- **Locked test split**: training, selection and the serving studies cannot read it (`train/tasks.py`). The only way in is `morrowglass final-test --runs … --reason "…"`, and every opening is appended to `reports/test_access_log.jsonl` (4 logged openings).
- **Optional LoRA track**: distilgpt2 with LoRA adapters against the frozen model, with full-validation bpb and intervals. One seed, validation only. `finetune/lora.py` *(Off by default: needs `--approved` and a model download)*

### Serving

- **`/v1/completions`**: OpenAI-style. Accepts text, token ids or a batch of either; `echo`, `logprobs` (0 to 5), `stop` (up to 4), `seed`, `temperature` (0 to 2), `top_p`, `top_k`, `max_tokens` (0 to 400) and SSE streaming. Deterministic for a given seed. Echo log-probabilities equal a direct forward pass, and a stream equals the non-stream answer (both tested). `serving/openai_api.py`
- **`/v1/chat/completions`**: exists, but every response says the model is not instruction-tuned. Default stop is a blank line.
- **`/v1/models`, `/tokenize`, `/detokenize`, `/tokenizer_info`.**
- **Loopback only**: the server refuses to bind to a non-loopback host (tested). Any API key is accepted.
- **Model backends**: a run id, or `hf:distilgpt2` from the local cache when enabled. `serving/backends.py`
- **KV-cache decoder** (`CachedDecoder`): reproduces the uncached logits and greedy tokens (max |Δlogit| about 2e-5); a CPU decode speed-up of 2.4x to 4.0x. On MPS there is no reliable gain at this size. `inference/generate.py`, `tests/test_kv_cache.py`
- **lm-evaluation-harness runner**: runs the pinned harness through `/v1/completions` from its own venv (`.venv-lmeval`), including a custom task over the validation split that must agree with the in-house evaluator before other numbers are trusted. Options: `--models`, `--limit`, `--port`, `--summarize-only`. `serving/harness.py`, `eval/tasks/morrowglass_shakespeare_val.yaml`
- **Dynamic int8 study**: quality held but decoding got slower (0.77x and 0.54x), so the hypothesis failed and is kept. `inference/quant.py`
- **Hub-compatible export**: safetensors, config and an auto-filled model card via `PyTorchModelHubMixin`; round trip gives identical logits. `export-hf` only writes locally and prints the publish command for a human. Nothing has been published. `hub.py`
- **Serving report generator**: `docs/SERVING_RESULTS.md` is generated from the saved records.
- **Latency benchmark**: `morrowglass bench <run_id>` (batch-1 forward, greedy decode, batch-32 throughput on MPS and CPU).

### API (FastAPI, local only)

`src/morrowglass/api/app.py`

- **Runs and records**: `/api/health`, `/api/hardware`, `/api/runs`, `/api/runs/{run_id}`, `/api/runs/{run_id}/metrics`, `/api/runs/{run_id}/blueprint`, `/api/compare`.
- **Findings**: `/api/experiments`, `/api/experiments/{name}`, `/api/experiments/{name}/plots/{plot}`, `/api/baselines`, `/api/dataset`, `/api/test-log`, `/api/serving`, `/api/paired`.
- **Report**: `/api/report` (text) and `/api/report/download`.
- **Live model calls**: `/api/generate` (with each token's probability), `/api/baseline-generate` (n-gram samples, with a background warm-up), `/api/attention`.
- **Bounded jobs**: `POST /api/jobs` starts only a smoke run in a subprocess; `GET /api/jobs/{id}`, `DELETE /api/jobs/{id}` (cancel) and `GET /api/jobs/{id}/events` (SSE progress). Real training is CLI only, so a browser cannot start a long GPU job. Path traversal returns 404 (tested).
- **Read-only contract for other local apps**: `/health`, `/capabilities`, `/runs`, `/runs/{run_id}` at the root. `api/contract.py`
- **Portable model-artifact manifest**: `export-artifact <run_id>` writes a manifest that validates against `contracts/model_artifact_v1.schema.json`; an example is in `contracts/examples/`. `contracts.py`

### Workbench (React, `web/`)

- **Layout**: blueprint pane, centre tabs, ledger pane.
- **Blueprint**: the architecture of the selected run, with clickable attention heads that open the Attention tab.
- **Ledger**: all measured runs with a filter, a "show smoke runs" tick and a test-split log. Runs whose headline number comes from the leaky no-mask model are marked `leaky`.
- **Evidence tab**: training and validation curves with overlay of up to 8 runs, against the baseline.
- **Findings tab**: one card per experiment with the verdict, numbers, caveats and a "show runs" jump. A seed-0 caveat is shown next to the scratchpad badge.
- **Break the Model tab**: the deliberate failures (learning rate, no mask, overfitting) and their fixes.
- **Generate tab**: sample from a checkpoint with per-token probabilities, next to an n-gram baseline.
- **Attention tab**: per-layer, per-head attention maps.
- **Serving tab**: the KV-cache, harness, int8 and export results, plus a live `/v1` try-it box.
- **Jobs tab**: start and cancel the bounded smoke job with live progress.
- **Honesty labels**: every item carries `MEASURED`, `SMOKE` or `LIVE SAMPLE`; Generate, Attention and the `/v1` card say which checkpoint they are using and why. Charts are hand-written SVG with no smoothing.
- **Keyboard**: `j`/`k` (or arrows in the ledger) move, `Enter` selects, `Space` adds or removes an overlay, `c` clears, `1` to `7` switch tabs, `t` cycles the theme (auto, dark, light; remembered in the browser), `?` shows help. Browser shortcuts such as copy are not intercepted.
- **Crash protection**: every tab sits in an error boundary; layer and head selections are clamped when the model changes.
- **Walkthrough recording**: `docs/media/walkthrough.webm`, made by `scripts/record_walkthrough.cjs`.

### Results on record

All numbers are bits per byte on held-out plays (lower is better), from `docs/RESULTS.md` and the run records.

| Model | Params | Validation | Test (logged opening) |
| --- | --- | --- | --- |
| Uniform over bytes | 0 | 8.000 | none |
| Kneser-Ney 4-gram on BPE-1024 (best baseline) | n/a | 2.107 [2.095, 2.120] | 2.191 |
| Byte, tiny | 0.27M | 2.335 | not opened |
| **Byte, small (pre-registered flagship)** | 1.87M | 2.086 [2.074, 2.098] | 2.171 |
| Same flagship, extended 10k steps | 1.87M | 1.993 | 2.089 |
| Byte, medium | 4.85M | 2.037 | 2.141 |
| **BPE-1024, small** | 2.02M | **2.007** | 2.081 |
| distilgpt2, frozen | 82.1M | 1.941 | not opened |
| distilgpt2 plus LoRA | 82.1M | 1.760 | not opened |

- The pre-registered `scratch` verdict is **not supported** (intervals overlap). The later paired test gives flagship minus KN-4 of -0.021 bpb with an interval of [-0.031, -0.011]. The registered verdict stays.
- Pretrained rows use a different tokenizer and context, and may have seen Shakespeare in pretraining, so the held-out guarantee is weaker.

## 5. Integrations

- **Project Gutenberg #100**: the only data source (public domain).
- **EleutherAI lm-evaluation-harness 0.4.13**: pinned in `eval/lm-eval-requirements.txt`, in its own venv; talks to `/v1/completions` over HTTP.
- **Hugging Face**: `transformers` 4.57.6 (optional group) for distilgpt2 and LoRA; Hub-format export (never uploaded); `tokenizers` for BPE.
- **OpenAI client libraries**: work against `base_url="http://127.0.0.1:5100/v1"` with any API key.
- **Other local apps**: can read `/health`, `/capabilities`, `/runs` and `/runs/{run_id}`; `scripts/demo.sh` adds user-local `uv` and `node` paths so a launcher with a minimal PATH can start it.
- **gitleaks**: pre-commit and pre-push hooks in `.githooks/`; `.gitleaks.toml` allows only exact 64-hex digests under `*_sha256` keys.

## 6. Data and privacy

- **Real vs synthetic**: language-model runs train on real public-domain text; digit-sum data is synthetic by design (generated from a seed).
- **In git**: the 320 KB offline sample, manifests and hashes, run records, verdicts, plots, serving records and model cards. Local paths are kept out of committed run records (#10).
- **Not in git**: raw and processed corpora, checkpoints (`artifacts/`, `*.pt`), token caches, the SQLite index, `web/node_modules`, `.env*`.
- **What leaves the machine**: only package downloads, the one-time corpus download, and (if the owner approves LoRA) the distilgpt2 download. The server binds loopback only.
- **Secrets**: none used.
- **Test-split access** is logged with a reason (`reports/test_access_log.jsonl`).

## 7. Tests and evidence

- **Counted from source on 2026-11-15, not run:** 114 `def test_` functions in 17 files under `tests/`. The README states 139 collected tests at `5921e39` (parametrised cases expand the count).
- **Command:** `make test` (CPU, a few minutes, no network); `make ci` runs lint, types, tests, smoke and the web build.

| File | Test functions | Covers |
| --- | --- | --- |
| `test_api.py` | 12 | API responses equal the files on disk; path traversal gives 404 |
| `test_data.py` | 11 | splits disjoint by work |
| `test_tokenizers_and_metrics.py` | 11 | lossless round trips, ECE on known cases |
| `test_trainer.py` | 11 | divergence detection, early stop, cancellation, resume |
| `test_model.py` | 10 | no future leakage, param count, fused equals explicit |
| `test_openai_api.py` | 14 | echo logprobs, stream equals non-stream, seed, loopback only |
| `test_evidence_integrity.py` | 8 | committed evidence stays byte-identical |
| `test_inference.py`, `test_kv_cache.py`, `test_paired.py`, `test_seed_followups.py` | 6 each | generation, cache equivalence, paired bits, seed follow-ups |
| `test_lora.py`, `test_portable_provenance.py` | 3 each | LoRA plumbing on a local random GPT-2, portable records |
| `test_hub_export.py`, `test_report_without_weights.py`, `test_synthetic_learning.py` | 2 each | Hub round trip, fresh-clone report, copy-rule control |
| `test_quant.py` | 1 | int8 |

- **Recorded results (from README, dated):**
  - 2026-11-10: a fresh clone of `513f958` passed `make setup`, `make smoke`, `make test` (134/134), ruff, mypy and the web build; GitHub Actions passed on the same commit.
  - 2026-11-11: `main` at `5921e39` collects 139 tests; the 111 that do not train models passed.
  - Harness agrees with the in-house evaluator to within 0.0012 bpb.
  - A fresh-clone `make reproduce-scratch` re-trained the flagship to 2.0840 bpb against the committed 2.0858.
- **Not tested:** the workbench has no component or end-to-end tests; it is checked by the TypeScript build and by hand.
- **Experiment evidence:** 14 verdicts in `reports/experiments/`, 4 serving studies in `reports/serving/`, `reports/lm_eval/`, `reports/paired/`, `reports/lora/`, and 17 verified claims in `resume_evidence/metrics.json`.

## 8. Known limits and gaps

- **Small held-out sets**: three validation plays and three test plays from one author. Results say nothing about general language ability.
- **Seeds**: the flagship has 3 seeds, the scratchpad 5 per arm, the optimisation study 3; every other arm has one.
- **Budget asymmetry**: the flagship only beats the n-gram on the pre-registered criterion at the extended budget, and only that recipe was re-run at that budget.
- **LoRA**: one seed, one rank, validation only; full fine-tuning was not compared.
- **Serving scope**: single process, batch 1, no continuous batching, auth, rate limits, tracing, container or deployment. *(Not built, on purpose)*
- **Cache cannot slide** past 256 tokens (learned absolute positions).
- **int8** uses the deprecated `torch.ao.quantization` API; cached and uncached int8 logits differ by about 6e-3.
- **Reproducibility**: MPS runs are not bit-reproducible (a re-train moved validation bpb by 0.0018).
- **Checkpoint selection** uses a 64-window validation subset, which is mildly optimistic when the best step is not the last.
- **Interval width**: per-run intervals treat neighbouring windows as independent; 8 KiB blocks give intervals about twice as wide.
- **Timings** were taken on a shared, busy machine; treat them as indicative.
- **Weights stay local**: a fresh clone needs `make reproduce-scratch` for Generate, Attention and `/v1`.
- **CUDA and multi-GPU**: *(Not built)*; the target is Apple silicon.
- **Hub publishing**: *(Waiting on owner)*; nothing has been uploaded.
- **Planned next steps** (README): seeds for tokenization and capacity arms; RoPE or ALiBi for the scratchpad track and a sliding KV cache; a LoRA rank sweep and full fine-tuning; continuous batching and request tracing for `/v1`.

## 9. Change log

| Date | PR | Change |
| --- | --- | --- |
| 2026-11-11 | #12 | README test count updated to 139 at `5921e39` |
| 2026-11-10 | #11 | `make report` works on a fresh clone without weights |
| 2026-11-10 | #10 | Local paths kept out of committed run records |
| 2026-11-10 | #9 | README synced with main after #8 |
| 2026-11-10 | #8 | Multi-seed follow-ups: flagship seeds 1 and 2, scratchpad seeds 2 to 4 (both supported) |
| 2026-11-10 | #7 | Post-hoc paired block-bootstrap test of scratch models against the best n-gram |
| 2026-11-10 | #6 | Workbench honesty and crash fixes: truthful checkpoint labels, error boundaries, no shortcut clashes |
| 2026-11-10 | #5 | Evidence protected from overwrite (`reproduce`), fresh weights served, clear errors on a fresh clone |
| 2026-11-10 | #4 | README rewritten as a full write-up |
| 2026-11-09 | #3 | Serving and evals: KV cache, `/v1` API, lm-eval-harness, Hub export, int8, Serving tab |
| 2026-11-09 | #2 | Read-only `/health`, `/capabilities`, `/runs` contract; `demo.sh` works from a minimal PATH |
| 2026-11-09 | #1 | First version: model, tokenizers, data pipeline, trainer, baselines, experiments, API, workbench, tests, CI, LoRA and the extended flagship run. The repo became public on 2026-11-09 |

## Keeping this file true

- **Same PR, same change.** A PR that adds, changes, removes or switches off something a person can see or use updates this file in that PR.
- **Status moves with reality.** Move a status word when the work moves; do not leave a stale one.
- **Only what is true.** Every feature here is found in code, tests, docs or merged-PR history. If unsure, write "unverified" or leave it out. Numbers come from saved records, not from memory.
- **Repo gates.** Commit evidence in the same PR as the code that produced it. Register any new experiment or serving study in `experiments/registry.py` before measuring it. No secrets or local paths in committed files; the gitleaks hooks must pass.
