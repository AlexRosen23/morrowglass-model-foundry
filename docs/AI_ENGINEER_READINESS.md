# AI Engineer readiness: what this repo shows, and what it does not

This page was written on 2026-11-09 on branch `feat/ai-engineer-readiness`. It maps the repo to what
late-2026 AI Engineer and Forward Deployed Engineer postings ask for. Every number here is copied from
a saved record named next to it. The tables are generated in [SERVING_RESULTS.md](SERVING_RESULTS.md).

## The postings this is measured against

The sample is 9 postings checked on 2026-11-09, mostly Chicago or US-remote. The keyword counts come
from a small script over the posting text, so they are approximate:

| ask | postings | source |
|---|---|---|
| Python | 9/9 | all |
| Prototype-to-production delivery | 7/9 | Endeavor FDE x2, OpenAI x2, Anthropic, Writer, Databricks |
| Agents / tool design | 6/9 | |
| Security, governance, human oversight | 5/9 | |
| Cloud, containers, CI/CD | 5/9 | Kirkland (Dice mirror), iManage |
| Evals as an engineering discipline | 4/9 | Kirkland, OpenAI x2, Anthropic |
| RAG; tracing | 4/9 each | |
| Cost and latency | 3/9 | |
| Fine-tuning / GPU / vLLM | 1/9 | iManage |

Sources, retrieved 2026-11-09:
- https://api.ashbyhq.com/posting-api/job-board/endeavorai
- https://www.dice.com/job-detail/6e8edb07-ec39-4fc1-886f-990b692abc49
- https://www.purpose.jobs/discover/companies/imanage-2-30d3782f-8d61-4fd1-aa6b-9695834634e0/jobs/84445120-senior-ai-engineer (closed)
- https://jobs.ashbyhq.com/openai/6fc25168-bc4c-47a0-8601-583dfaf7fcbd
- https://job-boards.greenhouse.io/anthropic/jobs/5390966008
- https://jobs.ashbyhq.com/WRITER/93fbe5ce-2ce9-41ea-a998-a433c0f7f085
- https://databricks.com/company/careers/open-positions/job?gh_jid=8592938002

Eval practice references:
- https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- https://hamel.dev/blog/posts/llm-judge/

## What this repo is evidence for

| posting ask | evidence here | where |
|---|---|---|
| Python in production-shaped code | Typed package (mypy with `disallow_untyped_defs`), ruff, a large pytest suite, CI on every PR (GitHub Actions, owner-approved) | `src/morrowglass`, `tests/`, `.github/workflows/ci.yml` |
| **Serving / latency / cost** (3/9) | KV cache with equivalence tests; tokens/s, TTFT and p90 latency, cache on/off, CPU and MPS; dynamic int8 quality-vs-speed trade-off | `model/transformer.py`, `inference/`, `reports/serving/` |
| **OpenAI-compatible API** | `/v1/completions` (echo, logprobs, stop, seed, SSE) and `/v1/chat/completions` on the existing FastAPI app; OpenAI-shaped errors; loopback only | `serving/openai_api.py`, `tests/test_openai_api.py` |
| **Evals as an engineering discipline** (4/9) | Pre-registered hypotheses and pass/fail criteria, committed before each run; bootstrap CIs; a locked, logged test split; negative results kept; now also the standard **EleutherAI lm-evaluation-harness**, run against this repo's own endpoint and cross-checked against the in-house evaluator | `experiments/registry.py`, `reports/lm_eval/`, `docs/EXPERIMENT_PROTOCOL.md` |
| Hub-compatible artifacts | `PyTorchModelHubMixin` export: safetensors, config.json, an auto-filled model card; bit-identical round trip; publishing left to the owner | `hub.py`, `reports/hf_export/` |
| Fine-tuning (1/9) | distilgpt2 + LoRA vs frozen, pre-registered (already on `main`) | `finetune/lora.py` |
| Reproducibility and provenance | Every record carries the command, commit, dirty flag, hardware and hashes; checkpoints are written with read-back hash verification | `repro.py`, `reports/runs/*/metadata.json` |

## What this branch added, with measured numbers

**1. KV cache.**
- Registered study `kv-cache`; files `reports/serving/kv-cache/<run>.json`.
- **Verdict: supported** for both models tested.
- Logits on CPU match within 2.3e-5 of the uncached path, and greedy tokens are identical.
- CPU decode throughput (batch 1, greedy, 64 new tokens, 2 threads):

  | run | 16-token prompt | 64-token prompt | 192-token prompt |
  |---|---|---|---|
  | flagship `lm-byte-small-2822813d-s0` | 364 → 904 tok/s (2.48x) | 240 → 786 tok/s (3.27x) | 109 → 399 tok/s (3.64x) |
  | BPE `lm-bpe1024-small-eb29b8fb-s0` | 2.40x | 2.67x | 3.99x |

- MPS shows no reliable gain (0.81-1.21x). At this size, kernel-launch overhead dominates.

**2. OpenAI-compatible endpoints.**
- Covered by 14 tests in `tests/test_openai_api.py`:
  - schema conformance;
  - seed determinism;
  - echo logprobs equal a direct forward pass;
  - streaming equals non-streaming;
  - stop sequences truncate text and logprobs;
  - OpenAI-shaped errors;
  - remote-tokenizer endpoints;
  - non-loopback hosts refused.

**3. lm-evaluation-harness** (pinned 0.4.13) through `/v1/completions`. Registered study
`lm-eval-harness`; files `reports/lm_eval/`.
- **Verdict: supported.**
- **Plumbing.** On our own validation split, the harness's bits/byte, measured through HTTP, the
  remote tokenizer and echo logprobs, matches each run's own evaluator within 0.0012:

  | run | harness | own evaluator |
  |---|---|---|
  | flagship | 2.0847 | 2.0858 |
  | extended | 1.9926 | 1.9933 |
  | BPE | 2.0080 | 2.0071 |

- **Expected weakness:**

  | model | lambada_openai accuracy | wikitext bits/byte |
  |---|---|---|
  | scratch models | 0.0000-0.0008 | 3.40-4.46 |
  | distilgpt2, through the same endpoint | 0.2498 | 1.069 |

- **Specialisation shows.** The extended flagship is *better* on Shakespeare (1.993) but *worse* on
  Wikipedia (4.46 vs 3.40 for the standard run). Longer training specialised it further.

**4. Hub-compatible export.**
- `morrowglass export-hf` for three runs, cards in `reports/hf_export/`.
- Round-trip max |Δlogit| is 0.0 for all three; `model.safetensors` is 7.5 MB (byte) or 8.1 MB (BPE).
- **Not published.**

**5. Dynamic int8 on CPU.** Registered study `int8-dynamic`; files `reports/serving/int8/`.
- **Verdict: not supported** for both runs.
- Quality held: validation bpb moved by +0.0016 (flagship, 2.0858 → 2.0875) and +0.0031 (BPE, 2.0071 → 2.0102).
- Weights shrank from 7.1 to 2.2 MB (flagship) and from 7.7 to 2.9 MB (BPE).
- Decoding got *slower*: 0.77x for the flagship and 0.54x for the BPE model.
- At about 2M parameters, the per-call cost of quantizing activations outweighs the savings from int8
  matmuls. The registered hypothesis was wrong, and the result stays in.

## What is still missing (honest)

- **Agents, tools, MCP and RAG (6/9, 4/9).** None here. This repo is about models, not LLM
  applications. The sibling portfolio repos cover those asks.
- **Cloud, containers, IaC (5/9).** No Dockerfile, no deployment. Everything is local and loopback
  only, by the owner's rule. CI runs on GitHub Actions.
- **Tracing / observability (4/9).** There is no OpenTelemetry. Requests are logged by uvicorn only.
- **Load and concurrency.**
  - One process with one lock per model, no batching across requests, no rate limiting and no auth
    (unneeded on loopback, but needed before any exposure).
  - There is no vLLM, continuous batching or paged KV cache. The latency numbers are batch-1.
- **Measurement quality.**
  - The machine was shared with other heavy jobs. Load average was about 30-46 on 18 cores during
    these runs (`uptime`, observed by hand, not recorded in the run files).
  - The timing numbers are indicative, not a controlled benchmark. In particular, the int8 run
    measured fp32 decoding at a much lower speed than the earlier KV-cache run on the same checkpoint
    did.
- **Model quality.**
  - A 2M-parameter Shakespeare model is not useful for general language tasks. The harness numbers
    say so in standard units.
  - The context window cannot slide with the cache (learned absolute positions). RoPE would fix this
    but needs retraining.
- **Human-labelled evals.** None. The evals here are likelihood-based and need no labels. No LLM
  judge was used.

## Owner decisions this branch does not take

- Publishing the exported models to the Hugging Face Hub (command in [SERVING_RESULTS.md](SERVING_RESULTS.md)).
- Merging this PR.
