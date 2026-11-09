---
license: mit
language: en
library_name: morrowglass
pipeline_tag: text-generation
tags:
  - pytorch_model_hub_mixin
  - model_hub_mixin
  - from-scratch
  - shakespeare
  - decoder-only
---

# morrowglass-lm-byte-small-2822813d-s0

A 1.87M-parameter decoder-only Transformer trained **from scratch** on public-domain Shakespeare by [Morrowglass Model Foundry](https://github.com/AlexRosen23/morrowglass-model-foundry), a reproducible miniature language-model lab. Every number below is copied by code from the saved run records; nothing is typed by hand.

> **Not instruction-tuned and not a general model.** It continues text in a Shakespeare-like style. Use it to study small-model training, evaluation and serving, not to generate text anyone relies on.

## Model

| | |
|---|---|
| run id | `lm-byte-small-2822813d-s0` (experiment `scratch`, profile `standard`, seed 0) |
| architecture | pre-LayerNorm decoder, 4 layers x 4 heads x d=192, context 256 tokens, learned positions, tied embeddings |
| tokenizer | `byte` (UTF-8 bytes, vocab 256, no file needed) |
| parameters | 1,869,504 |
| training | 2,500 steps, 20,480,000 tokens, AdamW lr 0.003, MPS on Apple M5 Pro, 291 s |
| code commit | `d72e0589db9f` |

## Data

Project Gutenberg eBook #100 (complete works of Shakespeare, public domain, boilerplate removed), split **by play** with a fixed seed: validation = *As You Like It*, *Hamlet*, *Titus Andronicus*; test = *Macbeth*, *Pericles*, *Cymbeline*; everything else (including the poems) is train. Model selection used validation only.

## Evaluation (held-out plays)

Bits per UTF-8 byte (lower is better), 95% bootstrap CI over evaluation windows.

| split | bits/byte | note |
|---|---|---|
| validation | 2.086 [2.074, 2.098] | top-1 ECE 0.029 |
| test | 2.171 [2.158, 2.183] | opened once on 2026-11-09, logged in reports/test_access_log.jsonl |

## Pre-registered experiments this run belongs to

Hypotheses and criteria were written before the runs (`src/morrowglass/experiments/registry.py`).

| experiment | verdict | detail |
|---|---|---|
| Model capacity at equal token budget (full data vs 240 KB scarce data) | **supported** | full-data bpb tiny 2.335 / small 2.086 / medium 2.037; scarce medium val loss rose +1.257 nats after step 550 |
| Scratch Transformer vs count-based baselines | **not supported** | Transformer 2.086 bpb vs best n-gram (kneser-ney-4, bpe1024) 2.107 |
| Byte vs BPE-1024 tokenization | **supported** | BPE 2.007 vs byte 2.086 val bpb |

## Serving measurements

KV cache (`morrowglass bench-kv lm-byte-small-2822813d-s0`, 2026-11-09, Apple M5 Pro, 2 threads, batch 1, greedy, 64 new tokens). Registered verdict: **supported**.

| device | prompt tokens | tokens/s, no cache | tokens/s, KV cache | speed-up |
|---|---|---|---|---|
| cpu | 16 | 364.4 | 903.8 | 2.48x |
| cpu | 64 | 240.3 | 785.5 | 3.27x |
| cpu | 192 | 109.4 | 398.6 | 3.64x |
| mps | 16 | 138.5 | 112.5 | 0.81x |
| mps | 64 | 134.9 | 129.9 | 0.96x |
| mps | 192 | 119.1 | 138.5 | 1.16x |

Dynamic int8 (CPU, `morrowglass quant-int8 lm-byte-small-2822813d-s0`): val bpb 2.0858 -> 2.0875 (+0.0016); decode 33.3 -> 25.6 tokens/s; weights 7.141 -> 2.169 MB. Registered verdict: **not supported**.

EleutherAI lm-evaluation-harness (0-shot, through the repo's OpenAI-compatible endpoint; expected to be weak, see limitations):

| task | metric | value |
|---|---|---|
| lambada_openai | acc | 0.0002 |
| wikitext (wikitext-2) | bits/byte | 3.396 |
| our Shakespeare validation split | bits/byte | 2.085 (own evaluator: 2.086) |

## Intended use

- Teaching and measurement: attention inspection, training-choice ablations, serving experiments (KV cache, quantization, OpenAI-compatible API) on a model small enough to run on a laptop CPU.
- Out of scope: any user-facing generation, factual answers, chat, or text in modern English.

## Limitations

- About 2M parameters trained on ~4.5 MB of one author's text: output is pastiche with frequent nonsense; it loses the thread within a sentence.
- Context is 256 tokens; longer prompts are cut to the last 256.
- Probabilities are model scores, not calibrated confidence for individual tokens.
- No safety tuning. Shakespeare's plays contain period violence and prejudice, which samples can reproduce.
- One training seed for this run; MPS training is not bit-reproducible.

## How to load

```python
from morrowglass.hub import MorrowglassHubLM   # pip install from the project repo
model = MorrowglassHubLM.from_pretrained("path/to/morrowglass-lm-byte-small-2822813d-s0")  # or the Hub repo id once published
```

Or serve it with an OpenAI-compatible API: `uv run morrowglass serve --model lm-byte-small-2822813d-s0`.

## Licence

Code and weights: MIT. Training text: public domain.
