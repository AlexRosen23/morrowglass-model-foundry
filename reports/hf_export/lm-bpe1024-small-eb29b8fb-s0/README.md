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

# morrowglass-lm-bpe1024-small-eb29b8fb-s0

A 2.02M-parameter decoder-only Transformer trained **from scratch** on public-domain Shakespeare by [Morrowglass Model Foundry](https://github.com/AlexRosen23/morrowglass-model-foundry), a reproducible miniature language-model lab. Every number below is copied by code from the saved run records; nothing is typed by hand.

> **Not instruction-tuned and not a general model.** It continues text in a Shakespeare-like style. Use it to study small-model training, evaluation and serving, not to generate text anyone relies on.

## Model

| | |
|---|---|
| run id | `lm-bpe1024-small-eb29b8fb-s0` (experiment `tokenization`, profile `standard`, seed 0) |
| architecture | pre-LayerNorm decoder, 4 layers x 4 heads x d=192, context 256 tokens, learned positions, tied embeddings |
| tokenizer | `bpe1024` (byte-level BPE fitted on the train split; tokenizer.json) |
| parameters | 2,016,960 |
| training | 2,500 steps, 20,480,000 tokens, AdamW lr 0.003, MPS on Apple M5 Pro, 297 s |
| code commit | `d72e0589db9f` |

## Data

Project Gutenberg eBook #100 (complete works of Shakespeare, public domain, boilerplate removed), split **by play** with a fixed seed: validation = *As You Like It*, *Hamlet*, *Titus Andronicus*; test = *Macbeth*, *Pericles*, *Cymbeline*; everything else (including the poems) is train. Model selection used validation only.

## Evaluation (held-out plays)

Bits per UTF-8 byte (lower is better), 95% bootstrap CI over evaluation windows.

| split | bits/byte | note |
|---|---|---|
| validation | 2.007 [1.993, 2.021] | top-1 ECE 0.055 |
| test | 2.081 [2.064, 2.095] | opened once on 2026-11-09, logged in reports/test_access_log.jsonl |

## Pre-registered experiments this run belongs to

Hypotheses and criteria were written before the runs (`src/morrowglass/experiments/registry.py`).

| experiment | verdict | detail |
|---|---|---|
| Byte vs BPE-1024 tokenization | **supported** | BPE 2.007 vs byte 2.086 val bpb |

## Serving measurements

KV cache (`morrowglass bench-kv lm-bpe1024-small-eb29b8fb-s0`, 2026-11-09, Apple M5 Pro, 2 threads, batch 1, greedy, 64 new tokens). Registered verdict: **supported**.

| device | prompt tokens | tokens/s, no cache | tokens/s, KV cache | speed-up |
|---|---|---|---|---|
| cpu | 16 | 154.0 | 369.8 | 2.40x |
| cpu | 64 | 160.1 | 427.3 | 2.67x |
| cpu | 192 | 85.2 | 340.3 | 3.99x |
| mps | 16 | 145.3 | 131.7 | 0.91x |
| mps | 64 | 61.1 | 74.2 | 1.21x |
| mps | 192 | 62.7 | 69.1 | 1.10x |

Dynamic int8 (CPU, `morrowglass quant-int8 lm-bpe1024-small-eb29b8fb-s0`): val bpb 2.0071 -> 2.0102 (+0.0031); decode 58.9 -> 31.7 tokens/s; weights 7.704 -> 2.875 MB. Registered verdict: **not supported**.

EleutherAI lm-evaluation-harness (0-shot, through the repo's OpenAI-compatible endpoint; expected to be weak, see limitations):

| task | metric | value |
|---|---|---|
| lambada_openai | acc | 0.0000 |
| wikitext (wikitext-2) | bits/byte | 3.891 |
| our Shakespeare validation split | bits/byte | 2.008 (own evaluator: 2.007) |

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
model = MorrowglassHubLM.from_pretrained("path/to/morrowglass-lm-bpe1024-small-eb29b8fb-s0")  # or the Hub repo id once published
```

Or serve it with an OpenAI-compatible API: `uv run morrowglass serve --model lm-bpe1024-small-eb29b8fb-s0`.

## Licence

Code and weights: MIT. Training text: public domain.
