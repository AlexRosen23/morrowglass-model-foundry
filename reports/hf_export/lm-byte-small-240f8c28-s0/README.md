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

# morrowglass-lm-byte-small-240f8c28-s0

A 1.87M-parameter decoder-only Transformer trained **from scratch** on public-domain Shakespeare by [Morrowglass Model Foundry](https://github.com/AlexRosen23/morrowglass-model-foundry), a reproducible miniature language-model lab. Every number below is copied by code from the saved run records; nothing is typed by hand.

> **Not instruction-tuned and not a general model.** It continues text in a Shakespeare-like style. Use it to study small-model training, evaluation and serving, not to generate text anyone relies on.

## Model

| | |
|---|---|
| run id | `lm-byte-small-240f8c28-s0` (experiment `scratch`, profile `extended`, seed 0) |
| architecture | pre-LayerNorm decoder, 4 layers x 4 heads x d=192, context 256 tokens, learned positions, tied embeddings |
| tokenizer | `byte` (UTF-8 bytes, vocab 256, no file needed) |
| parameters | 1,869,504 |
| training | 10,000 steps, 81,920,000 tokens, AdamW lr 0.003, MPS on Apple M5 Pro, 924 s |
| code commit | `667bb5fa5811` |

## Data

Project Gutenberg eBook #100 (complete works of Shakespeare, public domain, boilerplate removed), split **by play** with a fixed seed: validation = *As You Like It*, *Hamlet*, *Titus Andronicus*; test = *Macbeth*, *Pericles*, *Cymbeline*; everything else (including the poems) is train. Model selection used validation only.

## Evaluation (held-out plays)

Bits per UTF-8 byte (lower is better), 95% bootstrap CI over evaluation windows.

| split | bits/byte | note |
|---|---|---|
| validation | 1.993 [1.981, 2.006] | top-1 ECE 0.043 |
| test | 2.089 [2.075, 2.103] | opened once on 2026-11-09, logged in reports/test_access_log.jsonl |

## Pre-registered experiments this run belongs to

Hypotheses and criteria were written before the runs (`src/morrowglass/experiments/registry.py`).

| experiment | verdict | detail |
|---|---|---|
| Scratch Transformer vs count-based baselines (extended profile) | **supported** | Transformer 1.993 bpb vs best n-gram (kneser-ney-4, bpe1024) 2.107 |

## Serving measurements


EleutherAI lm-evaluation-harness (0-shot, through the repo's OpenAI-compatible endpoint; expected to be weak, see limitations):

| task | metric | value |
|---|---|---|
| lambada_openai | acc | 0.0008 |
| wikitext (wikitext-2) | bits/byte | 4.464 |
| our Shakespeare validation split | bits/byte | 1.993 (own evaluator: 1.993) |

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
model = MorrowglassHubLM.from_pretrained("path/to/morrowglass-lm-byte-small-240f8c28-s0")  # or the Hub repo id once published
```

Or serve it with an OpenAI-compatible API: `uv run morrowglass serve --model lm-byte-small-240f8c28-s0`.

## Licence

Code and weights: MIT. Training text: public domain.
