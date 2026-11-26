# Listing draft: Build a Transformer Language Model From Scratch on Your Laptop

Status: **DRAFT for the owner's review.** Not published or listed anywhere. Price and wording need the
owner's OK before anything goes on sale.

## Title

Build a Transformer Language Model From Scratch on Your Laptop: a Localpane Lab Kit

## Price

$39 (planned), one-time, single-developer licence.

## Blurb (100 words)

Build a small GPT-style language model in plain PyTorch, one tested step at a time. You write the byte
tokenizer, causal self-attention with multiple heads, Transformer blocks, the full model and its loss,
evaluation in bits per byte, the learning-rate schedule, AdamW setup, the training step, sampling with
temperature and top-k, and a key/value cache that must match uncached output exactly. Every step has a
starter with marked TODOs, tests that fail until your code works, plain explanations and hints. One
command checks your work. Then you train it on Shakespeare on your CPU in under a minute and watch it
write. Full solution included.

## What's inside

- WORKBOOK.md: 8 steps, each with the goal, key concepts in plain words, where the starter code is,
  the tests that must pass, hints and a check command
- 8 starter projects (`steps/step-01` to `step-08`): earlier steps solved, the current one stubbed
- The finished solution: about 900 lines of Python and 650 lines of tests
- 43 graded tests, and 6 tests for the code the kit provides; the whole suite runs in seconds on a CPU
- `check.sh` to check one step and everything before it
- Public-domain Shakespeare practice text (280 KB), split so the model is scored on plays it never saw
- A small command line to train, evaluate and sample from your model

The 8 steps:

1. Bytes as tokens, and training batches
2. Causal self-attention
3. The Transformer block
4. The whole model: embeddings, positions, logits, loss
5. Measuring: loss, perplexity, bits per byte
6. Training: schedule, AdamW, one step, a real run
7. Sampling: temperature, top-k, generation
8. The key/value cache

## Who it's for

Python developers who use language models and want to understand how one works inside by building
it: software engineers moving towards ML, students who have read "Attention Is All You Need" and want
to see it run, and anyone who prefers tests to videos.

Not for: anyone looking for a production model, a training framework, GPU or multi-machine training,
fine-tuning of existing models, or a chatbot. The model is deliberately small and trains on a CPU.

## Prerequisites

- Python 3.10 or later and PyTorch; tested with Python 3.12 and PyTorch 2.15.0 on macOS (Apple
  Silicon), CPU only. Linux and Windows not yet tested.
- Comfortable Python; knowing what a tensor, a matrix product and a gradient are helps
- No prior Transformer experience needed
- About 8 to 10 hours

## Open source, stated plainly

The finished code is adapted from Morrowglass Model Foundry, the author's open-source (MIT) miniature
language-model lab on GitHub. This kit is the guided path through it: the workbook, the step-by-step
starters, the graded tests and the checks.

## Licence line

Single-developer licence: use and modify the code, including in your own products; no
redistribution and no public repositories of the kit. Free updates within 1.x.

## Claims check (for the owner)

- "From scratch" means: no `nn.Transformer`, no `nn.MultiheadAttention`, no pretrained weights. PyTorch
  still provides tensors, autograd, Linear/Embedding/LayerNorm, GELU, cross-entropy and AdamW. The
  workbook and README-FIRST say so.
- "Under a minute" refers to the 300-step demo command, which we measured at about 12 seconds of
  training (27 s wall-clock including start-up) on an Apple Silicon Mac with 2 threads while other work
  was running. No other speed or quality numbers are claimed.
- No "production-grade", no quality guarantees. README-FIRST says the text only looks like
  Shakespeare from a distance.
- The code is not presented as secret; the open-source origin is stated in the listing, README-FIRST
  and the licence.
- Test counts and the step list match `labkit/verify.sh` output for 1.0.0.
