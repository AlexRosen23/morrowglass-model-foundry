# Changelog

All notable changes to this lab kit. Versions follow semantic versioning: 1.x updates are free for
existing buyers.

## 1.0.0 (unreleased draft, 2026-11-26)

- First version: 8 steps (byte tokenizer and batches, causal self-attention, the Transformer block,
  the full model, evaluation in bits per byte, training, sampling, the key/value cache), one starter
  project per step, the finished solution, `check.sh`, and public-domain Shakespeare practice text.
- 43 graded tests (`test_stepNN_*`) plus 6 tests for the provided code (`test_kit_*`). Every test runs
  a tiny model on the CPU; the whole suite takes a few seconds.
- Tested with Python 3.12.15, PyTorch 2.15.0 and pytest 9.2.0 on macOS (Apple Silicon), CPU only.
