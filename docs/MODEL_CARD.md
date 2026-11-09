# Model card: Morrowglass byte-level Transformer

## Model

| | |
|---|---|
| Name / run | `morrowglass-lm-byte-small`, run `lm-byte-small-2822813d-s0` (pre-registered flagship) |
| Variants measured | byte-tiny (0.27M), byte-medium (4.85M, `lm-byte-medium-b01e2a16-s0`), BPE-1024-small (2.02M, `lm-bpe1024-small-eb29b8fb-s0`) |
| Architecture | decoder-only causal Transformer, pre-LayerNorm, 4 layers × 4 heads × d=192, MLP 768, GELU, learned absolute positions, context 256, tied embeddings, no biases |
| Parameters | 1,869,504 (checked against the closed-form count in tests) |
| Written in | native PyTorch (`src/morrowglass/model/transformer.py`); no pretrained weights |
| Training | 2,500 steps × 32 × 256 tokens = 20.5M bytes (~4.5 epochs of 4.56 MB); AdamW (β 0.9/0.95, wd 0.1), lr 3e-3, warmup 100, cosine to 3e-4, grad-clip 1.0; seed 0 |
| Hardware / time | Apple M5 Pro, MPS, 291 s (shared machine) |
| Licence | code and weights MIT; training text public domain |

## Intended use

- A measurement and teaching instrument: inspecting attention, comparing training choices, and showing what a model from scratch learns from 4.5 MB of verse.
- **Not** for generating text for anyone to rely on, and not a general assistant.

## Performance

Held-out plays: validation is *As You Like It*, *Hamlet* and *Titus Andronicus*; test is *Macbeth*,
*Pericles* and *Cymbeline*. 95% bootstrap CIs are taken over windows.

| model | val bpb | test bpb | val ECE |
|---|---|---|---|
| flagship byte-small | 2.086 [2.074, 2.098] | 2.171 [2.158, 2.183] | 0.029 |
| flagship, extended (10k steps, `lm-byte-small-240f8c28-s0`) | 1.993 [1.981, 2.006] | 2.089 [2.075, 2.103] | 0.043 |
| byte-medium | 2.037 [2.025, 2.048] | 2.141 [2.128, 2.154] | 0.027 |
| BPE-1024-small | 2.007 [1.993, 2.021] | 2.081 [2.064, 2.095] | 0.055 |
| best n-gram (KN-4, BPE-1024) | 2.107 [2.095, 2.120] | 2.191 [2.176, 2.206] | — |

- The flagship's next-byte top-1 accuracy on validation is 57.0%.
- The test split was opened once, after selection, and logged in `reports/test_access_log.jsonl`.
- Inference cost on this machine is in `reports/runs/lm-byte-small-2822813d-s0/bench.json` (measured with `morrowglass bench`).

## Behaviour and failure modes

- **Pastiche, not meaning.** The model produces Shakespeare-shaped text: speaker names in capitals, verse line lengths, archaic words. It often invents words and loses the thread within a sentence. The UI labels samples `LIVE SAMPLE` and never presents them as quality metrics.
- **Overconfidence.** Its probabilities are model scores. Calibration is good on average (ECE 0.029), but individual tokens can be confidently wrong. The UI shows the raw probability and entropy per token, labelled uncalibrated.
- **Memorisation risk.** Train-subset bpb is 1.83 against 2.09 on validation. It may reproduce training lines verbatim (not measured), and on 240 KB the larger model memorises outright (see `capacity`).
- **Out of distribution.**
  - Modern English, code and non-Latin scripts are far from the training data. Non-ASCII text is rare in train (230 of 4.56M characters), so the model gives it very little probability.
  - Prompts longer than 256 bytes are truncated to the last 256.
- **Length.** It has learned absolute positions only. The digit-sum experiments show that this model family does not extrapolate to unseen positions without augmentation.

## Ethical considerations

The training text is public domain and contains no personal data. Shakespeare's plays include period
violence and prejudice, and samples can reproduce that language.

## Reproduce

```bash
make data && uv run morrowglass run scratch --profile standard --force
```

The run id is deterministic in the config. MPS kernels are not bit-deterministic, so expect small
differences in the last digits.
