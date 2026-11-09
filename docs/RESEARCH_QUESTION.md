# Research question

## Primary question

**On a fixed, public-domain corpus and a laptop-scale budget, how much does a small Transformer
written from scratch improve on strong count-based language models for held-out text? And which
choices (tokenization, capacity, optimisation) move that number by more than run-to-run noise?**

- **Task.** Next-token prediction (language modelling) on Shakespeare. Train on 38 works; validate on 3 unseen plays; test once on 3 other unseen plays.
- **Metric.** Bits per UTF-8 byte (bpb) on held-out text, with a 95% bootstrap interval over 256-token windows. Lower is better. It is tokenizer-independent, so byte-level and BPE models can be compared.
- **Secondary metrics.**
  - per-token cross-entropy and perplexity (only within one tokenizer);
  - top-1 expected calibration error (ECE);
  - inference latency, throughput and memory.
- **Utility.** This is a measurement bench, not a product. It shows which modelling decisions matter when data and compute are tiny, with every claim traceable to a run record.

## Baselines (all fitted on train only, order chosen on validation)

| kind | model |
|---|---|
| naive | uniform over the vocabulary (exactly 8 bpb for bytes) |
| majority / frequency | unigram with add-one smoothing |
| conventional ML | interpolated Kneser-Ney n-grams: byte orders 2-7, BPE-1024 orders 2-4 |
| neural (this project) | decoder-only Transformer, 0.3M-4.8M parameters, written in PyTorch |
| pretrained + LoRA | *planned; not executed* (needs owner approval for the model download, see "Open gaps" in the README) |

## What success means

These criteria were written in `src/morrowglass/experiments/registry.py` before any standard run.

1. **scratch.** The 1.9M-parameter byte-level Transformer's val bpb upper CI is below the best n-gram's lower CI.
2. **tokenization.** At an equal step budget, BPE-1024 beats byte-level on val bpb with non-overlapping CIs. Both caveats are stated up front: BPE sees about 3x more text per step and has more embedding parameters.
3. **capacity.** On the full corpus, bigger is better (tiny > small > medium in bpb). On 240 KB, the medium model overfits: val loss rises more than 0.05 nats after its minimum while train loss keeps falling.
4. **optimization.** For the tiny model, lr 3e-3 with warmup + cosine beats lr 1e-3 + cosine and lr 3e-3 constant without warmup, with seed ranges that do not overlap (3 seeds each).
5. **digitsum (algorithmic track).** The model reaches above 95% in-distribution accuracy but below 50% on 12-16-term inputs. Position augmentation improves OOD accuracy for both seeds.
6. **Break the Model.** Each induced failure (lr too high, no causal mask, memorising 240 KB) shows the predicted failure signature, and the documented fix measurably improves it.

A criterion that is not met is reported as **not supported**. It is not re-run with tweaks until it passes.

## Out of scope

Claims about general language ability, about reasoning, or about anything frontier-scale. A 2M-parameter
model trained on 4.5 MB of verse is a teaching and measurement instrument.
