# Case study: Morrowglass Model Foundry

**Problem.** Most "AI projects" call an API. I wanted evidence that I can build, train, evaluate,
debug and explain a neural network myself, and that I report results honestly when they disappoint.

**Architecture.**
- A decoder-only Transformer in plain PyTorch: pre-LayerNorm, explicit causal attention, learned positions, tied embeddings.
- A seeded trainer with sha256-verified checkpoints and bit-exact resume on CPU.
- Kneser-Ney n-gram baselines.
- An experiment registry where each hypothesis and pass criterion is written before the run.
- A FastAPI service and a React workbench that display only saved run records.
- Everything runs on one Apple-silicon laptop (MPS), with no paid services.

**Experiment.**
- Public-domain Shakespeare, split by play: 38 works train, 3 validate, 3 test.
- Metric: bits per byte, so byte-level and BPE models are comparable, with bootstrap 95% intervals.
- 31 standard training runs:
  - tokenization, capacity, learning rate × 3 seeds;
  - an algorithmic length-generalisation track;
  - four controlled "break the model" failures with measured fixes.
- The test split was opened once per decision, and every access is logged.

**Validated results.**

| model | test bits/byte (95% CI) |
|---|---|
| Kneser-Ney 4-gram (best baseline, chosen on validation) | 2.191 [2.176, 2.206] |
| Transformer, byte-level, 1.87M (pre-registered flagship) | 2.171 [2.158, 2.183] |
| Transformer, BPE-1024, 2.02M (best scratch model on validation) | **2.081** [2.064, 2.095] |

- **Mask removal.** Removing the causal mask produced a near-perfect *leaky* score (0.0135 bpb). Leak-free scoring showed the model was worse (3.38 vs 2.57).
- **Digit-sum.** A digit-sum task stayed at chance (parity-like plateau) until a scratchpad made it learnable. Length generalisation then depended on the seed.

**Limitation.** The pre-registered flagship's lead over the n-gram is inside the confidence interval,
and the report says "not supported". Most arms have one seed. The LoRA comparison (frozen distilgpt2 1.941 → 1.760 val bpb) has one seed, and its pretrained base may have seen Shakespeare.
