# Resume bullets (evidence only)

Every number below maps to a `status: "verified"` entry in
[`resume_evidence/metrics.json`](../resume_evidence/metrics.json), and each entry names its run id
and reproduction command. Do not add a number that is not in that file. If a claim is not verified,
use the measurement command instead of a number.

## ATS-friendly bullet (pick one)

- Built a decoder-only Transformer language model from scratch in PyTorch (1.87M parameters, causal attention, verified checkpointing) and an experiment framework with pre-registered hypotheses, locked test split and bootstrap confidence intervals; on held-out Shakespeare plays, the best-on-validation scratch model reached 2.081 bits/byte on the test set vs 2.191 for a tuned Kneser-Ney n-gram baseline.

- Designed and ran 30+ reproducible PyTorch training runs on Apple-silicon MPS (tokenization, model-capacity, learning-rate and schedule ablations across 3 seeds; controlled failure studies), with every metric traced to a git commit, seed, config hash and dataset hash, and 87 automated tests covering causal masking, checkpoint integrity and data-leakage guards.

Optional third bullet (verified):

- Adapted a pretrained distilgpt2 with LoRA (147K trainable of 82M parameters, under 2 minutes on Apple-silicon MPS), lowering held-out validation bits-per-byte from 1.941 to 1.760 with non-overlapping bootstrap intervals.

## LinkedIn / project blurb

> **Morrowglass Model Foundry.** A small language-model lab I built to show what's under the hood:
> a Transformer written from scratch in PyTorch and trained on public-domain Shakespeare. I measured
> it against strong n-gram baselines in bits per byte, with pre-registered hypotheses, a test split
> that is opened only once and logged, and an interactive workbench that shows only recorded data.
>
> Some hypotheses failed, and they're reported that way:
> - The 1.87M-parameter flagship's lead over the n-gram baseline is within the confidence interval.
> - A digit-sum task stayed at chance until a scratchpad format made it learnable.
>
> What held up: on the locked test split, the best-on-validation scratch model scored 2.081 bits/byte against
> the n-gram's 2.191.

## Claims that are NOT verified (do not use as achievements)

| claim | status | how to measure |
|---|---|---|
| Extended-profile flagship separates from the n-gram | planned | `uv run morrowglass run scratch --profile extended --approved` |
| Position augmentation reliably fixes length generalisation | not supported (1 of 2 seeds) | add seeds to `digitsum-scratchpad` in the registry |
