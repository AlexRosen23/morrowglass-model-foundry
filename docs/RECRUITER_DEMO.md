# 90-second demo script

Setup (before the call):

```bash
make setup
make demo
```

Then open http://127.0.0.1:5101. If the standard checkpoints are not on this machine, `make smoke`
first: the ledger still shows every measured run, and generation uses the smoke checkpoint.

| time | what to show | what to say |
|---|---|---|
| 0:00-0:10 | Header and badges | "This is a miniature language-model lab I wrote from scratch in PyTorch. Every number here comes from a saved run record. Green MEASURED badges are real experiment runs, dashed SMOKE runs are plumbing checks, and blue is text sampled live." |
| 0:10-0:25 | Left panel: blueprint of `lm-byte-small` | "Here's the model I wrote: 1.87M parameters, 4 pre-norm blocks, 4 heads of 48 dimensions. Every row shows the tensor shape and the parameter count, and a unit test checks that count against a closed-form formula. Clicking an attention layer…" |
| 0:25-0:40 | Attention tab: click a head | "…opens the real attention matrix for this input. The upper triangle is the causal mask, and a leakage test proves that future tokens can't change earlier outputs. The caveat under the heatmap is deliberate: attention isn't an explanation." |
| 0:40-0:55 | Evidence tab: curve + dashed n-gram line, held-out panel | "This is the training curve against a strong Kneser-Ney baseline, measured in bits per byte so different tokenizers are comparable. The held-out panel has the bootstrap interval and calibration. The Transformer is ahead on the point estimate, but the intervals overlap, so my pre-registered criterion says *not supported*, and that's what the UI shows." |
| 0:55-1:10 | Findings tab, then Break the Model | "Each experiment had a hypothesis and pass criterion written down before it ran: tokenization, capacity, learning rate across 3 seeds, length generalisation. Break the Model shows controlled failures: lr too high, the causal mask removed, memorising tiny data. Each one has the measured fix." |
| 1:10-1:25 | Ledger: tick two runs → config diff | "Any two runs overlay, with the exact config differences: commit, seed, config hash, dataset hash. The test split is locked: this log shows every time I opened it, and why." |
| 1:25-1:30 | Generate tab | "And it samples from a saved checkpoint, side by side with the baseline, with the model's per-token probabilities. Those are labelled as uncalibrated." |

Do not say "state of the art", "reasoning", or anything about large models. The interesting part is the
measurement discipline, not the size of the model.
