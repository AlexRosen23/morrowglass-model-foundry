# Ablations

Every ablation changes one factor against a fixed reference, and every arm is a saved run. Per-run
numbers are in [RESULTS.md](RESULTS.md); the interpretation lives in `reports/experiments/<exp>/note.md`.

| ablation | what changed | held fixed | outcome (val bpb unless stated) | verdict |
|---|---|---|---|---|
| **Tokenization** | byte (256) → byte-level BPE-1024, fitted on train | model shape, steps, batch × context tokens, lr, seed | 2.086 → **2.007** | supported, with caveats: BPE sees 2.38x more bytes per step and has +147K embedding params |
| **Capacity, full data** | 0.27M → 1.87M → 4.85M params | tokens (20.5M), data, schedule, seed | 2.335 → 2.086 → **2.037** | supported (equal tokens, not equal FLOPs: 2.1 / 13.3 / 33.4 MFLOPs per token) |
| **Capacity, scarce data** | 0.27M vs 4.85M on 240 KB | tokens (12.3M ≈ 51 epochs) | medium val loss: best 1.886 at step 550 → 3.143 at the end; tiny never rose | supported (overfit) |
| **Training budget** | 2,500 vs 10,000 steps (extended, owner-approved) | flagship recipe, seed | 2.086 → **1.993**; train/val gap 0.25 → 0.42 bpb; best step 8,400 | more steps help; overfitting begins |
| **Learning rate** | 1e-3 vs 3e-3 (warmup + cosine) | tiny model, 1,000 steps, 3 seeds | mean 3.188 vs **2.644**, non-overlapping ranges | lr matters a lot |
| **Schedule** | warmup + cosine vs constant without warmup at 3e-3 | as above | 2.644 [2.621, 2.660] vs 2.623 [2.552, 2.663] | no measurable difference at 1k steps (hypothesis not supported) |
| **Regularisation, scarce data** | none vs dropout 0.2 + early stopping | medium model, 240 KB, seed | best val loss 1.886 → **1.841**; stopped at step 1,200 | supported (modest) |
| **Stabilisers at lr 0.1** | none vs warmup + clip vs lr 3e-3 | tiny model, 400 steps | best 4.078 → 3.763 → **3.422** | supported |
| **Causal mask** | removed vs kept | tiny model, 400 and 2,000 steps | see `break-mask` and `break-mask-long` in RESULTS.md | 400 steps: no collapse yet |
| **Output format** (algorithmic) | answer only vs running-sum scratchpad | model, steps, data | in-distribution accuracy 11% (chance) → **100%** | the format, not capacity, made the task learnable |
| **LoRA adaptation** (pretrained) | frozen distilgpt2 vs + LoRA r=8 on c_attn, 300 steps | model, tokenizer, full validation split | 1.941 → **1.760** | supported (one seed; possible pretraining contamination) |
| **Position augmentation** (algorithmic) | learned positions vs random position offsets | scratchpad format, 2 seeds | 12-16-term accuracy: 10.3% / 9.8% → 10.5% / **88.8%** | seed-dependent: transfers for 1 of 2 seeds |

## What is not apples-to-apples, and why

- **Tokenization** is equalised per optimizer step and per token of context, not per byte of data or per parameter. A byte-equalised version is the next test.
- **Capacity** is equalised per token, not per FLOP.
- **One seed** is used per arm, except the optimization and digit-sum studies. Differences under ~0.02 bpb should be read as noise. The optimization study's seed spread (stdev 0.02-0.06 bpb) shows why.
