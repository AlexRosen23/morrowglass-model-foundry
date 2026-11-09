# Experiment protocol

## Commands

```bash
make setup                       # uv sync (Python 3.12, pinned torch 2.15.0) + npm ci in web/
make data                        # fetch PG #100 (5.6 MB) if missing, rebuild splits + manifest + sample
uv run morrowglass hw            # what hardware the runs will use (MPS if usable, else CPU)
uv run morrowglass sanity        # one-batch overfit check; must pass before any training
uv run morrowglass baselines     # uniform / unigram / Kneser-Ney on validation
uv run morrowglass estimate capacity          # measured time, peak memory and disk before running
uv run morrowglass run scratch --profile standard
./scripts/run_standard.sh        # every standard experiment (~45-60 min on an M5 Pro; idempotent)
uv run morrowglass analyze all   # verdicts + plots from saved records
uv run morrowglass final-test --runs <id,...> --reason "<why now>"   # opens the locked test split
uv run morrowglass report        # docs/RESULTS.md, docs/EXPERIMENT_REPORT.md, HTML report, resume_evidence/metrics.json
```

## Profiles

| profile | data | use | approval |
|---|---|---|---|
| `smoke` | 320 KB offline sample, nano model, 20-40 steps, CPU | plumbing check, CI | none |
| `standard` | full corpus, 0.3M-4.8M parameter models, 400-3000 steps | every number in the reports | none (each run under ~12 min) |
| `extended` | 4x the standard steps | longer training | `--approved` required; without it `run` prints the measured time/memory/disk estimate and exits with code 3 |

## Split policy

- **Shakespeare.**
  - Splits are made by work, with seed 1729: 3 validation plays, 3 test plays, everything else (including all poems) in train.
  - Tokenizers and baselines are fitted on train.
  - Model selection uses validation only. That covers the best checkpoint by validation bpb, the baseline order, and every experiment conclusion.
- **Digit-sum.**
  - Evaluation splits are generated first, and train excludes their expressions.
  - `val_iid` is for selection. `test_ood_near` and `test_ood_far` are locked test sets.

## The locked test

`Trainer.train` evaluates only train and validation, and a unit test asserts this. The test split is
opened only by `morrowglass final-test`, which:

1. requires a written reason;
2. evaluates the best-on-validation checkpoint of each named run, and the val-chosen baseline;
3. writes `reports/runs/<id>/test_eval.json`;
4. **appends** an entry to `reports/test_access_log.jsonl`, with time, runs, reason and commit.

The workbench and `docs/RESULTS.md` show that log. Each test opening is disclosed there, so there is no
silent peeking.

## Metrics

- **Bits per byte.**
  - Validation is cut into non-overlapping windows of `block_size` tokens, and every token after the first is predicted exactly once (`test_eval_windows_cover_every_target_once`).
  - bpb = Σ −log₂ p / Σ UTF-8 bytes of the predicted tokens.
  - Context restarts at each window, which is pessimistic for the early positions of each window.
- **n-gram scoring.** n-grams score every token after the first using up to n−1 previous tokens, with no window resets. This is slightly favourable to the n-gram.
- **95% CI.** Percentile bootstrap (1,000 resamples) over windows of the ratio statistic. It reflects sampling variability over text segments, not over training seeds.
- **Seed variation.** The optimization experiment runs 3 seeds per configuration and reports mean and [min, max]. Three seeds are a range, not an interval.
- **Calibration.** Top-1 ECE uses 15 equal-width confidence bins on the full validation split.
- **Digit-sum.** Exact-match accuracy of the answer digit; chance is 10%.

## Seeds and determinism

Each run's seed drives: Python, NumPy and torch (CPU and MPS) RNGs; model initialisation; a separate
`torch.Generator` for batch sampling; and dropout.

- **CPU.** Resuming from a checkpoint is bit-identical to uninterrupted training (`test_resume_matches_uninterrupted_training`).
- **MPS.** Kernels are not guaranteed deterministic, so repeated standard runs can differ in the last digits.

## Provenance recorded per run (`reports/runs/<run_id>/`)

- `config.json`: full train + model config.
- `metadata.json`: seed, config hash, git commit + dirty flag, command, hardware, dataset hash, tokenizer hash, token-id hashes, budget estimate, status.
- `metrics.jsonl`: every logged train step (loss, lr, grad-norm, tokens/s, memory) and eval.
- `summary.json`: final validation metrics on the best checkpoint, calibration, checkpoint sha256s.
- `test_eval.json` (only after `final-test`) and `bench.json` (only after `bench`).
- `run_id = <name>-<sha256(config identity)[:8]>-s<seed>`. Identical configurations share an id, so experiments that use the same arm reuse the run instead of retraining it.

## Checkpoints

Checkpoints hold model + optimizer + RNG states + configs. They are serialised in memory, hashed with
sha256, written to a temporary file, fsynced, read back and compared, then atomically renamed. A
sidecar `.sha256.json` records the hash. Every load re-verifies it and loads with
`torch.load(weights_only=True)`.

This matters because the development Mac has a known memory fault that can flip bits in buffered
writes. `test_corrupted_checkpoint_is_rejected` flips one bit and expects an `IntegrityError`.
