# Failure analysis

Each failure is traced as **problem → hypothesis → experiment → result → decision**. The same chains
appear in the workbench's *Break the Model* tab, read from `reports/experiments/*/result.json`.

## 1. The flagship does not clearly beat the n-gram (scratch: not supported)

- **Problem.** On validation the 1.87M-parameter byte model scores 2.086 bpb [2.074, 2.098]. KN-4 on BPE tokens scores 2.107 [2.095, 2.120]. The intervals overlap.
- **Hypothesis.** The model is capacity- and context-limited, not broken.
  - Its validation loss was still falling at step 2,500.
  - A 256-byte context covers only a couple of lines of verse, while BPE KN-4 conditions on only ~7 bytes of history (3 BPE tokens) but with very well-estimated counts.
- **Experiments.**
  - Capacity: the same recipe at 4.85M parameters.
  - Tokenization: the same model on BPE-1024 tokens, which gives about 610 bytes of context.
- **Result.**
  - Medium reaches 2.037 [2.025, 2.048] and BPE-small reaches 2.007 [1.993, 2.021]. Both beat the baseline with non-overlapping intervals on validation.
  - On the test split, opened once: 2.141 and 2.081 against the baseline's 2.191 [2.176, 2.206].
- **Decision.**
  - Report the pre-registered flagship result as not supported. Do not swap in the better model after the fact.
  - The extended-profile run (owner approval needed) is the direct test.

## 2. Digit-sum never left chance (digitsum: not supported)

- **Problem.** All four runs sat at a loss of exactly ln 10 = 2.303 and about 11% accuracy for 3,000 steps.
- **Hypotheses.**
  - (a) A pipeline bug (masking, target alignment).
  - (b) A sum mod 10 over many digits is parity-like: there is no partial credit until the full aggregation exists, so gradient descent plateaus.
- **Experiments.**
  - (a) The control test `test_sumtask_pipeline_learns_a_copy_rule`: same encoding, trainer and evaluator, with answer = first digit. It reaches above 90% in 250 steps, so (a) is rejected.
  - (b) The `digitsum-scratchpad` follow-up, registered after the failure and disclosed as such. It interleaves running sums so every step is local.
- **Result.** The scratchpad runs learn the rule perfectly in distribution: 100% for all four.
- **Decision.** Keep the negative result. The task format, not model capacity, determined learnability.

## 3. Learned positions do not extrapolate (digitsum-scratchpad)

- **Problem.** With plain learned positions, final-answer accuracy on 8-16-term inputs is at chance (9.8-10.8%), although per-step accuracy on 8-10 terms is about 70%.
  - The model solves the early running sums.
  - It breaks at absolute positions it never trained on.
- **Hypothesis.** Random position offsets in training expose every position embedding.
- **Result.** Seed 1 transfers (98.5% near, 88.8% far). Seed 0 barely moves (34.6% near, 10.5% far).
- **Decision.**
  - Position augmentation *can* fix it, but whether it does is seed-dependent, and two seeds cannot say how often.
  - The verdict code says "supported" because seed 0's far-OOD margin (+0.2 points) is technically positive. That margin is inside noise, and the note says so.
  - Next: 5+ seeds, and RoPE.

## 4. lr = 0.1 stalls rather than diverges (break-lr: supported)

- **Problem.** At 30x the usual learning rate, with no warmup or clipping, the model stalls at 4.08 bpb, worse than a byte bigram (3.60). There are no NaNs, because AdamW normalises step sizes.
- **Experiment.** Add warmup + clipping (3.76), then return to lr 3e-3 (3.42).
- **Decision.**
  - Stabilisers help but do not replace choosing the lr. Warmup and clipping stay on by default.
  - Non-finite handling is covered by its own test, at lr 1e6.

## 5. Removing the causal mask did not show up in 400 steps (break-mask: not supported)

- **Problem.** The mask-less model's training loss (2.44) was no lower than the causal model's (2.39). It had not yet found the copy-from-the-future shortcut.
- **Follow-up.** `break-mask-long` runs 2,000 steps (registered afterwards; see RESULTS.md).
  - The shortcut appeared between steps 600 and 700: train loss fell to 0.014, and *leaky* windowed val bpb to about 0.02.
  - That is the classic trap: a metric computed with the same leak looks almost perfect.
  - Leak-free last-position scoring exposes it. Those numbers are in RESULTS.md.
- **Decision.**
  - The unit test `test_non_causal_model_fails_leakage_check` is the guard. It catches a missing mask immediately, whatever the training dynamics.
  - Evaluation of any non-causal variant must use leak-free scoring.

## 6. Memorising 240 KB (capacity / break-overfit: supported)

- **Problem.** The 4.85M model's val loss bottoms at 1.886 nats (step 550), then climbs to 3.143 while train loss falls to 0.43.
- **Fix.** Dropout 0.2 + early stopping (patience 4) gives a best val loss of 1.841 and stops at step 1,200.
- **Decision.** Always ship the best-on-validation checkpoint, which the trainer does by default. Regularise when data is scarce.

## Operational failures during the build

- **Shared-machine load.** Load average reached 77 during the standard batch because other sessions were running test suites. Wall times in `summary.json` include that contention: the medium run took 675 s, against an estimated 8 min unloaded. Throughput is recorded per logged step, so slowdowns are visible in the evidence.
- **gitleaks false positive.** `tokenizer_sha256` matched the generic API-key rule. It was fixed with an allowlist for exact 64-hex digests under `*_sha256` keys only, documented in `.gitleaks.toml`.
- **MPS limitation.** The MPS backend has no float64, so evaluation accumulates in float64 on the CPU. MPS runs are not bit-deterministic; CPU resume is.
