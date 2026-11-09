# Interview defence: 15 hard questions

Answers point to code and to saved results. The numbers are in [RESULTS.md](RESULTS.md), which is
generated from run records. Nothing here relies on a run that was not executed.

**1. Show me where causality is enforced, and prove it works.**
- **Where.** `model/transformer.py`, `CausalSelfAttention.forward`. Scores at key positions j > i are set to −∞ before the softmax. The fused path uses `scaled_dot_product_attention(is_causal=True)`.
- **Proof.** `tests/test_model.py::test_causal_no_future_leakage` changes every token after position t and asserts that logits at positions ≤ t are bit-identical (atol 1e-5) on both attention paths.
- **The test can fail.** `test_non_causal_model_fails_leakage_check` builds a mask-less model and shows its early logits change. The `break-mask` experiment shows what training does without the mask.

**2. Why bits per byte, not perplexity?**
- Perplexity is per token, so a BPE model and a byte model have perplexities in different units.
- Bits per byte divides the total negative log-likelihood (in bits) by the UTF-8 bytes predicted. Both tokenizers are lossless, so that is the same text for every model.
- For the byte model, bpb = loss/ln 2 exactly (`test_bpb_equals_loss_over_ln2_for_bytes`).

**3. Is the Transformer really better than the n-gram?**
- Only as much as the pre-registered criterion allows.
- The Transformer's and the best Kneser-Ney model's 95% bootstrap intervals are both in RESULTS.md. If they overlap, the `scratch` verdict says "not supported", even when the Transformer's point estimate is lower.
- I chose a strong baseline: interpolated KN, order swept on validation, and also on BPE tokens, where it turned out best.
- The larger `capacity` arms and longer training are where the gap should grow. More data or steps is the next experiment, not a re-interpretation.

**4. How do you know your validation number is not leaking test or train?**
- Splits are by whole play, and test plays never enter training, tokenizer fitting, baseline fitting or model selection.
- Line-level overlap between held-out and train lines (≥ 30 chars) is measured and recorded (0.04% val, 0.02% test).
- The test split is opened only by `final-test`, and every opening is logged in `reports/test_access_log.jsonl`.

**5. What does your bootstrap CI capture, and what does it miss?**
- It resamples 256-token evaluation windows, so it captures variability from *which text* is evaluated, for one trained model.
- It misses training randomness (initialisation, batch order). The `optimization` experiment measures that with 3 seeds and reports mean and [min, max].
- Three seeds give a range, not a confidence interval. I say so in the report.

**6. Why did the BPE model look better? Is it fair?**
- It is not apples-to-apples, and the registry says so before the run.
- At equal steps and tokens per step, BPE covers about 3x more bytes of text and has a bigger embedding table.
- The comparison answers "per optimizer step at this context length", not "per byte of data" or "per parameter".
- A fair follow-up equalises bytes seen, or FLOPs.

**7. Walk me through one attention head's shapes.**
- x is (B, T, C=192). `qkv` gives (B, T, 576), split into three (B, T, 192) tensors.
- Each is viewed as (B, T, H=4, D=48) and transposed to (B, 4, T, 48).
- QKᵀ is (B, 4, T, T), scaled by 1/√48, masked and softmaxed.
- Multiplying by V gives (B, 4, T, 48), which is merged back to (B, T, 192) and projected by `W_o`.
- The UI blueprint shows these shapes per layer from the same config.

**8. Does the attention heatmap explain the prediction?**
- No, and the UI says so under every heatmap.
- Attention weights show where a position reads from in one layer. The value vectors, the MLPs, residual mixing and later layers all change what that read means.
- Showing what matters causally would need interventions: ablations or activation patching.

**9. What happens if the loss goes NaN?**
- The trainer checks the loss before backward and the gradient norm after clipping.
- If either is non-finite, it zeroes the gradients, skips the step, logs a `nonfinite` record, and either aborts (`nan_policy="abort"`) or allows up to `max_bad_steps` consecutive skips.
- The run then ends with `status="diverged"` and a recorded failure step. It does not crash.
- `test_divergence_is_detected_not_crashed` uses lr = 1e6. The `break-lr` experiment does it with lr = 0.1.

**10. How do you guarantee a checkpoint is what you trained?**
- It is serialised in memory, hashed with sha256, written to a temp file, fsynced, read back and re-hashed, then atomically renamed. A sidecar records the hash.
- Loading verifies the hash and uses `weights_only=True`.
- The dev Mac has a measured RAM fault that flips bits in buffered writes. `test_corrupted_checkpoint_is_rejected` flips one bit and expects `IntegrityError`.
- Resume equivalence is tested bit-for-bit on CPU.

**11. Why does the digit-sum model fail on longer inputs?**
- It learns the rule for the lengths and positions it saw (3-6 terms, answer at positions ≤ 13).
- With learned absolute positions, positions beyond those never get gradient at the answer slot, and summing more digits needs the attention pattern to extend to inputs it never saw.
- Position augmentation (random left-padding) gives every position embedding gradient. The OOD numbers in RESULTS.md show how much that helps.
- Synthetic success says nothing about general reasoning.

**12. Your early-stopping fix "helped". Compared to what?**
- Compared to the same model, data and seed without dropout or early stopping (`break-overfit`), judged on validation loss.
- I report both best and final validation loss, and the step at which each run stopped.
- The fix is judged on validation. The test split is not used for this.

**13. What would you do with 10x compute?**
1. Run the `extended` profile for the flagship. It is already defined and gated behind owner approval with a measured estimate.
2. Equalise FLOPs in the capacity study, to fit a tiny scaling curve.
3. Add RoPE vs learned positions on the digit-sum length split.
4. Compare LoRA ranks and full fine-tuning against frozen distilgpt2 (LoRA already gives 1.941 → 1.760 val bpb), and check its pretraining data for Shakespeare.

**14. How is the frontend prevented from showing fake numbers?**
- The web app has no metric constants. It fetches `/api/*`, and those handlers read `reports/**.json`.
- Each panel prints its source path, and runs carry a `MEASURED` / `SMOKE` / `LIVE SAMPLE` badge.
- API tests assert that the run detail equals the file on disk and that metrics lines match `metrics.jsonl`.

**15. What is the weakest part of this project?**
- Scale and seeds. Each main-track arm except the optimization study has one seed, so differences smaller than about 0.02 bpb should not be trusted.
- Validation and test are three plays each: one author, one era.
- MPS runs are not bit-reproducible.
- The `extended` run has not been executed. The LoRA run has one seed, and its pretrained base may have seen Shakespeare.
- These are listed as open gaps in the README instead of being glossed over.
