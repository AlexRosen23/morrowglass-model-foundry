Hypothesis: without the causal mask, training loss collapses (the model reads the next token from the future), while honest last-position scoring is worse than the causal model's.
Setup: tiny model, 400 steps, seed 0; identical except causal=False. Leak-free scoring uses only the final position of 2,048 random windows, where no model can see the future.
Observed: the predicted collapse did not happen within 400 steps. The non-causal model's final train loss was 2.441 vs 2.386 for the causal model, and its leak-free bpb was 3.593 vs 3.571. The non-causal model is slightly worse everywhere, but it had not yet learned to attend one position ahead. Not supported as stated.
Why: with learned absolute positions, the shortcut needs a query/key pattern that targets position t+1 for every t. At this size and lr, 400 steps are not enough to find it.
Limitations: one seed, a short horizon by design.
Next test: the same pair for 2,000 steps (break-mask-long), registered after this result and disclosed as a follow-up.
Decision: the unit test test_non_causal_model_fails_leakage_check is the guard that catches a missing mask immediately, independent of training dynamics.
