Hypothesis (registered after break-mask showed no collapse at 400 steps): given 2,000 steps, the mask-less model finds the copy-from-the-future shortcut; its training loss falls far below the causal model's, while leak-free bpb is worse.
Setup: tiny model, 2,000 steps, seed 0; arms identical except causal=False. Leak-free scoring: the final position of 2,048 random validation windows (no future tokens exist there for any model).
Observed: the shortcut appeared between steps 600 and 700. The non-causal model's final train loss was 0.010 nats against 1.564 for the causal model. Its windowed validation bpb, computed with the same leak, is 0.0135, a "perfect" score that is pure leakage. Leak-free scoring gives 3.379 bpb for the non-causal model against 2.571 for the causal one. Supported.
Lesson: a metric computed with the same leak as training looks almost perfect; only an evaluation that structurally cannot see the future reveals the failure.
Limitations: one seed; the windowed metric shown for the non-causal arm is deliberately the leaky one.
Decision: keep the mask, keep the leakage unit test as the guard, and evaluate any non-causal variant only with leak-free scoring.
