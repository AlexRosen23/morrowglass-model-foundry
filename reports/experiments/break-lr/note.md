Hypothesis: lr 0.1 with no warmup and no clipping stalls or diverges; warmup + clipping alone do not rescue it; lr 3e-3 does.
Setup: tiny model, 400 steps, seed 0, three arms that differ only in lr / warmup / clipping.
Observed: it did not produce NaNs (grad norm max 1.75; AdamW normalises step size). Instead it stalled: best val 4.078 bpb, worse than a bigram. Warmup + clipping at lr 0.1 reached 3.763 bpb, better but still poor. lr 3e-3 with warmup + clipping reached 3.422 bpb. Supported (broken > 3.0, fix 2 is > 0.5 bpb better).
Limitations: one seed, 400 steps; the "no NaN" outcome depends on Adam. Plain SGD at this lr would likely diverge.
Next test: the same arms with SGD + momentum, to show actual divergence and the non-finite handler.
Decision: warmup and clipping stay on by default, and the lr comes from the optimization experiment. The trainer's non-finite handling is covered by test_divergence_is_detected_not_crashed.
