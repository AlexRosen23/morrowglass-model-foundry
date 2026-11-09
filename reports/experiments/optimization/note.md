Hypothesis: for the tiny model at 1,000 steps, lr 3e-3 with warmup + cosine beats lr 1e-3 + cosine and beats lr 3e-3 constant without warmup, by more than the seed-to-seed range.
Setup: 3 configurations x seeds {0, 1, 2}; 1,000 steps x 8,192 tokens; seeds change both initialisation and batch order.
Observed: lr 1e-3 is clearly worse: mean 3.188 bpb [3.139, 3.213]. lr 3e-3 cosine has mean 2.644 [2.621, 2.660]; lr 3e-3 constant without warmup has mean 2.623 [2.552, 2.663]. The two lr 3e-3 schedules overlap, and the constant schedule's mean is slightly better. Its seed spread is 3x larger (stdev 0.061 vs 0.020) and its best seed (2.552) is an outlier. Not supported.
Interpretation (untested): in a short 1,000-step run, cosine decays the lr toward 3e-4, which may cost progress late in training; warmup did not matter for this model at 3e-3.
Limitations: 3 seeds give a range, not an interval; only one model size and one horizon.
Next test: same grid at 3,000 steps, plus lr 6e-3, to see whether the schedule ranking flips with a longer horizon.
Decision: keep warmup + cosine as the default for long runs (lower variance across seeds), and stop claiming it is better for short runs.
