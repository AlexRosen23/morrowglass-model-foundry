Hypothesis: on 4.5 MB, bigger models win at equal tokens; on 240 KB, the 4.85M-parameter model overfits.
Setup: tiny (0.27M) / small (1.87M) / medium (4.85M) byte models for 2,500 steps (20.5M tokens) on the full corpus; tiny and medium for 1,500 steps on the 240 KB sample (12.3M tokens, ~51 epochs).
Observed: full-data val bpb is 2.335 / 2.086 / 2.037, ordered as predicted, with non-overlapping intervals. On scarce data the medium model's best val loss was 1.886 nats at step 550, after which val loss climbed to 3.143 while train loss fell to 0.425: classic memorisation. Its best-checkpoint val bpb (2.756) is only slightly better than the tiny model's (2.771), which never overfit (final = best val loss, 1.918).
Limitations: equal tokens, not equal compute (medium spends ~2.5x the FLOPs per token of small: 33.4M vs 13.3M by the 6N + attention estimate); one seed each; the scarce validation set is the 40 KB sample of the same held-out plays.
Next test: equal-FLOPs sweep with 3 sizes x 2 seeds to fit a tiny scaling curve.
Decision: on scarce data, select the best validation checkpoint and use regularisation (see break-overfit); medium is the best byte model on full data.
