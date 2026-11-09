Hypothesis: the 4.85M-parameter model memorises the 240 KB sample; dropout 0.2 + early stopping (patience 4 evals) gives a lower best val loss and stops before the overfit regime.
Setup: scarce-byte-medium (no regularisation, 1,500 steps) vs scarce-byte-medium-dropout-es (dropout 0.2, early stopping), same seed and data.
Observed: without regularisation, the best val loss was 1.886 at step 550 and the final val loss 3.143. With the fix, the best val loss was 1.841 and training stopped itself at step 1,200 (status early_stopped). Supported.
Limitations: the improvement in best val loss is modest (0.045 nats). Most of the benefit is not wasting compute in the overfit regime and not shipping the last checkpoint. One seed.
Next test: a dropout sweep {0.1, 0.2, 0.3} x 2 seeds, plus weight decay 0.3.
Decision: always report the best-on-validation checkpoint (the trainer already does), and enable early stopping for scarce-data runs.
