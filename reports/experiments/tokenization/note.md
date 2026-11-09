Hypothesis: at an equal step budget, BPE-1024 reaches lower val bits-per-byte than byte-level.
Setup: identical 4x4x192 model, 2,500 steps, 32 x 256 tokens per step. Only the tokenizer differs: BPE-1024 is byte-level BPE trained on the train split only.
Observed: BPE 2.007 [1.993, 2.021] vs byte 2.086 [2.074, 2.098] bpb on validation, non-overlapping, so supported. BPE averages 2.38 bytes per token, so in the same steps it saw ~48.7M bytes of text vs 20.5M, and a 256-token window covers ~610 bytes instead of 256. It also has 147K more embedding parameters (2.02M vs 1.87M). ECE is worse for BPE (0.055 vs 0.029). The test split agrees: 2.081 vs 2.171.
Limitations: not apples-to-apples in text seen, context in bytes, or parameters (stated before the run). This answers "per optimizer step at fixed token context", not "which tokenizer is better per byte of data".
Next test: equalise bytes seen (run the byte model ~2.4x longer) and context in bytes (byte block 608 vs BPE 256).
Decision: BPE is the better default at this compute. The byte model stays the flagship because its per-byte attention maps are interpretable in the UI.
