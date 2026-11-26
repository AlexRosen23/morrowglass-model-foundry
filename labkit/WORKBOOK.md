# Workbook: Build a Transformer Language Model From Scratch on Your Laptop

You will build `scratchlm`, a small GPT-style language model, in plain PyTorch: a byte tokenizer,
batches, causal self-attention, Transformer blocks, the full model, evaluation in bits per byte, a
training loop, sampling and a key/value cache. Then you train it on Shakespeare on your laptop's CPU
and watch it go from random bytes to something that looks like a play.

**What you write and what you use.** You write every part of the model and its training and sampling
logic yourself. PyTorch provides tensors, automatic differentiation (`backward()`), the building blocks
`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `F.gelu`, `F.cross_entropy`, and the AdamW optimizer.
You do not use `nn.Transformer`, `nn.MultiheadAttention`, `F.scaled_dot_product_attention` (except
as a reference inside one test) or any pretrained weights. Checkpoint files, the training loop around
your code and the command line are given.

**What to expect from the model.** The default model has about 0.8 million parameters; step 6 trains
an even smaller one first. In our 12-second CPU test run (numbers in README-FIRST.md) the model had
learned line breaks, speaker-name lines and a few short words, and was still mostly misspelling.
Longer runs and the bigger default model should do better (we did not benchmark them for this kit). It will not learn meaning at this size. That is
the right size to understand every line of the code and to see each idea change the numbers.

## How the steps work

- `steps/step-NN/` is a complete Python project. Everything from steps before NN is solved, and the
  code for step NN and later is replaced by stubs that raise `NotImplementedError("TODO(step NN) ...")`.
- Find your work with `grep -rn "TODO(step 3)" scratchlm` inside the step folder. Each TODO comment
  says what the code must do; the function's docstring above it is the exact contract.
- Each step has tests named `test_stepNN_*`. They fail on the starter and pass when the step is done.
  Tests named `test_kit_*` cover code the kit gives you; they always pass.
- **Check your work:** from the kit folder, run `./check.sh N`. It runs step N's tests and then every
  earlier step's tests, so you also see if you broke something you had finished.
- You can stay in one folder the whole way (finish step 1 in `steps/step-01`, then search for
  `TODO(step 2)` in the same folder) or move to the next `steps/step-NN` folder, which has the earlier
  steps solved our way. `solution/` is the finished project.
- `python -m pytest` in a starter shows failures for every step not done yet. That is expected; use
  `./check.sh N` or `python -m pytest -k test_step03_`.
- Every test uses a tiny model on the CPU. The whole suite runs in a few seconds.

Times are rough estimates for someone who writes Python comfortably, knows what a tensor and a
gradient are, and is new to Transformers.

| Step | You build | Tests | Time |
|---|---|---|---|
| 1 | Bytes as tokens, and training batches | 5 | 30–45 min |
| 2 | Causal self-attention | 6 | 1.5–2 h |
| 3 | The Transformer block | 4 | 45 min |
| 4 | The whole model: embeddings, positions, logits, loss | 9 | 1.5 h |
| 5 | Measuring: loss, perplexity, bits per byte | 4 | 1 h |
| 6 | Training: schedule, AdamW, one step, a real run | 5 | 1.5 h |
| 7 | Sampling: temperature, top-k, generation | 6 | 1 h |
| 8 | The key/value cache | 4 | 1–1.5 h |

Shapes in this workbook use **B** = batch size, **T** = sequence length, **C** = model width
(`d_model`), **H** = number of heads, **D** = C / H (width of one head), **V** = vocabulary size (256).

---

## Step 1: Bytes as tokens, and training batches

**Goal.** Turn text into integer token ids and back, and cut random training examples out of a long
stream of ids.

**Key concepts.**
- *Tokens.* A language model predicts the next token from the tokens before it. Here a token is one
  byte of the text's UTF-8 encoding, so the vocabulary is exactly 256 ids. Every possible text can be
  encoded, nothing is ever "unknown", and the tokenizer has no training of its own. The price: a
  character like `é` is two tokens, and the model must learn to spell every word byte by byte.
- *Decoding can fail.* A model can generate half of a two-byte character. Decoding replaces broken
  bytes with `U+FFFD` instead of crashing.
- *Next-token targets.* One window of `block_size + 1` tokens gives `block_size` training examples at
  once: inputs `x = ids[s : s + T]` and targets `y = ids[s + 1 : s + T + 1]`. At every position t the
  model sees `x[0..t]` and must predict `y[t]`.
- *Random windows.* Each batch takes B windows at random start positions, using a seeded
  `torch.Generator` so a run can be repeated exactly.

**Starter.** `scratchlm/tokenizer.py`: `ByteTokenizer.encode` and `decode`. `scratchlm/data.py`:
`sample_batch`. `load_text`, `split_text` and `encode_file` are given.

**Tests that must pass.** `test_step01_encode_gives_utf8_bytes`,
`test_step01_decode_roundtrips_and_tolerates_broken_bytes`,
`test_step01_targets_are_the_inputs_shifted_by_one`,
`test_step01_batches_are_random_but_reproducible`, `test_step01_every_start_position_is_reachable`
(`tests/test_step01_tokens_and_batches.py`).

**Hints.**
- `text.encode("utf-8")` is a `bytes` object; `list()` of it is already a list of ints.
  `bytes(ids).decode("utf-8", errors="replace")` goes back.
- With N tokens there are `N - block_size` valid start positions: `0 .. N - block_size - 1`. The last
  one still leaves room for the final target.
- `torch.randint(0, n, (batch_size,), generator=gen)` draws the starts. Instead of a Python loop you
  can index with `starts[:, None] + torch.arange(block_size)`, a (B, T) matrix of positions.

**Check your work.** `./check.sh 1`

---

## Step 2: Causal self-attention

**Goal.** Let every position gather information from the positions before it (and never from the
ones after it).

**Key concepts.**
- *Queries, keys, values.* Each position's vector x is projected three ways (one `nn.Linear` to 3C,
  split in three). The query says what this position is looking for, the key says what it offers,
  and the value is what it hands over if chosen.
- *Scores.* The match between query i and key j is their dot product, divided by √D. Without the
  division, scores grow with the head width and the softmax becomes nearly one-hot, which makes
  gradients vanish.
- *The causal mask.* A language model must not see the token it is predicting. Before the softmax,
  every score where key j comes after query i is set to −∞, so its weight becomes exactly 0. This is
  what makes one window give T honest training examples at once.
- *Softmax and weighted sum.* Softmax over the keys turns each row of scores into weights that sum to 1.
  The output at position i is the weighted average of the values.
- *Heads.* Instead of one attention over C channels, H independent attentions run over D = C/H
  channels each (split, attend, merge back). Different heads can learn to look for different things,
  for example "the previous character" and "the start of the word".
- *Rectangular masks.* In step 8 only the newest positions are fed in, while keys and values cover the
  whole history. Then there are fewer queries than keys, and query i sits at absolute position
  `t_k - t_q + i`. Write `causal_mask` for that general case now; with t_q = t_k it is the usual lower
  triangle.

**Starter.** `scratchlm/attention.py`: `causal_mask`, `attention`, `split_heads`, `merge_heads`.
`CausalSelfAttention.forward` is given and calls your four functions.

**Tests that must pass.** `test_step02_causal_mask_square_and_rectangular`,
`test_step02_attention_matches_pytorch`, `test_step02_weights_are_a_distribution_over_the_past_only`,
`test_step02_scores_are_scaled_by_sqrt_head_dim`, `test_step02_split_and_merge_heads`,
`test_step02_future_tokens_do_not_change_the_past` (`tests/test_step02_attention.py`).

**Hints.**
- `k.transpose(-2, -1)` swaps the last two dimensions, so `q @ k.transpose(-2, -1)` is (…, T_q, T_k)
  for any number of leading dimensions.
- `scores.masked_fill(~mask, float("-inf"))` broadcasts a (T_q, T_k) mask over batch and heads.
- Every row of a causal mask has at least one True (each query sees itself), so the softmax never
  sees a row of only −∞.
- `split_heads`: `x.view(B, T, H, D).transpose(1, 2)`. `merge_heads` undoes it; after `transpose`
  the memory is no longer contiguous, so use `.contiguous().view(...)` or `.reshape(...)`.
- The first test compares your result with PyTorch's built-in fused attention. If it is off by a lot,
  print `w[0, 0]` for a 4-token input: it should be lower triangular with rows summing to 1.

**Check your work.** `./check.sh 2`

---

## Step 3: The Transformer block

**Goal.** Combine attention with a small feed-forward network, each wrapped in a LayerNorm and a
residual connection.

**Key concepts.**
- *Attention mixes, the MLP thinks.* Attention moves information between positions. The MLP
  (`Linear(C, 4C)`, GELU, `Linear(4C, C)`) then processes each position on its own.
- *Residual connections.* Each part adds its result to its input: `x = x + attn(…)`. The input flows
  through unchanged, and each layer only has to learn a correction. This keeps gradients healthy in
  deep stacks: if a branch outputs zero, the block is exactly the identity.
- *Pre-LayerNorm.* LayerNorm rescales each position's vector to mean 0 and variance 1 (then applies a
  learned scale and shift). GPT-2 and most later models normalise the *input* of each branch
  (`attn(LN1(x))`) rather than the output, which trains more stably without careful warmup.
- *Passing things through.* `Block.forward` hands `past`, `use_cache` and `return_weights` to the
  attention layer and returns the attention weights and cache next to the new x. They matter in
  steps 4 and 8.

**Starter.** `scratchlm/model.py`: `MLP.forward` and `Block.forward`. The layers themselves are
created for you in `__init__`.

**Tests that must pass.** `test_step03_mlp_is_linear_gelu_linear`,
`test_step03_block_matches_the_pre_norm_formula`, `test_step03_block_with_silent_branches_is_the_identity`,
`test_step03_block_is_causal` (`tests/test_step03_block.py`).

**Hints.**
- `self.attn(...)` returns three values: `(output, weights, present)`.
- Use `torch.nn.functional.gelu` (imported as `F`), the same activation as GPT-2.
- The formula test computes `h = x + attn(ln1(x))` and then `h + mlp(ln2(h))`. The second LayerNorm
  reads `h`, not the original x.

**Check your work.** `./check.sh 3`

---

## Step 4: The whole model: embeddings, positions, logits, loss

**Goal.** Turn token ids into next-token logits and a loss, and know exactly how many parameters the
model has.

**Key concepts.**
- *Token embeddings.* `nn.Embedding(256, C)` is a table with one learned vector per byte.
- *Positions.* Attention by itself does not know the order of its inputs: shuffle the keys and the
  weighted sum is the same. So each position gets a vector added to its token vector. Two choices:
  a *learned* table (`nn.Embedding(block_size, C)`, as in GPT-2) or the fixed *sinusoids* of the
  original Transformer, where column 2i is sin(p / 10000^(2i/C)) and column 2i+1 the matching cos. The
  sinusoids need no parameters; the learned table usually trains a little better at this size.
- *Logits.* After the last block and a final LayerNorm, `lm_head` maps each position's C-vector to
  256 scores, one per possible next byte. Here `lm_head` shares its weight matrix with the token
  embedding ("weight tying"): fewer parameters, and a byte's input and output vectors stay related.
- *The loss.* Cross-entropy is −ln p(true next token), averaged over every position of every window,
  measured in nats. A model that guesses uniformly scores ln 256 ≈ 5.545. Targets of −100 are skipped,
  which is how you would mask out padding.
- *Counting parameters.* Working the count out from the shapes is the best check that you know what
  is in the model. Each block has two LayerNorms (weight and bias), qkv (C × 3C plus bias), proj
  (C × C plus bias), fc (C × 4C plus bias) and out (4C × C plus bias).
- *The context limit.* The model has position vectors only up to `block_size`. A longer input is an
  error, not something to silently truncate.

**Starter.** `scratchlm/model.py`: `sinusoidal_table`, `LanguageModel.positions`,
`LanguageModel.forward` (the middle part) and `analytic_param_count`. Weight initialisation and
`count_parameters` are given.

**Tests that must pass.** `test_step04_logits_shape_and_a_near_uniform_start`,
`test_step04_loss_is_cross_entropy_and_ignores_minus_100`,
`test_step04_parameter_count_matches_the_formula` (4 configurations), `test_step04_sinusoidal_table`,
`test_step04_positions_learned_and_sinusoidal`,
`test_step04_position_changes_the_prediction_and_context_is_limited` (`tests/test_step04_model.py`).

**Hints.**
- `positions(T, offset)` returns rows `offset .. offset + T - 1`. In this step `offset` is always 0;
  in step 8 it is the number of cached positions. Raise ValueError when `offset + T > block_size`.
- Build the sinusoidal table once and keep it in `self._pos_table` (it is not a parameter, so it is not
  saved in checkpoints). Move it to `self.tok_emb.weight.device` when you return it.
- `F.cross_entropy` wants (N, V) logits and (N,) targets: `logits.view(-1, V)` and
  `targets.reshape(-1)`, with `ignore_index=-100`.
- In the loop over blocks, pass `past_kv[i] if past_kv else None`, and append the weights and presents
  that are not None to the two lists that are already defined. The given `return` line uses them.
- Freshly initialised, the model should give a loss close to 5.545. Much higher means the logits are
  too large (check the initialisation is untouched); NaN usually means a mask or shape problem.

**Check your work.** `./check.sh 4`. Then try `python -m scratchlm info` to see the default model's
size (about 0.8 million parameters).

---

## Step 5: Measuring: loss, perplexity, bits per byte

**Goal.** Score a model on text it has never trained on, in numbers you can compare.

**Key concepts.**
- *Held-out text.* Training loss says how well the model fits what it has seen. Only validation text
  (here: different plays) shows whether it generalises. The kit's validation file comes from plays
  that are not in the training file (no line of 30 or more characters appears in both).
- *Every token once.* For a fair score, cut the validation stream into non-overlapping windows that
  together predict every token after the first exactly once. Random windows would count some tokens
  twice and miss others.
- *Three views of one number.* Loss in nats per token; perplexity = e^loss ("as uncertain as choosing
  uniformly among this many bytes"); bits per byte = total −log₂ p ÷ bytes predicted. Bits per byte
  does not depend on the tokenizer, which is why it is the headline number here. With one byte per
  token it is simply loss / ln 2. Knowing nothing scores 8.0; a good character-level model of English
  gets well below 2.
- *Quick checks.* During training a full evaluation would be slow, so the trainer scores an evenly
  spaced subset of windows. Final numbers (`python -m scratchlm eval`) use all of them.

**Starter.** `scratchlm/evaluate.py`: `eval_windows` and `evaluate`.

**Tests that must pass.** `test_step05_windows_predict_every_token_exactly_once`,
`test_step05_max_windows_is_an_even_subset_with_both_ends`,
`test_step05_a_model_that_knows_nothing_scores_8_bits_per_byte`,
`test_step05_matches_a_direct_computation_and_keeps_the_mode` (`tests/test_step05_evaluate.py`).

**Hints.**
- N tokens give N − 1 predictions. Windows are (start, length) with length `min(block_size, N - 1 - s)`.
- For the subset, `round(i * (len(wins) - 1) / (max_windows - 1))` for i in `0 .. max_windows - 1`
  picks evenly spaced indices that include both ends.
- `torch.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1))` picks out log p of each true token.
  Sum the negative of that over everything, then divide by the number of tokens predicted (not by the
  number of windows).
- Only windows of the same length can be stacked into one batch; the last window is usually shorter.
- `evaluate` is decorated with `@torch.no_grad()` already. Remember `model.eval()` and restore the
  previous mode at the end (`model.train(was_training)`).

**Check your work.** `./check.sh 5`

---

## Step 6: Training: schedule, AdamW, one step, a real run

**Goal.** Make the model learn, then train it on Shakespeare.

**Key concepts.**
- *One training step.* Forward pass with targets → loss → `loss.backward()` fills every parameter's
  `.grad` → the optimizer updates the weights → clear the gradients for the next step.
- *Gradient clipping.* If the gradient's total norm is above a limit (1.0), scale it down to the
  limit. One unlucky batch then cannot throw the weights far off. `clip_grad_norm_` returns the norm
  *before* clipping, which is worth logging: spikes there come before most training failures.
- *Refusing NaN.* A non-finite loss or gradient norm means something diverged. Updating the weights
  with it would poison them, so the step raises `FloatingPointError` without touching any weight and
  the trainer stops with status "diverged".
- *AdamW.* Adam keeps a running average of each parameter's gradient and of its square, and scales
  every update by them, so all parameters move at a sensible rate. The "W" is decoupled weight decay:
  weights shrink slightly every step, which acts as regularisation. It is applied only to matrices
  (Linear and Embedding weights), not to biases or LayerNorm parameters, which do not cause
  overfitting and should stay free.
- *The learning-rate schedule.* Start small and ramp up linearly for the warmup steps (Adam's
  running averages are unreliable at first), then follow half a cosine from the peak down to a floor
  of 10% at the last step.

**Starter.** `scratchlm/train.py`: `lr_at`, `make_optimizer`, `train_step`. `train()`, the loop that
calls them, evaluates every `eval_interval` steps and saves `best.pt`, `last.pt` and `summary.json`,
is given.

**Tests that must pass.** `test_step06_warmup_then_cosine_down_to_the_floor`,
`test_step06_weight_decay_only_on_matrices_and_every_parameter_once`,
`test_step06_training_steps_learn_a_pattern`, `test_step06_a_non_finite_loss_raises_and_changes_nothing`,
`test_step06_a_short_run_beats_the_uniform_baseline` (`tests/test_step06_train.py`). The two training
tests run a tiny model for a second or so.

**Hints.**
- `lr_at` with `steps=110, warmup=10, lr=1`: step 0 → 0.1, step 9 → 1.0, step 10 → 1.0, step 60 → 0.55,
  step 110 and later → 0.1.
- `model.parameters()` already yields a tied weight only once, but the test also checks it ends up in
  the decay group: `p.dim() >= 2` is true for the shared embedding matrix.
- `torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
  lr=..., betas=(beta1, beta2))`.
- Call `opt.zero_grad(set_to_none=True)` before the forward pass, so a step that raises leaves no
  half-computed gradients behind. Check `torch.isfinite(out.loss)` before `backward()`.
- `out.loss.item()` gives a plain float (without the "requires_grad" warning that `float()` prints).

**Check your work.** `./check.sh 6`

**Then train it for real (optional, under a minute).** From your step folder (or `solution/`):

```sh
python -m scratchlm train --data ../../data/shakespeare-train.txt --val ../../data/shakespeare-val.txt \
  --out runs/first --n-layer 2 --d-model 64 --block-size 64 --steps 300 --eval-interval 50
```

(From `solution/`, the data is at `../data/...`.) Watch the validation bits per byte fall from 8
(README-FIRST.md has the numbers we measured for this exact command). Training uses 2 CPU threads by default (`--threads`);
more threads are faster if your machine is otherwise idle. The default model
(`--n-layer 4 --d-model 128 --block-size 128`) and more steps should give better text; they take
longer, and we did not benchmark them for this kit.

---

## Step 7: Sampling: temperature, top-k, generation

**Goal.** Make the model write.

**Key concepts.**
- *Autoregressive generation.* Feed the context, take the logits at the last position, choose a
  token, append it, repeat. Once the context is longer than `block_size`, only the last `block_size`
  tokens are fed (the given `Decoder` does that).
- *Greedy vs. sampling.* Always taking the most likely token (temperature 0) is deterministic and
  tends to loop. Sampling from the probabilities gives variety.
- *Temperature.* Dividing the logits by T before the softmax sharpens the distribution (T < 1) or
  flattens it (T > 1). 0.7–0.9 is a common middle ground.
- *Top-k.* Keep only the k most likely tokens and renormalise. This cuts the long tail of unlikely
  bytes that would otherwise produce the occasional garbage character.
- *Reproducibility.* Drawing with `torch.multinomial(p, 1, generator=gen)` and a seeded generator
  makes the same seed give the same text.

**Starter.** `scratchlm/sample.py`: `next_token_probs` and `generate`. `Decoder` (the uncached path) is
given.

**Tests that must pass.** `test_step07_temperature_zero_is_greedy`,
`test_step07_temperature_sharpens_or_flattens`, `test_step07_top_k_keeps_only_the_k_best`,
`test_step07_generation_is_reproducible_and_has_the_right_length`,
`test_step07_a_trained_model_continues_the_pattern` (2 temperatures) (`tests/test_step07_sampling.py`).
The last test trains a tiny model on "the cat sat on the mat." for about a second, then checks that it
finishes the sentence.

**Hints.**
- `torch.topk(scaled, k).values[-1]` is the k-th largest value; mask everything strictly below it.
- Greedy: return a one-hot vector, so callers can treat both cases the same way.
- `generate` returns only the new text, not the prompt.
- `Decoder(model, use_cache=use_cache).next_logits(ids)` returns the logits (V,) for the next token.
  Move them to the CPU (`.cpu()`) before sampling, so a model on another device still works.

**Check your work.** `./check.sh 7`. If you trained a model in step 6:

```sh
python -m scratchlm sample --ckpt runs/first/best.pt --prompt "ROMEO:" --tokens 300
```

---

## Step 8: The key/value cache

**Goal.** Generate without recomputing the whole context for every new token, and prove the result is
identical.

**Key concepts.**
- *Where the waste is.* Without a cache, generating token n runs the model over all n tokens of
  context, although the keys and values of the first n − 1 positions are the same as last time
  (causal attention means earlier positions never look at later ones).
- *The cache.* Keep each layer's keys and values, shape (B, H, T_past, D). For the next token compute
  q, k, v only for the new position, put the cached k and v in front of the new ones, and attend. The
  new query sits at absolute position T_past, so the rectangular mask from step 2 applies, and its
  position vector is `positions(1, offset=T_past)` (already wired in `forward`).
- *Prefill, then decode.* The first call runs the whole prompt once and fills the cache; every later
  call feeds one token.
- *The window limit.* With learned absolute positions, once the text is longer than `block_size` the
  window slides, every token's position changes, and the cache no longer applies. The given `Decoder`
  falls back to full recomputation then. (Models with relative positions, such as RoPE, keep caching;
  see "Going further".)
- *Same numbers, less work.* The cache is an optimisation only. The tests check that cached and
  uncached logits agree to 1e-5 at every step, across the block boundary, and that generation gives
  the same text with and without it.

**Starter.** `scratchlm/attention.py`: the cache part of `CausalSelfAttention.forward`.
`scratchlm/sample.py`: the cached path of `Decoder.next_logits`.

**Tests that must pass.** `test_step08_attention_with_past_equals_full_attention`,
`test_step08_model_token_by_token_equals_one_full_pass`,
`test_step08_decoder_logits_match_across_the_block_boundary`,
`test_step08_generation_with_and_without_the_cache_is_identical` (`tests/test_step08_kv_cache.py`).

**Hints.**
- `torch.cat([past[0], k], dim=2)`: dimension 2 is time in (B, H, T, D).
- `present` must hold *all* keys and values so far (past and new), so the next call can extend it.
- In `Decoder.next_logits`, the cache is valid only if it holds exactly `ids[:-1]`. Anything else
  (first call, a different context) is a prefill of all of `ids`. Keep `list(ids)`, a copy, because the
  caller appends to its list.
- The test expects the counters to read: 1 prefill, 13 cached steps, 8 full recomputes for context
  lengths 3 to 24 with `block_size` 16.

**Check your work.** `./check.sh 8`. Then compare speeds on a trained model:

```sh
python -m scratchlm sample --ckpt runs/first/best.pt --prompt "ROMEO:" --tokens 60
python -m scratchlm sample --ckpt runs/first/best.pt --prompt "ROMEO:" --tokens 60 --kv-cache
```

Both print the same text; the time is shown at the end. With a 64-token block the window slides
after 64 tokens, so the cache only helps for the first ones; the gap grows with `--block-size`.

---

## Done. Going further

You now have a complete, tested decoder-only Transformer. Ideas that build directly on it:

- **A BPE tokenizer.** Replace bytes with byte-pair encoding: start from the 256 bytes and repeatedly
  merge the most frequent adjacent pair into a new token. Compare models by bits per byte, which stays
  fair across tokenizers.
- **Dropout and a bigger run.** Add dropout after attention and in the MLP, train the default model
  for a few thousand steps and watch the gap between training and validation loss.
- **Fused attention.** Swap your `attention` for `F.scaled_dot_product_attention` in the no-weights
  path and measure the speed-up (the step 2 test already shows they agree).
- **Rotary position embeddings (RoPE).** Rotate q and k by position instead of adding position
  vectors. Then the KV cache keeps working after the window slides.
- **Break the mask on purpose.** Remove the causal mask, train, and watch the training loss fall to
  almost zero while generation becomes nonsense: the model learned to read the answer.
- **Read the attention.** `model(x, return_weights=True).weights` gives every layer's attention
  weights; plot them for a line of verse and look for heads that attend to the previous character or
  the start of a word. Keep in mind that attention weights alone are not a full explanation of a
  prediction.
