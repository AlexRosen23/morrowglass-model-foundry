# Serving: KV cache, OpenAI-compatible API, standard harness, Hub export, int8

This branch turns the research lab into something you can call like a model server. The measured
numbers are in [SERVING_RESULTS.md](SERVING_RESULTS.md), which `uv run morrowglass serving-report`
generates from the saved records. This page explains what was built and how to read those numbers.

All three measured studies were registered in `src/morrowglass/experiments/registry.py`
(`EVALUATIONS`), with a hypothesis and a pass/fail criterion. Each registration was committed before
the study's first measurement (commit `d7e6512`). The studies run on existing checkpoints and read
validation text only. The test split is never touched.

## 1. KV cache (`model/transformer.py`, `inference/generate.py`)

- **What it does.** `MorrowglassLM.forward(idx, past_kv=..., use_cache=True)` returns each layer's
  keys and values, and later calls feed only the new tokens.
- **Mask detail.** When a cache is present, new queries use an offset causal mask: query `i` sits at
  absolute position `t_past + i`. `scaled_dot_product_attention(is_causal=True)` would align the mask
  top-left, which is wrong when there are fewer queries than keys.
- **`CachedDecoder`.** It prefills once and then decodes one token per step. The API, `generate()` and
  the benchmark all use this same object.
- **The window limit.** The model uses learned absolute positions. Once a sequence is longer than
  `block_size` (256), the window slides and every position changes, so the cache cannot be reused.
  From that point the decoder recomputes the last 256 tokens on every step, exactly as the uncached
  path does.
  - This is a limitation of the architecture, not of the cache. RoPE or ALiBi would remove it. That
    is a possible next step, but it would mean retraining, so it is not done here.
- **Equivalence tests** (`tests/test_kv_cache.py`, CPU, `atol=1e-5`):
  - incremental forward equals full forward for learned, sinusoidal and no position encodings;
  - a multi-token chunk after a cache gives the same result as the full forward;
  - greedy logits are identical across the window overflow;
  - sampled generation is identical with and without the cache;
  - a stale cache is never reused when the history changes.
- **Benchmark** (`morrowglass bench-kv RUN_ID`). It covers tokens/s, time-to-first-token and
  per-token latency (median and p90), cache on and off, CPU and MPS, and prompts of 16, 64 and 192
  tokens with 64 new tokens. It also checks equivalence on the real checkpoint.
- **What was measured.**
  - **CPU.** The cache gives 2.5-3.6x the decode throughput for the flagship and 2.4-4.0x for the
    BPE model. The gain grows with prompt length. The verdict for both models is *supported*.
  - **MPS.** The cache gives 0.8-1.2x, so no reliable gain. At 2M parameters and batch 1, each step
    costs about 6 ms, which is set by per-kernel launch overhead and not by attention FLOPs.
    Recomputing 250 tokens costs MPS almost nothing extra. This was reported without a criterion,
    as registered.

## 2. OpenAI-compatible endpoints (`serving/openai_api.py`, mounted on the existing FastAPI app)

| route | notes |
|---|---|
| `GET /v1/models` | local LM checkpoints (+ `hf:distilgpt2` when enabled) |
| `POST /v1/completions` | prompt as text, token ids, or a batch of either; `max_tokens` 0-400; `temperature`, `top_p`, `top_k`; `stop` (≤4); `seed`; `logprobs` 0-5; `echo`; `stream` (SSE) |
| `POST /v1/chat/completions` | `messages` → a plain-text script template (`SYSTEM.` / `USER.` / `ASSISTANT.`), default stop `"\n\n"`, SSE streaming |
| `POST /tokenize`, `POST /detokenize`, `GET /tokenizer_info` | vLLM-style; the harness uses them as its remote tokenizer |

Details:

- **Logprobs** are natural-log probabilities of the *untempered* model distribution.
  - With `echo`, `token_logprobs[0]` is `null`, as in OpenAI's API, and every later prompt token is
    scored given its prefix.
  - This is the request shape lm-evaluation-harness's `local-completions` sends: `echo=true`,
    `logprobs=1`, `max_tokens=1`, `temperature=0`. `test_echo_logprobs_match_the_model` checks the
    values against a direct forward pass.
- **Determinism.** The sampler is a CPU `torch.Generator` seeded by `seed`. When `seed` is absent it
  is seeded with 0. This differs from OpenAI, where no seed means random. The same request gives the
  same text (`test_completion_schema_and_seed_determinism`), and streaming gives the same text as
  non-streaming.
- **Streaming** holds back text that could still turn into a stop sequence, and incomplete UTF-8
  bytes, which matters for the byte-level model.
- **The chat endpoint is a format adapter, not an assistant.**
  - The models are not instruction-tuned. Every chat response carries
    `morrowglass.note: "This model is not instruction-tuned. …"` and the template.
  - Replies are Shakespeare pastiche.
- **Errors** use OpenAI's `{"error": {message, type, param, code}}` shape. Penalties and `n>1` are
  rejected, not silently ignored.
- **Loopback only.** `morrowglass serve` refuses any host except `127.0.0.1`, `::1` and `localhost`
  (`test_serve_refuses_non_loopback_hosts`). There is no auth, because nothing is exposed.
- **Reference model.** `hf:distilgpt2` loads from the local Hugging Face cache only
  (`local_files_only=True`, `HF_HUB_OFFLINE=1`). It is off unless the server runs with `--enable-hf`.

```bash
uv run morrowglass serve --port 5100 --model lm-byte-small-2822813d-s0
curl -s 127.0.0.1:5100/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt": "HAMLET.\n", "max_tokens": 60, "temperature": 0.8, "seed": 0, "stop": ["\n\n"]}'
# Any OpenAI client works: base_url="http://127.0.0.1:5100/v1", api_key="unused"
```

## 3. EleutherAI lm-evaluation-harness through that endpoint (`serving/harness.py`)

```bash
uv venv .venv-lmeval --python 3.12 && VIRTUAL_ENV=.venv-lmeval uv pip sync eval/lm-eval-requirements.txt
uv run morrowglass lm-eval            # all registered models, one at a time, API on 127.0.0.1:5102
uv run morrowglass serving-report
```

- **Pinned harness.** lm-eval 0.4.13 and all its transitive dependencies are pinned in
  `eval/lm-eval-requirements.txt`. The harness lives in its own venv: it talks to the model only over
  HTTP, so it never touches the project's dependency set.
- **Tasks** (0-shot):
  - `lambada_openai`: all 5,153 docs; accuracy on the last word.
  - `wikitext`: wikitext-2 document-level test set, all 62 docs; rolling log-likelihood.
  - `morrowglass_shakespeare_val`: our own validation split, defined in `eval/tasks/`.
- **Why the custom task.** The validation task is the plumbing check: the harness, going through
  HTTP, the server's tokenizer and echo logprobs, should reproduce the bits/byte that the project's
  own evaluator recorded in `summary.json`.
  - It cannot match exactly. The harness scores 254-token windows after a newline prefix; ours scores
    256-token windows with no prefix.
  - The registered tolerance is 0.03 bpb. Measured differences were -0.0012, -0.0007 and +0.0009
    (verdict *supported*).
- **Wall time.** On a heavily loaded machine: 15-32 min per scratch model, 55 min for distilgpt2.

### Why these numbers are expected to be weak, and what they are for

- **What the models know.** The scratch models have about 2M parameters and were trained on 4.5 MB
  of one author's 400-year-old plays.
- **Why lambada is hard for them.** It asks for the exact last word of a modern English passage,
  which usually needs an entity from far back in the passage. At byte level the model sees only the
  last 256 bytes, roughly two sentences, and it has never seen modern prose. Near-zero accuracy is the
  expected result, and the registration says so in advance.
- **Wikitext** measures how surprised a Shakespeare model is by Wikipedia English. It is expected to
  be far worse than on Shakespeare, and worse than distilgpt2. distilgpt2 has 82M parameters, saw
  OpenWebText, and gets 1,024 tokens of context.
- **What the numbers are for:**
  1. **Comparability.** These are the same tasks, metric definitions and harness version that public
     leaderboards use, so any later model of this repo (bigger, RoPE, more data) can be placed on the
     same axis.
  2. **Plumbing.** They show that the server's logprobs are correct enough for a third-party
     evaluator to reproduce our own number.
  3. **Honesty about scope.** They show, in standard units, how narrow a from-scratch 2M-parameter
     model is.
- **What they are not.** They are not a claim of general language ability.

### Wikitext quirk

lm-eval's wikitext task divides by the bytes of the *original* page, but scores the detokenized
text. Its bits/byte is therefore not exactly comparable to our own evaluator's. Compare models only
within the harness table.

## 4. Hugging Face Hub-compatible export (`hub.py`)

- **The class.** `MorrowglassHubLM` is `MorrowglassLM` with `huggingface_hub.PyTorchModelHubMixin`
  added.
- **`uv run morrowglass export-hf RUN_ID`** writes `artifacts/hf_export/<run_id>/` (not in git)
  containing:
  - `model.safetensors`;
  - `config.json` (the `ModelConfig`);
  - `README.md`, a model card with YAML metadata, auto-filled from the run records: architecture,
    data, val/test bits-per-byte with CIs, the pre-registered verdicts the run belongs to, serving
    numbers, harness numbers, intended use, limitations and licence;
  - `morrowglass.json` (hashes, tokenizer, round-trip check);
  - `tokenizer.json` for BPE runs.
- **Committed copies.** The card, config and `morrowglass.json` are copied to
  `reports/hf_export/<run_id>/` so reviewers can read them without the weights.
- **Round trip.** `save_pretrained → from_pretrained` gives bit-identical logits, and tied weights
  survive (`tests/test_hub_export.py`). The test also patches `HfApi.upload_folder` and `create_repo`
  to fail, to prove the export never touches the network.
- **Not published.** Publishing needs the owner's approval. The command they would run is in
  [SERVING_RESULTS.md](SERVING_RESULTS.md) and in each `morrowglass.json`:

  ```bash
  uv run hf auth login     # owner's own token
  uv run hf upload AlexRosen23/morrowglass-lm-byte-small-2822813d-s0 artifacts/hf_export/lm-byte-small-2822813d-s0 . --repo-type model --private
  ```

## 5. Dynamic int8 quantization on CPU (`inference/quant.py`)

- **What it does.** `uv run morrowglass quant-int8 RUN_ID` applies
  `torch.ao.quantization.quantize_dynamic` on the qnnpack engine. Every `nn.Linear` is quantized to
  int8; embeddings and LayerNorms stay fp32.
- **What it measures:**
  - full-validation bits/byte with the project's own evaluator;
  - cached greedy decode tokens/s (64-token prompt + 64 new tokens, median of 5);
  - serialized weight size;
  - greedy-token agreement with fp32.
- **Caveats:**
  - `torch.ao.quantization` is deprecated in torch 2.15; torchao is its successor.
  - Dynamic int8 picks activation scales per call. Cached and uncached decoding of the *quantized*
    model therefore differ slightly: up to about 6e-3 in logits on a toy model in `tests/test_quant.py`,
    whereas fp32 agrees to 1e-5.
- **Results.** The registered hypothesis was **not supported**, for both runs:
  - Quality held: validation bpb moved by +0.0016 and +0.0031.
  - Weights shrank by about 3x.
  - Cached CPU decoding got *slower*: 0.77x and 0.54x.
  - At about 2M parameters, the per-call cost of quantizing activations outweighs the savings from int8
    matmuls.
  - These timings were taken while the machine was heavily loaded, and fp32 decoding itself ran far
    slower than in the KV-cache study. Treat the speed ratio as indicative only.
  - The full table is in [SERVING_RESULTS.md](SERVING_RESULTS.md).

## Workbench

The **Serving** tab (key `6`) shows:
- the KV-cache, int8 and harness tables, marked `MEASURED` and read from `/api/serving`, which serves
  the files above;
- a `LIVE SAMPLE` box that calls `POST /v1/completions` for the selected checkpoint and shows the
  equivalent `curl`.
