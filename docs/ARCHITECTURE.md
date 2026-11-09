# Architecture

Morrowglass is a small, typed Python package (`src/morrowglass`) plus a React workbench (`web/`).
Everything the UI shows is read from JSON records that the CLI writes; the UI never trains a
model except the bounded smoke job.

```
                 ┌──────────── CLI: morrowglass … (src/morrowglass/cli.py) ────────────┐
 data/raw/pg100 ─┤ data/corpus.py ──► data/processed/{train,val,test}.txt + manifest    │
 (Gutenberg #100)│ tokenizer/      ──► byte (256) | byte-level BPE (HF tokenizers, train) │
                 │ data/loader.py  ──► token ids (cached .npy, sha256) + batch sampler    │
                 │ model/          ──► MorrowglassLM (native PyTorch decoder-only)        │
                 │ train/          ──► Trainer: AdamW, warmup+cosine, accum, clip, early  │
                 │                     stop, non-finite handling, verified checkpoints     │
                 │ eval/           ──► bits/byte + bootstrap CI, ECE, Kneser-Ney baselines │
                 │ experiments/    ──► registry (pre-registered) → runner → analyze → report│
                 └────────────────────────────────┬────────────────────────────────────────┘
                                                  ▼
   reports/runs/<run_id>/{config,metadata,summary,test_eval,bench}.json + metrics.jsonl   (in git)
   reports/experiments/<exp>/{result.json,*.png,note.md}                                 (in git)
   artifacts/checkpoints/<run_id>/{best,last}.pt + .sha256.json                          (NOT in git)
   artifacts/ledger.sqlite  (derived index, rebuilt from the JSON)                        (NOT in git)
                                                  ▼
            api/ (FastAPI :5100, read-only + live sampling + bounded smoke job)
                 + serving/ (OpenAI-compatible /v1, KV-cached decoding; lm-eval harness talks to it)
                                                  ▼
            web/ (Vite + React + TypeScript :5101): blueprint │ evidence │ ledger
```

## Modules

| module | responsibility | key guarantees (tested) |
|---|---|---|
| `repro.py` | seeds, sha256, git provenance, `write_verified` | atomic write + read-back hash check |
| `hardware.py` | detect chip/RAM/MPS; `auto` → MPS if a real op runs, else CPU | never assumes CUDA |
| `data/corpus.py` | strip Gutenberg boilerplate, normalise, split **by play** | splits disjoint by work; overlap measured |
| `data/synthetic.py` | digit-sum mod 10 with length-shift splits | no expression in two splits |
| `tokenizer/tokenizers.py` | byte and byte-level BPE behind one protocol | lossless round trip; BPE fit on train only |
| `model/transformer.py` | the network, parameter count, FLOPs, blueprint | causal leakage test; analytic == actual params |
| `train/trainer.py` | the loop, checkpoints, resume | resume == uninterrupted (bit-identical, CPU) |
| `train/tasks.py` | LM task and digit-sum task; test splits locked | `final_eval` never touches test |
| `eval/metrics.py` | windowed held-out eval, bpb, bootstrap CI, ECE | bpb == loss/ln2 for bytes |
| `eval/ngram.py` | uniform, unigram, interpolated Kneser-Ney | distributions sum to 1 |
| `experiments/` | registry → runner → analyze → report; `final-test` | test access appended to a log |
| `inference/` | sampling with per-token scores; attention maps; latency bench | determinism under seed |
| `contracts.py` | portable model-artifact manifest (`contracts/model_artifact_v1.schema.json`) | export validates against schema |
| `api/` | FastAPI + SQLite index | path traversal → 404; only smoke jobs |
| `serving/openai_api.py` | OpenAI-compatible `/v1/completions` (echo, logprobs, stop, seed, SSE) and `/v1/chat/completions`; vLLM-style `/tokenize` | echo logprobs == direct forward; seed determinism; stream == non-stream; loopback only |
| `serving/backends.py` | scratch checkpoints (KV-cached) and an optional `hf:distilgpt2` reference (local cache only) | HF off unless enabled |
| `serving/harness.py` | runs the pinned lm-evaluation-harness against the API | `--limit` runs never write `reports/` |
| `hub.py` | `PyTorchModelHubMixin` export + auto-filled model card | save → `from_pretrained` gives identical logits; no network |
| `inference/quant.py` | dynamic int8 copy (CPU) | original model untouched |

## The model, line by line

`MorrowglassLM` (`model/transformer.py`) is a pre-LayerNorm GPT-style decoder:

1. **Token embedding** `tok_emb: V × C`. Its weight is tied to the output projection.
2. **Positions**: a learned `block_size × C` table (default), fixed sinusoids, or none (switch with `pos_encoding`).
3. **Blocks** × L, each `x = x + Attn(LN(x)); x = x + MLP(LN(x))`:
   - Q, K and V come from a single `C → 3C` linear layer and are reshaped to `(B, H, T, D)`;
   - `scores = QKᵀ/√D`, set to −∞ where key index > query index (the **causal mask**), then softmaxed;
   - `y = softmax · V`; heads are merged and passed through the output projection `W_o`;
   - MLP: `C → 4C`, GELU, `4C → C`.
4. **Final LayerNorm**, then logits `(B, T, V)`.

Training uses the fused `scaled_dot_product_attention(is_causal=True)` kernel. When attention weights are
requested (UI, tests), the explicit path above runs instead. `test_fused_and_explicit_attention_agree`
checks that the two paths give the same logits.

Residual projections are initialised with std `0.02/√(2L)`, following GPT-2, so the residual stream's
variance does not grow with depth.

## KV cache

With `use_cache=True` every block returns its keys and values. Later calls feed only the new tokens.

- Positions continue from the cache length.
- New queries use an offset causal mask, so query `i` may read keys `0..t_past+i`.
- Learned absolute positions mean the cache is only valid while the sequence fits in `block_size`.
  Past that, `CachedDecoder` recomputes the sliding window every step, exactly as the uncached path does.

See [SERVING.md](SERVING.md).

## Why these choices

See [ENGINEERING_DECISIONS.md](ENGINEERING_DECISIONS.md) for the trade-offs and the alternatives rejected.

## Ports

- API `127.0.0.1:5100`
- web `127.0.0.1:5101`

Both were checked free with `lsof` on 2026-11-08. The web dev server proxies `/api` to the API.
