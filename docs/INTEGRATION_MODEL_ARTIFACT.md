# Integrating a Morrowglass model artifact (contract v1)

A deployment or release system, for example a canary or rollback service, can ingest a Morrowglass
model without access to this repository. It needs only the exported directory and the schema in
`contracts/model_artifact_v1.schema.json`.

## Export

```bash
uv run morrowglass export-artifact <run_id>
# -> artifacts/exports/<run_id>/model_artifact.json, model.pt [, tokenizer.json]
```

The export is not in git because it contains weights. The manifest is small and self-describing.

## Manifest fields

| field | meaning |
|---|---|
| `schema`, `schema_version` | `morrowglass.model_artifact`, `1.x.y`; consumers should accept any 1.x |
| `model_name`, `model_version` | version = producing run id `<name>-<config-hash8>-s<seed>` (deterministic in the config) |
| `producer` | project, git commit, dirty flag, run id |
| `artifact` | `path` (relative to the manifest), `format: pytorch-state-dict`, `sha256`, `bytes`, how to load |
| `tokenizer` | name (`byte` or `bpeN`), sha256, optional `tokenizer.json` (HF tokenizers format) |
| `architecture.config` | `vocab_size`, `block_size`, `n_layer`, `n_head`, `d_model`, `pos_encoding`, `causal`, ... (the `ModelConfig` fields) |
| `io.input` | int64 token ids, shape `[batch, seq ≤ block_size]` |
| `io.output` | float32 logits, shape `[batch, seq, vocab_size]`; softmax gives model probabilities (uncalibrated) |
| `evaluation` | dataset + hash, primary metric (bits/byte), val bpb + CI + ECE, test bpb + CI, or `null` if the locked test was never opened for this run |
| `tests` | pytest ids that guard the properties the artifact relies on (causality, checkpoint integrity, determinism, schema) |
| `license` | code, weights, training data |
| `runtime` | Python, torch version, devices, approximate memory |
| `limitations` | free-text list; consumers should show it |

## Consumer checklist

1. Validate the manifest against the schema. The Python helper `morrowglass.contracts.validate_manifest` has no extra dependencies.
2. Verify `sha256(model.pt) == artifact.sha256` **before** loading. Reject on mismatch.
3. Load with `torch.load(path, weights_only=True)`, never full pickle.
4. Gate a release on `evaluation.val.bits_per_byte` (and `test` when present) against the currently deployed version's manifest. Lower is better. Treat differences inside the CIs as no change.
5. Keep the manifest with the deployment record so a rollback can restore the exact `sha256`.

## Versioning

- Additive fields bump the minor version (1.1.0).
- Renamed or removed fields bump the major version (2.0.0).
- The schema file is the source of truth; this note explains it.

No other repository is edited by this project. Integration happens only through this contract.
