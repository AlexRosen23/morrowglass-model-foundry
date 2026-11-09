# Contributing

## Setup

```bash
make setup          # uv sync --frozen (Python 3.12, torch 2.15.0) + npm ci in web/
make data           # downloads Project Gutenberg #100 (5.6 MB) once, builds splits
make smoke          # ~1 min, offline sample, CPU
```

Requirements:
- `uv` 0.12+
- Node 24+
- macOS on Apple silicon (MPS) or any CPU machine. CUDA is not used.

## Branches and commits

- `main` is the reviewed line. Work on `feat/…`, `fix/…`, `chore/…` or `docs/…` branches, push the branch, and open a PR with `gh pr create`.
- Nothing merges to `main` without the owner's approval. The `pre-push` hook blocks direct pushes to `main`; `ALLOW_MAIN_PUSH=1` is only for deliberate exceptions.
- Commit evidence (run JSON, experiment results, plots, notes) in the same PR as the code that produced it. Runs record the git commit they started from. A `*`/`dirty` flag means the working tree had uncommitted changes.

## Checks

| command | what |
|---|---|
| `make lint` | ruff |
| `make typecheck` | mypy |
| `make test` | pytest (CPU, a few minutes) |
| `make smoke` | end-to-end tiny experiments on the offline sample |
| `make web-check` | TypeScript + production build |
| `make ci` | all of the above, which is what `.github/workflows/ci.yml` runs |

**CI status.** GitHub Actions runs `.github/workflows/ci.yml` on every pull request and on pushes to
`main`. It was enabled with the owner's approval on 2026-11-08. Minutes count against the account's
shared 2,000/month private-repo budget, so still run `make ci` locally before pushing.

## Running experiments

- Add a new experiment to `src/morrowglass/experiments/registry.py` **before** running it: title, hypothesis, criterion, arms, caveats. Then add its verdict logic to `experiments/analyze.py`.
- Run `uv run morrowglass estimate <exp>` first. The `extended` profile refuses to run without `--approved`, which should only be passed after the owner has seen the estimate (time, memory, disk). The machine is shared and has a known memory fault, so keep runs short and checkpointed.
- Never open the test split in a loop. Use `morrowglass final-test --reason "…"`, which logs every access.
- After `make report`, write the interpretation in `reports/experiments/<exp>/note.md` (Hypothesis / Setup / Observed / Limitations / Next test / Decision), then run `make report` again.

## Serving and standard evals

| command | what |
|---|---|
| `make serve-openai` | API + OpenAI-compatible `/v1/completions`, `/v1/chat/completions` on 127.0.0.1:5100 |
| `make bench-kv` | registered KV-cache study → `reports/serving/kv-cache/` |
| `make quant-int8` | registered dynamic-int8 study → `reports/serving/int8/` |
| `make lm-eval-setup && make lm-eval` | pinned lm-evaluation-harness through the API → `reports/lm_eval/` |
| `make export-hf` | Hub-compatible folders in `artifacts/hf_export/` (cards copied to `reports/hf_export/`) |
| `make serving-report` | regenerate `docs/SERVING_RESULTS.md` |

- Register a new serving or eval study in `EVALUATIONS` (`experiments/registry.py`) and commit it **before** measuring.
- Serving studies use validation text only.
- Commit before measuring: each record stores the commit and a `dirty` flag.
- The harness lives in `.venv-lmeval` (gitignored), so it never changes the project's lock file. Its full pin list is `eval/lm-eval-requirements.txt` (regenerate with `uv pip compile eval/lm-eval-requirements.in`).
- **Never push to the Hugging Face Hub without the owner's approval.** The publish command is printed by `export-hf` and listed in `docs/SERVING_RESULTS.md`.

## Environment variables

- `MORROWGLASS_WARM_BASELINE=0` skips fitting the n-gram baseline when the API starts.
- `MORROWGLASS_DEFAULT_MODEL=<run_id>` (or `serve --model`) sets the model used when a `/v1` request or `/tokenize` names none.
- `MORROWGLASS_ENABLE_HF=1` (or `serve --enable-hf`) also serves `hf:distilgpt2`, from the local Hugging Face cache only.
- `API_PORT` / `WEB_PORT` move the API and the web preview (and the web proxy target), e.g. to run a second copy on test ports.

No secrets are used. `.env*` files are gitignored; nothing needs one.

## Never committed

- Raw or processed corpora (`data/raw`, `data/processed`), checkpoints (`artifacts/`, `*.pt`), token caches, SQLite indexes, `web/node_modules`, `.env*`.
- Only the offline sample, the manifests and hashes, and the JSON evidence are in git.
- The gitleaks `pre-commit`/`pre-push` hooks scan every commit.

## Releases and rollback

- **Releases.** Annotated tags (`git tag -a v0.1.0 -m …`).
- **Rollback.** `git revert` through a PR; never force-push.
- **Deployment.** Nothing is deployed. The API binds to `127.0.0.1` only.
