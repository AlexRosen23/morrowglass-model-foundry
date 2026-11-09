# Morrowglass Model Foundry. Every target is CPU/MPS-local and needs no paid service.
UV ?= uv
MG := $(UV) run --frozen morrowglass
API_PORT ?= 5100
WEB_PORT ?= 5101

.PHONY: help setup data smoke test test-fast lint typecheck web-check ci demo api web standard analyze report \
        final-test bench export reproduce-scratch clean distclean serve-openai bench-kv quant-int8 \
        lm-eval-setup lm-eval export-hf serving-report

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-18s %s\n", $$1, $$2}'

setup: ## Install pinned Python (uv) and web (npm ci) dependencies
	$(UV) sync --frozen
	cd web && npm ci --no-audit --no-fund

data: ## Fetch PG #100 (5.6 MB) if missing and rebuild splits, manifest and offline sample
	$(MG) data prepare --fetch

smoke: ## Offline smoke: sanity check, sample baselines, 3 tiny CPU experiments (~1 min)
	$(MG) sanity
	$(MG) baselines --dataset shakespeare-sample
	$(MG) run scratch --profile smoke --quiet
	$(MG) run digitsum --profile smoke --quiet
	$(MG) run break-mask --profile smoke --quiet
	@echo "smoke OK: outputs in reports/smoke/ (not evidence; plumbing only)"

test: ## Full pytest suite (CPU, a few minutes); writes reports/test_summary.json
	$(UV) run --frozen pytest -q -p no:warnings --junitxml=artifacts/junit.xml
	$(UV) run --frozen python scripts/junit_summary.py artifacts/junit.xml

test-fast: ## Model/data/metric tests only
	$(UV) run --frozen pytest -q -p no:warnings tests/test_model.py tests/test_data.py tests/test_tokenizers_and_metrics.py

lint: ## ruff
	$(UV) run --frozen ruff check src tests

typecheck: ## mypy (strict defs)
	$(UV) run --frozen mypy

web-check: ## TypeScript typecheck + production build
	cd web && npm run build

ci: lint typecheck test smoke web-check ## Everything .github/workflows/ci.yml runs (Actions are disabled; run locally)

api: ## API on 127.0.0.1:$(API_PORT)
	$(MG) serve --port $(API_PORT)

web: ## Vite dev server on 127.0.0.1:$(WEB_PORT) (proxies /api)
	cd web && npm run dev

demo: ## Build the UI and serve API + UI (http://127.0.0.1:5101)
	./scripts/demo.sh

standard: ## Every standard experiment (~45-60 min on Apple silicon; idempotent)
	./scripts/run_standard.sh

analyze: ## Verdicts + plots from saved runs
	$(MG) analyze all

report: analyze ## RESULTS.md, EXPERIMENT_REPORT.md, HTML report, resume_evidence/metrics.json
	$(MG) report

# ---- serving (needs local checkpoints of the flagship and BPE runs; see docs/SERVING.md)
FLAGSHIP ?= lm-byte-small-2822813d-s0
BPE ?= lm-bpe1024-small-eb29b8fb-s0

serve-openai: ## API + OpenAI-compatible /v1 on 127.0.0.1:$(API_PORT), default model $(FLAGSHIP)
	$(MG) serve --port $(API_PORT) --model $(FLAGSHIP)

bench-kv: ## Registered KV-cache study (equivalence + tokens/s, CPU and MPS)
	$(MG) bench-kv $(FLAGSHIP)
	$(MG) bench-kv $(BPE)

quant-int8: ## Registered dynamic-int8 study (CPU)
	$(MG) quant-int8 $(FLAGSHIP)
	$(MG) quant-int8 $(BPE)

lm-eval-setup: ## Pinned lm-evaluation-harness in .venv-lmeval (separate from the project env)
	$(UV) venv .venv-lmeval --python 3.12
	VIRTUAL_ENV=.venv-lmeval $(UV) pip sync eval/lm-eval-requirements.txt

lm-eval: ## Harness through /v1/completions on 127.0.0.1:5102, every registered model (~2.5 h on a busy M5 Pro)
	$(MG) lm-eval

export-hf: ## Hub-compatible folders (safetensors + config + model card); local only, never published
	$(MG) export-hf $(FLAGSHIP)
	$(MG) export-hf lm-byte-small-240f8c28-s0
	$(MG) export-hf $(BPE)

serving-report: ## docs/SERVING_RESULTS.md from the saved serving records
	$(MG) serving-report

reproduce-scratch: ## Re-train the flagship into reports/reproduce/, compare with the record, install weights if none (~5 min)
	$(MG) data prepare --fetch
	$(MG) reproduce $(FLAGSHIP)

clean: ## Remove caches, smoke outputs and the web build (keeps checkpoints and evidence)
	rm -rf .pytest_cache .ruff_cache .mypy_cache reports/smoke web/dist artifacts/tokens artifacts/ledger.sqlite
	find . -name __pycache__ -type d -prune -not -path './.venv/*' -exec rm -rf {} +

distclean: clean ## Also delete local checkpoints and processed data (evidence JSON in git is kept)
	@echo "This deletes artifacts/ (checkpoints) and data/processed. Ctrl-C within 5 s to abort."; sleep 5
	rm -rf artifacts data/processed
