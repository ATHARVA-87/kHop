# Plexus — one command per step. Run inside the `torch` conda env:
#   conda activate torch
PY ?= python
CONFIG ?= configs/a0_naive.yaml

.PHONY: help doctor install install-ml lint fmt test test-all ingest process index candidates golden eval eval-full judge-export judge-agreement compare gate gate-ci extract serve ui

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-12s %s\n", $$1, $$2}'

doctor: ## Check prerequisites and print how to fix anything missing
	$(PY) -m plexus.doctor --phase $${PHASE:-0}

install: ## Install core + dev deps (editable)
	$(PY) -m pip install -c constraints.txt -e '.[dev]'
	pre-commit install

install-ml: ## Install heavy ML deps (torch stays pinned at 2.5.1)
	$(PY) -m pip install -c constraints.txt -e '.[ml,dev]'

lint: ## Ruff lint + format check
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

fmt: ## Auto-format
	$(PY) -m ruff check --fix src tests
	$(PY) -m ruff format src tests

test: ## Fast tests (no ML models, no network) — what CI runs
	$(PY) -m pytest -m "not ml and not network"

test-all: ## All tests except network
	$(PY) -m pytest -m "not network"

ingest: ## Download the frozen EDGAR snapshot (resumable, cached)
	$(PY) -m plexus.ingest.edgar --config configs/corpus.yaml

process: ## Parse the snapshot into sections + write the parse report
	$(PY) -m plexus.ingest.process --snapshot $${SNAPSHOT:-semis-2025}

index: ## Chunk, embed and index the parsed snapshot for $(CONFIG)
	$(PY) -m plexus.index.build --config $(CONFIG)

candidates: ## Generate ~200 golden-set candidates into data/golden/review.csv (LLM, cached)
	$(PY) -m plexus.eval.candidates --config configs/golden_candidates.yaml

golden: ## Build data/golden/golden.jsonl from your reviewed review.csv (+ noise rate)
	$(PY) -m plexus.eval.golden build

extract: ## Graph extraction (Phase 3)
	@echo "extract: not implemented yet (Phase 3)"; exit 1

eval: ## Retrieval metrics for $(CONFIG) on the verified golden set (no LLM calls)
	$(PY) -m plexus.eval.run --config $(CONFIG)

eval-full: ## Retrieval + generation + judge metrics for $(CONFIG) (LLM calls, cached)
	$(PY) -m plexus.eval.run --config $(CONFIG) --judge

judge-export: ## Export answers from results/$${NAME:-a0_naive}.json for blind human labelling
	$(PY) -m plexus.eval.judge export --results results/$${NAME:-a0_naive}.json

judge-agreement: ## Cohen's kappa between the judge and your labels
	$(PY) -m plexus.eval.judge agreement --results results/$${NAME:-a0_naive}.json

gate: ## retrieval regression gate vs committed results (full set; needs GPU models)
	$(PY) -m plexus.eval.gate --config configs/a1h_bm25.yaml --config configs/a1h_hybrid.yaml --config configs/a3h_rerank.yaml

gate-ci: ## the CI subset of the gate: BM25 path only, no torch (needs data/processed)
	$(PY) -m plexus.index.build --config configs/a1h_bm25.yaml --chunks-only
	$(PY) -m plexus.eval.gate --config configs/a1h_bm25.yaml

compare: ## paired bootstrap of B vs A on the same queries: make compare A=a0_naive B=a0_bm25
	$(PY) -m plexus.eval.compare results/$(A).json results/$(B).json

serve: ## FastAPI server (Phase 6)
	@echo "serve: not implemented yet (Phase 6)"; exit 1

ui: ## Next.js UI (Phase 6)
	@echo "ui: not implemented yet (Phase 6)"; exit 1
