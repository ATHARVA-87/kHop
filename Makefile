# Plexus — one command per step. Run inside the `torch` conda env:
#   conda activate torch
PY ?= python
CONFIG ?= configs/a0_naive.yaml

.PHONY: help doctor install install-ml lint fmt test test-all ingest index extract eval serve ui

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

index: ## Parse, chunk and index the snapshot for $(CONFIG)
	$(PY) -m plexus.index.build --config $(CONFIG)

extract: ## Graph extraction (Phase 3)
	@echo "extract: not implemented yet (Phase 3)"; exit 1

eval: ## Score $(CONFIG) on the verified golden set
	$(PY) -m plexus.eval.run --config $(CONFIG)

serve: ## FastAPI server (Phase 6)
	@echo "serve: not implemented yet (Phase 6)"; exit 1

ui: ## Next.js UI (Phase 6)
	@echo "ui: not implemented yet (Phase 6)"; exit 1
