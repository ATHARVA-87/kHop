# Plexus — one command per step. Run inside the `torch` conda env:
#   conda activate torch
PY ?= python
CONFIG ?= configs/a0_naive.yaml

.PHONY: help doctor install install-ml lint fmt test test-all ingest process index candidates golden eval eval-full judge-export judge-agreement compare gate gate-ci golden-test test-candidates graph ragas ask flow-check security ui-dev extract serve ui

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

extract: ## Graph extraction over 10-K passages with local qwen3.5:4b (~4.5 h, resumable)
	$(PY) -m plexus.extract.extract

graph: ## Re-validate stored extractions (no LLM), resolve entities, build the LadybugDB graph
	$(PY) -m plexus.extract.extract --revalidate
	$(PY) -m plexus.extract.resolve
	$(PY) -m plexus.index.graph build

RAGAS_PY ?= $(HOME)/miniconda3/envs/ragas/bin/python
ragas: ## RAGAS Faithfulness + FactualCorrectness in the separate `ragas` env (RESULTS=results/a6_router.json)
	RAGAS_DO_NOT_TRACK=true $(RAGAS_PY) -m plexus.eval.ragas_eval --results $${RESULTS:-results/a6_router.json}

ask: ## One question through the LangGraph flow with its step trace (Q="...", CONFIG=configs/a6_router.yaml)
	$(PY) -m plexus.agents.flow --config $${CONFIG:-configs/a6_router.yaml} "$(Q)"

security: ## Secrets/home-path hooks on all files + known-vulnerability audit of uv.lock (docs/security.md)
	pre-commit run --all-files
	@mkdir -p .cache
	uv export --frozen --all-extras --no-hashes --no-emit-project -q -o .cache/requirements-audit.txt
	-uvx --from pip-audit==2.9.0 pip-audit -r .cache/requirements-audit.txt --no-deps --disable-pip --progress-spinner off -f json -o .cache/pip-audit.json
	$(PY) -m plexus.audit .cache/pip-audit.json
	cd web && NODE_OPTIONS=--dns-result-order=ipv4first npm audit --omit=dev --audit-level=low

flow-check: ## Cache-only: the LangGraph flow must reproduce every answer of results/a6_router.json
	$(PY) -m plexus.agents.flow --config configs/a6_router.yaml --check --results results/a6_router.json

golden-test: ## Build data/golden/golden_test.jsonl from your reviewed review_test.csv
	$(PY) -m plexus.eval.golden build --split test

test-candidates: ## Held-out test-set candidates (Gemini, user-style, excludes dev passages)
	$(PY) -m plexus.eval.candidates --config configs/golden_test.yaml --out data/golden/review_test.csv

eval: ## Retrieval metrics for $(CONFIG) on the verified golden set (no LLM calls)
	$(PY) -m plexus.eval.run --config $(CONFIG)

eval-full: ## Retrieval + generation + judge metrics for $(CONFIG) (LLM calls, cached)
	$(PY) -m plexus.eval.run --config $(CONFIG) --judge

judge-export: ## Export answers from results/$${NAME:-a0_naive}.json for blind human labelling
	$(PY) -m plexus.eval.judge export --results results/$${NAME:-a0_naive}.json

judge-agreement: ## Cohen's kappa between the judge and your labels
	$(PY) -m plexus.eval.judge agreement --results results/$${NAME:-a0_naive}.json

gate: ## retrieval regression gate vs committed results (full set; needs GPU models)
	$(PY) -m plexus.eval.gate --config configs/a1h_bm25.yaml --config configs/a1h_hybrid.yaml --config configs/a3h_rerank.yaml --config configs/a5_diverse.yaml

gate-ci: ## CI gate, no torch: BM25, and graph + A5 on BM25 (needs data/processed + graph extractions)
	$(PY) -m plexus.index.build --config configs/a1h_bm25.yaml --chunks-only
	$(PY) -m plexus.extract.extract --revalidate
	$(PY) -m plexus.extract.resolve
	$(PY) -m plexus.index.graph build
	$(PY) -m plexus.eval.gate --config configs/a1h_bm25.yaml --config configs/a5_ci_bm25.yaml

compare: ## paired bootstrap of B vs A on the same queries: make compare A=a0_naive B=a0_bm25
	$(PY) -m plexus.eval.compare results/$(A).json results/$(B).json

serve: ## Plexus app on http://127.0.0.1:8000 (API + UI; needs Ollama running). CONFIG=configs/app.yaml
	PLEXUS_CONFIG=$${CONFIG:-configs/app.yaml} $(PY) -m uvicorn plexus.serve.app:app --host 127.0.0.1 --port 8000

ui: ## Build the web page (static export into web/out; make serve then serves it). Telemetry off
	cd web && NEXT_TELEMETRY_DISABLED=1 NODE_OPTIONS=--dns-result-order=ipv4first npm run build

ui-dev: ## Live-reload UI on 127.0.0.1:3000 against make serve (dev only)
	cd web && NEXT_TELEMETRY_DISABLED=1 NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000 npm run dev
