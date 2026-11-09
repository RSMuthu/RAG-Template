# Everyday commands. `make check` is what CI runs.
.PHONY: install lint validate test eval check ingest serve up down integration \
	ollama-up ollama-test ollama-down
OLLAMA := docker compose -f compose.ollama.yaml

install:
	uv sync --all-extras

lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run pyright

validate:  # the spec and every overlay load, and reach their adoption level
	uv run harnessweaver spec validate harness.yaml -e offline
	uv run harnessweaver spec validate harness.yaml -e prod
	uv run harnessweaver spec validate harness.yaml -e ollama

test:  # offline: no network, no keys, no spend
	uv run harnessweaver test .
	uv run pytest -q -m harnessweaver_conformance --harnessweaver-conformance=app.plugin:plugin

eval:  # quality gates against the committed baselines in .harness/baselines
	HARNESSWEAVER_ENV=offline uv run rag ingest
	uv run harnessweaver eval run harness.yaml -e offline

check: lint validate test eval

ingest:
	uv run rag ingest

serve:
	uv run rag serve --reload

up:
	docker compose up -d --build

down:
	docker compose down

integration:  # the store contract against real Qdrant and Postgres
	docker compose --profile pgvector up -d qdrant postgres
	DATABASE_URL=postgresql://rag:rag@localhost:5432/rag uv run pytest -q -m integration

ollama-up:  # Qdrant + Ollama + the API, all local (the first run downloads ~2.3 GB of models)
	$(OLLAMA) up -d --build --wait
	$(OLLAMA) run --rm app rag ingest

ollama-test:  # live checks and the eval suites against the running local stack
	OLLAMA_API_KEY=ollama uv run pytest -q -m stack
	OLLAMA_API_KEY=ollama uv run harnessweaver eval run harness.yaml -e ollama

ollama-down:
	$(OLLAMA) down
