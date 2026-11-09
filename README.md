# RAG Template

A production-grade starting point for retrieval-augmented generation (RAG) services, governed by
[HarnessWeaver](https://harnessweaver.wisedreamerz.com/). Clone it, point it at your documents and
your models, and ship.

HarnessWeaver puts the controls around every request: PII redaction, prompt-injection checks on
the question **and** on retrieved passages, per-tenant ACLs re-checked on every passage, answers
that must cite what was retrieved, budgets, tracing and eval gates. This template adds what
HarnessWeaver deliberately leaves out: vector stores, embeddings, ingestion and an HTTP API.

- **Retrieval**: Qdrant, pgvector or an in-memory store; any OpenAI-compatible embedding endpoint;
  optional reranking through your gateway. Swap any piece with one line of config.
- **Ingestion**: folders and URLs; Markdown, text, HTML, PDF and DOCX; incremental (unchanged
  documents are skipped) with `--prune`; ACLs per source.
- **Serving**: FastAPI with cited answers, server-sent-event streaming, search, API-key auth mapped
  to tenants, health and readiness checks. Plus a `rag` CLI.
- **Quality**: offline tests, a golden and a red-team eval suite with committed baselines, and the
  HarnessWeaver plugin conformance suite, all in CI.
- **Operations**: Dockerfile, Docker Compose (app, LiteLLM gateway, Qdrant, pgvector, OpenTelemetry
  collector), GitHub Actions.

## Quickstart (offline: no keys, no network)

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync --all-extras
```

```bash
HARNESSWEAVER_ENV=offline uv run rag ingest
```

```bash
HARNESSWEAVER_ENV=offline uv run rag ask "How long does a refund take?"
```

```bash
HARNESSWEAVER_ENV=offline uv run rag serve
```

```bash
curl -s localhost:8000/v1/answer -H 'content-type: application/json' -d '{"question": "How is my data encrypted?"}'
```

The offline environment (`harness.offline.yaml`) swaps the model for a scripted one and the
embeddings for hashed words, so everything runs on your machine. `make check` runs what CI runs.

## Fully local with Ollama (real models, no keys)

`compose.ollama.yaml` runs the service with Qdrant and [Ollama](https://ollama.com):
`llama3.2:3b` answers and `nomic-embed-text` embeds. Nothing leaves your machine.

```bash
make ollama-up
```

```bash
curl -s localhost:8000/v1/answer -H 'content-type: application/json' -d '{"question": "How long does a refund take?"}'
```

```bash
make ollama-test
```

`make ollama-up` builds and starts the stack, downloads the models on the first run (about
2.3 GB) and indexes `data/sample`. `make ollama-test` runs the live checks (`pytest -m stack`)
and the eval suites against it. The stack uses `harness.ollama.yaml`, whose `localhost` URLs
also work from your machine, so the CLI can use it directly:

```bash
OLLAMA_API_KEY=ollama uv run rag ask "How fast does support reply?" -e ollama --tenant globex --principal ann
```

See [docs/deployment.md](docs/deployment.md#fully-local-with-ollama) for models, GPUs and tuning.

## Go live

1. `cp .env.example .env` and fill in the provider keys and the model behind each alias
   (`HARNESSWEAVER_UPSTREAM_RAG_CHAT`, `..._RAG_EMBED`).
2. `cp api-keys.example.yaml api-keys.yaml` and generate real keys.
3. Start the stack, then index your documents:

```bash
docker compose up -d --build
```

```bash
docker compose run --rm app rag ingest
```

```bash
curl -s localhost:8000/v1/answer -H 'Authorization: Bearer <your key>' -H 'content-type: application/json' -d '{"question": "How long does a refund take?"}'
```

To run outside Docker against your own gateway and Qdrant, run `uv run rag serve` with no
`HARNESSWEAVER_ENV`: `harness.yaml` points at `localhost`. See [docs/deployment.md](docs/deployment.md).

## Make it yours

1. **Documents**: replace `data/sample/` and list your folders and URLs in `ingest.yaml`, with an
   `acl` for anything that isn't for everyone.
2. **Models**: set the aliases in `harness.yaml` (`models.aliases`, `context.kb.embedder.model`)
   and what they map to in `.env`.
3. **Store**: `context.kb.store.kind` is `qdrant`, `pgvector` or `memory`.
4. **Prompt**: edit `prompts/answer@1.md`, or add `answer@2.md` and point the flow at it.
5. **Evals**: replace the cases in `evals/` with questions about your documents, then
   `uv run harnessweaver eval run harness.yaml -e offline --update-baseline`.
6. **Auth**: keep API keys, or replace `app/auth.py:current_caller` with your identity provider.

[docs/extending.md](docs/extending.md) has a recipe for each, plus adding your own store,
embedder, reranker, loader, guard or task.

## Configuration

| What | Where |
|---|---|
| Models, retrieval, guards, budgets, policy, evals | `harness.yaml` |
| Per-environment differences | `harness.offline.yaml`, `harness.ollama.yaml`, `harness.prod.yaml` (picked by `HARNESSWEAVER_ENV`) |
| What to index, chunking, ACLs | `ingest.yaml` |
| The answer prompt | `prompts/answer@1.md` |
| Gateway models and provider keys | `deploy/litellm.yaml`, `.env` |
| API keys → tenant, principal, roles | `api-keys.yaml` (`RAG_API_KEYS_FILE`) |
| Spec file for the API | `RAG_SPEC` (default `harness.yaml`) |

Retrieval settings live under `context.kb` in `harness.yaml`:

```yaml
context:
  kb:
    use: app.retriever
    max_items: 6          # passages per question
    max_tokens: 4000      # and at most this much text
    min_score: null       # drop weak matches (tune with `rag search`)
    store:    { kind: qdrant, url: http://localhost:6333, collection: kb }
    embedder: { kind: openai, model: rag-embed, base_url: http://localhost:4000 }
    rerank:   { kind: none }   # or gateway
```

`kind` takes a built-in name or an import path to your own class (`mypkg.stores:MyStore`).
`rag ingest` reads the same block, so indexing and retrieval always use the same store and embedder.

## API

| Endpoint | Does |
|---|---|
| `POST /v1/answer` | `{"question": "..."}` → `{run_id, found, answer, citations: [{id, title, uri, page}]}`. `Idempotency-Key` header supported. 422 when a control blocks the run |
| `POST /v1/answer/stream` | Server-sent events: `token` (text), `reset` (discard the text so far: the answer is being repaired), then `result` (as above) or `error` |
| `POST /v1/search` | `{"question": "..."}` → the passages this caller may see, with scores |
| `GET /healthz`, `GET /readyz` | Liveness; readiness (the vector store answers; the model is not called) |

Send the API key as `Authorization: Bearer <key>` or `X-API-Key`. Without `RAG_API_KEYS_FILE`
every caller is anonymous in the `default` tenant, and `HARNESSWEAVER_ENV=prod` refuses to start.

## Commands

| Command | Does |
|---|---|
| `uv run rag ingest [--prune]` | Index `ingest.yaml`'s sources (incremental) |
| `uv run rag ask "..." [--tenant t --principal p --role r]` | Answer one question |
| `uv run rag search "..."` | Show what retrieval returns, with scores |
| `uv run rag serve [--reload]` | Serve the API |
| `make check` | Lint, types, spec validation, offline tests, conformance, eval gates |
| `make integration` | The store contract against real Qdrant and Postgres (Docker) |
| `make ollama-up`, `make ollama-test`, `make ollama-down` | The fully local stack: start it, test it live, stop it |

## Project layout

```
harness.yaml, harness.*.yaml   the spec and its environment overlays
ingest.yaml                    what to index
prompts/  evals/  data/        prompts, eval suites, sample documents
app/flow.py                    the answer task: retrieve, then answer with citations
app/retrieval/                 the app.retriever context provider: stores, embedders, reranker
app/ingest/                    sources, loaders, chunking, the incremental pipeline
app/api.py  app/auth.py        the HTTP API and who is calling
app/cli.py  app/plugin.py      the rag command; the HarnessWeaver plugin
tests/                         offline tests (and integration tests for real stores)
deploy/  compose.yaml          gateway and collector configs; the stack with a gateway
compose.ollama.yaml            the fully local stack: Qdrant + Ollama
```

[docs/architecture.md](docs/architecture.md) explains how a request flows and why the code is
shaped the way it is.

## License

MIT. HarnessWeaver is Apache-2.0.
