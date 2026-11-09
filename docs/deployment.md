# Deployment

## Docker Compose

`compose.yaml` runs the service with everything it talks to. Every port is published on
127.0.0.1 only.

| Service | Port | Role |
|---|---|---|
| `app` | 8000 | The API (`HARNESSWEAVER_ENV=prod`, so `harness.prod.yaml` applies) |
| `litellm` | 4000 | The model gateway: holds the provider keys, owns retries |
| `qdrant` | 6333 | The default vector store |
| `postgres` | 5432 | pgvector, with `--profile pgvector` |
| `otel-collector` | 4318 | Receives traces and metrics from the app and the gateway |

```bash
cp .env.example .env
```

```bash
cp api-keys.example.yaml api-keys.yaml
```

```bash
docker compose up -d --build
```

```bash
docker compose run --rm app rag ingest
```

Mount your documents into the `app` container (or bake them into the image) and list them in
`ingest.yaml`. Run `rag ingest` whenever they change, for example from a scheduled job; it skips
unchanged documents.

## Fully local with Ollama

`compose.ollama.yaml` needs no keys and sends nothing off your machine. It runs with
`HARNESSWEAVER_ENV=ollama`, so `harness.ollama.yaml` applies.

| Service | Role |
|---|---|
| `net` | Holds the network the other services share, like a Kubernetes pod's pause container. It publishes 8000 (API), 6333 (Qdrant) and 11434 (Ollama) on 127.0.0.1 |
| `ollama` | Serves the chat model and the embeddings through its OpenAI-compatible API |
| `ollama-pull` | Downloads the models on the first start; later starts skip it, even offline |
| `qdrant` | The vector store |
| `app` | The API, with `data/` and `ingest.yaml` mounted from your checkout |

```bash
make ollama-up
```

```bash
make ollama-test
```

Because every service shares one network, the `localhost` URLs in `harness.ollama.yaml` work
inside the containers and on your machine alike. So you can edit a prompt and re-run the evals
against the real model without rebuilding (`OLLAMA_API_KEY=ollama uv run harnessweaver eval run
harness.yaml -e ollama`), or ingest from the host. After editing documents or `ingest.yaml`,
re-run `rag ingest`; after editing the spec or prompts, rebuild with
`docker compose -f compose.ollama.yaml up -d --build`.

**Models.** The defaults are `llama3.2:3b` for answers and `nomic-embed-text` for embeddings.
To change one, edit it in `harness.ollama.yaml` (`models.aliases.chat` and `models.prices`, or
`context.kb.embedder.model`) and in the `ollama-pull` service. A new embedding model changes
the vectors' meaning and usually their size: drop the collection and re-index
(`curl -X DELETE localhost:6333/collections/kb`, then `rag ingest`).

**Speed.** Docker on macOS has no GPU access, so models run on the CPU. On Linux with an
NVIDIA GPU and the NVIDIA Container Toolkit, uncomment the `deploy` block of the `ollama`
service. The overlay allows 300 s per run for slow CPUs. On a CPU that is busy with other work,
fewer threads can be faster, because llama.cpp waits for its slowest thread on every token:
on a loaded 10-core laptop, 6 threads generated 2.5 times faster than 10. Ollama takes the
setting per model (`PARAMETER num_thread 6` in a Modelfile, then `ollama create`).

**Tuning `min_score`.** With `nomic-embed-text` on the sample documents, the passage that
answers scored 0.69–0.77, same-domain questions the documents can't answer 0.50–0.59, and
unrelated ones 0.30–0.39. Hence 0.5: unrelated questions never reach the model, and borderline
ones reach it with weak passages and get `NOT_FOUND`. Check your own documents with
`rag search -e ollama "<question>"` before trusting the default.

**What `make ollama-test` checks** (`tests/test_stack.py`, against the running stack):
search ranks the answering passage first; answers cite it; an unrelated question isn't
answered; an injected question is blocked; the poisoned sample passage never comes back;
anonymous callers never see Globex's contract while a Globex caller does; streaming ends with
the full answer; an idempotency key replays the first run. Then it runs the eval suites with
the real model. Point `RAG_URL` (and `RAG_API_KEY`) at another deployment to run the HTTP
checks there.

**Health.** `/readyz` checks the vector store only. It doesn't call the model, so a probe never
costs a model call; a model outage shows up as 503 `provider_error` on the endpoints.

## Environment variables

| Variable | Read by | Meaning |
|---|---|---|
| `HARNESSWEAVER_ENV` | app | The overlay: `prod` applies `harness.prod.yaml` |
| `RAG_API_KEYS_FILE` | app | API keys → tenant, principal, roles. Required in prod |
| `RAG_SPEC` | app | The spec (default `harness.yaml`) |
| `RAG_NOT_FOUND_MESSAGE` | app | The reply when nothing answers the question |
| `LITELLM_API_KEY` | app | The key the app sends to the gateway |
| `QDRANT_API_KEY`, `DATABASE_URL` | app | Vector store credentials (names set in `harness.yaml`) |
| `LITELLM_MASTER_KEY`, provider keys | gateway | Never in the app's environment or spec |
| `HARNESSWEAVER_UPSTREAM_RAG_*` | gateway | The provider model behind each alias |

## Scaling

- The app is stateless apart from HarnessWeaver's run store: `harness.prod.yaml` uses SQLite at
  `/data/harness.db`, **one file per replica**. Idempotency keys and online eval samples are
  therefore per replica. Point `state` at a shared store when HarnessWeaver ships one (Postgres and
  Redis stores are planned), or route a client to one replica.
- Run one uvicorn worker per container and scale with replicas.
- The `memory` store is for development and tests: it's a JSON file read into each process.
  Use Qdrant or pgvector in production.

## Observability

Every run is traced with the OpenTelemetry GenAI conventions, with cost and token use, and
without any prompt or answer content (`observability.capture_content` stays false). The app and
the gateway export OTLP to `otel-collector`, which only logs a summary as shipped: add your
backend as an exporter in `deploy/otel-collector.yaml`. For a local Langfuse, run
`uv run harnessweaver dev harness.yaml --up` (see HarnessWeaver's
[dev stack](https://harnessweaver.wisedreamerz.com/docs/deployment/gateways/)).

`evals.online` in `harness.prod.yaml` grades 5% of production runs; read the report with:

```bash
uv run harnessweaver eval report harness.yaml -e prod
```

## Security checklist

- Generate long random API keys, keep `api-keys.yaml` and `.env` out of git (both are ignored),
  and prefer your platform's secret store in production.
- Give every restricted source an `acl` in `ingest.yaml`; set `require_acl: true` on
  `context.kb` if every document must have one.
- Keep the `rag` profile's guards on. Add new guards in shadow mode first (see extending.md).
- Pin images by digest for production builds (HarnessWeaver's RULES S5); the compose file pins
  by version tag.
- CI runs the red-team suite (`evals/redteam.yaml`); add a case for every attack you care about.

## CI

`.github/workflows/ci.yml` runs three jobs:

- **check**: `make check`, which covers lint, types, spec validation, the offline tests, plugin
  conformance and the eval gates against `.harness/baselines/`.
- **integration**: the store contract against real Qdrant and pgvector service containers.
- **image**: `docker build`.
