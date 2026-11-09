# Extending the template

Each recipe changes as little as possible. After any of them, run `make check`.

## Index your own documents

List folders and URLs in `ingest.yaml`. Each source can carry an `acl` (who may see it) and
`metadata` (kept on every chunk):

```yaml
sources:
  - type: files
    path: /srv/handbook          # relative to ingest.yaml, or absolute
    include: ["**/*.md", "**/*.pdf"]
    name: handbook               # id prefix: chunks are cited as [handbook/<path>#<n>]
    acl: { roles: [staff] }      # tenants, principals, roles; empty lists restrict nothing
  - type: urls
    urls: [https://example.com/pricing]
```

Run `uv run rag ingest`. Add `--prune` to delete documents that are no longer in any source.
A file format the template doesn't read gets a loader, a function from bytes to sections:

```python
# mypkg/loaders.py
from app.ingest.loaders import Section


def load_csv(data: bytes) -> list[Section]:
    return [Section(data.decode())]
```

```yaml
loaders:
  ".csv": mypkg.loaders:load_csv     # by extension, or by content type for URLs
```

## Switch the vector store

Change `context.kb.store.kind` in `harness.yaml`:

| kind | Settings | Install |
|---|---|---|
| `memory` | `path` (JSON file, relative to the working directory; `:memory:` for none) | built in |
| `qdrant` | `url`, `collection`, `api_key_env` | `qdrant` extra |
| `pgvector` | `collection` (the table), `dsn_env` (default `DATABASE_URL`) | `pgvector` extra |

The store is created on first write. Re-run `rag ingest` after switching.

## Add a vector store

Write a class that takes a `StoreConfig` and has the five methods of
`app.retrieval.stores.base.VectorStore`. Put its own settings under `options`:

```python
class MyStore:
    def __init__(self, config: StoreConfig) -> None:
        self.index = config.options["index"]

    async def upsert(self, chunks, vectors): ...
    async def delete(self, doc_ids): ...
    async def search(self, vector, k, env): ...  # only what env.tenant / env.principal may see
    async def get(self, ids): ...
    async def doc_hashes(self): ...
```

```yaml
store: { kind: mypkg.stores:MyStore, options: { index: docs } }
```

Then add your kind to the `store` fixture in `tests/test_retrieval.py`, so the store contract
(ranking, ACLs, get, delete, hashes) runs against it. Raise `harnessweaver.errors.ProviderError`
for failures, import the client library inside the class, and refuse the network when
`harnessweaver.ports.network_allowed()` is false.

Hybrid (keyword plus vector) search belongs in a store too: for example Postgres full-text
search fused with pgvector by reciprocal rank, or Qdrant's sparse vectors.

## Change the embedder

`kind: openai` speaks the OpenAI `/embeddings` API, so it covers most providers:

```yaml
embedder: { kind: openai, model: rag-embed, base_url: http://localhost:4000 }                  # LiteLLM gateway
embedder: { kind: openai, model: nomic-embed-text, base_url: http://localhost:11434/v1 }        # Ollama
embedder: { kind: openai, model: text-embedding-3-large, dimensions: 1024, api_key_env: OPENAI_API_KEY }
```

Your own embedder is a class with a `name` and `async def embed(self, texts) -> list[list[float]]`
(HarnessWeaver's `Embedder` port), selected with `kind: pkg.module:Class`. Changing the embedder
re-indexes everything on the next `rag ingest`; until then retrieval refuses the old index.

## Turn on reranking

```yaml
rerank: { kind: gateway, model: rag-rerank, base_url: http://localhost:4000, candidates: 30 }
```

The retriever fetches `candidates` hits, the gateway's `/rerank` orders them, and the best
`max_items` go to the model. Map `rag-rerank` in `deploy/litellm.yaml` (Cohere, Jina, Voyage, ...).

## Change the model

Point the alias at another model in `.env` (`HARNESSWEAVER_UPSTREAM_RAG_CHAT=...`): the app
doesn't change. To call Claude directly instead of through a gateway, add the `anthropic` extra
to the `harnessweaver` dependency in `pyproject.toml` and set:

```yaml
models:
  provider: anthropic.messages
  default: chat
  aliases: { chat: claude-sonnet-5-5 }
```

## Change the prompt

Prompts are versioned files. Add `prompts/answer@2.md` and change the reference in
`app/flow.py`, so the old version stays reproducible. A prompt must keep the `{sources}` and
`{question}` placeholders, ask for `[id]` citations and say when to reply `NOT_FOUND`.

Measure a prompt change against a real model before keeping it: with the local stack up,
`OLLAMA_API_KEY=ollama uv run harnessweaver eval run harness.yaml -e ollama` runs the eval
suites with the edited prompt, no rebuild needed. Small models copy what a prompt shows: an
example citation like `[guides/setup.md#0]` gets cited as if it were a source, and a
`<sentence> [<id>]` template gets its angle brackets copied. Describe the format in words,
and put the instructions after the sources.

## Add a guard

Add it in shadow mode, look at what it would have done, then enforce it:

```yaml
guardrails:
  input:
    - { id: no-legal-advice, use: topic.deny, topics: { legal: [lawsuit, attorney] }, mode: log_only }
```

`uv run harnessweaver eval run harness.yaml -e offline` reports how often a shadow guard would
fire and on which cases. Remove `mode: log_only` once that looks right. Your own guard is a
provider in `app/plugin.py` (see HarnessWeaver's "Providers and ports").

## Add a task (multi-turn chat, for example)

Every guard on a task runs on every model call in it, so a step that isn't a cited answer, such
as rewriting a follow-up question into a standalone one, goes in a task of its own:

```yaml
tasks:
  condense:
    runner: direct
    prompt: condense@1          # "Rewrite the last question so it stands on its own."
    output: str
    budget: { max_model_calls: 1 }

evals:
  condense: [evals/condense.yaml]   # every task needs a suite (adoption level L4)
```

The API then runs `condense` on the conversation and passes its output to `answer`.

## Replace authentication

`app/auth.py:current_caller` is the only place that decides who is calling. Return a `Caller`
from your JWT, OIDC or gateway headers; tenants and roles then drive retrieval ACLs and policy.

## Add policy

`run.start` rules decide who may start a run:

```yaml
policy:
  rules:
    - { id: staff-only, hook: run.start, when: "!('staff' in principal.roles)", effect: deny }
```

See HarnessWeaver's [Policy](https://harnessweaver.wisedreamerz.com/docs/tutorial/policy/) and
[expressions](https://harnessweaver.wisedreamerz.com/docs/reference/expressions/) references.
