"""The store contract (every backend), component selection and the retriever's checks.

The contract runs against the memory store always, and against Qdrant and pgvector with
``pytest -m integration`` once ``docker compose --profile qdrant --profile pgvector up -d``
is running (QDRANT_URL, DATABASE_URL override the defaults).
"""

import json
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import harnessweaver as hw
import pytest
from harnessweaver.errors import ProviderError, SpecError

from app.retrieval import Chunk, Retriever, RetrieverConfig, StoreConfig
from app.retrieval.config import EmbedderConfig, build

pytestmark = pytest.mark.anyio

CHUNKS = [
    (
        Chunk(id="d1#0", doc_id="d1", text="for everyone", metadata={"doc_hash": "h1"}),
        [1.0, 0.0, 0.0],
    ),
    (
        Chunk(
            id="d2#0",
            doc_id="d2",
            text="acme only",
            acl=hw.Acl(tenants=("acme",)),
            metadata={"doc_hash": "h2"},
        ),
        [0.9, 0.1, 0.0],
    ),
    (
        Chunk(
            id="d3#0",
            doc_id="d3",
            text="staff only",
            acl=hw.Acl(roles=("staff",)),
            metadata={"doc_hash": "h3"},
        ),
        [0.8, 0.2, 0.0],
    ),
    (
        Chunk(
            id="d3#1",
            doc_id="d3",
            text="ann only",
            acl=hw.Acl(principals=("ann",)),
            metadata={"doc_hash": "h3"},
        ),
        [0.7, 0.3, 0.0],
    ),
]


def envelope(
    tenant: str = "default", principal: str = "bob", roles: tuple[str, ...] = ()
) -> hw.Envelope:
    return hw.Envelope.new(tenant=tenant, principal=hw.Principal(id=principal, roles=roles))


@pytest.fixture(
    params=[
        "memory",
        "qdrant-in-process",
        pytest.param("qdrant", marks=pytest.mark.integration),
        pytest.param("pgvector", marks=pytest.mark.integration),
    ]
)
async def store(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Any]:
    kind: str = request.param.removesuffix("-in-process")
    collection = f"test_{uuid.uuid4().hex[:10]}"
    url = ":memory:" if request.param == "qdrant-in-process" else os.environ.get("QDRANT_URL")
    if request.param in ("qdrant", "pgvector"):
        monkeypatch.delenv("HARNESSWEAVER_TEST_MODE", raising=False)
        monkeypatch.setenv(
            "DATABASE_URL",
            os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag"),
        )
    config = StoreConfig(
        kind=kind,
        collection=collection,
        path=str(tmp_path / "kb.json"),
        url=url or "http://localhost:6333",
        dsn_env="DATABASE_URL",
    )
    instance = build("store", config)
    yield instance
    if kind == "qdrant":
        await instance._connect().delete_collection(collection)
    elif kind == "pgvector":
        await (await instance._connect()).execute(f"DROP TABLE IF EXISTS {collection}")


async def test_store_contract(store: Any) -> None:
    def ids(hits: list[Any]) -> list[str]:
        return [h.chunk.id for h in hits]

    query = [1.0, 0.0, 0.0]
    assert await store.search(query, 10, envelope()) == []  # an empty index is not an error
    assert await store.doc_hashes() == {}

    await store.upsert([c for c, _ in CHUNKS], [v for _, v in CHUNKS])
    assert ids(await store.search(query, 10, envelope())) == ["d1#0"]
    assert ids(await store.search(query, 10, envelope(tenant="acme"))) == ["d1#0", "d2#0"]
    assert ids(await store.search(query, 10, envelope(roles=("staff",)))) == ["d1#0", "d3#0"]
    assert ids(await store.search(query, 10, envelope(principal="ann"))) == ["d1#0", "d3#1"]
    assert ids(await store.search(query, 1, envelope(tenant="acme"))) == ["d1#0"]

    assert [c.id for c in await store.get(["d3#1", "missing", "d1#0"])] == ["d3#1", "d1#0"]
    [acme_only] = await store.get(["d2#0"])
    assert acme_only.acl == hw.Acl(tenants=("acme",))
    assert await store.doc_hashes() == {"d1": "h1", "d2": "h2", "d3": "h3"}

    await store.delete(["d3"])
    assert set(await store.doc_hashes()) == {"d1", "d2"}
    await store.upsert([CHUNKS[0][0].model_copy(update={"text": "updated"})], [CHUNKS[0][1]])
    [updated] = await store.get(["d1#0"])
    assert updated.text == "updated"


def test_kind_can_be_an_import_path() -> None:
    config = StoreConfig(kind="app.retrieval.stores.memory:MemoryStore", path=":memory:")
    assert type(build("store", config)).__name__ == "MemoryStore"


def test_an_unknown_kind_fails_loading() -> None:
    with pytest.raises(SpecError, match="unknown store kind"):
        Retriever(RetrieverConfig(store=StoreConfig(kind="nosuchdb")))


def test_the_offline_overlay_switches_kinds_without_leftover_settings() -> None:
    spec = hw.load_spec(Path(__file__).resolve().parents[1] / "harness.yaml", env="offline")
    retriever = Retriever.from_spec(spec)
    assert type(retriever.store).__name__ == "MemoryStore"
    assert retriever.embedder_name.startswith("hashing/")


async def _memory_retriever(**settings: Any) -> Retriever:
    retriever = Retriever(
        RetrieverConfig.model_validate(
            {"store": {"kind": "memory", "path": ":memory:"}, **settings}
        )
    )
    chunks = [
        Chunk(
            id="a#0",
            doc_id="a",
            text="refunds take five days",
            metadata={"embedder": retriever.embedder_name},
        ),
        Chunk(
            id="b#0",
            doc_id="b",
            text="shipping takes two days",
            metadata={"embedder": retriever.embedder_name},
        ),
    ]
    await retriever.store.upsert(chunks, await retriever.embedder.embed([c.text for c in chunks]))
    return retriever


async def test_fetch_ranks_and_honours_min_score() -> None:
    retriever = await _memory_retriever()
    items = await retriever.fetch("how long do refunds take", envelope())
    assert items[0].id == "a#0"
    assert "embedder" not in items[0].metadata and items[0].metadata["doc_id"] == "a"
    strict = await _memory_retriever(min_score=0.99)
    assert await strict.fetch("how long do refunds take", envelope()) == []


async def test_an_index_built_with_another_embedder_is_refused() -> None:
    retriever = await _memory_retriever()
    other = Retriever(retriever.config)
    other.store = retriever.store
    other.embedder.name = "openai/another-model/native"  # same vector size, different meaning
    with pytest.raises(ProviderError, match="another embedder"):
        await other.fetch("refunds", envelope())


async def test_live_embedders_refuse_the_network_in_test_mode() -> None:
    embedder = build("embedder", EmbedderConfig(kind="openai", model="any"))
    with pytest.raises(ProviderError, match="test mode"):
        await embedder.embed(["hello"])


async def test_the_gateway_reranker_reorders_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from app.retrieval import Hit, RerankConfig

    def gateway(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/rerank" and body["model"] == "rag-rerank"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            },
        )

    monkeypatch.delenv("HARNESSWEAVER_TEST_MODE")
    reranker = build(
        "rerank", RerankConfig(kind="gateway", model="rag-rerank", base_url="http://gw")
    )
    reranker._client = httpx.AsyncClient(
        base_url="http://gw", transport=httpx.MockTransport(gateway)
    )
    hits = [Hit(c, 0.5) for c, _ in CHUNKS[:2]]
    ranked = await reranker.rerank("question", hits)
    assert [(h.chunk.id, h.score) for h in ranked] == [("d2#0", 0.9), ("d1#0", 0.2)]
