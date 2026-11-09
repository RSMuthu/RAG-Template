"""``kind: qdrant``: a Qdrant collection (``pip install .[qdrant]``).

The collection is created on first write, sized to the embedder's vectors, with
cosine distance and keyword indexes on ``doc_id`` and the ACL fields. ACLs are a
payload filter, so Qdrant only returns what the caller may see.

``url: ":memory:"`` runs Qdrant in-process (qdrant-client's local mode), for tests.
"""

import os
import uuid
from collections.abc import Sequence
from typing import Any

import harnessweaver as hw
from harnessweaver.errors import ProviderError
from harnessweaver.ports import network_allowed

from app.retrieval.config import StoreConfig
from app.retrieval.stores.base import Chunk, Hit, acl_lists

__all__ = ["QdrantStore"]

_KEYWORD_FIELDS = ("doc_id", "acl_tenants", "acl_principals", "acl_roles")


def _point_id(chunk_id: str) -> str:
    """Qdrant ids are UUIDs or integers; chunk ids map to a stable UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag-template:{chunk_id}"))


def _payload(chunk: Chunk) -> dict[str, Any]:
    tenants, principals, roles = acl_lists(chunk.acl)
    return {
        "chunk_id": chunk.id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "acl": chunk.acl.model_dump(mode="json") if chunk.acl else None,
        "acl_tenants": tenants,
        "acl_principals": principals,
        "acl_roles": roles,
        "metadata": chunk.metadata,
    }


def _chunk(payload: dict[str, Any]) -> Chunk:
    return Chunk(
        id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        text=payload["text"],
        acl=payload.get("acl"),
        metadata=payload.get("metadata") or {},
    )


class QdrantStore:
    def __init__(self, config: StoreConfig) -> None:
        if not config.url:
            raise ValueError("store.url is required for kind 'qdrant'")
        self.config = config
        self.collection = config.collection
        self._client: Any = None
        self._exists = False

    def _connect(self) -> Any:
        if self.config.url != ":memory:" and not network_allowed():
            raise ProviderError("Qdrant calls are off in test mode (HARNESSWEAVER_TEST_MODE=1)")
        if self._client is None:
            from qdrant_client import AsyncQdrantClient  # imported on first use (S8)

            key = os.environ.get(self.config.api_key_env) if self.config.api_key_env else None
            self._client = AsyncQdrantClient(location=self.config.url, api_key=key)
        return self._client

    async def _collection_exists(self) -> bool:
        if not self._exists:
            self._exists = await self._connect().collection_exists(self.collection)
        return self._exists

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if not chunks:
            return
        from qdrant_client import models

        client = self._connect()
        if not await self._collection_exists():
            await client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(
                    size=len(vectors[0]), distance=models.Distance.COSINE
                ),
            )
            for name in _KEYWORD_FIELDS:
                await client.create_payload_index(
                    self.collection, name, field_schema=models.PayloadSchemaType.KEYWORD
                )
            self._exists = True
        points = [
            models.PointStruct(id=_point_id(c.id), vector=list(v), payload=_payload(c))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), 256):
            await client.upsert(self.collection, points=points[start : start + 256], wait=True)

    async def delete(self, doc_ids: Sequence[str]) -> None:
        if not doc_ids or not await self._collection_exists():
            return
        from qdrant_client import models

        selector = models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))]
            )
        )
        await self._connect().delete(self.collection, points_selector=selector, wait=True)

    async def search(self, vector: Sequence[float], k: int, env: hw.Envelope) -> list[Hit]:
        if not await self._collection_exists():
            return []
        response = await self._connect().query_points(
            self.collection,
            query=list(vector),
            limit=k,
            query_filter=_acl_filter(env),
            with_payload=True,
        )
        return [Hit(_chunk(p.payload or {}), float(p.score)) for p in response.points]

    async def get(self, ids: Sequence[str]) -> list[Chunk]:
        if not ids or not await self._collection_exists():
            return []
        points = await self._connect().retrieve(
            self.collection, ids=[_point_id(i) for i in ids], with_payload=True
        )
        found = {c.id: c for c in (_chunk(p.payload or {}) for p in points)}
        return [found[i] for i in ids if i in found]

    async def doc_hashes(self) -> dict[str, str]:
        if not await self._collection_exists():
            return {}
        hashes: dict[str, str] = {}
        offset = None
        while True:
            points, offset = await self._connect().scroll(
                self.collection,
                limit=1000,
                offset=offset,
                with_payload=["doc_id", "metadata"],
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                hashes[payload["doc_id"]] = str((payload.get("metadata") or {}).get("doc_hash", ""))
            if offset is None:
                return hashes


def _acl_filter(env: hw.Envelope) -> Any:
    """``hw.Acl.allows`` as a Qdrant filter: an empty list restricts nothing."""
    from qdrant_client import models

    def empty(key: str) -> Any:
        return models.IsEmptyCondition(is_empty=models.PayloadField(key=key))

    def has(key: str, value: str) -> Any:
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))

    tenant = models.Filter(should=[empty("acl_tenants"), has("acl_tenants", env.tenant)])
    who: list[Any] = [
        models.Filter(must=[empty("acl_principals"), empty("acl_roles")]),
        has("acl_principals", env.principal.id),
    ]
    if env.principal.roles:
        who.append(
            models.FieldCondition(
                key="acl_roles", match=models.MatchAny(any=list(env.principal.roles))
            )
        )
    return models.Filter(must=[tenant, models.Filter(should=who)])
