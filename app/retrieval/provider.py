"""``use: app.retriever``: vector retrieval behind HarnessWeaver's ``ContextProvider`` port.

The harness calls ``fetch`` from ``s.context("kb").fetch(query)`` and then governs what
comes back: it re-checks every item's ACL, labels it untrusted, applies ``max_items`` and
``max_tokens``, and runs the context guards before the model sees anything.
"""

from collections.abc import Sequence
from typing import Any, Self

import harnessweaver as hw
from harnessweaver.errors import HarnessError, ProviderError, SpecError

from app.retrieval.config import RetrieverConfig, build
from app.retrieval.stores.base import Chunk, Hit, VectorStore

__all__ = ["DEFAULT_K", "Retriever"]

DEFAULT_K = 6
"""Hits returned when the source sets no ``max_items``."""

_INTERNAL = ("doc_hash", "embedder")


@hw.provider(
    port="ContextProvider",
    name="app.retriever",
    config=RetrieverConfig,
    latency_class="medium",
    sample={"store": {"kind": "memory", "path": ":memory:"}},
)
class Retriever:
    """Embed the query, search the store, optionally rerank, and return the best chunks."""

    def __init__(self, config: RetrieverConfig) -> None:
        self.config = config
        try:
            self.store: VectorStore = build("store", config.store)
            self.embedder: Any = build("embedder", config.embedder)
            self.reranker: Any = build("rerank", config.rerank)
        except (ValueError, ImportError) as exc:
            raise SpecError(f"app.retriever: {exc}") from None

    @classmethod
    def from_spec(cls, spec: hw.EffectiveSpec, source: str = "kb") -> Self:
        """The same retriever the harness builds for ``context.<source>`` (for ingestion and the API)."""
        if source not in spec.spec.context:
            raise SpecError(f"the spec has no context source {source!r}")
        return cls(RetrieverConfig.model_validate(spec.spec.context[source].provider_settings()))

    @property
    def embedder_name(self) -> str:
        return getattr(self.embedder, "name", type(self.embedder).__qualname__)

    async def fetch(
        self, query: str, env: hw.Envelope, *, limit: int | None = None
    ) -> list[hw.ContextItem]:
        k = limit or DEFAULT_K
        depth = max(k, self.reranker.candidates) if self.reranker else k
        try:
            [vector] = await self.embedder.embed([query])
            hits = await self.store.search(vector, depth, env)
            if self.reranker and hits:
                hits = await self.reranker.rerank(query, hits)
        except HarnessError:
            raise
        except Exception as exc:  # a client library's own errors never cross the boundary (C5)
            raise ProviderError(f"retrieval failed: {type(exc).__name__}") from None
        self._check_embedder(hits)
        if self.config.min_score is not None:
            hits = [h for h in hits if h.score >= self.config.min_score]
        return [self._item(h) for h in hits[:k]]

    async def cite(
        self, ids: Sequence[str], *, tenant: str, principal: hw.Principal
    ) -> list[Chunk]:
        """The cited chunks the caller may see, for showing titles and links next to an answer."""
        chunks = await self.store.get(list(ids))
        return [c for c in chunks if c.acl is None or c.acl.allows(tenant, principal)]

    def _check_embedder(self, hits: Sequence[Hit]) -> None:
        for hit in hits:
            built_with = hit.chunk.metadata.get("embedder")
            if built_with is not None and built_with != self.embedder_name:
                raise ProviderError(
                    "the index was built with another embedder; re-run ingestion",
                    index_embedder=built_with,
                    embedder=self.embedder_name,
                )

    @staticmethod
    def _item(hit: Hit) -> hw.ContextItem:
        chunk = hit.chunk
        metadata = {k: v for k, v in chunk.metadata.items() if k not in _INTERNAL}
        return hw.ContextItem(
            id=chunk.id,
            content=chunk.text,
            source="app.retriever",
            score=round(hit.score, 6),
            acl=chunk.acl,
            metadata={"doc_id": chunk.doc_id, **metadata},
        )
