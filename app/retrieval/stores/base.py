"""What a vector store stores and the five methods every store implements."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import harnessweaver as hw
from pydantic import BaseModel, ConfigDict, Field, JsonValue

__all__ = ["Chunk", "Hit", "VectorStore", "acl_lists", "visible"]


class Chunk(BaseModel):
    """One indexed piece of a document.

    ``id`` is ``<doc_id>#<n>``, such as ``public/refunds.md#0``: the answer cites it as
    ``[public/refunds.md#0]``, so it only uses characters the citation pattern allows.
    ``metadata`` holds ``title``, ``uri``, ``page`` and ``chunk`` for people, and
    ``doc_hash`` and ``embedder`` for ingestion's bookkeeping.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    text: str
    acl: hw.Acl | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    """A store of chunks and their vectors.

    ``search`` returns only chunks the envelope's tenant and principal may see, with the
    semantics of ``hw.Acl.allows``: an empty list restricts nothing, and a chunk without
    an ACL is visible to everyone. HarnessWeaver re-checks every item anyway.
    """

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    async def delete(self, doc_ids: Sequence[str]) -> None: ...

    async def search(self, vector: Sequence[float], k: int, env: hw.Envelope) -> list[Hit]: ...

    async def get(self, ids: Sequence[str]) -> list[Chunk]: ...

    async def doc_hashes(self) -> dict[str, str]:
        """``doc_id -> doc_hash`` for every indexed document (incremental ingestion)."""
        ...


def visible(acl: hw.Acl | None, env: hw.Envelope) -> bool:
    """Whether the run's tenant and principal may see an item with this ACL."""
    return acl is None or acl.allows(env.tenant, env.principal)


def acl_lists(acl: hw.Acl | None) -> tuple[list[str], list[str], list[str]]:
    """``(tenants, principals, roles)`` for stores that filter on flat fields."""
    if acl is None:
        return [], [], []
    return list(acl.tenants), list(acl.principals), list(acl.roles)
