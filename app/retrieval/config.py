"""Retrieval settings: the keys under ``context.<source>`` in ``harness.yaml``.

HarnessWeaver owns ``use``, ``label``, ``max_items``, ``max_tokens`` and ``require_acl``;
everything else in the block is validated against :class:`RetrieverConfig`.

Each component picks its implementation with ``kind``: a built-in name, or an import
path (``mypkg.stores:MyStore``) to a class built as ``Cls(config)``. The models are flat
on purpose: an overlay that switches ``kind`` merges key by key with the base spec, so
settings another kind uses (a Qdrant ``url`` when the offline store is ``memory``) must
be accepted and ignored rather than rejected.
"""

import importlib
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["EmbedderConfig", "RerankConfig", "RetrieverConfig", "StoreConfig", "build"]


class _Component(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    options: dict[str, Any] = Field(
        default_factory=dict, description="Settings for a custom kind; built-in kinds ignore it."
    )


class StoreConfig(_Component):
    kind: str = "memory"
    collection: str = Field(default="kb", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    url: str | None = Field(default=None, description="qdrant: the server URL.")
    api_key_env: str | None = Field(default=None, description="qdrant: env var with the API key.")
    dsn_env: str | None = Field(default=None, description="pgvector: env var with the DSN.")
    path: str | None = Field(
        default=None,
        description="memory: JSON file, relative to the working directory; ':memory:' keeps "
        "nothing on disk. Default .harness/index/<collection>.json.",
    )


class EmbedderConfig(_Component):
    kind: str = "hashing"
    model: str | None = Field(default=None, description="openai: the embedding model or alias.")
    base_url: str | None = Field(default=None, description="openai: any /embeddings endpoint.")
    api_key_env: str | None = Field(default=None, description="openai: env var with the key.")
    dimensions: int | None = Field(default=None, gt=0, description="Vector size, if settable.")
    batch_size: int = Field(default=64, gt=0)
    max_retries: int = Field(default=2, ge=0, description="SDK retries; 0 behind a gateway.")


class RerankConfig(_Component):
    kind: str = "none"
    model: str | None = None
    base_url: str | None = Field(default=None, description="gateway: serves POST /rerank.")
    api_key_env: str | None = None
    candidates: int = Field(default=30, gt=0, description="How many hits to rerank.")


class RetrieverConfig(BaseModel):
    """The ``app.retriever`` context provider's settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    store: StoreConfig = StoreConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    rerank: RerankConfig = RerankConfig()
    min_score: float | None = Field(
        default=None, description="Drop hits scoring below this (after reranking, if any)."
    )


BUILTINS: dict[str, dict[str, str | None]] = {
    "store": {
        "memory": "app.retrieval.stores.memory:MemoryStore",
        "qdrant": "app.retrieval.stores.qdrant:QdrantStore",
        "pgvector": "app.retrieval.stores.pgvector:PgVectorStore",
    },
    "embedder": {
        "hashing": "app.retrieval.embedders:HashingEmbedder",
        "openai": "app.retrieval.embedders:OpenAIEmbedder",
    },
    "rerank": {
        "none": None,
        "gateway": "app.retrieval.rerank:GatewayReranker",
    },
}


def build(component: str, config: _Component) -> Any:
    """Build the class ``config.kind`` names (lazily imported), or ``None`` for ``none``.

    Raises:
        ValueError: For a kind that is neither built in nor an import path.
    """
    target = BUILTINS[component].get(config.kind, config.kind)
    if target is None:
        return None
    module, _, name = target.partition(":")
    if not module or not name:
        known = ", ".join(BUILTINS[component])
        raise ValueError(
            f"unknown {component} kind {config.kind!r}; use one of {known} or 'pkg.mod:Class'"
        )
    return getattr(importlib.import_module(module), name)(config)
