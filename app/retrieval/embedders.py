"""Embedders. Each implements HarnessWeaver's ``Embedder`` port: ``async embed(texts)``.

An embedder's ``name`` is written on every chunk it embeds, so ingestion re-embeds
everything when it changes and retrieval refuses an index built with another one.
"""

import hashlib
import math
import os
import re
from collections.abc import Sequence
from typing import Any

from harnessweaver.errors import ProviderError
from harnessweaver.ports import network_allowed

from app.retrieval.config import EmbedderConfig

__all__ = ["HashingEmbedder", "OpenAIEmbedder"]

_WORD = re.compile(r"[^\W_]+")
_STOPWORDS = frozenset(
    (  # noqa: SIM905 - one readable line beats a 45-line list
        "a an and are as at be by can do does for from has have how i in is it its my of on or "
        "our so that the their there this to was we what when where which who why will with you "
        "your"
    ).split()
)


class HashingEmbedder:
    """``kind: hashing``: words hashed into a fixed-size vector. Deterministic and free.

    For offline runs and tests only: it matches words, not meaning.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        self.dims = config.dimensions or 1024
        self.name = f"hashing/{self.dims}"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        for word in _WORD.findall(text.lower()):
            if word in _STOPWORDS:
                continue
            h = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big")
            vector[h % self.dims] += 1.0 if h >> 63 else -1.0
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector] if norm else vector


class OpenAIEmbedder:
    """``kind: openai``: any OpenAI-compatible ``/embeddings`` endpoint.

    That covers OpenAI itself, a LiteLLM gateway (and every provider behind it), vLLM,
    Ollama and most hosted embedding APIs.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        if not config.model:
            raise ValueError("embedder.model is required for kind 'openai'")
        self.config = config
        self.name = f"openai/{config.model}/{config.dimensions or 'native'}"
        self._client: Any = None

    def _connect(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI  # imported on first use (S8)

            key = os.environ.get(self.config.api_key_env or "OPENAI_API_KEY") or "none"
            self._client = AsyncOpenAI(
                base_url=self.config.base_url, api_key=key, max_retries=self.config.max_retries
            )
        return self._client

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not network_allowed():
            raise ProviderError("embedding calls are off in test mode (HARNESSWEAVER_TEST_MODE=1)")
        client = self._connect()
        extra = {"dimensions": self.config.dimensions} if self.config.dimensions else {}
        size = self.config.batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(texts), size):
            batch = list(texts[start : start + size])
            try:
                response = await client.embeddings.create(
                    model=self.config.model, input=batch, **extra
                )
            except Exception as exc:  # the SDK's errors never cross the boundary (C5)
                status = getattr(exc, "status_code", None)
                raise ProviderError(
                    f"embedding request failed: {type(exc).__name__}", status=status
                ) from None
            vectors += [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        return vectors
