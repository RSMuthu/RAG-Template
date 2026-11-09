"""``rerank.kind: gateway``: a Cohere-style ``POST /rerank``, as LiteLLM serves it.

Point ``base_url`` at your gateway and ``model`` at a rerank model it routes (Cohere,
Jina, Voyage, a self-hosted cross-encoder, ...).
"""

import os
from typing import Any

import httpx
from harnessweaver.errors import ProviderError
from harnessweaver.ports import network_allowed

from app.retrieval.config import RerankConfig
from app.retrieval.stores.base import Hit

__all__ = ["GatewayReranker"]


class GatewayReranker:
    def __init__(self, config: RerankConfig) -> None:
        if not config.model or not config.base_url:
            raise ValueError("rerank.model and rerank.base_url are required for kind 'gateway'")
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.candidates = config.candidates
        self._client: httpx.AsyncClient | None = None

    async def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not network_allowed():
            raise ProviderError("rerank calls are off in test mode (HARNESSWEAVER_TEST_MODE=1)")
        if self._client is None:
            key = os.environ.get(self.config.api_key_env or "", "")
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=30)
        body: dict[str, Any] = {
            "model": self.config.model,
            "query": query,
            "documents": [h.chunk.text for h in hits],
            "top_n": len(hits),
        }
        try:
            response = await self._client.post("/rerank", json=body)
            response.raise_for_status()
            results = response.json()["results"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise ProviderError(f"rerank request failed: {type(exc).__name__}") from None
        ranked = sorted(results, key=lambda r: -float(r["relevance_score"]))
        return [Hit(hits[r["index"]].chunk, float(r["relevance_score"])) for r in ranked]
