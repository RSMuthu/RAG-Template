"""``kind: memory``: chunks in a Python dict, saved to a JSON file.

For development, tests and small corpora. Search is a linear cosine scan. The file is
re-read when it changes, so ``rag ingest`` is picked up by a running server.
"""

import json
import math
from collections.abc import Sequence
from pathlib import Path

import harnessweaver as hw

from app.retrieval.config import StoreConfig
from app.retrieval.stores.base import Chunk, Hit, visible

__all__ = ["MemoryStore"]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class MemoryStore:
    def __init__(self, config: StoreConfig) -> None:
        path = config.path or f".harness/index/{config.collection}.json"
        self.path = None if path == ":memory:" else Path(path)
        self._rows: dict[str, tuple[Chunk, list[float]]] = {}
        self._mtime: float | None = None

    def _load(self) -> dict[str, tuple[Chunk, list[float]]]:
        if self.path is None or not self.path.is_file():
            return self._rows
        mtime = self.path.stat().st_mtime
        if mtime != self._mtime:
            rows = json.loads(self.path.read_text(encoding="utf-8"))["chunks"]
            self._rows = {
                row["chunk"]["id"]: (Chunk.model_validate(row["chunk"]), row["vector"])
                for row in rows
            }
            self._mtime = mtime
        return self._rows

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"chunk": c.model_dump(mode="json"), "vector": v} for c, v in self._rows.values()]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"chunks": rows}), encoding="utf-8")
        tmp.replace(self.path)
        self._mtime = self.path.stat().st_mtime

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        rows = self._load()
        for chunk, vector in zip(chunks, vectors, strict=True):
            rows[chunk.id] = (chunk, list(vector))
        self._save()

    async def delete(self, doc_ids: Sequence[str]) -> None:
        gone = set(doc_ids)
        self._rows = {k: row for k, row in self._load().items() if row[0].doc_id not in gone}
        self._save()

    async def search(self, vector: Sequence[float], k: int, env: hw.Envelope) -> list[Hit]:
        hits = [
            Hit(chunk, _cosine(vector, stored))
            for chunk, stored in self._load().values()
            if visible(chunk.acl, env)
        ]
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return [h for h in hits[:k] if h.score > 0]

    async def get(self, ids: Sequence[str]) -> list[Chunk]:
        rows = self._load()
        return [rows[i][0] for i in ids if i in rows]

    async def doc_hashes(self) -> dict[str, str]:
        return {
            chunk.doc_id: str(chunk.metadata.get("doc_hash", ""))
            for chunk, _ in self._load().values()
        }
