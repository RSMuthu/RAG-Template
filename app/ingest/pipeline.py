"""``rag ingest``: read sources, chunk, embed and upsert into the store ``harness.yaml`` names.

Incremental: each document's hash covers its bytes, ACL, metadata, the chunking settings
and the embedder, so an unchanged document is skipped and changing any of those
re-indexes it. ``prune`` deletes documents that are no longer in any source.
"""

import hashlib
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import harnessweaver as hw
import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ingest.chunking import split
from app.ingest.loaders import BY_CONTENT_TYPE, BY_EXTENSION, Loader, Section
from app.ingest.sources import RawDocument, Source, read_sources
from app.retrieval import Chunk, Retriever

__all__ = ["Chunking", "IngestConfig", "Report", "ingest", "make_chunks"]


class Chunking(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    size: int = Field(default=1200, gt=0, description="Characters per chunk.")
    overlap: int = Field(
        default=150, ge=0, description="Characters shared with the previous chunk."
    )

    @model_validator(mode="after")
    def _overlap_fits(self) -> Self:
        if self.overlap >= self.size:
            raise ValueError("chunking.overlap must be smaller than chunking.size")
        return self


class IngestConfig(BaseModel):
    """ingest.yaml."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str = Field(default="kb", description="The context source in harness.yaml.")
    chunking: Chunking = Chunking()
    sources: list[Source]
    loaders: dict[str, str] = Field(
        default_factory=dict, description="Extension or content type -> 'pkg.module:function'."
    )

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def loader_table(self) -> dict[str, Loader]:
        custom = {}
        for key, target in self.loaders.items():
            module, _, name = target.partition(":")
            custom[key.lower()] = getattr(importlib.import_module(module), name)
        return {**BY_EXTENSION, **BY_CONTENT_TYPE, **custom}


@dataclass
class Report:
    indexed: int = 0
    unchanged: int = 0
    deleted: int = 0
    chunks: int = 0
    failed: dict[str, str] = field(default_factory=dict)


def doc_hash(raw: RawDocument, chunking: Chunking, embedder: str) -> str:
    acl = raw.acl.model_dump(mode="json") if raw.acl else None
    settings = json.dumps([acl, raw.metadata, chunking.model_dump(), embedder], sort_keys=True)
    return hashlib.sha256(raw.data + settings.encode()).hexdigest()


def _title(sections: list[Section], fallback: str) -> str:
    for section in sections:
        for line in section.text.splitlines():
            if line.strip():
                return line.strip().lstrip("#").strip()[:200]
    return fallback


def make_chunks(
    raw: RawDocument, sections: list[Section], chunking: Chunking, *, digest: str, embedder: str
) -> list[Chunk]:
    title = _title(sections, Path(raw.uri).stem)
    texts = [
        (section.page, text)
        for section in sections
        for text in split(section.text, chunking.size, chunking.overlap)
    ]
    return [
        Chunk(
            id=f"{raw.doc_id}#{n}",
            doc_id=raw.doc_id,
            text=text,
            acl=raw.acl,
            metadata={
                **raw.metadata,
                "title": title,
                "uri": raw.uri,
                "chunk": n,
                **({"page": page} if page is not None else {}),
                "doc_hash": digest,
                "embedder": embedder,
            },
        )
        for n, (page, text) in enumerate(texts)
    ]


async def ingest(
    config_path: str | Path = "ingest.yaml",
    *,
    spec: str | Path = "harness.yaml",
    env: str | None = None,
    prune: bool = False,
) -> Report:
    """Index every source in ``config_path`` into ``context.<target>`` of ``spec``."""
    path = Path(config_path)
    config = IngestConfig.load(path)
    retriever = Retriever.from_spec(hw.load_spec(spec, env=env), config.target)
    store, embedder = retriever.store, retriever.embedder
    loaders = config.loader_table()
    existing = await store.doc_hashes()
    report, seen = Report(), set[str]()

    async for raw in read_sources(config.sources, path.parent):
        seen.add(raw.doc_id)
        digest = doc_hash(raw, config.chunking, retriever.embedder_name)
        if raw.error is None and existing.get(raw.doc_id) == digest:
            report.unchanged += 1
            continue
        try:
            if raw.error is not None:
                raise RuntimeError(raw.error)
            loader = loaders.get(raw.kind)
            if loader is None:
                raise ValueError(f"no loader for {raw.kind or 'unknown type'}")
            chunks = make_chunks(
                raw,
                loader(raw.data),
                config.chunking,
                digest=digest,
                embedder=retriever.embedder_name,
            )
        except Exception as exc:  # one bad document doesn't stop the rest
            report.failed[raw.doc_id] = f"{type(exc).__name__}: {exc}"
            logger.warning("skipped {}: {}", raw.doc_id, type(exc).__name__)
            continue
        vectors = await embedder.embed([c.text for c in chunks]) if chunks else []
        await store.delete([raw.doc_id])
        if chunks:
            await store.upsert(chunks, vectors)
        report.indexed += 1
        report.chunks += len(chunks)

    if prune:
        stale = sorted(set(existing) - seen)
        if stale:
            await store.delete(stale)
        report.deleted = len(stale)
    return report
