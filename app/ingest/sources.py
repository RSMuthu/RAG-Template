"""Where documents come from: local folders and web pages."""

import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import harnessweaver as hw
import httpx
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["FilesSource", "RawDocument", "Source", "UrlsSource", "read_sources", "safe_id"]


class _Source(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    acl: hw.Acl | None = Field(default=None, description="Who may see these documents.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Kept on every chunk.")


class FilesSource(_Source):
    type: Literal["files"]
    path: str = Field(description="A folder, relative to ingest.yaml.")
    include: list[str] = Field(default_factory=lambda: ["**/*"], description="Glob patterns.")
    name: str | None = Field(default=None, description="Id prefix; defaults to the folder name.")


class UrlsSource(_Source):
    type: Literal["urls"]
    urls: list[str]
    timeout: float = 30.0


type Source = Annotated[FilesSource | UrlsSource, Field(discriminator="type")]


@dataclass(frozen=True, slots=True)
class RawDocument:
    doc_id: str
    uri: str
    kind: str  # a file extension (".pdf") or a content type ("text/html")
    data: bytes
    acl: hw.Acl | None
    metadata: dict[str, Any]
    error: str | None = None  # set when it couldn't be read; ingestion reports it and keeps it


def safe_id(text: str) -> str:
    """Only characters the citation pattern allows; ``#`` is kept for chunk numbers."""
    return re.sub(r"[^A-Za-z0-9_.:/-]", "-", text).strip("-/") or "doc"


async def read_sources(
    sources: Sequence[FilesSource | UrlsSource], base: Path
) -> AsyncIterator[RawDocument]:
    for source in sources:
        if isinstance(source, FilesSource):
            for document in _read_files(source, base):
                yield document
        else:
            async for document in _read_urls(source):
                yield document


def _read_files(source: FilesSource, base: Path) -> list[RawDocument]:
    root = (base / source.path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"source folder not found: {root}")
    name = source.name or root.name
    home = base.resolve()
    paths = sorted({p for pattern in source.include for p in root.glob(pattern) if p.is_file()})
    return [
        RawDocument(
            doc_id=safe_id(f"{name}/{path.relative_to(root).as_posix()}"),
            uri=path.relative_to(home).as_posix() if path.is_relative_to(home) else str(path),
            kind=path.suffix.lower(),
            data=path.read_bytes(),
            acl=source.acl,
            metadata=source.metadata,
        )
        for path in paths
    ]


async def _read_urls(source: UrlsSource) -> AsyncIterator[RawDocument]:
    headers = {"User-Agent": "rag-template-ingest"}
    async with httpx.AsyncClient(
        timeout=source.timeout, follow_redirects=True, headers=headers
    ) as client:
        for url in source.urls:
            parts = urlsplit(url)
            doc_id = safe_id(parts.netloc + parts.path.rstrip("/"))
            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                yield RawDocument(
                    doc_id, url, "", b"", source.acl, source.metadata, type(exc).__name__
                )
                continue
            kind = response.headers.get("content-type", "").split(";")[0].strip().lower()
            yield RawDocument(
                doc_id=doc_id,
                uri=url,
                kind=kind or Path(parts.path).suffix.lower(),
                data=response.content,
                acl=source.acl,
                metadata=source.metadata,
            )
