"""Parsers: bytes in, text sections out. One section per PDF page, one for everything else.

Add a format with ``loaders: {".csv": "mypkg.loaders:load_csv"}`` in ingest.yaml; a
loader is any function ``(data: bytes) -> list[Section]``.
"""

import io
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["BY_CONTENT_TYPE", "BY_EXTENSION", "Loader", "Section"]


@dataclass(frozen=True, slots=True)
class Section:
    text: str
    page: int | None = None


type Loader = Callable[[bytes], list[Section]]


def load_text(data: bytes) -> list[Section]:
    return [Section(data.decode("utf-8", errors="replace"))]


def load_html(data: bytes) -> list[Section]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
        tag.decompose()
    return [Section(soup.get_text("\n"))]


def load_pdf(data: bytes) -> list[Section]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return [Section(page.extract_text() or "", page=n) for n, page in enumerate(reader.pages, 1)]


def load_docx(data: bytes) -> list[Section]:
    try:
        import docx  # python-docx, the `docx` extra
    except ImportError as exc:
        raise RuntimeError("install the docx extra to index .docx files") from exc
    document = docx.Document(io.BytesIO(data))
    return [Section("\n".join(p.text for p in document.paragraphs))]


BY_EXTENSION: dict[str, Loader] = {
    ".txt": load_text,
    ".md": load_text,
    ".markdown": load_text,
    ".html": load_html,
    ".htm": load_html,
    ".pdf": load_pdf,
    ".docx": load_docx,
}

BY_CONTENT_TYPE: dict[str, Loader] = {
    "text/plain": load_text,
    "text/markdown": load_text,
    "text/html": load_html,
    "application/pdf": load_pdf,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": load_docx,
}
