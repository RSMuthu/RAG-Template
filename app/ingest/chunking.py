"""One recursive splitter: paragraphs, then lines, then sentences, then words."""

import re

__all__ = ["normalize", "split"]

_SEPARATORS = ("\n\n", "\n", ". ", " ")


def normalize(text: str) -> str:
    """Unify line endings, trim each line, and collapse runs of blank lines and spaces."""
    lines = [
        re.sub(r"[ \t]+", " ", line).strip() for line in text.replace("\r\n", "\n").split("\n")
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def split(text: str, size: int, overlap: int) -> list[str]:
    """Chunks of about ``size`` characters; each starts with the last ``overlap`` of the one before."""
    text = normalize(text)
    if not text:
        return []
    return _merge(_pieces(text, size, _SEPARATORS), size, overlap)


def _pieces(text: str, size: int, separators: tuple[str, ...]) -> list[str]:
    """Split ``text`` at the coarsest separator until every piece fits in ``size``."""
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, finer = separators[0], separators[1:]
    parts = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        piece = part + sep if i < len(parts) - 1 else part
        pieces += _pieces(piece, size, finer) if len(piece) > size else [piece]
    return pieces


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current.strip() and len(current) + len(piece) > size:
            chunks.append(current.strip())
            current = _tail(current, overlap)
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _tail(text: str, overlap: int) -> str:
    """The last ``overlap`` characters, starting at a word boundary."""
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail
