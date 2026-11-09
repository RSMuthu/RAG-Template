"""Chunking, parsing and incremental ingestion."""

from pathlib import Path

import harnessweaver as hw
import pytest
import yaml

from app.ingest.chunking import normalize, split
from app.ingest.loaders import BY_EXTENSION
from app.ingest.pipeline import ingest
from app.retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "harness.yaml"
SAMPLE = ROOT / "data" / "sample" / "public"


def test_chunks_respect_size_and_carry_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(400))
    chunks = split(text, size=200, overlap=40)
    assert len(chunks) > 5
    assert all(len(c) <= 240 for c in chunks)
    assert chunks[1].split()[0] in chunks[0]  # the next chunk starts inside the previous one


def test_chunks_break_at_paragraphs_first() -> None:
    assert split("First paragraph.\n\nSecond paragraph.", size=20, overlap=0) == [
        "First paragraph.",
        "Second paragraph.",
    ]


def test_normalize_collapses_whitespace() -> None:
    assert normalize("  a\t\tb  \r\n\n\n\nc ") == "a b\n\nc"


def test_pdf_pages_are_read() -> None:
    [page] = BY_EXTENSION[".pdf"]((SAMPLE / "security.pdf").read_bytes())
    assert page.page == 1
    assert "AES-256" in page.text


def test_html_drops_navigation_and_styles() -> None:
    [section] = BY_EXTENSION[".html"]((SAMPLE / "invoices.html").read_bytes())
    assert "download it as a PDF" in section.text
    assert "Home | Help" not in section.text
    assert "font-family" not in section.text


def _write_config(tmp: Path, **extra: object) -> Path:
    config = tmp / "ingest.yaml"
    config.write_text(yaml.safe_dump({"sources": [{"type": "files", "path": "docs"}], **extra}))
    return config


@pytest.mark.anyio
async def test_ingest_is_incremental_and_prunes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # a fresh .harness/index for this test
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\nAlpha facts.")
    (docs / "b.md").write_text("# B\n\nBeta facts.")
    config = _write_config(tmp_path)

    first = await ingest(config, spec=SPEC)
    assert (first.indexed, first.chunks, first.unchanged) == (2, 2, 0)
    again = await ingest(config, spec=SPEC)
    assert (again.indexed, again.unchanged) == (0, 2)

    (docs / "a.md").write_text("# A\n\nAlpha facts, revised.")
    (docs / "b.md").unlink()
    third = await ingest(config, spec=SPEC, prune=True)
    assert (third.indexed, third.unchanged, third.deleted) == (1, 0, 1)

    store = Retriever.from_spec(hw.load_spec(SPEC)).store
    assert set(await store.doc_hashes()) == {"docs/a.md"}
    [chunk] = await store.get(["docs/a.md#0"])
    assert chunk.metadata["title"] == "A"
    assert "revised" in chunk.text


@pytest.mark.anyio
async def test_a_broken_document_is_reported_and_the_rest_indexed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "good.md").write_text("Fine.")
    (docs / "broken.pdf").write_bytes(b"not a pdf")
    report = await ingest(_write_config(tmp_path), spec=SPEC)
    assert report.indexed == 1
    assert list(report.failed) == ["docs/broken.pdf"]


@pytest.mark.anyio
async def test_source_acl_and_custom_loaders_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "prices.csv").write_text("plan,price\npro,20")
    config = tmp_path / "ingest.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "sources": [{"type": "files", "path": "docs", "acl": {"roles": ["staff"]}}],
                "loaders": {".csv": "app.ingest.loaders:load_text"},
            }
        )
    )
    assert (await ingest(config, spec=SPEC)).indexed == 1
    [chunk] = await Retriever.from_spec(hw.load_spec(SPEC)).store.get(["docs/prices.csv#0"])
    assert chunk.acl == hw.Acl(roles=("staff",))
