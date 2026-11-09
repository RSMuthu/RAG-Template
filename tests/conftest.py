"""Tests run offline: the scripted model, the hashing embedder and the memory store.

HARNESSWEAVER_TEST_MODE=1 makes every live provider refuse the network, so no test can
reach a real model or spend money (integration tests lift it for their store only).
"""

import asyncio
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ["HARNESSWEAVER_ENV"] = "offline"
os.environ["HARNESSWEAVER_TEST_MODE"] = "1"
os.environ.pop("RAG_API_KEYS_FILE", None)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def offline_index(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Index the sample corpus in a scratch working directory, so tests never touch yours.

    The offline store's path (.harness/index/kb.json) is relative to the working directory.
    """
    from app.ingest import ingest

    work = tmp_path_factory.mktemp("work")
    previous = Path.cwd()
    os.chdir(work)
    report = asyncio.run(ingest(ROOT / "ingest.yaml", spec=ROOT / "harness.yaml"))
    assert report.indexed and not report.failed, report
    yield work
    os.chdir(previous)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
