"""Ingestion: sources -> loaders -> chunks -> embeddings -> the vector store."""

from app.ingest.pipeline import IngestConfig, Report, ingest

__all__ = ["IngestConfig", "Report", "ingest"]
