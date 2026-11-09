"""The app's HarnessWeaver plugin, loaded by ``plugins.local: ["app.plugin"]`` in ``harness.yaml``.

It contributes ``app.retriever``, the context provider behind ``context.kb``. Add your own
guards, middleware or providers here (see docs/extending.md).
"""

import harnessweaver as hw

from app.retrieval.provider import Retriever

plugin = hw.Plugin(
    namespace="app",
    version="0.1.0",
    providers=[Retriever],
    description="Vector retrieval (memory, Qdrant, pgvector) for the RAG template",
)
