"""``kind: pgvector``: a Postgres table with the pgvector extension (``pip install .[pgvector]``).

The table (named after ``collection``) is created on first write, with an HNSW index for
cosine distance. ACLs are filtered in SQL, so Postgres only returns what the caller may
see. The DSN comes from the environment variable ``dsn_env`` names (default DATABASE_URL).
"""

import json
import os
from collections.abc import Sequence
from typing import Any

import harnessweaver as hw
from harnessweaver.errors import ProviderError
from harnessweaver.ports import network_allowed

from app.retrieval.config import StoreConfig
from app.retrieval.stores.base import Chunk, Hit, acl_lists

__all__ = ["PgVectorStore"]

_COLUMNS = "id, doc_id, text, acl, metadata"


def _vector(values: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def _chunk(row: Any) -> Chunk:
    return Chunk(
        id=row["id"],
        doc_id=row["doc_id"],
        text=row["text"],
        acl=row["acl"],
        metadata=row["metadata"],
    )


class PgVectorStore:
    def __init__(self, config: StoreConfig) -> None:
        self.table = config.collection  # validated as an SQL identifier by StoreConfig
        self.dsn_env = config.dsn_env or "DATABASE_URL"
        self._pool: Any = None

    async def _connect(self) -> Any:
        if not network_allowed():
            raise ProviderError("Postgres calls are off in test mode (HARNESSWEAVER_TEST_MODE=1)")
        if self._pool is None:
            import asyncpg  # imported on first use (S8)

            dsn = os.environ.get(self.dsn_env)
            if not dsn:
                raise ProviderError(f"set ${self.dsn_env} to the Postgres DSN")

            async def init(conn: Any) -> None:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

            self._pool = await asyncpg.create_pool(dsn, min_size=1, max_size=10, init=init)
        return self._pool

    async def _missing_table(self, pool: Any) -> bool:
        return await pool.fetchval("SELECT to_regclass($1) IS NULL", self.table)

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if not chunks:
            return
        pool = await self._connect()
        t = self.table
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {t} (
                    id text PRIMARY KEY,
                    doc_id text NOT NULL,
                    text text NOT NULL,
                    acl jsonb,
                    acl_tenants text[] NOT NULL DEFAULT '{{}}',
                    acl_principals text[] NOT NULL DEFAULT '{{}}',
                    acl_roles text[] NOT NULL DEFAULT '{{}}',
                    metadata jsonb NOT NULL DEFAULT '{{}}',
                    embedding vector({len(vectors[0])}) NOT NULL
                )"""
            )
            await conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_doc_id ON {t} (doc_id)")
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {t}_embedding ON {t} USING hnsw (embedding vector_cosine_ops)"
            )
            await conn.executemany(
                f"""INSERT INTO {t} (id, doc_id, text, acl, acl_tenants, acl_principals, acl_roles,
                                     metadata, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        doc_id = EXCLUDED.doc_id, text = EXCLUDED.text, acl = EXCLUDED.acl,
                        acl_tenants = EXCLUDED.acl_tenants, acl_principals = EXCLUDED.acl_principals,
                        acl_roles = EXCLUDED.acl_roles, metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding""",
                [
                    (
                        c.id,
                        c.doc_id,
                        c.text,
                        c.acl.model_dump(mode="json") if c.acl else None,
                        *acl_lists(c.acl),
                        c.metadata,
                        _vector(v),
                    )
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )

    async def delete(self, doc_ids: Sequence[str]) -> None:
        pool = await self._connect()
        if doc_ids and not await self._missing_table(pool):
            await pool.execute(
                f"DELETE FROM {self.table} WHERE doc_id = ANY($1::text[])", list(doc_ids)
            )

    async def search(self, vector: Sequence[float], k: int, env: hw.Envelope) -> list[Hit]:
        pool = await self._connect()
        if await self._missing_table(pool):
            return []
        rows = await pool.fetch(
            f"""SELECT {_COLUMNS}, 1 - (embedding <=> $1::vector) AS score
                FROM {self.table}
                WHERE (cardinality(acl_tenants) = 0 OR $2 = ANY(acl_tenants))
                  AND ((cardinality(acl_principals) = 0 AND cardinality(acl_roles) = 0)
                       OR $3 = ANY(acl_principals) OR acl_roles && $4::text[])
                ORDER BY embedding <=> $1::vector
                LIMIT $5""",
            _vector(vector),
            env.tenant,
            env.principal.id,
            list(env.principal.roles),
            k,
        )
        return [Hit(_chunk(r), float(r["score"])) for r in rows]

    async def get(self, ids: Sequence[str]) -> list[Chunk]:
        pool = await self._connect()
        if not ids or await self._missing_table(pool):
            return []
        rows = await pool.fetch(
            f"SELECT {_COLUMNS} FROM {self.table} WHERE id = ANY($1::text[])", list(ids)
        )
        found = {r["id"]: _chunk(r) for r in rows}
        return [found[i] for i in ids if i in found]

    async def doc_hashes(self) -> dict[str, str]:
        pool = await self._connect()
        if await self._missing_table(pool):
            return {}
        rows = await pool.fetch(
            f"SELECT DISTINCT ON (doc_id) doc_id, metadata->>'doc_hash' AS doc_hash FROM {self.table}"
        )
        return {r["doc_id"]: r["doc_hash"] or "" for r in rows}
