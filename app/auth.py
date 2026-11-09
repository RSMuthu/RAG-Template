"""Who is calling: one FastAPI dependency, ``current_caller``. Replace it to use JWT, OIDC,
mTLS or your gateway's identity headers; everything else only needs a :class:`Caller`.

The default reads API keys from the YAML file ``RAG_API_KEYS_FILE`` names::

    # key: who the key acts as (tenant and roles drive retrieval ACLs and policy)
    "replace-with-a-long-random-key": { tenant: acme, principal: ann, roles: [staff] }

Without the file every caller is ``anonymous`` in the ``default`` tenant, which is fine
on your machine and refused when ``HARNESSWEAVER_ENV=prod``.
"""

import hmac
import os
from dataclasses import dataclass
from pathlib import Path

import harnessweaver as hw
import yaml
from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict

__all__ = ["ANONYMOUS", "Caller", "current_caller", "load_api_keys"]


@dataclass(frozen=True, slots=True)
class Caller:
    tenant: str
    principal: hw.Principal


ANONYMOUS = Caller(tenant="default", principal=hw.Principal(id="anonymous"))


class _Key(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant: str = "default"
    principal: str
    roles: list[str] = []


def load_api_keys() -> dict[str, Caller] | None:
    """The configured keys, or ``None`` when no file is configured (anonymous mode).

    Raises:
        RuntimeError: In prod without a keys file, so a misconfigured service never starts open.
    """
    path = os.environ.get("RAG_API_KEYS_FILE")
    if not path:
        if os.environ.get("HARNESSWEAVER_ENV") == "prod":
            raise RuntimeError("set RAG_API_KEYS_FILE: prod refuses anonymous callers")
        return None
    entries = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    keys = {}
    for key, raw in entries.items():
        entry = _Key.model_validate(raw)
        keys[str(key)] = Caller(
            entry.tenant, hw.Principal(id=entry.principal, roles=tuple(entry.roles))
        )
    return keys


async def current_caller(request: Request) -> Caller:
    """The caller of an API request, from ``Authorization: Bearer <key>`` or ``X-API-Key``."""
    keys: dict[str, Caller] | None = request.app.state.keys
    if keys is None:
        return ANONYMOUS
    header = request.headers.get("authorization", "")
    given = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""
    given = given or request.headers.get("x-api-key", "")
    for key, caller in keys.items():
        if given and hmac.compare_digest(given.encode(), key.encode()):
            return caller
    raise HTTPException(status_code=401, detail="a valid API key is required")
