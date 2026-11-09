"""The HTTP API. Run it with ``rag serve`` (or ``uvicorn app.api:create_app --factory``).

    POST /v1/answer          a cited answer
    POST /v1/answer/stream   the same, as server-sent events: token, reset, result
    POST /v1/search          the passages the caller may see, without an answer
    GET  /healthz            the process is up
    GET  /readyz             the vector store answers

Every request is a governed HarnessWeaver run: the caller's tenant and principal drive
retrieval ACLs and policy, and the run's trace, cost and guards apply as in harness.yaml.
"""

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import harnessweaver as hw
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from harnessweaver.errors import HarnessError, RunInterrupted
from pydantic import BaseModel, Field

from app.auth import Caller, current_caller, load_api_keys
from app.models import Answer
from app.retrieval import Retriever

__all__ = ["create_app"]

NOT_FOUND_MESSAGE = os.environ.get(
    "RAG_NOT_FOUND_MESSAGE", "I couldn't find that in the knowledge base."
)


CallerDep = Annotated[Caller, Depends(current_caller)]


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class Citation(BaseModel):
    id: str
    title: str | None = None
    uri: str | None = None
    page: int | None = None


class AnswerOut(BaseModel):
    run_id: str
    found: bool
    answer: str
    citations: list[Citation]


class Passage(Citation):
    score: float | None
    text: str


class SearchOut(BaseModel):
    passages: list[Passage]


def _meta(metadata: dict[str, Any]) -> dict[str, Any]:
    return {k: metadata.get(k) for k in ("title", "uri", "page")}


async def answer_out(retriever: Retriever, result: hw.RunResult, caller: Caller) -> AnswerOut:
    """A completed ``answer`` run as the API returns it, with titles and links for its citations."""
    answer = Answer.model_validate(result.output)
    chunks = await retriever.cite(
        answer.cited_ids(), tenant=caller.tenant, principal=caller.principal
    )
    return AnswerOut(
        run_id=result.run_id,
        found=answer.answer is not None,
        answer=answer.answer or NOT_FOUND_MESSAGE,
        citations=[Citation(id=c.id, **_meta(c.metadata)) for c in chunks],
    )


def failure_of(result: hw.RunResult) -> dict[str, Any]:
    """Why a run didn't complete: a control's code (``policy_denied``) or the error's (``provider_error``)."""
    error = result.error
    code = (error.details.get("code") or error.code.value) if error else None
    return {"run_id": result.run_id, "status": result.status.value, "code": code}


def create_app(
    spec: str | Path | None = None, env: str | None = None, *, harness: hw.Harness | None = None
) -> FastAPI:
    """Build the app.

    Args:
        spec: The harness spec; defaults to ``$RAG_SPEC``, then harness.yaml.
        env: The overlay; defaults to ``$HARNESSWEAVER_ENV``.
        harness: A harness you built yourself (tests, or embedding the API in a larger app).
    """
    spec_path = Path(spec or os.environ.get("RAG_SPEC", "harness.yaml"))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.keys = load_api_keys()
        governed = harness or hw.load(spec_path, env=env)
        async with governed:  # owns the background runs that /v1/answer/stream submits
            app.state.harness = governed
            app.state.retriever = Retriever.from_spec(governed.spec)
            yield

    app = FastAPI(title="RAG service", version="0.1.0", lifespan=lifespan)

    @app.post("/v1/answer", response_model=AnswerOut)
    async def answer(
        body: Question,
        request: Request,
        caller: CallerDep,
        idempotency_key: Annotated[str | None, Header()] = None,
    ) -> AnswerOut:
        harness: hw.Harness = request.app.state.harness
        result = await harness.run(
            "answer",
            body.question,
            principal=caller.principal,
            tenant=caller.tenant,
            idempotency_key=idempotency_key,
        )
        if not result.ok:
            blocked = result.status is hw.RunStatus.BLOCKED
            raise HTTPException(status_code=422 if blocked else 503, detail=failure_of(result))
        return await answer_out(request.app.state.retriever, result, caller)

    @app.post("/v1/answer/stream")
    async def answer_stream(
        body: Question, request: Request, caller: CallerDep
    ) -> StreamingResponse:
        harness: hw.Harness = request.app.state.harness
        retriever: Retriever = request.app.state.retriever

        async def events() -> AsyncIterator[str]:
            # Subscribe right after submitting, with no await in between, so no event is missed.
            run_id = await harness.submit(
                "answer", body.question, principal=caller.principal, tenant=caller.tenant
            )
            async for event in harness.stream(run_id):
                if event.type == "chunk" and event.chunk is not None:
                    yield _sse("token", {"text": event.chunk.delta})
                elif (
                    event.type == "trace"
                    and event.trace is not None
                    and event.trace.code == "repair"
                ):
                    yield _sse("reset", {})  # the answer so far was rejected; a new one follows
            result = await harness.result(run_id)
            if result.ok:
                yield _sse("result", (await answer_out(retriever, result, caller)).model_dump())
            else:
                yield _sse("error", failure_of(result))

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(events(), media_type="text/event-stream", headers=headers)

    @app.post("/v1/search", response_model=SearchOut)
    async def search(body: Question, request: Request, caller: CallerDep) -> SearchOut:
        harness: hw.Harness = request.app.state.harness
        try:
            async with harness.session(principal=caller.principal, tenant=caller.tenant) as s:
                query = await s.guard("input", body.question)
                items = await s.context("kb").fetch(str(query))
        except RunInterrupted as exc:  # a control stopped it: policy, an input guard
            raise HTTPException(status_code=422, detail={"code": exc.details.get("code")}) from None
        except HarnessError as exc:  # the embedder or the store failed
            raise HTTPException(status_code=503, detail={"code": exc.code.value}) from None
        return SearchOut(
            passages=[
                Passage(id=i.id, score=i.score, text=i.content, **_meta(dict(i.metadata)))
                for i in items
            ]
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> dict[str, str]:
        retriever: Retriever = request.app.state.retriever
        try:
            await retriever.store.get(["readyz"])
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"vector store: {type(exc).__name__}"
            ) from None
        return {"status": "ready"}

    return app


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
