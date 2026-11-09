"""rag: ingest documents, ask and search from the terminal, and serve the API.

Every command takes --env (or $HARNESSWEAVER_ENV) to pick the overlay harness.<env>.yaml.
"""

import asyncio
import os
from pathlib import Path
from typing import Annotated

import harnessweaver as hw
import typer
from harnessweaver.errors import HarnessError

from app.api import answer_out, failure_of
from app.auth import Caller
from app.ingest import ingest as run_ingest
from app.retrieval import Retriever

cli = typer.Typer(help=__doc__, no_args_is_help=True, add_completion=False)

Spec = Annotated[Path, typer.Option(help="The harness spec.")]
Env = Annotated[str | None, typer.Option("--env", "-e", help="Overlay: harness.<env>.yaml.")]
Tenant = Annotated[str, typer.Option(help="Who the run acts for (retrieval ACLs).")]
PrincipalId = Annotated[str, typer.Option("--principal", help="The caller's id.")]
Roles = Annotated[
    list[str] | None, typer.Option("--role", help="A role the caller holds (repeatable).")
]


def _caller(tenant: str, principal: str, roles: list[str] | None) -> Caller:
    return Caller(tenant, hw.Principal(id=principal, roles=tuple(roles or ())))


@cli.command()
def ingest(
    config: Annotated[Path, typer.Option(help="What to index.")] = Path("ingest.yaml"),
    spec: Spec = Path("harness.yaml"),
    env: Env = None,
    prune: Annotated[bool, typer.Option(help="Delete documents no longer in any source.")] = False,
) -> None:
    """Index the sources in ingest.yaml (incremental: unchanged documents are skipped)."""
    report = asyncio.run(run_ingest(config, spec=spec, env=env, prune=prune))
    typer.echo(
        f"indexed {report.indexed} document(s) ({report.chunks} chunks), "
        f"unchanged {report.unchanged}, deleted {report.deleted}, failed {len(report.failed)}"
    )
    for doc_id, reason in report.failed.items():
        typer.echo(f"  failed {doc_id}: {reason}", err=True)
    if report.failed:
        raise typer.Exit(1)


@cli.command()
def ask(
    question: str,
    spec: Spec = Path("harness.yaml"),
    env: Env = None,
    tenant: Tenant = "default",
    principal: PrincipalId = "cli",
    role: Roles = None,
) -> None:
    """Answer a question with citations, as POST /v1/answer does."""
    caller = _caller(tenant, principal, role)

    async def main() -> None:
        harness = hw.load(spec, env=env)
        result = await harness.run("answer", question, principal=caller.principal, tenant=tenant)
        if not result.ok:
            failure = failure_of(result)
            typer.echo(f"{failure['status']}: {failure['code']}", err=True)
            raise typer.Exit(1)
        out = await answer_out(Retriever.from_spec(harness.spec), result, caller)
        typer.echo(out.answer)
        for c in out.citations:
            typer.echo(f"  [{c.id}] {c.title} ({c.uri})")

    asyncio.run(main())


@cli.command()
def search(
    query: str,
    spec: Spec = Path("harness.yaml"),
    env: Env = None,
    tenant: Tenant = "default",
    principal: PrincipalId = "cli",
    role: Roles = None,
) -> None:
    """Show the passages a caller would retrieve, with scores (tune min_score and chunking)."""
    caller = _caller(tenant, principal, role)

    async def main() -> None:
        harness = hw.load(spec, env=env)
        async with harness.session(principal=caller.principal, tenant=tenant) as s:
            items = await s.context("kb").fetch(query)
        for item in items:
            preview = " ".join(item.content.split())[:100]
            typer.echo(f"{item.score:.3f}  [{item.id}]  {preview}")
        if not items:
            typer.echo("no passages")

    asyncio.run(main())


@cli.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    env: Env = None,
    reload: Annotated[bool, typer.Option(help="Restart on code changes (development).")] = False,
) -> None:
    """Serve the HTTP API with uvicorn."""
    import uvicorn

    if env:
        os.environ["HARNESSWEAVER_ENV"] = env
    uvicorn.run("app.api:create_app", factory=True, host=host, port=port, reload=reload)


def main() -> None:
    try:
        cli()
    except HarnessError as exc:  # a provider or spec problem: say what failed, not where
        typer.echo(f"error: {exc.message}", err=True)
        raise SystemExit(1) from None
