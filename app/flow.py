"""The ``answer`` task: retrieve, then answer with citations. Run by the ``session`` runner.

The app owns the steps and the harness governs each one: the question has passed the
input guards before this function sees it, ``fetch`` returns only items the caller may
see (ACL-checked, injection-screened, capped), and ``generate`` runs the output guards,
so an answer that cites nothing it retrieved is sent back for one repair.
"""

import harnessweaver as hw

from app.models import Answer

__all__ = ["NOT_FOUND", "answer"]

NOT_FOUND = "NOT_FOUND"
"""What the prompt tells the model to reply when the sources don't answer the question.
Too short for the citation guard to ask for a citation."""


async def answer(s: hw.Session, question: str) -> Answer:
    passages = await s.context("kb").fetch(question)
    if not passages:  # nothing the caller may see answers it: don't let the model guess
        return Answer(answer=None)
    sources = "\n\n".join(f"[{p.id}] {p.content}" for p in passages)
    text = await s.generate(prompt="answer@1", question=question, sources=sources, stream=True)
    return Answer(answer=None if text.strip() == NOT_FOUND else text)
