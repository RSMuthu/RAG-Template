"""What the ``answer`` task returns."""

import re

from pydantic import BaseModel, Field

__all__ = ["CITATION", "Answer"]

CITATION = re.compile(r"\[([A-Za-z0-9_.:#/-]+)\]")
"""How an answer cites a chunk: its id in square brackets (grounded.citations' default)."""


class Answer(BaseModel):
    """The cited answer, or ``None`` when the knowledge base doesn't answer the question.

    Keep string fields out of this model unless they cite their sources. HarnessWeaver
    runs the task's output guards on every string in it, and ``grounded.citations`` asks
    for a repair on any string of four or more words that cites nothing, which this late
    in a run ends it blocked. That's why a missing answer is ``None``, not a sentence, and
    why titles and links are looked up by id outside the run (see ``app.api``).
    """

    answer: str | None = Field(description="The answer, citing chunks as [chunk-id].")

    def cited_ids(self) -> list[str]:
        """The chunk ids the answer cites, in order of first citation.

        The citation guard has already checked that each one was retrieved in this run.
        """
        return list(dict.fromkeys(CITATION.findall(self.answer or "")))
