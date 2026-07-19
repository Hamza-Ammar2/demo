"""Compatibility wrapper — fundamentals now come from the foundation graph.

Prefer `cyclebench.foundation.query.assemble_read`. This module keeps the old
`match_fundamentals` API working by projecting FoundationCards into the prior shape.
"""

from __future__ import annotations

from cyclebench.foundation.query import assemble_read


def match_fundamentals(intake: dict) -> list[dict]:
    """Legacy shape used by older callers / tests."""
    read = assemble_read(intake)
    out = []
    for c in read.cards:
        out.append({
            "id": c.association_id,
            "title": c.title,
            "talking_point": c.foundation_fact,
            "ask_doctor": c.ask_doctor,
            "source": c.source,
            "when": ", ".join(c.datasets) if c.datasets else c.relation,
            "evidence_summaries": c.evidence_summaries,
            "personal_pattern": c.personal_pattern,
        })
    return out


def fundamentals_as_questions(hits: list[dict]) -> list[str]:
    return [h["ask_doctor"] for h in hits if h.get("ask_doctor")]
