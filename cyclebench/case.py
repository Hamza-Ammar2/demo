"""Case container: the in-memory bundle the engine operates on.

Kept separate from the 6 published schema entities (schema.py) because a Case is
a working aggregate, not a persisted record type.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from cyclebench.schema import (
    SCHEMA_VERSION,
    CycleContext,
    HealthEvent,
    SourceReference,
    SubjectProfile,
)


class Case(BaseModel):
    schema_version: str = SCHEMA_VERSION
    subject: SubjectProfile
    events: list[HealthEvent] = Field(default_factory=list)
    cycles: list[CycleContext] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)

    def events_of(self, *types) -> list[HealthEvent]:
        wanted = {t.value if hasattr(t, "value") else t for t in types}
        return [e for e in self.events if e.event_type.value in wanted]

    def source(self, source_id: Optional[str]) -> Optional[SourceReference]:
        if source_id is None:
            return None
        for s in self.sources:
            if s.source_id == source_id:
                return s
        return None
