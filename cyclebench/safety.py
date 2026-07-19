"""Medical-safety language guard.

Enforces the project's conservative-language rules on any generated text before it
reaches a DoctorBrief. Fails loudly so unsafe language cannot ship silently.
"""

from __future__ import annotations

import re

# Affirmative diagnostic / causal / treatment CLAIMS. Matched case-insensitively.
# Deliberately targets assertions (verbs/claims), not bare nouns: the word "diagnosis"
# is fine in "no diagnosis is established"; "we diagnose you" is not. Negation-aware
# matching (see find_violations) additionally allows explicitly negated phrasings.
FORBIDDEN_PATTERNS = [
    r"\byou have\b",
    r"\bdiagnosed with\b",
    r"\bdiagnose you\b",
    r"\bwe (can |)diagnose\b",
    r"\byou (should|must) (take|stop|start|change|switch|increase|decrease)\b",
    r"\bwe recommend (taking|stopping|starting|changing|switching|you)\b",
    r"\bis caused by\b",
    r"\bcaused by\b",
    r"\bis causing\b",
    r"\bcauses your\b",
    r"\bprescrib(e|ed|ing)\b",
    r"\btreatment (plan|recommendation)\b",
    r"\byou are suffering from\b",
    r"\bconfirms? (that )?(you|the patient) (have|has)\b",
    r"\bdisease (risk|probability)\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_PATTERNS]
_NEGATION = re.compile(r"\b(no|not|n't|without|cannot|never|isn't|aren't|don't|doesn't)\b", re.IGNORECASE)


class SafetyViolation(Exception):
    """Raised when generated text contains diagnostic/causal/treatment language."""


def _is_negated(text: str, start: int) -> bool:
    """True if a negation cue appears in the ~25 chars preceding a match."""
    window = text[max(0, start - 25):start]
    return bool(_NEGATION.search(window))


def find_violations(text: str) -> list[str]:
    """Return affirmative forbidden phrases in `text` (empty if clean).

    Negated phrasings (e.g. 'not caused by', 'does not diagnose') are allowed.
    """
    hits: list[str] = []
    text = text or ""
    for pat in _COMPILED:
        for m in pat.finditer(text):
            if _is_negated(text, m.start()):
                continue
            hits.append(m.group(0))
    return hits


def assert_safe(text: str, where: str = "text") -> None:
    """Raise SafetyViolation if `text` contains forbidden language."""
    hits = find_violations(text)
    if hits:
        raise SafetyViolation(
            f"forbidden medical language in {where}: {sorted(set(h.lower() for h in hits))}"
        )


def assert_brief_safe(brief) -> None:
    """Validate every text field of a DoctorBrief."""
    fields = (
        [brief.opening_statement, brief.disclaimer]
        + brief.strongest_findings
        + brief.unresolved_questions
        + brief.established
        + brief.possible
        + brief.not_established
        + brief.missing
    )
    for i, txt in enumerate(fields):
        assert_safe(txt, where=f"doctor_brief.field[{i}]")
