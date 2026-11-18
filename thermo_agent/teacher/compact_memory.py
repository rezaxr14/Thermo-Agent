"""Token-efficient failure memory for the frontier agent (D11) + behavior guidance (D10).

The full memory ledger (data/memory/*.jsonl) is for OUR analysis: it keeps model ids, timestamps,
cost, latency, run ids and the detailed verifier log. The AI does not need any of that — it needs
only *what failed and what works*. This module distils each retrieved Note to a single short line:
    fault_class: symptom -> correct pattern
and drops everything that does not change the model's reasoning (source model, id, record_id,
timestamp, token/cost/latency fields in `extra`). The compact block is what build_context feeds
the model; the full ledger remains intact for provenance and our reports.
"""
from __future__ import annotations

import re

from . import memory as mem

# Failure causes worth stating once, independent of any retrieved note (D10).
COMMON_FAILURE_CAUSES = (
    "UNIT MISMATCH: MW vs W, kJ/kg vs J/kg, fraction vs percent (eta=0.85 not 85), "
    "bar vs Pa, K vs Celsius; SIGN FLIP; SWAPPED PORT endpoints; WRONG FLUID class; "
    "efficiency or COP above its reversible limit; negative exergy destruction; "
    "parameters outside their valid range; tear-stream temperature not converged."
)

GOOD_PRACTICE = (
    "METHOD: (1) FIRST inventory the skills and physics tools available to you and what each "
    "checks; (2) THEN read the plant/cycle and compare it against the retrieved failure modes; "
    "(3) propose a candidate; (4) call first_law_check, second_law_check and validity_check; "
    "(5) read the numbers, locate the residual/violation, revise, and re-check; "
    "(6) when the residuals are ~0 and all checks pass, STOP and return the final XML answer."
)


def _clean(text: str) -> str:
    """Strip verbosity in a lesson while keeping its reasoning content."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    # Drop provenance/timing tails that sometimes leak into lesson strings.
    text = re.sub(r"\b(google/|openai/|anthropic/)[A-Za-z0-9._-]+", "", text)
    text = re.sub(r"\b(ts|latency|cost|tokens?)\s*=\s*[0-9.eE+-]+", "", text)
    return text.strip()


def compact_notes(notes: list[mem.Note], k: int = 5) -> str:
    """Return a short 'what failed -> what works' block; no model/time/cost/id metadata."""
    if not notes:
        return ""
    lines = []
    for n in notes[:k]:
        fault = _clean(n.fault_type or n.task)
        symptom = _clean(n.symptoms)
        fix = _clean(n.fix)
        lesson = _clean(n.lesson)
        # Keep the lesson's 'Correct pattern' hint only if there is no explicit fix.
        hint = fix or (lesson.split(". Correct", 1)[-1].lstrip() if ". Correct" in lesson else "")
        line = f"- {fault}: {symptom}"
        if hint:
            line += f" -> {hint}"
        lines.append(line)
    if not lines:
        return ""
    return ("FAILURE MEMORY (what failed -> what works; classes only, not this row's answer):\n"
            + "\n".join(lines))


def agent_guidance_block(include_causes: bool = True) -> str:
    parts = [GOOD_PRACTICE]
    if include_causes:
        parts.append(COMMON_FAILURE_CAUSES)
    return "\n".join(parts)
