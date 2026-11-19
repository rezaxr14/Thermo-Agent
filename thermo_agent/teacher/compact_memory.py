"""Token-efficient failure memory for the frontier agent (D11) + behavior guidance (D10).

The full memory ledger (data/memory/*.jsonl) is for OUR analysis: it keeps model ids, timestamps,
cost, latency, run ids and the detailed verifier log. The AI does not need any of that — it needs
only *what failed and what works*. This module distils each retrieved Note to a single short, marked line:
    OK/FAIL fault_class: correct pattern
and drops everything that does not change the model's reasoning (source model, id, record_id,
architecture, symptoms, timestamp, token/cost/latency fields in `extra`). The compact block is what
build_context feeds the model; the full ledger remains intact for provenance and our reports.
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
    # Drop provider/model slugs, whether '/' or '-' separated. The old rule only
    # handled the slash form, so `google-gemini-...` and bare `luna-2` leaked.
    text = re.sub(r"\b(?:google|openai|anthropic)[/-][A-Za-z0-9._-]+", "", text, flags=re.I)
    # Drop bare model-name tokens that can leak into lesson/symptom/fix strings.
    text = re.sub(r"\b(?:gemini|gpt-[A-Za-z0-9._-]+|claude|haiku|luna)[A-Za-z0-9._-]*",
                  "", text, flags=re.I)
    # Drop bookkeeping key=value tails (ts, cost, tokens, latency, run/record ids).
    text = re.sub(r"\b(?:ts|cost|tokens?|latency|run[_-]?id|record[_-]?id)\s*=\s*[A-Za-z0-9._eE+-]+",
                  "", text, flags=re.I)
    # Drop any remaining bare 'latency' word (with an optional numeric value).
    text = re.sub(r"\blatency\b(?:\s*[=:]?\s*[0-9.eE+-]+)?", "", text, flags=re.I)
    # Drop ISO-ish timestamps (2026-01-14, 2026-01-14T09:12:00, ...).
    text = re.sub(r"\b20\d{2}-\d{2}-\d{2}(?:[T ][0-9:.+-]+)?", "", text)
    # Collapse whitespace left behind by the removals.
    return re.sub(r"\s+", " ", text).strip()


def compact_notes(notes: list[mem.Note], k: int = 5) -> str:
    """Return a short 'what failed -> what works' block; no model/time/cost/id/symptom metadata.

    Only the fault class and the reusable fix pattern reach the model.  Bookkeeping fields (record/id/source,
    architecture, symptoms, extra) and any model name, timestamp, cost or token figure are deliberately dropped.
    Each line is marked OK (verifier-accepted) or FAIL (verifier-rejected) so the two kinds are distinguishable.
    """
    if not notes:
        return ""
    lines = []
    for n in notes[:k]:
        fault = _clean(n.fault_type or n.task)
        fix = _clean(n.fix)
        lesson = _clean(n.lesson)
        # Keep the lesson's 'Correct pattern' hint only if there is no explicit fix.
        hint = fix or (lesson.split(". Correct", 1)[-1].lstrip() if ". Correct" in lesson else "")
        marker = "OK  " if mem._accepted(n) else "FAIL"
        line = f"- {marker} {fault}:"
        if hint:
            line += f" {hint}"
        elif lesson:
            line += f" {lesson}"
        lines.append(line)
    if not lines:
        return ""
    return ("FAILURE MEMORY (what failed -> what works; OK=accepted, FAIL=rejected; classes only, not this "
            "row's answer):\n" + "\n".join(lines))


def agent_guidance_block(include_causes: bool = True) -> str:
    parts = [GOOD_PRACTICE]
    if include_causes:
        parts.append(COMMON_FAILURE_CAUSES)
    return "\n".join(parts)
