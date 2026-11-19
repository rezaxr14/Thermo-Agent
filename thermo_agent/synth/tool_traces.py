"""Deterministic tool-using traces built from existing ground truth (Gate 3a).

The frontier CoT harness produces *model* tool traces; this module produces the other
half of the Gate 3a / Gate 4 mix: traces generated with no model at all.  Every tool
result embedded in an SFT completion is the **verbatim** JSON string returned by the
real ``teacher.toolkit`` functions (they run the project's solver / verifiers).  The
only text we author is short ``<think>`` scaffolding assembled from fields of those
JSON results, so no physical number is ever invented here.

Variants
--------
``direct``
    Call the physics tools on the row's own ground-truth answer and finish with that
    answer.  Repair inspects the broken plant first; audit inspects, ranks and re-checks
    the fuel effect of the proposed ``SET`` change; ledger inspects the broken lines and
    confirms closure; design verifies the candidate spec and solves it.
``recover``
    Same shape, but first call a tool on a *known wrong* candidate so the tool exposes a
    failure, then call it again on the correct answer.  For repair the wrong candidate
    comes from ``teacher.validate.adversarial_candidates``; a specific one can be chosen
    with ``variant="recover:wrong_scale"`` (etc.).  For the other tasks, whose
    adversarial generator is repair-only, the wrong candidate is a deliberately empty
    answer (``recover:empty``), which the tools reject uniformly.

Every trace is validated with ``teacher.validate.validate(task, final_answer, row)``
before it is returned.  A rejected row is reported (``accepted=False``) rather than
silently emitted; with ``strict=True`` (the default for direct callers) it raises.

    from thermo_agent.synth.tool_traces import build_trace
    trace = build_trace(row, "repair", variant="direct")
    trace["sft_completion"]  # ends with row["completion"][0]["content"]
"""
from __future__ import annotations

import json
import re

from ..teacher import validate
from ..teacher.toolkit import make_tools
from . import combine

# A stream-state table from a large plant can dwarf the rest of the row.  We never
# truncate a tool result (that would stop it being verbatim): a tool whose JSON is
# longer than this cap is dropped as a whole and recorded in `dropped_large_tools`.
DEFAULT_MAX_TOOL_RESULT_CHARS = 4500
DEFAULT_MAX_TOKENS = 8000
TOKEN_RESERVE = combine.TOKEN_RESERVE

# Used only by the test that checks no ground-truth *answer block* was pasted into a
# tool result.  Fragments such as a SET line are legitimate tool facts (the audit tool
# echoes the change it applied); whole answer blocks are not.
ANSWER_BLOCK = re.compile(
    r"<(diagnosis|analysis|ranking|change|prediction|edits|fix|rationale|design)>.*?</\1>",
    re.S,
)

ADVERSARIES = ("no_edits", "echo_broken", "wrong_scale", "third_fluid", "bad_port")


def ground_truth(row: dict) -> str:
    """The row's existing ground-truth completion text (never rewritten here)."""
    c = row.get("completion")
    if isinstance(c, list) and c:
        return c[0]["content"]
    if isinstance(c, str):
        return c
    raise ValueError("row has no assistant completion")


def _tools(task: str, row: dict) -> dict:
    return {fn.__name__: fn for fn in make_tools(task, row)}


def _ntok(text: str) -> int | None:
    tok = combine.tokenizer()
    return len(tok.encode(text, add_special_tokens=False).ids) if tok is not None else None


def _facts(payload: str) -> str:
    """Terse reasoning copied from a tool's JSON result; no new numbers are introduced."""
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        return "tool returned a non-JSON payload"
    bits: list[str] = []
    for key in ("solver_status", "converged", "solved", "valid"):
        if key in data and data[key] is not None:
            value = data[key]
            bits.append(f"{key}={str(value).lower() if isinstance(value, bool) else value}")
    for key in ("error", "solver_message"):
        if data.get(key):
            bits.append(f"{key}={json.dumps(data[key], ensure_ascii=False)}")
    for key in ("failed_checks", "failed_lines", "unit_range_flags"):
        if data.get(key):
            bits.append(f"{key}={json.dumps(data[key], ensure_ascii=False)}")
    if "closure_residual_w" in data:
        bits.append(f"closure_residual_w={data['closure_residual_w']}")
    if "ledger_closure_residual_W" in data:
        bits.append(f"ledger_closure_residual_W={data['ledger_closure_residual_W']}")
    eff = data.get("candidate_effect")
    if isinstance(eff, dict):
        keep = {k: eff[k] for k in ("applied", "change_valid", "delta_fuel_W", "reason") if k in eff}
        if keep:
            bits.append("candidate_effect=" + json.dumps(keep, ensure_ascii=False))
    if "n_streams" in data:
        bits.append(f"n_streams={data['n_streams']}")
    return "; ".join(bits) or "tool returned an empty result"


def _think(*lines: str) -> str:
    return "<think>\n" + "\n".join(line for line in lines if line) + "\n</think>"


def _turn(reasoning: str = "", assistant: str = "", name: str | None = None,
          candidate: str | None = None) -> dict:
    calls = [] if name is None else [{"name": name, "arguments": {"candidate": candidate or ""}}]
    return {"assistant": assistant, "reasoning": reasoning,
            "tool_calls": calls, "tool_results": []}


def _wrong_candidate(task: str, row: dict, variant: str) -> tuple[str, str]:
    """(label, wrong answer) for a recover variant; label goes into provenance."""
    if task == "repair":
        candidates = dict(validate.adversarial_candidates(row))
        wanted = variant.split(":", 1)[1] if ":" in variant else ""
        if wanted and wanted in candidates:
            return wanted, candidates[wanted]
        for name in ADVERSARIES:
            if name in candidates:
                return name, candidates[name]
    # Other tasks have no adversarial generator; an empty answer is rejected by the
    # tools with the uniform {"ok": false, "failed_checks": [...]} shape.
    return "empty", ""


def _plan(task: str, variant: str, final: str, wrong_label: str, wrong: str) -> list[tuple[str, str, str]]:
    """Ordered (tool_name, candidate, templated reasoning lead) steps."""
    recover = variant.startswith("recover")
    solve_plant = ("solve_plant", final, "Inspect the stream states of the verified plant.")
    if task == "repair":
        inspect = ("inspect_plant", "", "Inspect the broken plant as given before editing.")
        check = ("validity_check", final, "Apply the corrected edits and check the repaired plant.")
        law = ("first_law_check", final, "Confirm the mass and energy balances of the repaired plant.")
        if not recover:
            return [inspect, check, law]
        bad = ("validity_check", wrong,
               f"Test the {wrong_label} candidate first and read the failure reported by the tool.")
        return [inspect, bad, check]
    if task == "audit":
        inspect = ("inspect_plant", "", "Inspect the solved plant as given before ranking it.")
        rank = ("second_law_check", final,
                "Rank the components by exergy destruction, propose the SET change and re-check its fuel effect.")
        confirm = ("validity_check", final, "Confirm the re-solved plant with the change applied is valid.")
        if not recover:
            return [inspect, rank, confirm]
        bad = ("second_law_check", wrong,
               "Propose no SET change first and read the candidate effect reported by the tool.")
        return [inspect, bad, rank, confirm]
    if task == "ledger":
        inspect = ("inspect_plant", "",
                   "Inspect the ledger as given for closure failure and negative exergy destruction.")
        confirm = ("validity_check", final,
                   "Apply the line fixes and confirm closure and non-negative destruction.")
        if not recover:
            return [inspect, confirm]
        bad = ("validity_check", wrong, "Try an empty fix list first and read the tool's failure.")
        return [inspect, bad, confirm]
    if task == "design":
        check = ("validity_check", final, "Build and verify the candidate specification.")
        if not recover:
            return [check, solve_plant]
        bad = ("validity_check", wrong, "Try an empty specification first and read the tool's failure.")
        return [bad, check, solve_plant]
    raise KeyError(f"unknown task {task!r}")


def build_trace(row: dict, task: str, *, variant: str = "direct",
                max_tool_result_chars: int = DEFAULT_MAX_TOOL_RESULT_CHARS,
                max_tokens: int = DEFAULT_MAX_TOKENS, strict: bool = True,
                prompt_override: list[dict] | None = None) -> dict:
    """Build one deterministic tool trace and validate its final answer.

    Returns a dict with ``sft_completion``, ``turns``, ``tool_calls``, ``accepted``,
    ``source="deterministic_tools"``, ``final_answer``, ``verdict``, ``char_length``,
    ``tokens`` and bookkeeping for dropped (over-cap) tool results.  On a rejected final
    answer it returns ``accepted=False`` with an ``error`` (or raises when ``strict``).

    ``prompt_override`` is the FINAL prompt messages that will actually be emitted
    (e.g. after the harness block has been appended).  When given, ``tokens`` and
    ``over_budget`` are measured from those messages, not from the row's plain
    ``prompt``.  The plain-prompt measurement is still returned as ``plain_tokens``
    so the manifest can show the difference.  With no override, behaviour is unchanged.
    """
    final = ground_truth(row)
    if task not in validate.TASKS:
        raise KeyError(f"unknown task {task!r}")
    recover = variant.startswith("recover")
    wrong_label, wrong = _wrong_candidate(task, row, variant) if recover else ("", "")
    tools = _tools(task, row)
    dropped: list[dict] = []
    turns: list[dict] = []
    last = ""
    for name, candidate, lead in _plan(task, variant, final, wrong_label, wrong):
        reasoning = _think(lead, f"Previous tool result: {_facts(last)}" if last else "")
        try:
            raw = tools[name](candidate or "")
        except Exception as exc:  # a tool failure must never invent a result
            dropped.append({"tool": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if len(raw) > max_tool_result_chars:
            dropped.append({"tool": name, "chars": len(raw), "cap": max_tool_result_chars})
            continue
        turn = _turn(reasoning=reasoning, name=name, candidate=candidate or "")
        turn["tool_results"] = [raw]
        turns.append(turn)
        last = raw

    if last:
        final_reasoning = _think("The tool evidence is in; return the verifier-accepted answer.",
                                 f"Last tool result: {_facts(last)}")
    else:
        final_reasoning = ""
    turns.append(_turn(reasoning=final_reasoning, assistant=final))

    parts: list[str] = []
    for turn in turns:
        if turn["reasoning"]:
            parts.append(turn["reasoning"])
        if turn["assistant"]:
            parts.append(turn["assistant"])
        for call, result in zip(turn["tool_calls"], turn["tool_results"]):
            args = json.dumps(call["arguments"], ensure_ascii=False)
            parts.append(f"<tool_call name=\"{call['name']}\">{args}</tool_call>")
            parts.append(f"<tool_result>{result}</tool_result>")
    sft = "\n".join(parts)

    verdict = validate.validate(task, final, row)
    n_calls = sum(len(t["tool_calls"]) for t in turns)
    prompt_messages = row.get("prompt", []) if prompt_override is None else prompt_override
    n_prompt = sum(_ntok(m.get("content", "")) for m in prompt_messages)
    n_prompt_plain = sum(_ntok(m.get("content", "")) for m in row.get("prompt", []))
    n_completion = _ntok(sft)
    tokens = None
    plain_tokens = None
    if n_prompt is not None and n_completion is not None:
        tokens = n_prompt + n_completion + TOKEN_RESERVE
    if n_prompt_plain is not None and n_completion is not None:
        plain_tokens = n_prompt_plain + n_completion + TOKEN_RESERVE

    out = {
        "sft_completion": sft,
        "turns": turns,
        "tool_calls": n_calls,
        "accepted": bool(verdict["accepted"]),
        "source": "deterministic_tools",
        "task": task,
        "variant": variant,
        "recover_candidate": wrong_label or None,
        "final_answer": final,
        "verdict": verdict,
        "char_length": len(sft),
        "prompt_tokens": n_prompt,
        "completion_tokens": n_completion,
        "tokens": tokens,
        "over_budget": bool(tokens is not None and tokens > max_tokens),
        "prompt_tokens_plain": n_prompt_plain,
        "plain_tokens": plain_tokens,
        "dropped_large_tools": dropped,
        "record_id": str(row.get("id", "")),
    }
    if not out["accepted"]:
        out["error"] = "final answer rejected: " + "; ".join(verdict["reasons"])
        if strict:
            raise AssertionError(f"deterministic {variant} trace failed validation: {out['error']}")
    elif n_calls == 0:
        out["error"] = "all tool results were dropped by the length cap"
        if strict:
            raise AssertionError(f"deterministic {variant} trace has no tool calls: {dropped}")
    elif not sft.rstrip().endswith(final.rstrip()):
        raise AssertionError("SFT completion does not end with the accepted final answer")
    return out


def leaky_tool_results(trace: dict) -> list[str]:
    """Answer blocks found inside tool results (should always be empty)."""
    return [m.group(0)[:80] for turn in trace.get("turns", [])
            for raw in turn.get("tool_results", [])
            for m in ANSWER_BLOCK.finditer(raw)]
