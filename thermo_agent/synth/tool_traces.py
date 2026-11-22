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

import hashlib
import json
import re

from ..agent import tool_protocol
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


_RENDER_TOK = None


def render_tokenizer():
    """Chat-template tokenizer used for the real 8,192-token gate (cached, local only)."""
    global _RENDER_TOK
    if _RENDER_TOK is None:
        import pathlib
        path = combine.tokenizer_path()
        config = pathlib.Path(path).parent / "tokenizer_config.json" if path else None
        if config and config.is_file():
            from transformers import AutoTokenizer
            _RENDER_TOK = AutoTokenizer.from_pretrained(str(config.parent), local_files_only=True)
        else:
            _RENDER_TOK = False
    return _RENDER_TOK or None


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
    """Ordered (tool_name, candidate, reasoning kind) steps; the reasoning text is built from real tool facts."""
    recover = variant.startswith("recover")
    if task == "repair":
        steps = [("inspect_plant", "", "inspect")]
        if recover:
            steps.append(("validity_check", wrong, "bad_guess"))
        steps += [("validity_check", final, "apply_fix"), ("first_law_check", final, "confirm")]
        return steps if not recover else steps[:3]
    if task == "audit":
        steps = [("inspect_plant", "", "inspect")]
        if recover:
            steps.append(("second_law_check", wrong, "bad_guess"))
        steps += [("second_law_check", final, "rank_and_change"), ("validity_check", final, "confirm")]
        return steps
    if task == "ledger":
        steps = [("inspect_plant", "", "inspect")]
        if recover:
            steps.append(("validity_check", wrong, "bad_guess"))
        steps.append(("validity_check", final, "apply_fix"))
        return steps
    if task == "design":
        steps = []
        if recover:
            steps.append(("validity_check", wrong, "bad_guess"))
        steps += [("validity_check", final, "propose_design"), ("solve_plant", final, "confirm")]
        return steps
    raise KeyError(f"unknown task {task!r}")


# ---- reasoning built from the real tool results (v3, 2026-11-20) -------------------------------------------
# Every number below is copied from a tool's JSON or from the ground-truth explanation; nothing is invented.
_OPENERS = {
    "repair": ("Before editing anything I will inspect the plant as given; the solver status and failed checks should show what is wrong.",
               "First I inspect the broken plant to see the solver status, the failed checks and which stream or component they point at.",
               "I start by solving the plant exactly as given and reading what fails before I propose any edit."),
    "audit": ("Before proposing a change I will inspect the solved plant: total destruction and fuel input show how much room there is.",
              "First I read the solved plant's totals (exergy destruction, fuel, net work) so the change I propose can be sized against them.",
              "I start from the plant as given and read its overall balances before ranking components."),
    "ledger": ("Before fixing any line I will inspect the ledger as given and compare each line with its recomputed value.",
               "First I read the (line, value, recomputed, residual) table as given and look for the entry that does not close.",
               "I start by reading the full ledger table and its closure residual before choosing any fix."),
}
_HYPOTHESIS = {
    "no_edits": "As a baseline I submit no edit at all, to see whether the checks fail without any change.",
    "echo_broken": "As a control I resubmit the plant unchanged, to confirm that the failure comes from the current wiring.",
    "wrong_scale": "I try the obvious parameter correction with a rescaled value to see whether the failure is only a unit or scale issue.",
    "third_fluid": "I try swapping to a different fluid class to test whether the mismatch is about the fluid label.",
    "bad_port": "I try rewiring a port to test whether the connection, not the fluid, is the problem.",
    "empty": "As a first probe I submit an empty answer to see exactly what the checker demands.",
}


def _data(payload: str) -> dict:
    try:
        d = json.loads(payload)
        return d if isinstance(d, dict) else {}
    except (TypeError, ValueError):
        return {}


def _as_list(x):
    if isinstance(x, str):
        try:
            import ast
            x = ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    return x if isinstance(x, list) else []


def _inner(final: str, tags=("diagnosis", "analysis", "rationale")) -> str:
    for t in tags:
        m = re.search(rf"<{t}>(.*?)</{t}>", final, re.S)
        if m:
            return " ".join(m.group(1).split())
    return ""


def _block(final: str, tag: str, limit: int = 2) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", final, re.S)
    lines = [ln.strip() for ln in (m.group(1) if m else "").splitlines() if ln.strip()]
    return "; ".join(lines[:limit])


def _tool_facts(task: str, tool: str, d: dict) -> str:
    """One readable sentence of the numbers a tool result actually contains."""
    if not d:
        return "the tool returned nothing usable"
    if d.get("ok") is False:
        return f"the tool rejected the input ({d.get('error') or ', '.join(d.get('failed_checks') or []) or 'no detail'})"
    bits = []
    if d.get("solver_status") not in (None, "OK"):
        bits.append(f"solver status {d['solver_status']}" + (f" ({d['solver_message']})" if d.get("solver_message") else ""))
    if d.get("failed_checks"):
        bits.append("failed checks " + ", ".join(map(str, d["failed_checks"][:4])))
    flags = d.get("unit_range_flags")
    if flags:
        bits.append("unit/range flags " + ", ".join(map(str, flags[:3])))
    lines = _as_list(d.get("per_line_residuals"))
    if lines:
        rows = []
        for x in lines:
            if isinstance(x, dict):
                rows.append(f"({x.get('line')},{x.get('value_MW')},"
                            f"{x.get('recomputed_MW')},{x.get('residual_W')})")
        if rows:
            bits.append("lines (line,value,recomputed,residual): " + "; ".join(rows[:12]))
    comps = _as_list(d.get("per_component_exergy_destruction_MW"))
    if comps:
        top = sorted((c for c in comps if isinstance(c, dict)), key=lambda c: -c.get("exergy_destruction_MW", 0))[:3]
        bits.append("largest exergy destruction " + ", ".join(f"{c['component']} {c['exergy_destruction_MW']} MW" for c in top))
    elif d.get("total_exergy_destruction_MW") and tool == "inspect_plant":
        bits.append(f"total exergy destruction {d['total_exergy_destruction_MW']} MW"
                    + (f" for {round(d['Q_fuel_W'] / 1e6, 2)} MW of fuel" if d.get("Q_fuel_W") else ""))
    eff = d.get("candidate_effect")
    eff = eff if isinstance(eff, dict) else (_as_list(eff) and {} or {})
    if not eff and isinstance(d.get("candidate_effect"), str):
        try:
            import ast
            eff = ast.literal_eval(d["candidate_effect"])
        except (ValueError, SyntaxError):
            eff = {}
    if isinstance(eff, dict) and eff.get("delta_fuel_W") is not None:
        if eff.get("change"):
            bits.append(f"the change {eff['change']} shifts fuel by {round(eff['delta_fuel_W'] / 1e6, 4)} MW"
                        + ("" if eff.get("change_valid", True) else " but is not valid"))
        else:
            bits.append("with no change proposed the fuel input is unchanged")
    if d.get("valid") is True and not d.get("failed_checks"):
        bits.append("all checks pass")
    elif d.get("valid") is False and not d.get("failed_checks"):
        bits.append("the plant is not valid")
    if d.get("closure_residual_w") not in (None, 0.0) or d.get("ledger_closure_residual_W") not in (None, 0.0):
        bits.append(f"closure residual {d.get('ledger_closure_residual_W', d.get('closure_residual_w'))} W")
    appr = d.get("min_approach_K")
    if isinstance(appr, list) and appr and isinstance(appr[0], dict):
        bits.append("minimum approach " + ", ".join(f"{a['component']} {a['min_approach_K']} K" for a in appr[:3]))
    if tool in ("solve_plant", "inspect_plant") and d.get("Q_fuel_W") and d.get("W_net_W"):
        bits.append(f"fuel {round(d['Q_fuel_W'] / 1e6, 2)} MW, net work {round(d['W_net_W'] / 1e6, 2)} MW")
    if tool == "solve_plant" and d.get("n_streams"):
        bits.append(f"{d['n_streams']} streams solved")
    if d.get("T_max_K"):
        bits.append(f"peak temperature {d['T_max_K']} K")
    return "; ".join(bits) or "no check fails"


def _reasoning(kind: str, task: str, tool: str, last: dict, final: str, wrong_label: str, pick: int) -> str:
    gt = _inner(final)
    facts = _tool_facts(task, tool, last) if last else ""
    if kind == "inspect":
        return _think(_OPENERS[task][pick % len(_OPENERS[task])])
    if kind == "bad_guess":
        return _think(_HYPOTHESIS.get(wrong_label, _HYPOTHESIS["empty"]), f"What I know so far: {facts}." if facts else "")
    if kind == "apply_fix":
        lead = (f"That probe fails: {facts}." if wrong_label else f"The result says: {facts}.") if facts else ""
        edit = _block(final, "edits") or _block(final, "fix")
        return _think(lead, gt, f"So I apply: {edit}, and verify the result." if edit else "So I apply the correction and verify the result.")
    if kind == "rank_and_change":
        change = _block(final, "change", 1)
        return _think(f"The latest result shows: {facts}.", gt, f"I test the proposed change ({change}) against the solver." if change else "")
    if kind == "propose_design":
        return _think(gt, "I write this specification and check it against the verifier." if not wrong_label else "The probe taught me what the checker demands; now the specification from the site analysis.")
    if kind == "confirm":
        return _think(f"The last check reports: {facts}.", "Next I confirm the balances/state of the result before answering.")
    return _think(facts)


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
    last_raw, last_tool = "", ""
    seen_facts: list[str] = []
    pick = int(hashlib.md5(str(row.get("id", "")).encode()).hexdigest(), 16)
    for name, candidate, kind in _plan(task, variant, final, wrong_label, wrong):
        reasoning = _reasoning(kind, task, last_tool, _data(last_raw), final, wrong_label if recover else "", pick)
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
        last_raw, last_tool = raw, name
        seen_facts.append(_tool_facts(task, name, _data(raw)))
    last = last_raw

    if last:
        trivial = {"no check fails", "all checks pass"}
        informative = [f for f in seen_facts if f not in trivial]
        problem = informative[0] if informative else ""
        verified = "; ".join(f for f in dict.fromkeys(seen_facts[-2:]) if f != problem) or "all checks pass"
        final_reasoning = _think(
            (f"What was wrong: {problem}." if problem and len(seen_facts) > 1 else ""),
            f"What the final checks confirm: {verified}.",
            "I report the verified result below.")
    else:
        final_reasoning = ""
    turns.append(_turn(reasoning=final_reasoning, assistant=final))

    messages = tool_protocol.messages_from_turns(turns)
    sft = tool_protocol.native_text(messages)

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

    # Exact gate: the native template injects all 16 tool schemas into the system
    # message, so the old plain prompt+completion count under-measures the training
    # sequence by thousands of tokens. Measure the ACTUAL rendered conversation.
    rt = render_tokenizer()
    if rt is not None and isinstance(prompt_messages, list):
        try:
            tools = tool_protocol.tool_schemas(task)
            prefix = tool_protocol.render_messages(
                rt, prompt_messages, tools, add_generation_prompt=True)
            full = tool_protocol.render_messages(rt, prompt_messages + messages, tools)
            if full.startswith(prefix):
                n_prompt = len(rt.encode(prefix, add_special_tokens=False))
                n_completion = len(rt.encode(full[len(prefix):], add_special_tokens=False))
                tokens = len(rt.encode(full, add_special_tokens=False)) + TOKEN_RESERVE
        except (TypeError, ValueError):
            pass

    out = {
        "sft_completion": sft,
        "messages": messages,
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
