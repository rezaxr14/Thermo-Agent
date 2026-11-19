"""Verifiable reward functions for GRPO (TRL contract: f(prompts, completions, **dataset_columns) -> list[float | None]).
Dataset columns used: plant_broken, plant_original, site (JSON strings; '' when absent), deducible, min_edits."""
from __future__ import annotations
import json, re
from functools import lru_cache
from pydantic import ValidationError
from ..core.schema import Plant, Site
from ..core.solve import solve
from ..verify.verifiers import run_all
from ..verify.grounding import check_numbers
from ..agent.dsl import parse_edits, edits_to_patch, EditError
from ..synth.faults import apply_patch, _dump
from ..eval.repair import _close


_TRUNCATED = "truncated rollout: "

# Both tool-call formats are recognised by the shared parser in `agent/toolcall.py`,
# including Qwen3.5's native <tool_call><function=...><parameter=...> rendering. The permissive
# regex below is only a fallback for *truncated* markup (an incomplete block is not "well formed"
# for the parser, but must still be rejected rather than graded as a final answer).
_NATIVE_TOOL_CALL = re.compile(r"<tool_call\b|<function=|\bto=functions\.", re.IGNORECASE)

try:
    from ..agent.toolcall import is_tool_call as _is_well_formed_tool_call
except ImportError:  # pragma: no cover - keep the fallback if the helper is absent
    _is_well_formed_tool_call = None


def is_tool_call_content(text) -> bool:
    """True for our legacy inline tag, Qwen3.5's native tool call, or truncated tool markup."""
    if not text:
        return False
    value = str(text)
    if _is_well_formed_tool_call is not None and _is_well_formed_tool_call(value):
        return True
    return bool(_NATIVE_TOOL_CALL.search(value))


class TruncatedRollout(ValueError):
    """Raised by :func:`completion_text` with ``strict=True`` when there is no final answer."""


def truncation_reason(c) -> str | None:
    """Why a completion cannot be scored, or ``None`` when its last message is a final answer.

    TRL's multi-turn loop appends tool-result messages to the completion and only appends the
    post-tool assistant message when generation produced one (`grpo_trainer.py:2314-2318`). A
    rollout that ran out of ``max_completion_length`` therefore ends on a tool result, on an
    assistant turn that still carries unanswered ``tool_calls`` (including Qwen3.5's native
    ``<tool_call>...<function=name>...`` rendering in the message content), or on an empty final
    message. None of those is the verifier-scored final answer, so they must never be rewarded.
    """
    if isinstance(c, str):
        return None                      # legacy single-turn text (SFT/eval replay)
    if not isinstance(c, (list, tuple)) or not c:
        return _TRUNCATED + "empty completion"
    last = c[-1]
    if not isinstance(last, dict):
        return _TRUNCATED + "last completion item is not a message"
    if last.get("tool_calls"):
        return _TRUNCATED + "last assistant turn ends with an unanswered tool call"
    if last.get("role") == "tool":
        return _TRUNCATED + "last message is a tool result; no final assistant answer"
    if last.get("role") not in (None, "assistant"):
        return _TRUNCATED + f"last message has role {last.get('role')!r}, not 'assistant'"
    if is_tool_call_content(last.get("content")):
        return _TRUNCATED + "last assistant message is a tool call, not a final answer"
    if not str(last.get("content") or "").strip():
        return _TRUNCATED + "final assistant answer is empty"
    return None


def completion_text(c, *, strict: bool = False) -> str:
    """The text to score: the *final assistant* message, never a partial tool turn.

    Full rollouts (one- or multi-turn) return ``c[-1]['content']`` as before. Truncated rollouts
    have no final answer: by default they return ``""`` (which ``evaluate`` scores as zero) and
    with ``strict=True`` they raise :class:`TruncatedRollout` carrying the clear reason.
    """
    if isinstance(c, str):
        return c
    reason = truncation_reason(c)
    if reason is not None:
        if strict:
            raise TruncatedRollout(reason)
        return ""
    return c[-1]["content"] or ""


def _diagnosis(text: str):
    m = re.search(r"<diagnosis>(.*?)</diagnosis>", text, flags=re.S)
    return m.group(1).strip() if m else None


@lru_cache(maxsize=8192)
def evaluate(text: str, broken_js: str, original_js: str, site_js: str, deducible: bool) -> dict:
    out = {"parsed": False, "applied": False, "solved": False, "physics_ok": False, "site_ok": None, "exact": False,
           "n_edits": 0, "score": 0.0, "grounded": 0.0, "has_diag": False}
    broken = Plant.model_validate_json(broken_js); original = Plant.model_validate_json(original_js)
    site = Site.model_validate_json(site_js) if site_js else None
    diag = _diagnosis(text); out["has_diag"] = diag is not None
    try:
        edits = parse_edits(text); out["parsed"] = True; out["n_edits"] = len(edits)
        patch = edits_to_patch(broken, edits)
        fixed = Plant.model_validate(apply_patch(_dump(broken), patch)); out["applied"] = True
    except (EditError, ValidationError, KeyError, IndexError, ValueError, TypeError):
        fixed = None
    bres = solve(broken)
    if diag is not None:
        out["grounded"] = check_numbers(diag, broken, bres)["fraction"]
    if fixed is None: return out
    res = solve(fixed); out["solved"] = res.ok
    if res.ok:
        rep = run_all(fixed, res, site)
        phys = [c for c in rep.checks if not c.name.startswith("V8")]
        out["physics_ok"] = all(c.ok for c in phys if c.hard)
        out["score"] = sum(c.ok for c in phys) / max(len(phys), 1)
        if site is not None: out["site_ok"] = all(c.ok for c in rep.checks if c.name.startswith("V8"))
    out["exact"] = _close(_dump(fixed), _dump(original))
    return out


def _each(completions, kw, fn):
    n = len(completions)
    site = kw.get("site") or [""] * n
    reasons = [truncation_reason(c) for c in completions]
    out = []
    for i, c in enumerate(completions):
        # A truncated rollout has no verifier-scored final answer: 0 for every reward channel.
        if reasons[i] is not None:
            out.append(0.0)
            continue
        out.append(fn(evaluate(completion_text(c), kw["plant_broken"][i], kw["plant_original"][i],
                               site[i], bool(kw["deducible"][i])), i, kw))
    return out


def _log_rejections(completions, kw) -> None:
    """Log the rejection reason/rate once per batch (called only from `r_format`).

    TRL passes `log_metric(name, value)` and `log_extra(column, values)` to every reward function
    (`grpo_trainer.py:1710-1714`); values from `log_metric` are averaged over the logging step.
    """
    reasons = [truncation_reason(c) or "" for c in completions]
    metric = kw.get("log_metric")
    if callable(metric):
        metric("reward_diagnostics/truncated_rollout_fraction",
               sum(bool(r) for r in reasons) / max(len(reasons), 1))
    extra = kw.get("log_extra")
    if callable(extra):
        extra("reject_reason", reasons)


def r_format(prompts, completions, **kw):
    _log_rejections(completions, kw)
    return _each(completions, kw, lambda e, i, k: 1.0 if (e["applied"] and e["has_diag"]) else (0.3 if e["parsed"] else 0.0))

def r_valid(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: (1.0 if e["physics_ok"] else 0.5 * e["score"]) if e["solved"] else 0.0)

def r_site(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: None if e["site_ok"] is None else float(bool(e["site_ok"] and e["physics_ok"])))

def r_exact(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: float(e["exact"]) if kw["deducible"][i] else None)

def r_minimal(prompts, completions, **kw):
    def f(e, i, k):
        ok = e["physics_ok"] and e["site_ok"] is not False
        return 1.0 / (1 + max(0, e["n_edits"] - int(k["min_edits"][i]))) if ok else 0.0
    return _each(completions, kw, f)

def r_grounded(prompts, completions, **kw):
    return _each(completions, kw, lambda e, i, k: e["grounded"] if e["has_diag"] else 0.0)


REWARD_FUNCS = [r_format, r_valid, r_site, r_exact, r_minimal, r_grounded]
REWARD_WEIGHTS = [0.5, 2.0, 1.0, 1.0, 0.5, 0.5]
