"""Validate a candidate answer with the same tool suite that scores the model.

Nothing enters the training set on the strength of looking plausible: a repair candidate is parsed, applied to
the broken plant, re-solved and run through the physics verifiers; a ledger candidate is evaluated by the
ledger program; an audit candidate by the ranking/change/prediction evaluators; a design candidate by building
and checking the plant. Rejections are not thrown away: they become hindsight-memory notes.

    verdict = validate("repair", completion, row)      # {"accepted": bool, "reasons": [...], "reward": float, ...}
    row2 = teacher_row("repair", completion, row, verdict)   # SFT-shaped row, only when accepted
    note = rejection_note("repair", row, verdict)            # memory note, only when rejected
"""
from __future__ import annotations

import hashlib
import re

from ..agent.dsl import EditError, edits_to_patch, parse_edits
from ..audit import rewards as audit_rewards
from ..core.schema import Plant, Site
from ..design import rewards as design_rewards
from ..eval.repair import score_patch
from ..ledger import rewards as ledger_rewards
from ..synth import paraphrase
from ..synth.paraphrase import FLUIDS
from . import memory

TASKS = ("repair", "ledger", "audit", "design")

KEEP = {
    "repair": ("plant_broken", "plant_original", "site", "deducible", "min_edits", "topology", "fault_type"),
    "ledger": ("ledger", "topology", "fault_types", "min_edits"),
    "audit": ("plant", "site", "ed_json", "fuel_w", "net_w", "best_gain_w", "topology"),
    "design": ("site", "ref_fuel_w", "ref_spec"),
}

# Scoring-only columns the local verifier needs in order to return a trustworthy
# verdict.  Hosted prompt shards deliberately strip these answer-bearing fields
# (see scripts/make_hosted_prompts.FORBIDDEN_KEYS), so when any is absent the
# verdict is DEFERRED to ``transfer.ingest``, which joins the full export by id.
# Every local corpus row carries all of them, so real local validation is
# unchanged.  Declared here (next to KEEP) so every caller shares ONE mapping.
REQUIRED_SCORING_FIELDS = {
    "repair": ("plant_broken", "plant_original", "min_edits"),
    "ledger": ("ledger", "fault_types", "min_edits"),
    "audit": ("plant", "ed_json", "fuel_w", "net_w"),
    "design": ("site", "ref_spec"),
}


def missing_scoring_fields(task: str, row: dict) -> list[str]:
    """The scoring-only columns absent from `row` (prompt shards omit them by design)."""
    return [k for k in REQUIRED_SCORING_FIELDS.get(task, ()) if k not in row]


def deferred_verdict(missing: list[str]) -> dict:
    """Distinct, non-raising verdict for a row whose grading belongs to ingest."""
    names = ", ".join(sorted(missing))
    return {"accepted": False, "reward": 0.0,
            "reasons": [f"validation deferred: missing scoring fields {names}"],
            "deferred": True, "failed": [], "exact": False}

CLASS_HINT = {
    "wrong_fluid": "Copying the broken label back, or inventing a third fluid, does not work; the destination component fixes the fluid class.",
    "unit_slip": "Correct the power of ten, not the digits.",
    "sign_flip": "Flip the sign back; the magnitude is already right.",
    "efficiency_gt_1": "Efficiencies and second-law fractions are fractions in (0, 1]; a percentage must be divided by 100.",
    "swapped_ports": "Swap the two stream endpoints back; do not change parameters at the same time.",
}


def _verdict(accepted: bool, reasons: list[str], reward: float = 0.0, failed: list[str] | None = None,
             exact: bool = False) -> dict:
    return {"accepted": bool(accepted), "reasons": [r for r in reasons if r], "reward": float(reward),
            "failed": list(failed or []), "exact": bool(exact), "deferred": False}


def validate_repair(completion: str, row: dict) -> dict:
    plant = Plant.model_validate_json(row["plant_broken"])
    original = Plant.model_validate_json(row["plant_original"])
    site = Site.model_validate_json(row["site"]) if row.get("site") else None
    try:
        edits = parse_edits(completion)
    except EditError as exc:
        return _verdict(False, [f"format: {exc}"])
    if not edits:
        return _verdict(False, ["format: empty <edits> block"])
    try:
        patch = edits_to_patch(plant, edits)
    except EditError as exc:
        return _verdict(False, [f"edit: {exc}"])
    s = score_patch(plant, original, patch, site)
    if not s["applied"]:
        return _verdict(False, [s.get("reason", "patch did not apply")])
    if not s["success"]:
        return _verdict(False, ["verifier: " + (", ".join(s.get("failed", [])) or "invalid plant")],
                        s.get("reward", 0.0), s.get("failed"))
    return _verdict(True, ["restored a valid plant" + (" exactly" if s.get("exact") else "")],
                    s.get("reward", 1.0), [], bool(s.get("exact")))


def validate_ledger(completion: str, row: dict) -> dict:
    e = ledger_rewards.evaluate(completion, row["ledger"])
    if not e["parsed"]:
        return _verdict(False, ["format: no parseable <fix> block"])
    reasons = []
    if not e.get("all_correct"):
        if e["closure"] < 0.999:
            reasons.append("closure: the energy balance does not close")
        if e["nonneg"] < 1.0:
            reasons.append("nonneg: a component has negative exergy destruction")
        if e["exact"] < 1.0:
            reasons.append("exact: the fixed lines do not reproduce the truth")
    if not e["has_diag"]:
        reasons.append("format: no <diagnosis> block")
    accepted = not reasons
    return _verdict(accepted, reasons or ["ledger repaired"], float(e.get("exact", 0.0)))


def validate_audit(completion: str, row: dict) -> dict:
    e = audit_rewards.evaluate(completion, row["plant"], row.get("site", ""), row["ed_json"],
                               float(row["fuel_w"]), float(row["net_w"]))
    if not e["parsed"]:
        return _verdict(False, [f"format: {e['reason'] or 'unparseable audit'}"])
    reasons = []
    if not e["change_ok"]:
        reasons.append(f"change: {e['reason'] or 'change is not valid'}")
    if e["ndcg"] < 0.5:
        reasons.append(f"ranking: nDCG {e['ndcg']:.2f} is below 0.5")
    if e["pred_err"] is None:
        reasons.append("prediction: no usable prediction")
    elif e["pred_err"] > 0.5:
        reasons.append(f"prediction: relative error {e['pred_err']:.2f} is above 0.5")
    if e["grounded"] < 0.8:
        reasons.append(f"grounding: only {e['grounded']:.0%} of the numbers in <analysis> are in the plant")
    return _verdict(not reasons, reasons or ["audit accepted"], 1.0 - e["pred_err"] if e["pred_err"] is not None else 0.0)


def validate_design(completion: str, row: dict) -> dict:
    e = design_rewards.evaluate(completion, row["site"])
    if not e["parsed"]:
        return _verdict(False, [f"format: {e['reason'] or 'unparseable design'}"])
    reasons = []
    if not e["valid"]:
        reasons.append(f"plant: {e['reason'] or 'design does not build or verify'}")
    if e["grounded"] < 0.8:
        reasons.append(f"grounding: only {e['grounded']:.0%} of the numbers in <rationale> are in the plant")
    return _verdict(not reasons, reasons or ["design accepted"], 1.0 if e["valid"] else 0.0)


VALIDATORS = {"repair": validate_repair, "ledger": validate_ledger, "audit": validate_audit, "design": validate_design}


def validate(task: str, completion: str, row: dict) -> dict:
    if task not in VALIDATORS:
        raise KeyError(f"unknown task {task!r}")
    return VALIDATORS[task](completion, row)


MIN_COT_CHARS = 30        # a tool turn needs at least this much reasoning text to count


def _turn_cot(turn: dict) -> str:
    """The reasoning of one assistant turn: the provider's reasoning trace if any, else the visible prose.

    Models that refuse a reasoning parameter (e.g. gpt-6-luna on chat-completions with tools) think out loud in
    the visible message; that prose, across the multi-round conversation, IS their chain of thought.
    """
    r = str(turn.get("reasoning") or "").strip()
    a = re.sub(r"<tool_call.*?</tool_call>", "", str(turn.get("assistant") or ""), flags=re.S).strip()
    return r or a


def reasoning_report(trajectory: dict | None) -> dict:
    """Is this tool-using trajectory a usable chain of thought? (gate used at ingest, 2026-11-20)

    Usable means: the first tool turn carries reasoning, at least 75% of tool turns carry >= MIN_COT_CHARS of
    reasoning, and the reasoning is not one repeated sentence (distinct >= 60% of the reasoned turns).
    """
    turns = (trajectory or {}).get("turns") or []
    tool_turns = [t for t in turns if t.get("tool_calls")]
    texts = [_turn_cot(t) for t in tool_turns]
    reasoned = [x for x in texts if len(x) >= MIN_COT_CHARS]
    distinct = len(set(reasoned))
    first_ok = bool(texts) and len(texts[0]) >= MIN_COT_CHARS
    ok = bool(tool_turns) and first_ok and len(reasoned) >= 0.75 * len(texts) and distinct >= 0.6 * max(len(reasoned), 1)
    return {"tool_turns": len(tool_turns), "reasoned_turns": len(reasoned), "distinct": distinct,
            "cot_chars": sum(len(x) for x in reasoned), "first_turn_reasoned": first_ok, "ok": ok}


def _has_tool_turns(text: str) -> bool:
    """True only when the stored SFT completion actually contains a tool call/result turn."""
    return "<tool_call" in text or "<tool_result" in text


def teacher_row(task: str, completion: str, row: dict, verdict: dict, tag: str = "",
                trajectory: dict | None = None) -> dict:
    """SFT-shaped row from a validated teacher answer; task columns are kept for the GRPO reward functions.

    When `trajectory` is supplied (a self-correcting tool-using CoT), its full reasoning +
    tool calls + numeric results + revisions become the SFT completion, so the student learns
    to use tools/skills/memory and improve its answer, not just to emit a zero-shot answer.
    The final verifier-accepted answer is always the last block of the completion.
    """
    if not verdict["accepted"]:
        raise ValueError("refusing to build a training row from a rejected answer")
    if trajectory is not None and trajectory.get("final"):
        sft_text = trajectory["sft_completion"]
        completion_messages = trajectory.get("messages") or [
            {"role": "assistant", "content": sft_text}]
    else:
        sft_text = completion
        completion_messages = [{"role": "assistant", "content": sft_text}]
    suffix = tag or hashlib.md5(sft_text.encode()).hexdigest()[:8]
    out = {"id": f"{row['id']}-t-{suffix}", "record_id": row["id"], "split": row["split"], "task": task, "source": "frontier",
           "prompt": row["prompt"], "completion": completion_messages,
           "validation": {"reward": verdict["reward"], "reasons": verdict["reasons"]}}
    if trajectory is not None:
        out["trajectory"] = {k: v for k, v in trajectory.items() if k != "sft_completion"}
        # Only a completion that really contains tool turns is a tool-using CoT. A harness
        # record with no tool calls must not be mislabelled just because it carried a trajectory.
        if _has_tool_turns(sft_text):
            out["source"] = "frontier_tools"
    for k in KEEP[task]:
        if k in row:
            out[k] = row[k]
    return out


def rejection_note(task: str, row: dict, verdict: dict, source: str = "validator") -> memory.Note:
    """Turn a rejected attempt into a reusable lesson about a class of mistakes."""
    fault = row.get("fault_type") or (row.get("fault_types") or [""])[0]
    reason = "; ".join(verdict["reasons"]) or "rejected by the verifiers"
    hint = CLASS_HINT.get(fault, "")
    lesson = f"Attempt rejected ({reason})." + (f" {hint}" if hint else "")
    fix = ""
    try:
        text = row["completion"][0]["content"]
        if "<edits>" in text:
            fix = " ".join(text.split("<edits>")[1].split("</edits>")[0].split())
    except (KeyError, IndexError, TypeError):
        pass
    rid = str(row.get("id", ""))
    return memory.Note(
        id=f"{task}-{rid}-{hashlib.md5((reason + source).encode()).hexdigest()[:8]}",
        task=task, fault_type=fault, architecture=str(row.get("topology", "")),
        symptoms=reason, lesson=lesson, fix=fix, source=source, accepted=False, record_id=rid,
    )


# Positive, class-level phrasings of the same mistake classes the failure memory teaches.  They describe the
# pattern that *worked*, never this plant's numbers (those live in `fix`, which is taken from the completion).
_SUCCESS_HINT = {
    "wrong_fluid": "fixed by correcting the destination-component fluid, not the label",
    "unit_slip": "fixed by correcting the power of ten, not the digits",
    "sign_flip": "fixed by flipping the sign back; the magnitude was already right",
    "efficiency_gt_1": "fixed by writing the efficiency as a fraction in (0, 1], not a percentage",
    "swapped_ports": "fixed by swapping the two stream endpoints back, without touching parameters",
}


def _cap(text: str, limit: int = 240) -> str:
    """Keep notes short; class-level lessons must never turn into answer dumps."""
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _completion_text(row: dict) -> str:
    """The candidate's own completion (not the ground-truth plant/spec)."""
    value = row.get("completion")
    if isinstance(value, list) and value:
        value = value[0].get("content") if isinstance(value[0], dict) else None
    return value if isinstance(value, str) else ""


def _block(text: str, tag: str) -> str:
    open_tag, close_tag = f"<{tag}>", f"</{tag}>"
    if open_tag in text and close_tag in text:
        return text.split(open_tag, 1)[1].split(close_tag, 1)[0].strip()
    return ""


def _lines(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def _success_fix(task: str, text: str) -> str:
    """The reusable edit pattern, taken only from the completion the verifier accepted."""
    if task == "repair":
        return "; ".join(_lines(_block(text, "edits")))
    if task == "ledger":
        return "; ".join(_lines(_block(text, "fix")))
    if task == "audit":
        ranked = " ".join(_block(text, "ranking").split())
        change = " ".join(_block(text, "change").split())
        return "; ".join(p for p in (ranked, change) if p)
    if task == "design":
        # Type + key capacities only -- never the whole <design> block as JSON.
        keep = ("prime_mover", "cooling", "rejection", "bottoming",
                "P_orc_bar", "T_hw", "dT_hw", "dT_rej", "orc_frac")
        pairs = []
        for line in _lines(_block(text, "design")):
            if "=" not in line:
                continue
            key, value = (p.strip() for p in line.split("=", 1))
            if key in keep:
                pairs.append(f"{key} = {value}")
        return "; ".join(pairs)
    return ""


def success_note(task: str, row: dict, verdict: dict, source: str = "validator") -> memory.Note:
    """Turn a verifier-accepted attempt into a short lesson about what worked.

    The lesson is class-level and built only from the verifier's verdict (reasons / reward / exact) plus a count
    of the candidate's own edits.  `fix` is the reusable edit pattern lifted from the accepted *completion*,
    never from ``row["plant_original"]`` / ``ref_spec`` / ``ed_json`` -- so no ground-truth answer, stream table,
    model id, timestamp, cost or token figure is ever written to memory.
    """
    fault = row.get("fault_type") or (row.get("fault_types") or [""])[0]
    reasons = "; ".join(verdict.get("reasons", [])) or "verifier accepted"
    text = _completion_text(row)

    if task == "repair":
        n_edits = len(_lines(_block(text, "edits")))
        detail = "plant verified" + (" exactly" if verdict.get("exact") else "")
        if n_edits:
            detail += f", {n_edits} edit{'' if n_edits == 1 else 's'}"
        hint = _SUCCESS_HINT.get(fault, "")
        lesson = f"Attempt accepted ({detail})." + (f" Fault class {fault} is {hint}." if fault and hint else "")
    elif task == "audit":
        detail = "ranking + change valid"
        reward = verdict.get("reward")
        if isinstance(reward, (int, float)):
            err = 1.0 - float(reward)
            if err >= 0:
                detail += f", pred_err {err:.2f}"
        lesson = f"Attempt accepted ({detail})."
    elif task == "ledger":
        lesson = "Attempt accepted (closure residual ~0, ledger repaired)."
    elif task == "design":
        lesson = "Attempt accepted (spec valid and grounded)."
    else:
        lesson = f"Attempt accepted ({reasons})."

    lesson = _cap(lesson)
    fix = _cap(_success_fix(task, text))
    rid = str(row.get("id", ""))
    digest = hashlib.md5((lesson + fix + source).encode()).hexdigest()[:8]
    return memory.Note(
        id=f"{task}-{rid}-ok-{digest}",
        task=task, fault_type=fault, architecture=str(row.get("topology", "")),
        symptoms=reasons, lesson=lesson, fix=fix, source=source, accepted=True, record_id=rid,
    )


def _current(plant: Plant, e: dict):
    if e["kind"] == "fluid":
        return next(s.fluid for s in plant.streams if s.id == e["id"])
    if e["kind"] == "rewire":
        s = next(s for s in plant.streams if s.id == e["id"])
        return s.dst if e["end"] == "to" else s.src
    if e["stream"]:
        s = next(s for s in plant.streams if s.id == e["id"])
        if e["field"] == "m_dot":
            return s.m_dot
        return getattr(s.fixed, e["field"].split(".", 1)[1])
    return next(c for c in plant.components if c.id == e["id"]).params[e["field"]]


def _render(e: dict, value) -> str:
    if e["kind"] == "fluid":
        return f"FLUID @{e['id']} = {value}"
    if e["kind"] == "rewire":
        return f"REWIRE @{e['id']}.{e['end']} = {value}"
    return f"SET {'@' if e['stream'] else ''}{e['id']}.{e['field']} = {value!r}"


def _wrap(lines: list[str]) -> str:
    return "<diagnosis>Candidate repair.</diagnosis>\n<edits>\n" + "\n".join(lines) + "\n</edits>"


def adversarial_candidates(row: dict) -> list[tuple[str, str]]:
    """Plausible wrong answers for a repair row, built without any model.

    They are the mistakes a model actually makes: no edit block, echoing the broken value back, fixing the
    digits but not the scale, swapping in a third fluid, naming a port that does not exist. Each is run through
    the verifiers; the ones that fail become hindsight memory.
    """
    plant = Plant.model_validate_json(row["plant_broken"])
    truth = parse_edits(row["completion"][0]["content"])
    out = [("no_edits", "<diagnosis>Candidate repair.</diagnosis>")]
    echo = []
    for e in truth:
        v = _current(plant, e)
        if v is None:
            echo = []
            break
        echo.append(_render(e, v))
    if echo:
        out.append(("echo_broken", _wrap(echo)))
    scaled = []
    for e in truth:
        v = _current(plant, e)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            scaled.append(_render(e, v * 10.0))
    if scaled:
        out.append(("wrong_scale", _wrap(scaled)))
    fluid = [e for e in truth if e["kind"] == "fluid"]
    if fluid:
        e = fluid[0]
        broken = _current(plant, e)
        # keep the wrong class but change the label: a same-class fluid is still the wrong class for the port
        wrong = next((f for f in FLUIDS if f != e["value"] and f != broken
                      and (f in paraphrase.GAS) == (broken in paraphrase.GAS)), None)
        if wrong:
            out.append(("third_fluid", _wrap([_render(e, wrong)])))
    rewire = [e for e in truth if e["kind"] == "rewire"]
    if rewire:
        out.append(("bad_port", _wrap([f"REWIRE @{rewire[0]['id']}.{rewire[0]['end']} = nope.port"])))
    return out


def mine_failures(rows: list[dict], task: str = "repair", limit: int | None = None,
                  source: str = "adversarial") -> list[memory.Note]:
    """Run the adversarial candidates through the verifiers and keep the rejections as memory notes."""
    notes = []
    for row in rows[:limit] if limit else rows:
        if task != "repair":
            continue
        for name, text in adversarial_candidates(row):
            v = validate(task, text, row)
            if not v["accepted"]:
                n = rejection_note(task, row, v, source=f"{source}:{name}")
                notes.append(n)
    return memory.dedupe_lessons(notes)


def main():
    """Mine hindsight memory from the verifier's rejections of programmatic adversarial attempts."""
    import argparse
    import json
    import pathlib

    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", required=True, help="processed data dir with {task}_{split}.jsonl")
    ap.add_argument("--out", required=True, help="memory dir to write adversarial_{task}.jsonl into")
    ap.add_argument("--task", default="repair", choices=list(TASKS))
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()
    src = pathlib.Path(a.mine) / f"{a.task}_{a.split}.jsonl"
    rows = [json.loads(l) for l in src.open(encoding="utf-8")][: a.limit]
    notes = mine_failures(rows, a.task, limit=None, source="adversarial")
    path = pathlib.Path(a.out) / f"adversarial_{a.task}.jsonl"
    memory.save(notes, path)
    print(json.dumps({"rows": len(rows), "notes": len(notes), "path": str(path)}, indent=1))


if __name__ == "__main__":
    main()
