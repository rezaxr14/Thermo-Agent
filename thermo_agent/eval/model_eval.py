"""Evaluate a model (base or LoRA adapter) on Repair, Design and Audit, with an optional agent-harness mode.

    python -m thermo_agent.eval.model_eval DATA_DIR --model Qwen/Qwen3.5-4B [--adapter runs/grpo_repair] \\
        [--tasks repair design audit] [--limit 40] [--harness none|agents|skills|agents+skills] --out reports/base.json

Nothing here depends on torch except HFGenerator; every evaluator takes any callable generate(messages) -> str."""
from __future__ import annotations
import argparse, json, pathlib, platform, re, statistics as st, time
from typing import Callable
from ..agent import harness as H
from ..agent.loop import repair_loop
from ..core.schema import Plant
from ..design.loop import design_loop, design_refine
from ..audit import rewards as audit_rewards
from ..ledger import rewards as ledger_rewards

DEFAULT_SPLITS = {"repair": ["test_id", "test_ood_fault", "test_ood_topology"],
                  "design": ["test_id", "test_ood_climate", "test_ood_scale", "test_ood_load"],
                  "audit": ["test_id", "test_ood_topology"], "ledger": ["test_id", "test_ood_fault", "test_ood_topology"]}
_THINK = re.compile(r"<think>.*?</think>", re.S)


class HFGenerator:
    """Greedy chat generation with transformers (+ peft adapter). Untested in the authoring environment (no GPU)."""
    def __init__(self, model_id, adapter=None, dtype="bfloat16", load_4bit=False, max_new_tokens=400):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(model_id); self.max_new_tokens = max_new_tokens
        kw = {"device_map": "auto"}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, dtype))
        try: self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype), **kw)
        except TypeError: self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=getattr(torch, dtype), **kw)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def __call__(self, messages: list[dict]) -> str:
        import torch
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        inputs = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return _THINK.sub("", self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)).strip()


def wilson(k: int, n: int, z: float = 1.96):
    """95 % Wilson score interval for a proportion k/n."""
    if n == 0: return None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 4) if xs else None


def _wrap(generate, mode, task, plant, log, rid):
    def g(messages):
        text = generate(H.augment(messages, mode, task, plant)); log.append({"id": rid, "task": task, "completion": text}); return text
    return g


def eval_repair(generate, rows, mode="none", max_turns=3, log=None):
    log = [] if log is None else log; res = []
    for r in rows:
        g = _wrap(generate, mode, "repair", Plant.model_validate_json(r["plant_broken"]), log, r["id"])
        res.append({**repair_loop(g, r, max_turns), "deducible": r["deducible"], "fault": r["fault_type"], "id": r["id"]})
    ded = [x for x in res if x["deducible"]]
    return {"n": len(res), "success": _mean(x["success"] for x in res), "success_ci95": wilson(sum(x["success"] for x in res), len(res)),
            "first_turn_success": _mean(x["success"] and x["turns"] == 1 for x in res),
            "exact_on_deducible": _mean(x["exact"] for x in ded), "mean_turns": _mean(x["turns"] for x in res), "mean_solves": _mean(x["solves"] for x in res),
            "by_fault": {f: _mean(x["success"] for x in res if x["fault"] == f) for f in sorted({x["fault"] for x in res})},
            "examples": [{"id": x["id"], "success": bool(x["success"]), "first_turn_success": bool(x["success"] and x["turns"] == 1),
                          "exact": bool(x["exact"]) if x["deducible"] else None, "solves": x["solves"]} for x in res]}


def eval_design(generate, rows, mode="none", max_turns=3, polish_evals=0, log=None, refine_turns=0):
    from ..core.schema import Site
    log = [] if log is None else log; res = []
    for r in rows:
        g = _wrap(generate, mode, "design", None, log, r["id"])
        res.append({**(design_refine(g, r, refine_turns) if refine_turns else design_loop(g, r, max_turns, polish_evals)), "id": r["id"]})
    ok = [x["gap"] for x in res if x["valid"]]
    w1 = sum(bool(x["valid"] and x["gap"] <= 0.01) for x in res)
    return {"n": len(res), "valid": _mean(x["valid"] for x in res), "valid_ci95": wilson(sum(x["valid"] for x in res), len(res)), "mean_gap_pct": _mean(100 * g for g in ok),
            "within_1pct": _mean((x["valid"] and x["gap"] <= 0.01) for x in res), "within_1pct_ci95": wilson(w1, len(res)), "mean_evals": _mean(x["evals"] for x in res),
            "polish_evals": polish_evals, "refine_turns": refine_turns,
            "examples": [{"id": x["id"], "valid": bool(x["valid"]), "gap": x["gap"], "within_1pct": bool(x["valid"] and x["gap"] <= 0.01), "evals": x["evals"],
                          "anytime": x.get("anytime")} for x in res]}


def eval_audit(generate, rows, mode="none", log=None):
    log = [] if log is None else log; comps = []
    for r in rows:
        g = _wrap(generate, mode, "audit", Plant.model_validate_json(r["plant"]), log, r["id"])
        comps.append(g(r["prompt"]))
    kw = {k: [r[k] for r in rows] for k in ("plant", "site", "ed_json", "fuel_w", "net_w", "best_gain_w")}
    out = {"n": len(rows)}; per = {}
    for f in audit_rewards.REWARD_FUNCS: per[f.__name__[2:]] = f(None, comps, **kw); out[f.__name__[2:]] = _mean(per[f.__name__[2:]])
    out["examples"] = [{"id": r["id"], **{k: v[i] for k, v in per.items()}} for i, r in enumerate(rows)]
    return out


def eval_ledger(generate, rows, mode="none", log=None):
    log = [] if log is None else log; comps = []
    for r in rows:
        g = _wrap(generate, mode, "ledger", None, log, r["id"]); comps.append(g(r["prompt"]))
    kw = {"ledger": [r["ledger"] for r in rows]}; out = {"n": len(rows)}
    per = {}
    for f in ledger_rewards.REWARD_FUNCS: per[f.__name__[2:]] = f(None, comps, **kw); out[f.__name__[2:]] = _mean(per[f.__name__[2:]])
    flags = [bool(ledger_rewards.evaluate(c, r["ledger"]).get("all_correct")) for c, r in zip(comps, rows)]; ok = sum(flags)
    out["all_correct"] = round(ok / max(len(rows), 1), 4); out["all_correct_ci95"] = wilson(ok, len(rows))
    out["examples"] = [{"id": r["id"], "all_correct": flags[i], **{k: v[i] for k, v in per.items()}} for i, r in enumerate(rows)]
    return out


def load_rows(data_dir, task, split, limit):
    p = pathlib.Path(data_dir) / f"{task}_{split}.jsonl"
    return [json.loads(l) for l in open(p)][:limit] if p.exists() else None


def run(generate, data_dir, tasks, splits=None, limit=40, mode="none", max_turns=3, polish_evals=0, refine_turns=0) -> tuple[dict, list]:
    report, log = {}, []
    for task in tasks:
        report[task] = {}
        for sp in (splits or DEFAULT_SPLITS[task]):
            rows = load_rows(data_dir, task, sp, limit)
            if not rows: report[task][sp] = None; continue
            t = time.time()
            if task == "repair": m = eval_repair(generate, rows, mode, max_turns, log)
            elif task == "design": m = eval_design(generate, rows, mode, max_turns, polish_evals, log, refine_turns)
            elif task == "ledger": m = eval_ledger(generate, rows, mode, log)
            else: m = eval_audit(generate, rows, mode, log)
            m["seconds"] = round(time.time() - t, 1); report[task][sp] = m
    return report, log


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("--model", default="Qwen/Qwen3.5-4B"); ap.add_argument("--adapter")
    ap.add_argument("--tasks", nargs="+", default=["repair", "design", "audit", "ledger"]); ap.add_argument("--splits", nargs="*")
    ap.add_argument("--limit", type=int, default=40); ap.add_argument("--harness", default="none", choices=list(H.MODES))
    ap.add_argument("--max-turns", type=int, default=3); ap.add_argument("--polish-evals", type=int, default=0)
    ap.add_argument("--refine-turns", type=int, default=0, help="Design: anytime protocol, the model sees the fuel of each valid design and may improve it (records an anytime curve)")
    ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--load-4bit", action="store_true"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gen = HFGenerator(a.model, a.adapter, a.dtype, a.load_4bit)
    report, log = run(gen, a.data, a.tasks, a.splits, a.limit, a.harness, a.max_turns, a.polish_evals, a.refine_turns)
    import importlib.metadata as md
    meta = {"model": a.model, "adapter": a.adapter, "harness": a.harness, "limit": a.limit, "max_turns": a.max_turns, "polish_evals": a.polish_evals, "refine_turns": a.refine_turns,
            "python": platform.python_version(), **{p: (md.version(p) if _has(p) else None) for p in ("torch", "transformers", "peft", "trl", "CoolProp")}}
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, **report}, indent=1)); out.with_suffix(".completions.jsonl").write_text("\n".join(json.dumps(x) for x in log))
    print(json.dumps({"meta": meta, **report}, indent=1))


def _has(p):
    import importlib.metadata as md
    try: md.version(p); return True
    except md.PackageNotFoundError: return False


if __name__ == "__main__":
    main()
