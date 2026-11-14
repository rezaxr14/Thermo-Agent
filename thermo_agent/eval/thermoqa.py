"""ThermoQA (Duzkar 2026, huggingface.co/datasets/olivenet/thermoqa): accuracy of a model before and after training, as a forgetting check.
The dataset schema has not been inspected here: run `describe` first, then pass explicit column names.
    python -m thermo_agent.eval.thermoqa describe
    python -m thermo_agent.eval.thermoqa run --question-col Q --answer-col A --model Qwen/Qwen3.5-4B [--adapter ...] --limit 100 --out r.json
Answers are compared numerically (relative tolerance) when both sides parse as numbers, otherwise as case-insensitive strings."""
from __future__ import annotations
import argparse, json, re

_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
SYSTEM = "You are a thermodynamics expert. Answer with the final numerical value or short answer on the last line."


def last_number(text: str):
    m = _NUM.findall(text.replace(",", ""))
    return float(m[-1]) if m else None


def correct(response: str, answer, rel_tol: float = 0.02) -> bool:
    a = last_number(str(answer)) if _NUM.search(str(answer).replace(",", "")) else None
    r = last_number(response)
    if a is not None and r is not None: return abs(r - a) <= rel_tol * max(abs(a), 1e-9)
    return str(answer).strip().lower() in response.strip().lower()


def evaluate(generate, rows, question_col: str, answer_col: str, rel_tol: float = 0.02) -> dict:
    ok = 0; details = []
    for r in rows:
        resp = generate([{"role": "system", "content": SYSTEM}, {"role": "user", "content": str(r[question_col])}])
        c = correct(resp, r[answer_col], rel_tol); ok += c; details.append({"correct": c, "response": resp[-200:], "answer": str(r[answer_col])})
    return {"n": len(rows), "accuracy": round(ok / max(len(rows), 1), 4), "details": details}


def _load(split=None):
    from datasets import load_dataset
    ds = load_dataset("olivenet/thermoqa")
    return ds[split or list(ds.keys())[0]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["describe", "run"]); ap.add_argument("--split")
    ap.add_argument("--question-col"); ap.add_argument("--answer-col"); ap.add_argument("--model", default="Qwen/Qwen3.5-4B"); ap.add_argument("--adapter")
    ap.add_argument("--limit", type=int, default=100); ap.add_argument("--rel-tol", type=float, default=0.02); ap.add_argument("--out")
    a = ap.parse_args(); ds = _load(a.split)
    if a.cmd == "describe":
        print(ds); print(json.dumps(ds[0], indent=1, default=str)[:1500]); return
    from .model_eval import HFGenerator
    res = evaluate(HFGenerator(a.model, a.adapter), [ds[i] for i in range(min(a.limit, len(ds)))], a.question_col, a.answer_col, a.rel_tol)
    res["meta"] = {"model": a.model, "adapter": a.adapter}
    if a.out: open(a.out, "w").write(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "details"}))


if __name__ == "__main__":
    main()
