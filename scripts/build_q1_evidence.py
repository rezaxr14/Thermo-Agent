"""Generate Q1 evidence assets that do not require trained weights.

Outputs (all derived from committed report JSON, never transcribed):
  reports/significance.json          paired McNemar / bootstrap / sign-flip tests with Holm
  paper/assets/cost_quality_frontier.json  recorded provider cost vs verifier success per arm
  reports/gpu_hours.json             GPU-hour accounting from recorded run metadata
"""
from __future__ import annotations

import json
from pathlib import Path

from thermo_agent.eval.compare import compare

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
ASSETS = ROOT / "paper" / "assets"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def significance():
    pairs = {
        "tools_vs_zero": ("eval_frontier_tools.json", "eval_frontier_zero.json"),
        "frontier_vs_other": ("eval_frontier_zero.json", "eval_other_model.json"),
    }
    out = {}
    for name, (a, b) in pairs.items():
        ra, rb = load(REPORTS / a), load(REPORTS / b)
        rows = compare(ra, rb)
        out[name] = {
            "a": a, "b": b, "n_comparisons": len(rows),
            "significant_holm_005": [
                {k: r[k] for k in ("task", "split", "metric", "mean_a", "mean_b", "diff", "p_holm")}
                for r in rows if r["p_holm"] < 0.05
            ],
            "rows": rows,
        }
    (REPORTS / "significance.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote significance.json: " + ", ".join(f"{k}={v['n_comparisons']} rows" for k, v in out.items()))
    return out


def cost_quality():
    arms = {
        "frontier_zero": ("eval_frontier_zero.json", "data/teacher_runs/eval_all.jsonl"),
        "frontier_tools": ("eval_frontier_tools.json", "data/teacher_runs/harness_eval_completions2.jsonl"),
        "other_model": ("eval_other_model.json", "data/teacher_runs/eval_gptnano_replay.jsonl"),
    }
    # metric used as the "quality" coordinate for each task
    quality_metric = {
        "repair": "success", "design": "valid", "audit": "format", "ledger": "all_correct",
    }
    out = {}
    for arm, (report_f, replay_f) in arms.items():
        rep = load(REPORTS / report_f)
        usage = {}
        for line in (ROOT / replay_f).open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            u = r.get("usage") or {}
            usage[r["id"]] = int(u.get("total_cost_nanodollars") or 0)
        per_task = {}
        for task, metric in quality_metric.items():
            ids, vals = [], []
            for split, v in (rep.get(task) or {}).items():
                for ex in v.get("examples", []):
                    if ex.get(metric) is not None and ex["id"] in usage:
                        ids.append(ex["id"])
                        vals.append(float(ex[metric]))
            if not vals:
                continue
            cost = sum(usage[i] for i in ids)
            per_task[task] = {
                "n": len(vals), "quality_metric": metric,
                "quality": round(sum(vals) / len(vals), 4),
                "cost_usd": round(cost / 1e9, 6),
                "usd_per_solved_row": round(cost / 1e9 / max(sum(vals), 1e-9), 4),
            }
        out[arm] = {
            "model_family": rep["meta"]["model_family"],
            "tool_access": rep["meta"]["tool_access"],
            "per_task": per_task,
        }
    payload = {
        "note": "Quality is the verifier success rate on the same 30-per-split reported rows; "
                "cost is the provider-reported nanodollar total for exactly those replayed answers.",
        "arms": out,
    }
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "cost_quality_frontier.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print("wrote cost_quality_frontier.json")
    return payload


def gpu_hours():
    # Kaggle charges by the second of accelerator time. Recorded run wall-times are kept in
    # the teacher usage and eval meta; training GPU-hours are filled from kernel logs when the
    # full run exists. Until then the accounting file records what is measured and what is not.
    tool = load(REPORTS / "tool_usage.json")
    evidence = load(ROOT / "data" / "provenance" / "stats.json")
    out = {
        "measured": {
            "frontier_tool_eval_seconds": round(tool["usage"]["latency_ms"] / 1000, 1),
            "frontier_tool_eval_usd": tool["usage"]["cost_usd"],
            "api_spend_total_usd": round(evidence["cost"]["total_nanodollars"] / 1e9, 6),
        },
        "pending": [
            "full SFT GPU-hours (dual T4) — fill from thermo-train-run kernel log",
            "GRPO GPU-hours per task — fill from grpo_* kernel logs",
            "ours_* eval GPU-hours — fill from thermo-eval-run logs",
        ],
    }
    (REPORTS / "gpu_hours.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("wrote gpu_hours.json")
    return out


if __name__ == "__main__":
    significance()
    cost_quality()
    gpu_hours()
