"""Normalise model-evaluation reports into one comparison table for the paper.

Each arm is a `model_eval` report JSON plus the axes we want to compare: model family (ours / frontier /
domain model / non-model baseline), inference text (harness mode), tool access, and zero-shot vs trained. The
manifest is a small JSON file so a new arm never needs a code change.

    python -m thermo_agent.eval.comparison paper/assets/model_results_manifest.json \
        --out paper/assets/model_results.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

KEYS = {
    "repair": ["success", "first_turn_success", "exact_on_deducible", "mean_solves"],
    "design": ["valid", "within_1pct", "mean_gap_pct", "mean_evals"],
    "audit": ["rank", "valid", "gain", "pred"],
    "ledger": ["exact", "closure", "all_correct"],
}


def load_manifest(path, root=None):
    """Accept `{arms: {label: {...}}}` or a bare `{label: {...}}` mapping."""
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    arms = d.get("arms", d)
    out = {}
    for label, spec in arms.items():
        if not isinstance(spec, dict) or "report" not in spec:
            continue
        report = pathlib.Path(spec["report"])
        if root and not report.is_absolute():
            report = pathlib.Path(root) / report
        out[label] = {**spec, "report": str(report)}
    return out


def build(manifest: dict) -> dict:
    """Return one row per (arm, task, split) with the axes and the task's headline metrics."""
    rows, provenance = [], {}
    for label, spec in manifest.items():
        path = pathlib.Path(spec["report"])
        if not path.exists():
            provenance[label] = f"missing report: {path}"
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        meta = report.get("meta", {})
        for task, keys in KEYS.items():
            for split, metrics in (report.get(task) or {}).items():
                if not metrics:
                    continue
                rows.append({
                    "arm": label,
                    "display_name": spec.get("display_name") or meta.get("display_name") or label,
                    "model_family": spec.get("family") or meta.get("model_family") or "unknown",
                    "harness": spec.get("harness") or meta.get("harness", "none"),
                    "tool_access": spec.get("tool_access") or meta.get("tool_access", "none"),
                    "zero_shot": bool(spec.get("zero_shot", meta.get("zero_shot", False))),
                    "model": meta.get("model"), "adapter": meta.get("adapter"),
                    "report": str(path), "task": task, "split": split, "n": metrics.get("n"),
                    "metrics": {k: metrics.get(k) for k in keys},
                    "ci95": {k: metrics.get(k + "_ci95") for k in keys if metrics.get(k + "_ci95")},
                })
        provenance[label] = str(path)
    return {"arms": len(manifest), "rows": rows, "provenance": provenance}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None, help="base dir for relative report paths (default: manifest's dir)")
    a = ap.parse_args()
    manifest = load_manifest(a.manifest, a.root or pathlib.Path(a.manifest).resolve().parent)
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    result = build(manifest)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{result['arms']} arms, {len(result['rows'])} rows -> {out}")


if __name__ == "__main__":
    main()
