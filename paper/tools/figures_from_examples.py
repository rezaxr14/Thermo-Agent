#!/usr/bin/env python3
"""Draw a paper figure from the measured example traces (vector only: SVG + PDF).

    python paper/tools/figures_from_examples.py --examples paper/assets/examples.json \
        [--tool-usage reports/tool_usage.json] [--models paper/assets/model_results.json] \
        --out paper/figures

The panels show (a) a real fault, its diagnosis and the applied edits, (b) how many physics tools the frontier model
called and what it reasoned, (c) a scored GRPO rollout, and (d) the comparison arms. Every string and number comes
from a stage output; nothing is drawn by hand.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import textwrap


def _wrap(text, width=54, lines=6):
    out = []
    for chunk in str(text or "").split("\n"):
        out += textwrap.wrap(chunk, width)[:lines]
        if len(out) >= lines:
            break
    return "\n".join(out[:lines]) or "--"


def _load(path):
    p = pathlib.Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def draw(examples, tool_usage, models, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ex = examples.get("examples", [])
    pick = lambda src: next((e for e in ex if e["source"] == src), None)
    dat, harness, grpo = pick("datagen"), pick("harness"), pick("grpo")
    fig = plt.figure(figsize=(11.5, 8.2), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28, left=0.07, right=0.97, top=0.92, bottom=0.07)
    blue, red, green = "#2E5E8C", "#B5383C", "#2E7F5E"

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); ax.set_title("(a) A fault from the deterministic corpus", loc="left", color=blue, weight="bold")
    if dat:
        f = dat["fields"]
        ax.text(0.02, 0.92, f"{dat.get('fault')} on {dat.get('topology')}", transform=ax.transAxes, weight="bold")
        ax.text(0.02, 0.74, "Diagnosis\n" + _wrap(f.get("diagnosis")), transform=ax.transAxes, va="top", fontsize=9, family="monospace")
        ax.text(0.02, 0.28, "Expert edits are applied in the answer; the verifiers decide.", transform=ax.transAxes, fontsize=8, color="#666")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = fig.add_subplot(gs[0, 1]); ax.axis("off"); ax.set_title("(b) Frontier model with our physics tools", loc="left", color=green, weight="bold")
    if harness:
        calls = harness["fields"].get("tool_calls") or []
        ax.text(0.02, 0.92, f"{len(calls)} calls to check_answer; usage recorded per call", transform=ax.transAxes, weight="bold", fontsize=10)
        ax.text(0.02, 0.76, "Reasoning\n" + _wrap(harness["fields"].get("reasoning")), transform=ax.transAxes, va="top", fontsize=9)
        ax.text(0.02, 0.30, "Answer\n" + _wrap(harness["fields"].get("completion")), transform=ax.transAxes, va="top", fontsize=9, family="monospace")
    tu = (tool_usage or {}).get("by_task") or {}
    if tu:
        tasks = sorted(tu)
        calls = [tu[t]["calls"] for t in tasks]
        ax2 = ax.inset_axes([0.08, 0.05, 0.84, 0.18])
        ax2.bar(range(len(tasks)), calls, color=green, alpha=0.75)
        ax2.set_xticks(range(len(tasks)), tasks, fontsize=8); ax2.set_ylabel("calls", fontsize=8); ax2.tick_params(labelsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = fig.add_subplot(gs[1, 0]); ax.set_title("(c) A scored GRPO rollout", loc="left", color=red, weight="bold")
    if grpo:
        rewards = grpo["fields"].get("rewards") or []
        names = grpo["fields"].get("reward_functions") or []
        for i, vals in enumerate(rewards):
            if not vals:
                continue
            ax.plot(range(len(vals)), vals, "o-", label=f"{names[i] if i < len(names) else 'reward'}", alpha=0.8)
        ax.set_xlabel("completion in the group"); ax.set_ylabel("reward"); ax.legend(fontsize=7); ax.grid(alpha=0.25)
        ax.text(0.02, 0.03, f"spread {grpo['fields'].get('spread')}: a group with signal, so the policy can learn",
                transform=ax.transAxes, fontsize=8, color="#555")

    ax = fig.add_subplot(gs[1, 1]); ax.axis("off"); ax.set_title("(d) Comparison arms (from result JSON)", loc="left", color=blue, weight="bold")
    arms = (models or {}).get("rows") or []
    if arms:
        labels = []
        for r in arms[:12]:
            labels.append(f"{r['display_name']} | {r['model_family']} | {r['harness']} | tools {r['tool_access']}")
        ax.text(0.02, 0.94, "\n".join(_wrap(s, 62, 1) for s in labels), transform=ax.transAxes, va="top", fontsize=9, family="monospace")
    else:
        ax.text(0.02, 0.90, "No trained-model reports yet.\n\nThe axes are recorded in\nassets/model_results_manifest.json:\n  model family\n  inference text\n  tool access\n  zero-shot flag", transform=ax.transAxes, va="top", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle("Measured interactions: corpus, frontier tools, GRPO and evaluation", fontsize=14, weight="bold")
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "svg").mkdir(exist_ok=True); (out / "pdf").mkdir(exist_ok=True)
    paths = []
    for ext in ("svg", "pdf"):
        p = out / ext / f"fig_trace_walkthrough.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", default="paper/assets/examples.json")
    ap.add_argument("--tool-usage", default=None)
    ap.add_argument("--models", default=None)
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    paths = draw(_load(a.examples), _load(a.tool_usage) if a.tool_usage else {},
                 _load(a.models) if a.models else {}, a.out)
    print("wrote", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
