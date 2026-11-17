#!/usr/bin/env python3
"""Draw a paper figure from measured traces, in the project's house SVG style.

    python paper/tools/figures_from_examples.py --examples paper/assets/examples.json \
        [--tool-usage reports/tool_usage.json] [--models paper/assets/model_results.json] \
        --out paper/figures

Four cards: a fault and its diagnosis, a frontier interaction with tool calls and cost, a scored GRPO group, and
the comparison arms. Every string and number is copied from a stage output. The palette, rounded cards, colour bars
and typography match `paper/figures/svg/fig_overview.svg`; SVG and PDF are drawn by the same code, so they cannot
drift.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import textwrap

BLUE = "#4C72B0"; DARK = "#34548A"; TEAL = "#2E7F96"; GREEN = "#55A868"
ORANGE = "#DD8452"; INK = "#333333"; GREY = "#666666"
PALE = {"blue": "#EEF3FA", "teal": "#EAF6FA", "green": "#EAF6EE", "orange": "#FDF1E7"}
ACCENT = {"blue": BLUE, "teal": TEAL, "green": GREEN, "orange": ORANGE}


def _load(path):
    p = pathlib.Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _plain(value):
    """Text may be a string or a chat-style list of dicts; take the assistant content when present."""
    if isinstance(value, str):
        if value.startswith("[{'") or value.startswith('[{"'):
            try:
                import ast
                value = ast.literal_eval(value)
            except Exception:
                return value
        else:
            return value
    if isinstance(value, list):
        return "\n".join(str(m.get("content", "")) for m in value if isinstance(m, dict))
    return str(value or "")


def _wrap(text, width):
    lines = []
    for chunk in _plain(text).replace("<diagnosis>", "").replace("</diagnosis>", "").split("\n"):
        lines += textwrap.wrap(chunk, width) or [""]
    return lines


def draw(examples, tool_usage, models, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    ex = examples.get("examples", [])
    pick = lambda src: next((e for e in ex if e["source"] == src), None)
    dat, harness, grpo = pick("datagen"), pick("harness"), pick("grpo")

    fig = plt.figure(figsize=(10, 8.6), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1000); ax.set_ylim(860, 0); ax.axis("off")

    def text(x, y, s, size=11.5, color=INK, weight="normal", mono=False, va="top"):
        ax.text(x, y, s, fontsize=size, color=color, weight=weight, va=va, ha="left",
                family="DejaVu Sans Mono" if mono else "DejaVu Sans", zorder=6)

    def block(x, y, w, value, size=10.5, color=INK, mono=False, width_chars=None):
        # DejaVu glyphs in this 1000x860 coordinate system are about 0.78 * fontsize pixels wide (sans).
        chars = width_chars or max(12, int((w - 16) / (0.84 * size if mono else 0.78 * size)))
        lines = _wrap(value, chars)
        for i, line in enumerate(lines):
            text(x, y + i * (size + 3.4), line, size, color, mono=mono)

    def card(x, y, w, h, key, title, subtitle, tag):
        c = ACCENT[key]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=12",
                                    facecolor="white", edgecolor=c, linewidth=2, zorder=1))
        ax.add_patch(FancyBboxPatch((x + 1, y + 1), 6, h - 2, boxstyle="round,pad=0,rounding_size=3",
                                    facecolor=c, edgecolor="none", zorder=2))
        text(x + 22, y + 24, title, 15, c, "bold")
        text(x + 22, y + 45, subtitle, 11.5, GREY)
        ax.add_patch(FancyBboxPatch((x + w - 14 - 12 * len(tag), y + 16), 12 * len(tag) + 4, 22,
                                    boxstyle="round,pad=0,rounding_size=5",
                                    facecolor=PALE[key], edgecolor=c, linewidth=1.1, zorder=3))
        ax.text(x + w - 14 - 6 * len(tag) - 2, y + 27, tag, fontsize=9.5, color=c, weight="bold",
                ha="center", va="center", family="DejaVu Sans", zorder=6)

    def subbox(x, y, w, h, key, label, value, mono=False, size=10.2):
        c = ACCENT[key]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                                    facecolor=PALE[key], edgecolor=c, linewidth=1.4, zorder=3))
        text(x + 12, y + 12, label, 11.5, c, "bold")
        block(x + 12, y + 32, w - 24, value, size, INK, mono)

    text(16, 20, "Measured interactions: corpus, frontier tools, GRPO and evaluation", 16, DARK, "bold")
    text(16, 40, "Every panel is generated from a stage output; see assets/examples.json and examples_schema.json", 11, GREY)

    # (a) deterministic fault
    card(16, 58, 470, 360, "blue", "(a) Fault from the corpus", "deterministic ground truth, no model", "DATA")
    if dat:
        f = dat["fields"]; g = dat.get("graph") or {}
        block(38, 136, 426, f"{dat.get('fault', '?')} on {dat.get('topology', '?')}", 13, DARK)
        comps = " | ".join(str(c) for c in g.get("components", [])[:5]) or "plant graph"
        block(38, 160, 426, f"{len(g.get('components', []))} components / {g.get('streams', '?')} streams: {comps}", 10.5, GREY)
        subbox(38, 188, 426, 132, "blue", "Diagnosis (verbatim)", f.get("diagnosis", ""))
        block(38, 330, 426, "Expert edits are applied in the answer; the verifiers judge the result.", 10.5, GREY)
        block(38, 356, 426, "file: " + pathlib.Path(dat.get("file", "")).name, 9.5, GREY, mono=True)
    else:
        text(38, 140, "No ground-truth example available.", 11, GREY)

    # (b) frontier tools
    card(502, 58, 482, 360, "teal", "(b) Frontier + physics tools", "tool calls, reasoning and cost per answer", "TOOLS")
    if harness:
        calls = harness["fields"].get("tool_calls") or []
        usage = harness["fields"].get("usage") or {}
        block(524, 136, 438, f"{len(calls)} calls to check_answer", 13, TEAL)
        block(524, 160, 438,
              f"cost: {usage.get('total_cost_nanodollars', '?')} nanodollars | tokens {usage.get('input_tokens', '?')} in / {usage.get('output_tokens', '?')} out",
              10.5, GREY)
        subbox(524, 188, 438, 100, "teal", "Reasoning trace", harness["fields"].get("reasoning", ""))
        subbox(524, 298, 438, 96, "teal", "Final answer", harness["fields"].get("completion", ""), mono=True, size=9.5)
    else:
        text(524, 140, "Frontier tool traces appear after the Kaggle harness run.", 11, GREY)

    # (c) scored rollout
    card(16, 436, 470, 388, "green", "(c) Scored GRPO rollout", "group-normalized advantage needs reward spread", "GRPO")
    rewards = (grpo or {}).get("fields", {}).get("rewards") or []
    if rewards:
        names = grpo["fields"].get("reward_functions") or []
        spread = grpo["fields"].get("spread")
        block(38, 514, 426, f"reward spread {spread}: " +
              ("signal for learning" if spread else "flat group, no gradient"), 12, GREEN)
        y = 556
        for i, vals in enumerate(rewards):
            if not vals:
                continue
            block(38, y, 140, names[i] if i < len(names) else "reward", 10, INK, mono=True)
            x0 = 196
            for j, v in enumerate(vals):
                w = max(4.0, min(24.0, float(v) * 22))
                ax.add_patch(Rectangle((x0 + j * 30, y + 1), w, 13,
                                       facecolor=[BLUE, TEAL, ORANGE, GREEN][i % 4], edgecolor="none", zorder=4))
                text(x0 + j * 30 + w + 3, y, f"{float(v):.2f}", 9, GREY)
            y += 32
        block(38, 770, 426, "Prompt and completions are stored verbatim in assets/examples.json.", 10, GREY)
    else:
        text(38, 518, "Rollout bars appear after the first GRPO session.", 11, GREY)

    # (d) comparison arms
    card(502, 436, 482, 388, "orange", "(d) Comparison arms", "family / inference text / tool access", "EVAL")
    arms = (models or {}).get("rows") or []
    y = 512
    if arms:
        for r in arms[:9]:
            label = f"{r['display_name']} | {r['model_family']} | {r['harness']} | tools {r['tool_access']}"
            ax.add_patch(FancyBboxPatch((524, y - 5), 438, 25, boxstyle="round,pad=0,rounding_size=5",
                                        facecolor=PALE["orange"], edgecolor="none", zorder=3))
            block(534, y, 420, label, 9.8, INK, mono=True)
            y += 30
    else:
        for line in ("No trained-model reports yet.", "The axes are fixed in", "assets/model_results_manifest.json:",
                     "  model family", "  inference text", "  tool access", "  zero-shot flag"):
            text(524, y, line, 11.5, GREY if line.startswith("  ") else INK, mono=line.startswith("  "))
            y += 27
    block(524, 790, 438, "Tables are generated from model_eval reports, never transcribed.", 10, GREY)

    text(16, 842, "Sources: assets/examples.json, examples_schema.json, reports/tool_usage.json, assets/model_results.json", 9.5, GREY)

    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "svg").mkdir(exist_ok=True); (out / "pdf").mkdir(exist_ok=True)
    paths = []
    for ext in ("svg", "pdf"):
        p = out / ext / f"fig_trace_walkthrough.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor="white")
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
