"""Turn report files into LaTeX tables and bar charts.
    python -m thermo_agent.eval.paper_tables reports/*.json --out paper/generated
Writes tables.tex (booktabs, one table per task) and one PNG per task and metric (bars with 95 % intervals where the report has them)."""
from __future__ import annotations
import argparse, json, pathlib

KEYS = {"repair": ["success", "first_turn_success", "exact_on_deducible", "mean_solves"], "design": ["valid", "within_1pct", "mean_gap_pct", "mean_evals"],
        "audit": ["rank", "valid", "gain", "pred"], "ledger": ["exact", "closure", "all_correct"]}
INTERVAL = {"success", "valid", "within_1pct", "all_correct"}


def _name(path, rep):
    m = rep["meta"]; return f"{pathlib.Path(path).stem}: {m.get('adapter') or 'base'}, {m['harness']}".replace("_", r"\_")


def _cell(v, k):
    x = v.get(k)
    if x is None: return "--"
    ci = v.get(k + "_ci95")
    return f"{x:.2f} ({ci[0]:.2f}--{ci[1]:.2f})" if ci else f"{x:.2f}" if isinstance(x, float) else str(x)


def latex(reports: dict) -> str:
    out = []
    for task, keys in KEYS.items():
        rows = []
        for path, rep in reports.items():
            for sp, v in (rep.get(task) or {}).items():
                if v: rows.append(f"{_name(path, rep)} & {sp.replace('_', ' ')} & {v['n']} & " + " & ".join(_cell(v, k) for k in keys) + r" \\")
        if not rows: continue
        head = " & ".join(k.replace("_", " ") for k in keys)
        out.append("\\begin{table}[t]\\centering\\small\n\\begin{tabular}{llr" + "r" * len(keys) + "}\\toprule\n"
                   f"run & split & n & {head} \\\\\\midrule\n" + "\n".join(rows) + f"\n\\bottomrule\\end{{tabular}}\n\\caption{{{task} results.}}\\end{{table}}\n")
    return "\n".join(out)


def plots(reports: dict, out_dir: pathlib.Path) -> list[pathlib.Path]:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    for task, keys in KEYS.items():
        for k in keys:
            if k not in INTERVAL and k not in ("rank", "gain", "pred", "exact", "closure"): continue
            series = {}
            for path, rep in reports.items():
                for sp, v in (rep.get(task) or {}).items():
                    if v and v.get(k) is not None: series.setdefault(sp, []).append((_name(path, rep).replace("\\", ""), v[k], v.get(k + "_ci95")))
            if not series: continue
            fig, ax = plt.subplots(figsize=(1.6 * len(series) + 2, 3)); w = 0.8 / max(len(r) for r in series.values())
            for j, (sp, rows) in enumerate(series.items()):
                for i, (label, val, ci) in enumerate(rows):
                    err = [[val - ci[0]], [ci[1] - val]] if ci else None
                    ax.bar(j + i * w, val, w, yerr=err, capsize=2, color=f"C{i}", label=label if j == 0 else None)
            ax.set_xticks([j + 0.4 - w / 2 for j in range(len(series))]); ax.set_xticklabels([s.replace("_", " ") for s in series], rotation=20, ha="right")
            ax.set_ylabel(k.replace("_", " ")); ax.set_title(task); ax.legend(fontsize=6); fig.tight_layout()
            p = out_dir / f"{task}_{k}.png"; fig.savefig(p, dpi=150); plt.close(fig); files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("reports", nargs="+"); ap.add_argument("--out", default="paper/generated")
    a = ap.parse_args(); reports = {p: json.loads(pathlib.Path(p).read_text()) for p in a.reports}
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "tables.tex").write_text(latex(reports)); print(f"wrote {out / 'tables.tex'} and {len(plots(reports, out))} figures")


if __name__ == "__main__":
    main()
