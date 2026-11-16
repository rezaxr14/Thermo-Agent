"""Will GRPO get a learning signal? Simulates policies of several qualities (expert answers damaged with probability `level`),
draws groups of G completions per prompt, and reports what the training reward would look like. CPU only, no model.
    python -m thermo_agent.train.reward_diagnostics DATA_DIR [DATA_DIR ...] [--tasks repair design audit ledger] [--groups 12] [--levels 0 0.3 0.6 1] --out reports/reward_diagnostics.json
A group whose rewards are all equal gives zero advantage (TRL normalises by the group's standard deviation), so it teaches nothing."""
from __future__ import annotations
import argparse, importlib, json, pathlib, random, statistics as st, time
from .grpo import TASKS
from .perturb import simulate_policy
from ..eval.rows import load_split


def total_reward(per_func: list[list], weights: list[float], i: int) -> float:
    vals = [(w, f[i]) for w, f in zip(weights, per_func) if f[i] is not None]
    return sum(w * v for w, v in vals)


def diagnose(rows: list[dict], task: str, levels, groups: int = 12, G: int = 4, seed: int = 0) -> dict:
    funcs, weights, cols = TASKS[task]; cols = [c for c in cols if c != "prompt"]; rng = random.Random(seed)
    chosen = rng.sample(rows, min(groups, len(rows))); out = {}
    for level in levels:
        means, stds, zero, ms, chan = [], [], 0, [], {f.__name__: [] for f in funcs}
        for r in chosen:
            expert = r["completion"][0]["content"]; comps = [simulate_policy(expert, rng, level) for _ in range(G)]
            kw = {c: [r[c]] * G for c in cols}; t = time.perf_counter()
            per = [f(None, comps, **kw) for f in funcs]; ms.append(1000 * (time.perf_counter() - t) / G)
            tot = [total_reward(per, weights, i) for i in range(G)]
            means.append(st.mean(tot)); sd = st.pstdev(tot); stds.append(sd); zero += sd < 1e-9
            for f, vals in zip(funcs, per):
                v = [x for x in vals if x is not None]
                if v: chan[f.__name__].append(st.mean(v))
        out[str(level)] = {"mean_reward": round(st.mean(means), 3), "mean_within_group_std": round(st.mean(stds), 3), "zero_variance_groups": round(zero / len(chosen), 3),
                           "zero_variance_expected_if_all_correct_or_all_wrong_alike": round((1 - level) ** G, 3),
                           "ms_per_completion": round(st.mean(ms), 1), "channel_means": {k: round(st.mean(v), 3) for k, v in chan.items() if v}}
    mod = importlib.import_module(funcs[0].__module__); fresh = []          # cost of an evaluation that is not a cache hit
    for r in chosen[:5]:
        mod.evaluate.cache_clear(); kw = {c: [r[c]] for c in cols}; comps = [r["completion"][0]["content"]]; t = time.perf_counter()
        for f in funcs: f(None, comps, **kw)
        fresh.append(1000 * (time.perf_counter() - t))
    return {"groups": len(chosen), "G": G, "by_policy_error_rate": out, "max_total_reward": round(sum(weights), 2), "fresh_evaluation_ms": round(st.mean(fresh), 1)}


def advise(task: str, d: dict) -> list[str]:
    notes = []; lv = d["by_policy_error_rate"]
    mid = lv.get("0.3") or next(iter(lv.values()))
    if mid["zero_variance_groups"] > 0.5: notes.append(f"{task}: at 30 percent error rate {mid['zero_variance_groups']:.0%} of groups have identical rewards; GRPO will learn little unless the SFT policy is worse than that (check zero-variance share in the training log).")
    if mid["mean_within_group_std"] < 0.05: notes.append(f"{task}: reward spread within groups is very small at 30 percent errors.")
    if d["fresh_evaluation_ms"] > 400: notes.append(f"{task}: a fresh reward evaluation costs {d['fresh_evaluation_ms']:.0f} ms; with 16 completions per step that is up to {16 * d['fresh_evaluation_ms'] / 1000:.1f} s of CPU per step.")
    return notes


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data", nargs="+", help="ground-truth dirs, e.g. data/processed data/processed_arch")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--groups", type=int, default=12); ap.add_argument("--levels", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.3, 0.6, 1.0]); ap.add_argument("--out")
    ap.add_argument("--generations", type=int, default=4, help="group size G, must match GrpoSettings.num_generations")
    a = ap.parse_args(); res = {}; notes = []
    for t in a.tasks:
        rows = load_split(a.data, t, "train")
        if not rows: continue
        res[t] = diagnose(rows, t, a.levels, a.groups, a.generations); notes += advise(t, res[t])
    res["notes"] = notes
    if a.out: pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
