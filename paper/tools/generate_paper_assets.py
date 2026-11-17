#!/usr/bin/env python3
"""Regenerate every number, table fragment and value macro used in the paper.

    python tools/generate_paper_assets.py --project PATH/TO/thermo-agent --scratch DIR [--stage STAGE ...] [--force]

Stages (default: all). Each study writes assets/<name>.json; `derive` turns the JSON files into assets_summary.json,
generated/values.tex (one \\val{key} macro per number) and generated/table_*.tex.

    environment   component registry, constants, library versions, test count (from --tests-log)
    crosscheck    generic engine against the independent reference plant
    example       one solved plant with absorption and with electric cooling
    mechanisms    feasibility mechanisms of the Design catalog (small sweeps)
    repair        non-model Repair baseline on the ground-truth datasets (seed 0)
    audit         non-model Audit baselines on the ground-truth datasets (seed 0)
    splits        row counts per split of the ground-truth datasets (baselines, split figure)
    derive        assets_summary.json, values.tex, tables

Files in assets/ that carry the key `kind` and `command` and no `study` entry were produced by the figure build
(figure_manifest.json lists them); derive reads them as they are.
"""
from __future__ import annotations
import argparse, collections, inspect, json, math, pathlib, re, statistics as st, subprocess, sys, tempfile, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS, GEN = ROOT / "assets", ROOT / "generated"
FIG_ASSETS = ["ledger_symptoms", "audit_first_vs_second_law", "design_baselines", "design_feasibility", "part_load",
              "reward_signal", "exergy_flow", "splits", "harness", "plant_graph", "excerpts"]
SEED = 0


def jdump(name, obj):
    ASSETS.mkdir(exist_ok=True)
    (ASSETS / name).write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n")


def jload(name):
    return json.loads((ASSETS / name).read_text())


def have(name):
    return (ASSETS / name).exists()


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# --------------------------------------------------------------------------------------------- studies
def study_environment(project, tests_log):
    import numpy, scipy, pydantic, pandas, CoolProp
    from thermo_agent.core import components as C
    from thermo_agent.synth.topologies import dc_chp, ORC_T_MAX
    from thermo_agent.synth import faults
    from thermo_agent.train import config as TC
    from thermo_agent.design import spec as DS
    from thermo_agent.design.search import enumerate_designs  # noqa: F401  (import check)
    comps = []
    for t, ports in C.REQUIRED_PORTS.items():
        comps.append({"type": t, "ports": [{"name": p, "class": C.PORT_CLASS.get((t, p), "?")} for p in ports],
                      "params": {k: {"doc": v[0], "default": v[1]} for k, v in C.PARAMS.get(t, {}).items()}})
    sig = inspect.signature(dc_chp)
    defaults = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
    cfg = {"sft": {k: getattr(TC.SftSettings(), k) for k in ("lr", "epochs", "max_length", "per_device_batch", "grad_accum", "max_grad_norm")},
           "sft_lora": vars(TC.SftSettings().lora), "grpo": {k: getattr(TC.GrpoSettings(), k) for k in
                                                              ("lr", "num_generations", "max_completion_length", "temperature", "per_device_batch", "grad_accum", "max_steps", "loss_type", "beta", "mask_truncated", "max_grad_norm")},
           "grpo_lora": vars(TC.GrpoSettings().lora), "base_model": TC.BASE_MODEL}
    tests = None
    if tests_log and pathlib.Path(tests_log).exists():
        m = re.findall(r"(\d+) passed(?:, (\d+) (?:warnings?|deselected|skipped))?.* in ([\d.]+)s", pathlib.Path(tests_log).read_text())
        if m:
            tests = {"passed": int(m[-1][0]), "seconds": float(m[-1][2]), "command": "python -m pytest -q"}
    n_catalog = 1
    for k, v in DS.DISCRETE.items():
        if k != "heat_source":
            n_catalog *= len(v)
    jdump("environment.json", {
        "study": "environment", "command": "thermo_agent.core.components (REQUIRED_PORTS, PORT_CLASS, PARAMS); inspect.signature(synth.topologies.dc_chp); train.config; design.spec", "seed": None,
        "n": len(comps), "components": comps, "dc_chp_defaults": defaults, "orc_t_max_K": ORC_T_MAX, "training_config": cfg,
        "fault_injectors": sorted(faults.INJECTORS), "discrete_catalog": {k: list(v) for k, v in DS.DISCRETE.items()},
        "continuous_parameters": {k: list(v) for k, v in DS.CONTINUOUS.items()}, "n_catalog_specs": n_catalog,
        "versions": {"CoolProp": CoolProp.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__, "pydantic": pydantic.__version__,
                     "pandas": pandas.__version__, "python": sys.version.split()[0]},
        "tests": tests})


def study_crosscheck():
    from thermo_agent.core.cgam_plant import solve as ref_solve
    from thermo_agent.core.solve import solve
    from thermo_agent.synth.topologies import cgam_orc
    R = ref_solve(); T11 = R["T"][11]
    P = cgam_orc(n_air=R["n_a"] * 1e3, T_cond=T11, T_cw_in=T11, dT_cw=10.0); RES = solve(P)
    rel = lambda a, b: (a - b) / abs(b)
    b = {x.component: x for x in RES.balances}
    ed = {c: rel(b[c].exergy_destruction_w / 1e6, R["ED"][c]) for c in ("AC", "AP", "GT", "HRSG", "EVA", "CC")}
    jdump("crosscheck.json", {
        "study": "crosscheck", "command": "synth.topologies.cgam_orc(n_air=R['n_a']*1e3, T_cond=T11, T_cw_in=T11, dT_cw=10) solved by core.solve.solve; reference core.cgam_plant.solve()", "seed": None, "n": 1,
        "solved": bool(RES.ok), "w_net_rel_err": rel(RES.kpis["W_net_w"] / 1e6, R["W_net"] + R["W_orc_net"]),
        "eta_I_rel_err": rel(RES.kpis["eta_I"], R["eta_I"]), "eta_II_rel_diff": rel(RES.kpis["eta_II"], R["eta_II"]),
        "hrsg_pinch_diff_K": b["HRSG"].min_approach_K - R["dT_pinch_hrsg"], "closure_residual_w": RES.kpis["closure_residual_w"],
        "ed_rel_diff": ed, "max_mass_residual": max(abs(x.mass_residual) for x in RES.balances),
        "max_energy_residual_w": max(abs(x.energy_residual) for x in RES.balances)})


def study_example():
    from thermo_agent.synth.topologies import dc_chp, size_generation
    from thermo_agent.core.solve import solve
    out = {}
    for cool in ("absorption", "electric"):
        p = size_generation(dc_chp, P_it=20e6, chilled_fraction=0.5, cooling=cool, prime_mover="turbine", rejection="ext")
        r = solve(p)
        out[cool] = {"ok": bool(r.ok), "kpis": {k: v for k, v in r.kpis.items() if isinstance(v, (int, float))},
                     "ed_mw": {b.component: b.exergy_destruction_w / 1e6 for b in r.balances}}
    jdump("example_plant.json", {"study": "example", "command": "size_generation(dc_chp, P_it=20e6, chilled_fraction=0.5, cooling=C, prime_mover='turbine', rejection='ext'); core.solve.solve",
                                 "seed": None, "n": 2, "plants": out})


def _site(f, T=298.15, mw=20.0, **kw):
    from thermo_agent.core.schema import Site, Ambient
    return Site(it_load_w=mw * 1e6, cooling_load_w=f * mw * 1e6, grid_import_allowed=False, ambient=Ambient(T0=T, RH=0.5),
                t_supply_k=291.15, t_return_k=301.15, **kw)


def _stack_T(e):
    """Temperature of the stream that leaves the plant carrying flue gas."""
    if e.result is None or e.plant is None:
        return None
    d = e.plant.model_dump(by_alias=True, mode="json")
    names = [s["id"] for s in d["streams"] if str(s["to"]).startswith("ext:") and s["fluid"] == "FLUE"]
    for sp in e.result.states:
        if sp.stream in names:
            return sp.T
    return None


def _exhaust_T(e):
    """Temperature of the exhaust stream entering the first exhaust heat exchanger (`HX.hot_in`)."""
    d = e.plant.model_dump(by_alias=True, mode="json")
    names = [x["id"] for x in d["streams"] if x["to"] == "HX.hot_in"]
    return next((sp.T for sp in e.result.states if sp.stream in names), None)


def _norm(s):
    return re.sub(r"\d+(\.\d+)?", "#", s or "")


def study_mechanisms():
    from thermo_agent.design.spec import default_spec
    from thermo_agent.design.build import evaluate_design
    rec = lambda e: {"valid": bool(e.valid), "reason": _norm(e.reason), "fuel_mw": None if e.fuel_w is None else e.fuel_w / 1e6, "stack_K": _stack_T(e)}
    out = {"sweep_f": [], "orc_turbine": {}, "engine_orc": {}, "ambient_305": {}, "water_limit": {}, "exhaust_K": {}}
    for pm in ("turbine", "engine"):
        out["exhaust_K"][pm] = _exhaust_T(evaluate_design(_site(0.5), default_spec(pm, "absorption", "wet")))
    for f in [round(0.3 + 0.05 * i, 2) for i in range(15)]:
        for pm in ("engine", "turbine"):
            e = evaluate_design(_site(f), default_spec(pm, "absorption", "wet")); out["sweep_f"].append({"prime_mover": pm, "chilled_fraction": f, **rec(e)})
    for b in ("orc_r245fa", "orc_cyclopentane", "orc_toluene"):
        out["orc_turbine"][b] = rec(evaluate_design(_site(0.5), default_spec("turbine", "absorption", "wet", b)))
        out["engine_orc"][b] = rec(evaluate_design(_site(0.5), default_spec("engine", "absorption", "wet", b)))
    out["engine_orc"]["none"] = rec(evaluate_design(_site(0.5), default_spec("engine", "absorption", "wet", "none")))
    e = evaluate_design(_site(0.5), default_spec("engine", "absorption", "wet", "orc_cyclopentane"))
    out["engine_orc_net_orc_mw"] = None
    if e.result is not None and e.valid:
        out["engine_orc_net_orc_mw"] = {k: v / 1e6 for k, v in e.result.kpis.items() if "orc" in k.lower() and k.endswith("_w")}
        out["engine_orc_kpis_w"] = {k: v for k, v in e.result.kpis.items() if isinstance(v, (int, float))}
    for rej in ("dry", "wet"):
        out["ambient_305"][rej] = rec(evaluate_design(_site(0.5, T=305.15), default_spec("engine", "absorption", rej)))
    for rej in ("dry", "wet"):
        out["water_limit"][rej] = rec(evaluate_design(_site(0.5, max_water_kg_s=0.5), default_spec("turbine", "absorption", rej)))
    out["water_limit"]["unlimited_wet_kpis"] = {k: v for k, v in evaluate_design(_site(0.5), default_spec("turbine", "absorption", "wet")).result.kpis.items() if k == "water_use_kg_s"}
    jdump("mechanisms.json", {"study": "mechanisms", "command": "design.build.evaluate_design(Site(it_load_w=20e6, cooling_load_w=f*20e6, T0, RH=0.5, supply 291.15 K, return 301.15 K, no grid import), design.spec.default_spec(...))",
                              "seed": None, "n": len(out["sweep_f"]), **out})


def _src_default(path, pattern, cast=float):
    m = re.search(pattern, path.read_text())
    if not m:
        raise RuntimeError(f"pattern not found in {path}: {pattern}")
    return cast(m.group(1))


def study_corpus(project, scratch):
    from thermo_agent.synth import architectures as A, paraphrase as P
    from thermo_agent.synth.splits import is_ood_topology
    project = pathlib.Path(project) if project else pathlib.Path(__file__).resolve().parents[2]
    arch = A.ARCHITECTURES
    # Prefer the corpus that is actually on disk: a fresh 400-row sample of the architecture generator can miss a
    # whole split (no held-out diagram in that draw), and then the paper would quote 0 rows for test_ood_topology.
    real_arch, real_repair = project / "data" / "processed_arch", project / "data" / "processed"
    source = "data/processed_arch" if (real_arch / "repair_train.jsonl").exists() else "generated (seed 0, n=400)"
    if source == "data/processed_arch":
        d = real_arch
    else:
        d = scratch / "arch"; d.mkdir(parents=True, exist_ok=True)
        if not (d / "repair_train.jsonl").exists():
            subprocess.run([sys.executable, "-m", "thermo_agent.synth.architectures", str(d), "-n", "1200", "--seed", str(SEED), "--workers", "1"], cwd=project, check=True)
    counts, archs = {}, {}
    for f in sorted(d.glob("repair_*.jsonl")):
        sp = f.stem.replace("repair_", ""); rows = [json.loads(l) for l in open(f)]
        counts[sp] = len(rows); archs[sp] = sorted({r["topology"] for r in rows})
    requested = 1200
    meta = d / "meta_repair_arch.json"
    if meta.exists():
        requested = json.loads(meta.read_text())["params"]["n"]
    elif (d / "meta_repair.json").exists():
        requested = json.loads((d / "meta_repair.json").read_text())["params"]["n"]
    if source == "data/processed_arch":
        rep = real_repair / "repair_train.jsonl"
    else:
        rep = _ensure_data(project, scratch, "repair", 400, ("--workers", "1")) / "repair_train.jsonl"
    var = collections.defaultdict(list)
    for l in open(rep):
        r = json.loads(l); var[r["fault_type"]].append(P.variant_count(r))
    corpus = None
    sft_report = project / "data" / "sft" / "sft_report.json"
    if sft_report.exists():
        rr = json.loads(sft_report.read_text(encoding="utf-8"))
        corpus = {"source": "data/sft/sft_report.json (scripts/build_corpus.ps1, harness 0.35, memory 3)",
                  "splits": rr["splits"],
                  "tasks": {t: {sp: v["n"] for sp, v in d.items()} for t, d in rr["tasks"].items()}}
    t = project / "thermo_agent" / "teacher"; sy = project / "thermo_agent" / "synth"
    jdump("corpus.json", {
        "study": "corpus", "command": ("synth.architectures.ARCHITECTURES; row counts from data/processed_arch (gen_data.ps1, seed 0); " if source == "data/processed_arch" else "synth.architectures OUT -n 1200 --seed 0; ") + "synth.paraphrase.variant_count on the seed-0 Repair training rows; defaults read from teacher/*.py and synth/combine.py",
        "seed": SEED, "n": len(arch),
        "architectures": [{**a, "held_out": is_ood_topology(a["name"])} for a in arch],
        "n_architectures": len(arch), "n_dc": sum(1 for a in arch if a["kind"] == "dc"), "n_turbine": sum(1 for a in arch if a["prime"] == "turbine"),
        "n_engine": sum(1 for a in arch if a["prime"] == "engine"), "n_orc": sum(1 for a in arch if a["orc"]), "n_heat": sum(1 for a in arch if a["heat"] not in ("none",)),
        "by_rejection": dict(collections.Counter(a["rejection"] for a in arch if a["kind"] == "dc")), "by_cooling": dict(collections.Counter(a["cooling"] for a in arch if a["kind"] == "dc")),
        "arch_rows_requested": requested, "arch_rows_by_split": counts, "arch_distinct_by_split": {k: len(v) for k, v in archs.items()}, "data_source": source,
        "paraphrase_variants_by_fault": {k: {"min": min(v), "max": max(v), "n": len(v)} for k, v in var.items()},
        "corpus": corpus,
        "combine": {"variants_default": _src_default(sy / "combine.py", r'"--variants", type=int, default=(\d+)', int), "harness_fraction": _src_default(sy / "combine.py", r'"--harness-fraction", type=float, default=([\d.]+)'),
                    "k_memory": _src_default(sy / "combine.py", r'"--k-memory", type=int, default=(\d+)', int), "max_prompt_chars": _src_default(sy / "combine.py", r'"--max-prompt-chars", type=int, default=(\d+)', int),
                    "max_prompt_tokens": _src_default(sy / "combine.py", r'"--max-prompt-tokens", type=int, default=(\d+)', int)},
        "sampler": {"temperature": _src_default(t / "sample.py", r'"--temperature", type=float, default=([\d.]+)'), "max_tokens": _src_default(t / "sample.py", r'"--max-tokens", type=int, default=(\d+)', int)},
        "validation": {"audit_ndcg_min": _src_default(t / "validate.py", r'e\["ndcg"\] < ([\d.]+)'), "audit_pred_err_max": _src_default(t / "validate.py", r'e\["pred_err"\] > ([\d.]+)'),
                       "grounding_min": _src_default(t / "validate.py", r'e\["grounded"\] < ([\d.]+)')},
        "memory": {"min_score": _src_default(t / "memory.py", r"min_score: float = ([\d.]+)"), "fault_boost": _src_default(t / "memory.py", r"base \+= ([\d.]+)\n    if note.architecture")}})


def _count_splits(d, task):
    return {p.stem[len(task) + 1:]: sum(1 for l in open(p, encoding="utf-8") if l.strip())
            for p in sorted(pathlib.Path(d).glob(f"{task}_*.jsonl"))}


def _ground_truth(project):
    project = pathlib.Path(project) if project else pathlib.Path(__file__).resolve().parents[2]
    dirs = [project / "data" / d for d in ("processed", "processed_arch")]
    missing = [str(d) for d in dirs if not d.exists()]
    if missing:
        raise SystemExit(f"generate the ground-truth datasets first (scripts/gen_data.ps1); missing: {missing}")
    return dirs


def study_splits(project, scratch):
    """Row counts per split of the ground-truth datasets that the baselines and evaluations read.

    The counts merge `data/processed` (five plant templates) with `data/processed_arch` (the diagram catalog),
    exactly as `eval.rows.load_split` does, so the table describes the data the experiments sample from. The
    held-out definitions come from the generators (`synth.splits`, `ledger.faults`, `design.data.assign_split`).
    """
    from thermo_agent.ledger import faults as LF
    from thermo_agent.synth.architectures import ARCHITECTURES
    from thermo_agent.synth.splits import is_ood_topology
    project = pathlib.Path(project) if project else pathlib.Path(__file__).resolve().parents[2]
    dirs = _ground_truth(project)
    counts = {}
    for task in ("repair", "design", "audit", "ledger"):
        counts[task] = {}
        for d in dirs:
            for sp, n in _count_splits(d, task).items():
                counts[task][sp] = counts[task].get(sp, 0) + n
    held = [a for a in ARCHITECTURES if is_ood_topology(a["name"])]
    n_engine = sum(1 for a in held if a["prime"] == "engine")
    held_txt = (f"{len(held)} of {len(ARCHITECTURES)} diagrams: the steam plant and the {n_engine} gas-engine "
                f"plants with ideal rejection")
    definitions = {
        "repair": {"test_ood_fault": "sign flip, swapped ports",
                   "test_ood_topology": held_txt},
        "design": {"test_ood_scale": "IT load of 45 MW or more", "test_ood_climate": "ambient 305 K or more",
                   "test_ood_load": "chilled fraction 0.85 or more"},
        "audit": {"test_ood_topology": held_txt},
        "ledger": {"test_ood_fault": ", ".join(sorted(LF.HELD_OUT)), "test_ood_topology": held_txt},
    }
    jdump("splits.json", {
        "command": "row counts read from data/processed and data/processed_arch (scripts/gen_data.ps1, seed 0); "
                   "held-out definitions from synth.splits, ledger.faults and design.data.assign_split",
        "seed": SEED, "n": sum(sum(v.values()) for v in counts.values()), "kind": "synthetic",
        "counts": counts, "definitions": definitions,
        "in_distribution": "train, val and test_id are separated by a hash of the example id"})


def _ensure_data(project, scratch, task, n, extra=()):
    d = scratch / task; d.mkdir(parents=True, exist_ok=True)
    first = {"repair": "repair_test_id.jsonl", "audit": "audit_test_id.jsonl"}[task]
    if not (d / first).exists():
        mod = {"repair": ["thermo_agent.synth.build_repair_data"], "audit": ["thermo_agent.audit.data"]}[task]
        subprocess.run([sys.executable, "-m", *mod, str(d), "-n", str(n), "--seed", str(SEED), *extra], cwd=project, check=True)
    return d


def study_repair(project, scratch, limits):
    from thermo_agent.eval.run_eval import run
    dirs = _ground_truth(project)
    res = {}
    for sp, lim in limits.items():
        t = time.time(); res[sp] = {"all": run(dirs, sp, lim), "by_fault": run(dirs, sp, lim, by_fault=True), "limit": lim, "seconds": round(time.time() - t, 1)}
    counts = {}
    for d in dirs:
        for sp, n in _count_splits(d, "repair").items(): counts[sp] = counts.get(sp, 0) + n
    jdump("repair_baselines.json", {"study": "repair", "command": "eval.run_eval.run([data/processed, data/processed_arch], split, limit, by_fault)", "seed": SEED,
                                    "n": sum(r["all"]["n"] for r in res.values()), "split_rows": counts, "splits": res})


def study_audit(project, scratch):
    from thermo_agent.audit.run_eval import score
    from thermo_agent.eval.rows import load_split
    dirs = _ground_truth(project)
    res, shares = {}, {}
    for sp in ("test_id", "test_ood_topology"):
        rows = load_split(dirs, "audit", sp)[:100]
        res[sp] = score(rows, seed=SEED)
        shares[sp] = [r["best_gain_w"] / r["fuel_w"] for r in rows]
    summ = lambda v: {"median": st.median(v), "min": min(v), "max": max(v), "n": len(v)}
    jdump("audit_baselines.json", {"study": "audit", "command": "eval.rows.load_split([data/processed, data/processed_arch], 'audit', split)[:100]; audit.run_eval.score(rows, seed=0)", "seed": SEED,
                                   "n": sum(len(v) for v in shares.values()), "splits": res,
                                   "best_change_saving_share": {**{k: summ(v) for k, v in shares.items()}, "all": summ(shares["test_id"] + shares["test_ood_topology"])}})


# --------------------------------------------------------------------------------------------- derive
V: dict[str, dict] = {}


def put(key, value, tex=None, src="", cmd="", seed=None, n=None):
    if tex is None:
        tex = fmt_int(value) if isinstance(value, int) and not isinstance(value, bool) else str(value)
    V[key] = {"value": value, "tex": tex, "source": src, "command": cmd, "seed": seed, "n": n}


def fmt_int(x):
    s = f"{int(x):,}".replace(",", "{,}")
    return s


def f1(x): return f"{x:.1f}".replace("-", "$-$")
def f2(x): return f"{x:.2f}".replace("-", "$-$")
def f0(x): return f"{x:.0f}".replace("-", "$-$")
def f3(x): return f"{x:.3f}".replace("-", "$-$")
def f4(x): return f"{x:.4f}".replace("-", "$-$")


def putf(key, x, nd, src, cmd, seed=None, n=None, scale=1.0):
    v = x * scale
    put(key, round(v, nd), {0: f0, 1: f1, 2: f2, 3: f3, 4: f4}[nd](v), src, cmd, seed, n)


def tex(s):
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def tt(s): return r"\texttt{" + tex(s) + "}"


def derive_values():
    E = jload("environment.json"); X = jload("crosscheck.json"); P = jload("example_plant.json"); M = jload("mechanisms.json")
    L = jload("ledger_symptoms.json"); A = jload("audit_first_vs_second_law.json"); D = jload("design_baselines.json"); F = jload("design_feasibility.json")
    PL = jload("part_load.json"); RS = jload("reward_signal.json"); S = jload("splits.json"); XF = jload("exergy_flow.json"); H = jload("harness.json")
    RP = jload("repair_baselines.json"); AB = jload("audit_baselines.json"); CO = jload("corpus.json")
    # ---- environment
    s = "assets/environment.json"; c = E["command"]
    put("env.n_components", len(E["components"]), src=s, cmd=c, n=len(E["components"]))
    put("env.n_catalog", E["n_catalog_specs"], src=s, cmd=c)
    put("env.n_catalog_heat", E["n_catalog_specs"] * 2, src=s, cmd=c)
    put("env.n_repair_faults", len(E["fault_injectors"]), src=s, cmd=c)
    put("env.n_ledger_faults", sum(len(v) for v in L["layers"].values()), src="assets/ledger_symptoms.json", cmd=L["command"])
    put("env.n_skills", len(H["skills"]), src="assets/harness.json", cmd=H["command"])
    put("env.n_params", sum(len(c_["params"]) for c_ in E["components"]), src=s, cmd=c)
    if E.get("tests"):
        put("env.n_tests", E["tests"]["passed"], src=s, cmd=E["tests"]["command"])
        putf("env.tests_minutes", E["tests"]["seconds"] / 60, 0, s, E["tests"]["command"])
    for k, v in E["versions"].items():
        put(f"env.ver.{k}", v, src=s, cmd="library versions in the authoring sandbox")
    # ---- corpus (read from the repository)
    s = "assets/corpus.json"; c = CO["command"]; sd = CO["seed"]
    for k in ("n_architectures", "n_dc", "n_turbine", "n_engine", "n_orc", "n_heat"):
        put("corp." + k.replace("n_architectures", "n_arch"), CO[k], src=s, cmd=c)
    put("corp.n_held", sum(1 for a in CO["architectures"] if a["held_out"]), src=s, cmd=c)
    for k, v in CO["combine"].items():
        put("corp." + k, v if k != "harness_fraction" else int(round(v * 100)), src=s, cmd=c)
    for k, v in CO["sampler"].items():
        put("corp.samp_" + k, v, src=s, cmd=c)
    put("corp.ndcg_min", CO["validation"]["audit_ndcg_min"], src=s, cmd=c); put("corp.pred_err_max", CO["validation"]["audit_pred_err_max"], src=s, cmd=c); put("corp.ground_min_pct", int(round(CO["validation"]["grounding_min"] * 100)), src=s, cmd=c)
    put("corp.mem_min_score", CO["memory"]["min_score"], src=s, cmd=c)
    vv = CO["paraphrase_variants_by_fault"]
    put("corp.var_min", min(v["min"] for v in vv.values()), src=s, cmd=c, seed=sd); put("corp.var_max", max(v["max"] for v in vv.values()), src=s, cmd=c, seed=sd)
    for f_, v in vv.items():
        put(f"corp.var.{f_}.min", v["min"], src=s, cmd=c, seed=sd, n=v["n"]); put(f"corp.var.{f_}.max", v["max"], src=s, cmd=c, seed=sd, n=v["n"])
    put("corp.arch_requested", CO["arch_rows_requested"], src=s, cmd=c, seed=sd); put("corp.arch_kept", sum(CO["arch_rows_by_split"].values()), src=s, cmd=c, seed=sd)
    putf("corp.arch_kept_pct", sum(CO["arch_rows_by_split"].values()) / CO["arch_rows_requested"] * 100, 0, s, c, sd, CO["arch_rows_requested"])
    for sp_, v in CO["arch_rows_by_split"].items():
        put(f"corp.arch_rows.{sp_}", v, src=s, cmd=c, seed=sd, n=v); put(f"corp.arch_distinct.{sp_}", CO["arch_distinct_by_split"][sp_], src=s, cmd=c, seed=sd, n=v)
    if CO.get("corpus"):
        C = CO["corpus"]; cs = "assets/corpus.json"
        put("corp.rows", sum(C["splits"].values()), src=cs, cmd=C["source"])
        put("corp.rows_train", C["splits"].get("train", 0), src=cs, cmd=C["source"])
        for t_, d_ in C["tasks"].items():
            for sp_, n_ in d_.items():
                put(f"corp.rows.{t_}.{sp_}", n_, src=cs, cmd=C["source"])
    tc = E["training_config"]; s = "assets/environment.json"; c = E["command"]
    put("tr.group", tc["grpo"]["num_generations"], src=s, cmd=c); put("tr.completion_len", tc["grpo"]["max_completion_length"], src=s, cmd=c); put("tr.sft_len", tc["sft"]["max_length"], src=s, cmd=c)
    put("tr.sft_batch_eff", tc["sft"]["per_device_batch"] * tc["sft"]["grad_accum"] * 2, src=s, cmd=c + "; two ranks from the dual-T4 recipe")
    put("tr.grpo_batch_eff", tc["grpo"]["per_device_batch"] * tc["grpo"]["grad_accum"] * 2, src=s, cmd=c + "; two ranks from the dual-T4 recipe")
    # ---- cross-check
    s = "assets/crosscheck.json"; c = X["command"]
    putf("xchk.wnet_rel_pct", abs(X["w_net_rel_err"]), 4, s, c, scale=100)
    putf("xchk.etaI_rel_pct", abs(X["eta_I_rel_err"]), 3, s, c, scale=100)
    putf("xchk.etaII_diff_pct", X["eta_II_rel_diff"], 2, s, c, scale=100)
    putf("xchk.pinch_diff_K", abs(X["hrsg_pinch_diff_K"]), 3, s, c)
    cw = X["closure_residual_w"]; mant, exp = (f"{abs(cw):.1e}").split("e")
    put("xchk.closure_w", cw, f"${mant}\\times10^{{{int(exp)}}}$" if cw != 0 else "0", s, c)
    putf("xchk.ed_max_rel_pct", max(abs(v) for v in X["ed_rel_diff"].values()), 2, s, c, scale=100)
    # ---- example plant (own solve) and exergy-flow asset
    a, e_ = P["plants"]["absorption"], P["plants"]["electric"]; s = "assets/example_plant.json"; c = P["command"]
    putf("ex.fuel_abs_mw", a["kpis"]["Q_fuel_w"] / 1e6, 1, s, c); putf("ex.fuel_el_mw", e_["kpis"]["Q_fuel_w"] / 1e6, 1, s, c)
    putf("ex.etaII_abs_pct", a["kpis"]["eta_II"] * 100, 1, s, c); putf("ex.etaII_el_pct", e_["kpis"]["eta_II"] * 100, 1, s, c)
    putf("ex.cop_abs", a["kpis"]["COP_CH"], 2, s, c); putf("ex.cop_el", e_["kpis"]["COP_CH"], 1, s, c)
    for comp, key in (("CC", "cc"), ("HX", "hx"), ("GT", "gt"), ("CH", "ch"), ("AC", "ac"), ("DC", "dc")):
        putf(f"ex.ed_{key}_mw", a["ed_mw"][comp], 1, s, c)
    s = "assets/exergy_flow.json"; c = XF["command"]
    d = XF["destruction"]
    putf("exf.fuel_mw", XF["fuel_exergy"], 1, s, c, n=1); putf("exf.net_el_mw", XF["net_electricity"], 1, s, c, n=1)
    putf("exf.cc_mw", d["CC"]["mw"], 1, s, c, n=1); putf("exf.cc_share_pct", d["CC"]["mw"] / XF["fuel_exergy"] * 100, 0, s, c, n=1)
    putf("exf.stack_mw", XF["stack_exergy"], 1, s, c, n=1); putf("exf.stack_share_pct", XF["stack_exergy"] / XF["fuel_exergy"] * 100, 0, s, c, n=1)
    putf("exf.hx_mw", d["HX"]["mw"], 1, s, c, n=1); putf("exf.dc_mw", d["DC"]["mw"], 1, s, c, n=1)
    putf("exf.plant_total_mw", XF["destruction_plant_total"], 1, s, c, n=1)
    putf("exf.cool_ex_mw", XF["delivered_cooling_exergy"], 2, s, c, n=1); putf("exf.heat_ex_mw", XF["delivered_heat_exergy"], 2, s, c, n=1)
    putf("exf.cool_mw", XF["cooling_load_mw"], 1, s, c, n=1); putf("exf.heat_mw", XF["heat_delivered_mw"], 1, s, c, n=1)
    putf("exf.rej_energy_mw", XF["heat_rejector_dissipated_mw"], 1, s, c, n=1); putf("exf.rej_ed_mw", d["REJ"]["mw"], 1, s, c, n=1)
    putf("exf.solve_ms", XF["solve_ms_mean_of_10"], 0, s, c, n=1)
    # ---- mechanisms
    s = "assets/mechanisms.json"; c = M["command"]
    eng = [r for r in M["sweep_f"] if r["prime_mover"] == "engine"]; tur = [r for r in M["sweep_f"] if r["prime_mover"] == "turbine"]
    ok = [r for r in eng if r["valid"]]
    last_ok = max(r["chilled_fraction"] for r in ok)
    first_bad = min(r["chilled_fraction"] for r in eng if not r["valid"] and r["chilled_fraction"] > last_ok)
    put("mech.engine_max_frac_pct", int(round(last_ok * 100)), src=s, cmd=c, n=len(eng))
    put("mech.engine_first_fail_pct", int(round(first_bad * 100)), src=s, cmd=c, n=len(eng))
    r_ok = next(r for r in eng if r["chilled_fraction"] == last_ok)
    putf("mech.engine_stack_K", r_ok["stack_K"], 0, s, c)
    putf("mech.engine_fuel_max_mw", r_ok["fuel_mw"], 1, s, c)
    r_bad = next(r for r in eng if r["chilled_fraction"] == first_bad)
    put("mech.engine_fail_reason", r_bad["reason"], tex(r_bad["reason"]), s, c)
    r_100 = next(r for r in eng if r["chilled_fraction"] == 1.0)
    put("mech.engine_100_reason", r_100["reason"], tex(r_100["reason"]), s, c)
    putf("mech.turbine_exhaust_K", M["exhaust_K"]["turbine"], 0, s, c); putf("mech.engine_exhaust_K", M["exhaust_K"]["engine"], 0, s, c)
    t100 = next(r for r in tur if r["chilled_fraction"] == 1.0)
    put("mech.turbine_100_valid", int(t100["valid"]), src=s, cmd=c)
    putf("mech.turbine_100_stack_K", t100["stack_K"], 0, s, c); putf("mech.turbine_100_fuel_mw", t100["fuel_mw"], 1, s, c)
    put("mech.turbine_valid_all", int(all(r["valid"] for r in tur)), src=s, cmd=c, n=len(tur))
    put("mech.orc_turbine_valid", sum(1 for v in M["orc_turbine"].values() if v["valid"]), src=s, cmd=c, n=3)
    put("mech.orc_turbine_n", len(M["orc_turbine"]), src=s, cmd=c)
    reasons = sorted({v["reason"] for v in M["orc_turbine"].values()})
    put("mech.orc_turbine_reason", "; ".join(reasons), tex("; ".join(reasons)), s, c)
    en = M["engine_orc"]
    putf("mech.eng_none_fuel_mw", en["none"]["fuel_mw"], 1, s, c)
    if en["orc_cyclopentane"]["valid"]:
        putf("mech.eng_cyc_fuel_mw", en["orc_cyclopentane"]["fuel_mw"], 1, s, c)
        putf("mech.eng_cyc_saving_pct", (1 - en["orc_cyclopentane"]["fuel_mw"] / en["none"]["fuel_mw"]) * 100, 1, s, c)
    for b, nm in (("orc_r245fa", "r245fa"), ("orc_toluene", "toluene")):
        put(f"mech.eng_{nm}_valid", int(en[b]["valid"]), src=s, cmd=c)
        if en[b]["valid"]:
            putf(f"mech.eng_{nm}_fuel_mw", en[b]["fuel_mw"], 1, s, c)
        else:
            put(f"mech.eng_{nm}_reason", en[b]["reason"], tex(en[b]["reason"]), s, c)
    kp = M.get("engine_orc_kpis_w") or {}
    for k, v in kp.items():
        if k.startswith("W_orc") and k.endswith("net_w"):
            putf("mech.eng_cyc_orc_net_mw", v / 1e6, 2, s, c)
    for rej in ("dry", "wet"):
        r = M["ambient_305"][rej]
        put(f"mech.amb305_{rej}_valid", int(r["valid"]), src=s, cmd=c)
        if not r["valid"]:
            put(f"mech.amb305_{rej}_reason", r["reason"], tex(r["reason"]), s, c)
        else:
            putf(f"mech.amb305_{rej}_fuel_mw", r["fuel_mw"], 1, s, c)
    for rej in ("dry", "wet"):
        r = M["water_limit"][rej]
        put(f"mech.water_{rej}_valid", int(r["valid"]), src=s, cmd=c)
        if not r["valid"]:
            put(f"mech.water_{rej}_reason", r["reason"], tex(r["reason"]), s, c)
    wu = M["water_limit"]["unlimited_wet_kpis"].get("water_use_kg_s")
    if wu is not None:
        putf("mech.water_wet_use_kg_s", wu, 1, s, c)
    # ---- feasibility grid (figure asset)
    s = "assets/design_feasibility.json"; c = F["command"]; cells = F["cells"]
    nv = [x["n_valid"] for x in cells]; fam = collections.Counter((x["best"]["prime_mover"], x["best"]["cooling"]) for x in cells if x["best"])
    put("feas.n_cells", len(cells), src=s, cmd=c, n=len(cells)); put("feas.n_specs", cells[0]["n_specs"], src=s, cmd=c)
    put("feas.n_cells_none", sum(1 for x in cells if x["n_valid"] == 0), src=s, cmd=c)
    putf("feas.mean_valid", st.mean(x for x in nv if x), 1, s, c)
    put("feas.n_winner_types", len({tuple(sorted(x["best"].items())) for x in cells if x["best"]}), src=s, cmd=c)
    put("feas.n_cells_valid", sum(1 for x in cells if x["n_valid"]), src=s, cmd=c)
    put("feas.n_ea_none", sum(1 for x in cells if x["engine_absorption_valid"] == 0 and x["n_valid"] > 0), src=s, cmd=c)
    r0 = F["reasons_in_cells_without_valid_design"]; tot = sum(r0.values()); top = max(r0.values())
    put("feas.n_cells_283", sum(1 for x in cells if x["T_amb_K"] == min(F["T_amb_K"]) and x["n_valid"] == 0), src=s, cmd=c)
    put("feas.chiller_reason_share_pct", int(round(top / tot * 100)), src=s, cmd=c)
    for k, v in sorted(fam.items()):
        put(f"feas.win.{k[0]}_{k[1]}", v, src=s, cmd=c)
    # ---- ledger
    s = "assets/ledger_symptoms.json"; c = L["command"]; sd = L["seed"]
    put("led.n", L["n"], src=s, cmd=c, seed=sd, n=L["n"]); put("led.n_single", L["rows_with_one_fault"], src=s, cmd=c, seed=sd, n=L["n"]); put("led.n_double", L["rows_with_two_faults"], src=s, cmd=c, seed=sd, n=L["n"])
    putf("led.any_pct", L["rows_with_any_symptom"] * 100, 0, s, c, sd, L["n"]); putf("led.none_pct", L["rows_with_no_symptom"] * 100, 0, s, c, sd, L["n"])
    put("led.range_checks", L["parameter_range_checks_seeing_equation_faults"], src=s, cmd=c, seed=sd, n=L["n"])
    g = L["closure_by_row_group"]
    put("led.comp_rows", g["component_line"]["rows"], src=s, cmd=c, seed=sd, n=L["n"]); put("led.comp_closure", g["component_line"]["closure_residual"], src=s, cmd=c, seed=sd, n=L["n"])
    put("led.stream_rows", g["stream_or_constant_only"]["rows"], src=s, cmd=c, seed=sd, n=L["n"]); put("led.stream_closure", g["stream_or_constant_only"]["closure_residual"], src=s, cmd=c, seed=sd, n=L["n"])
    put("led.stream_either", g["stream_or_constant_only"]["either"], src=s, cmd=c, seed=sd, n=L["n"]); put("led.stream_neg", g["stream_or_constant_only"]["negative_ed"], src=s, cmd=c, seed=sd, n=L["n"])
    sf = L["by_fault_single_fault_rows"]
    put("led.t0_celsius_n", sf["t0_celsius"]["n"], src=s, cmd=c, seed=sd, n=L["n"]); put("led.t0_offset_n", sf["t0_offset"]["n"], src=s, cmd=c, seed=sd, n=L["n"])
    putf("led.mean_prompt_chars", L["mean_prompt_chars"], 0, s, c, sd, L["n"])
    cd = L["closure_demo"]
    putf("led.demo_drop_work_mw", abs(cd["drop_work_term"]["closure"]), 1, s, c, sd, 1)
    put("led.held_out_n", sum(sf[k]["n"] for k in L["held_out_fault_types"]), src=s, cmd=c, seed=sd, n=L["n"])
    # ---- audit, one plant (figure asset)
    s = "assets/audit_first_vs_second_law.json"; c = A["command"]; sd = A["seed"]
    rej = max(A["components"], key=lambda x: x["energy_dissipated_mw"]); cc = max(A["components"], key=lambda x: x["exergy_destroyed_mw"])
    non_pm = [x for x in A["components"] if x["id"] != A["prime_mover"]]
    ex_top = max(non_pm, key=lambda x: x["exergy_destroyed_mw"])
    put("aud.ex_rej_id", rej["id"], tt(rej["id"]), s, c, sd, 1); putf("aud.ex_rej_energy_mw", rej["energy_dissipated_mw"], 1, s, c, sd, 1); putf("aud.ex_rej_ed_mw", rej["exergy_destroyed_mw"], 1, s, c, sd, 1)
    put("aud.ex_pm_id", cc["id"], tt(cc["id"]), s, c, sd, 1); putf("aud.ex_pm_ed_mw", cc["exergy_destroyed_mw"], 1, s, c, sd, 1); putf("aud.ex_pm_energy_mw", cc["energy_dissipated_mw"], 1, s, c, sd, 1)
    put("aud.ex_top_id", ex_top["id"], tt(ex_top["id"]), s, c, sd, 1); putf("aud.ex_top_ed_mw", ex_top["exergy_destroyed_mw"], 1, s, c, sd, 1)
    putf("aud.ex_fuel_mw", A["fuel_mw"], 0, s, c, sd, 1)
    # ---- audit baselines (own run): the ranking macros of the Audit figure and text come from the same rows as the
    # baseline table, so that both describe one measurement; only the example plant above comes from the figure asset.
    s = "assets/audit_baselines.json"; c = AB["command"]
    for sp, k in (("test_id", "id"), ("test_ood_topology", "ood")):
        r = AB["splits"][sp]; n_ = r["n"]
        rule = r["rule_change (raise turbine inlet T 10%, else cut engine loss 10%)"]
        nd = r["ranking_ndcg3_excluding_prime_mover"]; t1 = r["ranking_top1_excluding_prime_mover"]
        kk = int(round(t1["first_law"] * n_)); lo, hi = wilson(kk, n_)
        put(f"aud.n_{k}", n_, src=s, cmd=c, seed=SEED, n=n_); put(f"aud.fl_hits_{k}", kk, src=s, cmd=c, seed=SEED, n=n_)
        putf(f"aud.fl_top1_{k}_pct", kk / n_ * 100, 0, s, c, SEED, n_)
        putf(f"aud.fl_ci_lo_{k}_pct", lo * 100, 0, s, c, SEED, n_); putf(f"aud.fl_ci_hi_{k}_pct", hi * 100, 0, s, c, SEED, n_)
        putf(f"aud.rnd_top1_{k}_pct", t1["random"] * 100, 0, s, c, SEED, n_)
        putf(f"aud.ndcg_rnd_{k}", nd["random"], 3, s, c, SEED, n_); putf(f"aud.ndcg_fl_{k}", nd["first_law"], 3, s, c, SEED, n_)
        put(f"audb.n_{k}", n_, src=s, cmd=c, seed=SEED, n=n_)
        putf(f"audb.fl_top1_{k}_pct", t1["first_law"] * 100, 0, s, c, SEED, n_)
        putf(f"audb.rnd_top1_{k}_pct", t1["random"] * 100, 0, s, c, SEED, n_)
        putf(f"audb.rule_gain_{k}_pct", (rule["mean_gain_vs_best"] or 0) * 100, 0, s, c, SEED, n_)
        put(f"audb.rule_valid_{k}", rule["valid"].split("/")[0], src=s, cmd=c, seed=SEED, n=n_)
        pb = r["prediction_baselines_for_the_best_change (a_pred score)"]
        putf(f"audb.pred_zero_{k}", pb["predict_zero"], 2, s, c, SEED, n_); putf(f"audb.pred_mean_{k}", pb["predict_mean_relative_change"], 2, s, c, SEED, n_)
        putf(f"audb.search_evals_{k}", r["change_search_evaluations"], 0, s, c, SEED, n_)
    for sp, k in (("test_id", "id"), ("test_ood_topology", "ood"), ("all", "all")):
        b = AB["best_change_saving_share"][sp]
        putf(f"audb.saving_median_{k}_pct", b["median"] * 100, 1, s, c, SEED, b["n"]); putf(f"audb.saving_min_{k}_pct", b["min"] * 100, 2, s, c, SEED, b["n"]); putf(f"audb.saving_max_{k}_pct", b["max"] * 100, 1, s, c, SEED, b["n"])
    put("audb.n_total", AB["best_change_saving_share"]["all"]["n"], src=s, cmd=c, seed=SEED, n=AB["best_change_saving_share"]["all"]["n"])
    # ---- design baselines (figure asset)
    s = "assets/design_baselines.json"; c = D["command"]; sd = D["seed"]; n_ = D["n"]; C = D["curves"]
    put("des.n_sites", n_, src=s, cmd=c, seed=sd, n=n_); put("des.n_heat_sites", D["sites_with_heat_demand"], src=s, cmd=c, seed=sd, n=n_)
    for k, v in D["split_counts"].items():
        put(f"des.split.{k}", v, src=s, cmd=c, seed=sd, n=n_)
    at = lambda name, b_: next(r for r in C[name] if r["budget"] == b_)
    put("des.enum_len_min", C["enumerate_defaults_lengths"]["min"], src=s, cmd=c, seed=sd, n=n_); put("des.enum_len_max", C["enumerate_defaults_lengths"]["max"], src=s, cmd=c, seed=sd, n=n_)
    put("des.polish_len_min", C["enumerate_then_polish_lengths"]["min"], src=s, cmd=c, seed=sd, n=n_); put("des.polish_len_max", C["enumerate_then_polish_lengths"]["max"], src=s, cmd=c, seed=sd, n=n_)
    put("des.random_len", C["random_lengths"]["max"], src=s, cmd=c, seed=sd, n=n_)
    for name, short in (("enumerate_defaults", "enum"), ("random", "rand"), ("enumerate_then_polish", "pol")):
        for b_ in (12, 24, 32, 48, 64, 96, 128):
            r = at(name, b_)
            putf(f"des.{short}.b{b_}.within_pct", r["within_tol"] * 100, 0, s, c, sd, n_); putf(f"des.{short}.b{b_}.gap_pct", r["mean_gap_pct"], 1, s, c, sd, n_)
            putf(f"des.{short}.b{b_}.valid_pct", r["valid"] * 100, 0, s, c, sd, n_)
    r = at("enumerate_defaults", 1); putf("des.enum.b1.valid_pct", r["valid"] * 100, 0, s, c, sd, n_)
    r = at("random", 1); putf("des.rand.b1.valid_pct", r["valid"] * 100, 0, s, c, sd, n_)
    putf("des.tolerance_pct", D["tolerance"] * 100, 0, s, c, sd, n_)
    for b_ in (96,):
        putf(f"des.pol.b{b_}.within_pct", at("enumerate_then_polish", b_)["within_tol"] * 100, 0, s, c, sd, n_)
    s = "assets/harness.json"; c = H["command"]
    put("har.agents_lines", H["agents_md_lines"], src=s, cmd=c, seed=H["seed"]); put("har.agents_chars", H["agents_md_chars"], src=s, cmd=c, seed=H["seed"])
    put("har.engine_n_ports", len(H["gas_engine_ports"]), src=s, cmd=c, seed=H["seed"])
    for t_, d_ in H["test_ood_topology_topologies"].items():
        put(f"har.heldout_{t_}_rows", sum(d_.values()), src=s, cmd=c, seed=H["seed"])
        put(f"har.heldout_{t_}_engine_rows", sum(v for k, v in d_.items() if "engine" in k), src=s, cmd=c, seed=H["seed"])
    # ---- part load
    s = "assets/part_load.json"; c = PL["command"]; sd = PL["seed"]
    for pm in ("turbine", "engine"):
        for r in PL["scan"][pm]:
            putf(f"pl.{pm}.u{r['n_units']}_pct", r["modularity_penalty"] * 100, 1, s, c, sd, PL["n"])
    put("pl.n_samples", PL["n"], src=s, cmd=c, seed=sd, n=PL["n"]); put("pl.bins", PL["bins"], src=s, cmd=c, seed=sd, n=PL["n"])
    t, e2 = PL["scan"]["turbine"][0], PL["scan"]["engine"][0]
    putf("pl.turbine.ideal_mwh", t["ideal_fuel_mwh"], 0, s, c, sd, PL["n"]); putf("pl.turbine.fuel1_mwh", t["fuel_mwh"], 0, s, c, sd, PL["n"])
    put("pl.units_max", max(r["n_units"] for r in PL["scan"]["turbine"]), src=s, cmd=c, seed=sd, n=PL["n"])
    put("pl.same_penalty", int(all(abs(a_["modularity_penalty"] - b_["modularity_penalty"]) < 5e-4 for a_, b_ in zip(PL["scan"]["turbine"], PL["scan"]["engine"]))), src=s, cmd=c, seed=sd, n=PL["n"])
    # ---- reward diagnostics
    s = "assets/reward_signal.json"; c = RS["command"]; sd = RS["seed"]
    for t_, d_ in RS["tasks"].items():
        put(f"rd.{t_}.groups", d_["groups"], src=s, cmd=c, seed=sd, n=d_["groups"]); putf(f"rd.{t_}.eval_ms", d_["fresh_evaluation_ms"], 1, s, c, sd, d_["groups"])
        for lv, row in d_["by_error_rate"].items():
            putf(f"rd.{t_}.zv_e{int(round(float(lv) * 100))}_pct", row["zero_variance_groups"] * 100, 0, s, c, sd, d_["groups"])
    put("rd.G", RS["tasks"]["repair"]["G"], src=s, cmd=c, seed=sd)
    for t_, d_ in RS["tasks_g8"].items():
        for lv, row in d_["by_error_rate"].items():
            putf(f"rd8.{t_}.zv_e{int(round(float(lv) * 100))}_pct", row["zero_variance_groups"] * 100, 0, s, c, sd, d_["groups"])
    # ---- splits
    s = "assets/splits.json"; c = S["command"]; sd = S["seed"]
    tot = 0
    for t_, d_ in S["counts"].items():
        for k, v in d_.items():
            put(f"split.{t_}.{k}", v, src=s, cmd=c, seed=sd, n=v); tot += v
        put(f"split.{t_}.total", sum(d_.values()), src=s, cmd=c, seed=sd, n=sum(d_.values()))
    put("split.total", tot, src=s, cmd=c, seed=sd, n=tot)
    # ---- repair baselines (own run)
    s = "assets/repair_baselines.json"; c = RP["command"]
    for sp, r in RP["splits"].items():
        a_ = r["all"]; n_ = a_["n"]
        put(f"rep.{sp}.n", n_, src=s, cmd=c, seed=SEED, n=n_); put(f"rep.{sp}.rows", RP["split_rows"][sp], src=s, cmd=c, seed=SEED, n=RP["split_rows"][sp])
        putf(f"rep.{sp}.noop_pct", a_["noop"] / n_ * 100, 0, s, c, SEED, n_); putf(f"rep.{sp}.search_pct", a_["search_success"] / n_ * 100, 0, s, c, SEED, n_)
        lo, hi = wilson(a_["search_success"], n_)
        putf(f"rep.{sp}.search_lo_pct", lo * 100, 0, s, c, SEED, n_); putf(f"rep.{sp}.search_hi_pct", hi * 100, 0, s, c, SEED, n_)
        putf(f"rep.{sp}.exact_pct", a_["search_exact"] / n_ * 100, 0, s, c, SEED, n_)
        if a_.get("search_mean_solves_when_solved") is not None:
            putf(f"rep.{sp}.calls", a_["search_mean_solves_when_solved"], 0, s, c, SEED, n_)
        putf(f"rep.{sp}.expert_pct", a_["expert_success"] / n_ * 100, 0, s, c, SEED, n_)
        for ft, fr in r["by_fault"].items():
            put(f"rep.{sp}.{ft}.n", fr["n"], src=s, cmd=c, seed=SEED, n=fr["n"])
    allf = collections.defaultdict(lambda: [0, 0, []])
    for sp, r in RP["splits"].items():
        for ft, fr in r["by_fault"].items():
            allf[ft][0] += fr["n"]; allf[ft][1] += fr["search_success"]
            if fr.get("search_mean_solves_when_solved") is not None:
                allf[ft][2].append(fr["search_mean_solves_when_solved"])
    for ft, (n_, k_, calls) in allf.items():
        put(f"rep.f.{ft}.n", n_, src=s, cmd=c, seed=SEED, n=n_); putf(f"rep.f.{ft}.search_pct", k_ / n_ * 100, 0, s, c, SEED, n_)
        if calls:
            putf(f"rep.f.{ft}.calls", st.mean(calls), 0, s, c, SEED, n_)
    put("rep.total_n", sum(r["all"]["n"] for r in RP["splits"].values()), src=s, cmd=c, seed=SEED)
    return E, L, S, H, RP, D, AB, A, PL, RS


# ---------------------------------------------------------------------------------------------- tables
def hline(): return ""


def table_components(E):
    rows = []
    for c in E["components"]:
        ports = ", ".join(f"{tt(p['name'])} ({'g' if p['class'] == 'gas' else 'l' if p['class'] == 'liquid' else p['class']})" for p in c["ports"])
        pars = ", ".join(tt(k) for k in c["params"]) or "--"
        rows.append(f"{tt(c['type'])} & {ports} & {pars} \\\\")
    return ("\\begingroup\\scriptsize\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.15}\n\\begin{tabular}{@{}p{3.2cm}p{4.9cm}p{7.0cm}@{}}\n\\toprule\n"
            "Type & Ports (fluid class) & Parameters \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\\endgroup\n")


def table_params(E):
    rows = []
    for c in E["components"]:
        first = True
        for k, v in c["params"].items():
            d = v["default"]
            ds = "required" if isinstance(d, str) and d in ("required",) else (tex(d) if isinstance(d, str) else (f"{d:g}" if isinstance(d, (int, float)) else tex(d)))
            rows.append(f"{tt(c['type']) if first else ''} & {tt(k)} & {tex(v['doc'])} & {ds} \\\\"); first = False
    return ("\\begingroup\\scriptsize\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.1}\n\\begin{longtable}{@{}p{3.1cm}p{2.9cm}p{6.6cm}p{2.0cm}@{}}\n"
            "\\caption{Component parameter registry (\\texttt{core.components.PARAMS}). ``required'' means no default; other entries are the code default.}\\label{tab:params}\\\\\n\\toprule\n"
            "Type & Parameter & Meaning & Default \\\\\n\\midrule\n\\endfirsthead\n\\toprule\nType & Parameter & Meaning & Default \\\\\n\\midrule\n\\endhead\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}\\endgroup\n")


VERIFIERS = [
    ("V1", "mass balance", "per component, residual $\\le 10^{-6}$ of the largest flow", "hard"),
    ("V2", "energy balance", "per component, residual $\\le 10^{-6}\\,(|\\dot W_{\\text{net}}|+\\dot Q_{\\text{fuel}}+1~\\mathrm{MW})$", "hard"),
    ("V3a", "non-negative destruction", "$\\ED \\ge -10^{-5}\\,\\dot E_{\\text{fuel}}$ in every component", "hard"),
    ("V3b", "exergy closure", "$|\\sum \\ED - (\\text{in} - \\text{out} - \\dot W_{\\text{net}})| \\le 10^{-6}\\,\\dot E_{\\text{fuel}}$", "hard"),
    ("V4a", "temperature cross", "minimum approach $\\ge 0$ K in every heat exchanger", "hard"),
    ("V4b", "practical pinch", "minimum approach $\\ge 1$ K", "soft"),
    ("V5", "second-law bounds", "$0\\le\\eta_{\\mathrm{II}}\\le 1$; power efficiency $<1-\\Tzero/T_{\\max}$; ORC efficiency $<1-T_{\\mathrm{cond}}/T_{\\mathrm{hot}}$; every COP at or below its reversible limit", "hard"),
    ("V6", "parameter and state ranges", "efficiencies in $(0,1]$, pressure ratio $>1$, drops in $[0,0.3]$, $T$ in 150--2500 K, $P$ in 1 kPa--100 MPa, pressure-change direction", "hard"),
    ("V7", "number grounding", "numbers with units in reasoning text match solver or plant values within 0.5\\% (3\\% for predicted Design quantities)", "reward channel"),
    ("V8", "site requirements", "IT load within 1\\%, electricity balance (no grid import unless allowed), cooling and heat $\\ge 99\\%$, water limit", "hard (site)"),
]


def table_verifiers():
    rows = [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in VERIFIERS]
    return ("\\begingroup\\small\\renewcommand{\\arraystretch}{1.15}\n\\begin{tabular}{@{}lp{3.3cm}p{7.6cm}p{1.9cm}@{}}\n\\toprule\nID & Check & Definition and tolerance & Kind \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\\endgroup\n")


def table_tasks(S, ex):
    ch = lambda t: ", ".join(f"{c['name']}~{c['weight']:g}" for c in ex[t]["channels_parsed"])
    sp = lambda t: "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in S["definitions"][t].items())
    rows = [
        ("Repair", "plant text, site, solver error or failed checks", "\\texttt{<diagnosis>}, at most 6 edits", ch("repair"), sp("repair")),
        ("Design", "site: IT load, chilled load, ambient $T$ and RH, coolant temperatures, optional heat demand and water limit", "\\texttt{<rationale>}, \\texttt{<design>} block", ch("design"), sp("design")),
        ("Audit", "solved plant with an energy-level summary", "\\texttt{<analysis>}, \\texttt{<ranking>}, one \\texttt{<change>}, \\texttt{<prediction>}", ch("audit"), sp("audit")),
        ("Ledger repair", "exergy ledger of a solved plant (arithmetic lines)", "\\texttt{<diagnosis>}, \\texttt{FIX} lines", ch("ledger"), sp("ledger")),
    ]
    body = "\n".join(f"{a} & {b} & {c} & {tex(d)} & {tex(e)} \\\\" for a, b, c, d, e in rows)
    return ("\\begingroup\\scriptsize\\renewcommand{\\arraystretch}{1.2}\n\\begin{tabular}{@{}p{1.5cm}p{3.2cm}p{2.8cm}p{3.3cm}p{3.3cm}@{}}\n\\toprule\n"
            "Task & Input & Output & Reward channels (weight) & Held-out splits \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\\endgroup\n")


LEDGER_DESC = {"missing_m": "flow factor missing in a stream line", "missing_ech": "chemical term missing in a stream line", "unit_slip": "$/10^3$ for $/10^6$ in a stream line, or a factor $10^{\\pm3}$ in a destruction line",
               "sign_T0": "wrong sign on the $T_0$ term in a stream line", "wrong_ref": "another stream's reference state in a stream line", "t0_literal": "the number 25 in place of $T_0$",
               "sign_flip": "flipped sign of a stream term in a destruction line", "dropped_term": "a stream dropped from a destruction line", "dropped_work": "work term dropped from a destruction line",
               "wrong_factor": "destruction line scaled by 0.1, 10, 2 or 0.5", "wrong_stream": "a different stream in a destruction line", "t0_celsius": "$T_0$ in Celsius", "t0_offset": "$T_0 = 273.15$"}


def table_faults(E, L, S):
    desc_r = {"unit_slip": "value off by a factor of $10^{\\pm k}$ or a unit offset", "sign_flip": "sign of a parameter or flow reversed", "efficiency_gt_1": "efficiency above 1",
              "swapped_ports": "destinations of two same-fluid streams exchanged", "wrong_fluid": "stream labelled with the wrong fluid"}
    held_r = {"sign_flip", "swapped_ports"}
    rows = [f"Repair & {tt(f)} & {desc_r.get(f, '')} & {'yes' if f in held_r else 'no'} \\\\" for f in E["fault_injectors"]]
    held = set(L["held_out_fault_types"])
    for layer, fl in L["layers"].items():
        for f in fl:
            rows.append(f"Ledger, {tex(layer)} & {tt(f)} & {LEDGER_DESC.get(f, '')} & {'yes' if f in held else 'no'} \\\\")
    first_led = len(E["fault_injectors"])
    rows.insert(first_led, "\\midrule")
    return ("\\begingroup\\small\\renewcommand{\\arraystretch}{1.12}\n\\begin{tabular}{@{}p{3.3cm}p{3.0cm}p{7.2cm}l@{}}\n\\toprule\nTask, layer & Fault & Effect & Held out \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\\endgroup\n")


def table_baselines(RP, AB, D):
    def rep_row(sp, label):
        r = RP["splits"][sp]["all"]; n = r["n"]
        lo, hi = wilson(r["search_success"], n)
        calls = r.get("search_mean_solves_when_solved")
        return (f"Repair: blind search, {label} & {n} & {100 * r['search_success'] / n:.0f}\\% ({100 * lo:.0f}--{100 * hi:.0f}) & "
                f"{'--' if calls is None else f'{calls:.0f} solver calls'} & seed 0 \\\\")
    rows = [rep_row("test_id", "in-distribution"), rep_row("test_ood_fault", "held-out faults"), rep_row("test_ood_topology", "held-out topologies")]
    C = D["curves"]; at = lambda name, b: next(r for r in C[name] if r["budget"] == b)
    for name, label, bs in (("enumerate_defaults", "Design: enumerate defaults", (64,)), ("random", "Design: random search", (12, 128)), ("enumerate_then_polish", "Design: enumerate, then polish", (64, 128))):
        for b in bs:
            r = at(name, b)
            rows.append(f"{label}, {b} evaluations & {D['n']} & {100 * r['within_tol']:.0f}\\% within {100 * D['tolerance']:.0f}\\% & gap {r['mean_gap_pct']:.1f}\\% & seed 0 \\\\")
    for sp, lab in (("test_id", "in-distribution"), ("test_ood_topology", "held-out topologies")):
        r = AB["splits"][sp]; n = r["n"]
        rule = r["rule_change (raise turbine inlet T 10%, else cut engine loss 10%)"]
        k1 = round(r["ranking_top1_excluding_prime_mover"]["first_law"] * n); lo, hi = wilson(k1, n)
        rows.append(f"Audit: first-law ranking top-1, {lab} & {n} & {100 * k1 / n:.0f}\\% ({100 * lo:.0f}--{100 * hi:.0f}) & random {100 * r['ranking_top1_excluding_prime_mover']['random']:.0f}\\% & seed 0 \\\\")
        rows.append(f"Audit: fixed rule change, {lab} & {n} & {100 * (rule['mean_gain_vs_best'] or 0):.0f}\\% of best gain & valid {rule['valid']} & seed 0 \\\\")
    return ("\\begingroup\\scriptsize\\renewcommand{\\arraystretch}{1.15}\n\\begin{tabular}{@{}p{5.7cm}rp{3.1cm}p{3.1cm}l@{}}\n\\toprule\nBaseline (no language model) & $n$ & Main result (95\\% Wilson) & Cost or reference & Seed \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\\endgroup\n")


def table_constants(E):
    d = E["dc_chp_defaults"]; orc = E["orc_t_max_K"]
    g = lambda k: f"{d[k]:g}"
    rows = [
        ("Compressor isentropic efficiency", r"\texttt{eta\_c}", g("eta_c")), ("Turbine isentropic efficiency", r"\texttt{eta\_t}", g("eta_t")),
        ("Pressure ratio", r"\texttt{PR}", g("PR")), ("Turbine inlet temperature (K)", r"\texttt{T4}", g("T4")),
        ("Combustor pressure drop; heat loss fraction", r"\texttt{dp\_cc}, \texttt{q\_loss}", f"{g('dp_cc')}; {g('q_loss')}"),
        ("Gas engine electrical efficiency", r"\texttt{eta\_el}", g("eta_el")), ("Gas engine loss fraction (jacket and radiation)", r"\texttt{f\_loss}", g("f_loss")),
        ("Air excess", r"\texttt{air\_excess}", g("air_excess")),
        ("Absorption chiller: fraction of reversible COP; COP cap", r"\texttt{eps\_abs}, \texttt{COP\_max}", f"{g('eps_abs')}; {g('COP_max')}"),
        ("Electric chiller and heat pump: fraction of reversible COP", r"\texttt{eps\_ec}, \texttt{eps\_hp}", f"{g('eps_ec')}; {g('eps_hp')}"),
        ("Chilled-water supply and return (K)", r"\texttt{T\_supply}, \texttt{T\_return}", f"{g('T_supply')}; {g('T_return')}"),
        ("Hot-water loop supply and return (K)", r"\texttt{T\_hw}, \texttt{T\_gen\_out}", f"{g('T_hw')}; {g('T_gen_out')}"),
        ("Rejector approach to ambient (dry or wet; K)", r"\texttt{dT\_rej}", "8 (dry), 5 (wet)"), ("Rejector fan power (share of heat rejected)", r"\texttt{fan\_frac}", g("fan_frac")),
        ("ORC working-fluid vapour limits (K)", r"\texttt{ORC\_T\_MAX}", "; ".join(f"{k} {v:g}" for k, v in orc.items())),
        ("ORC condenser approach (K); pump and turbine efficiency", r"\texttt{orc\_cond\_approach}, \texttt{eta\_pump}, \texttt{eta\_orct}", f"{g('orc_cond_approach')}; {g('eta_pump')}; {g('eta_orct')}"),
        ("Stack temperature floor for exhaust recovery (K)", r"\texttt{T\_stack\_min}", g("T_stack_min")),
        ("Fraction of the IT heat on the chilled-water loop", r"\texttt{chilled\_fraction}", g("chilled_fraction") + " (Design: site input)"),
        ("Sizing margin on net electricity", r"\texttt{margin}", "1.03"),
    ]
    body = "\n".join(f"{a} & {b} & {tex(c) if not c.startswith('\\') else c} \\\\" for a, b, c in rows)
    return ("\\begingroup\\small\\renewcommand{\\arraystretch}{1.12}\n\\begin{tabular}{@{}p{7.2cm}p{4.6cm}p{3.6cm}@{}}\n\\toprule\nConstant (assumed, not fitted) & Code name & Default \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\\endgroup\n")


def table_training(E):
    c = E["training_config"]; sl, gl = c["sft_lora"], c["grpo_lora"]; s, g = c["sft"], c["grpo"]
    ranks = 2
    rows = [("Base model", tt(c["base_model"]), tt(c["base_model"])), ("Method", "LoRA SFT, all tasks in one adapter", "GRPO per task on the merged SFT model"),
            ("Precision and parallelism", f"fp16 LoRA, DDP over {ranks} ranks (2 T4), no 4-bit", f"fp16 LoRA, DDP over {ranks} ranks, no 4-bit"),
            ("LoRA rank / alpha / dropout", f"{sl['r']} / {sl['alpha']} / {sl['dropout']:g}", f"{gl['r']} / {gl['alpha']} / {gl['dropout']:g}"),
            ("Target modules", tex(sl["target_modules"]), tex(gl["target_modules"])), ("Learning rate", f"{s['lr']:g}", f"{g['lr']:g}"),
            ("Epochs or steps", f"{s['epochs']} epoch" + ("s" if s["epochs"] != 1 else ""), f"{g['max_steps']} steps (maximum)"), ("Maximum length (tokens)", f"{s['max_length']}", f"completion {g['max_completion_length']}"),
            ("Batch per device / accumulation / ranks", f"{s['per_device_batch']} / {s['grad_accum']} / {ranks}", f"{g['per_device_batch']} / {g['grad_accum']} / {ranks}"),
            ("Effective batch", f"{s['per_device_batch'] * s['grad_accum'] * ranks} sequences", f"{g['per_device_batch'] * g['grad_accum'] * ranks} prompts, {g['num_generations']} completions each"),
            ("Group size / temperature", "--", f"{g['num_generations']} / {g['temperature']:g}"), ("Gradient clipping", f"max norm {s['max_grad_norm']:g}", f"max norm {g['max_grad_norm']:g}"),
            ("Loss", "causal LM, completion tokens only", f"{tex(g['loss_type'])}-style, KL coefficient {g['beta']:g}, truncated completions masked"),
            ("Prompt rendering", "tokenizer generation prompt, thinking disabled", "same text as SFT and evaluation")]
    body = "\n".join(f"{a} & {b} & {c_} \\\\" for a, b, c_ in rows)
    return ("\\begingroup\\small\\renewcommand{\\arraystretch}{1.12}\n\\begin{tabular}{@{}p{4.2cm}p{5.4cm}p{5.6cm}@{}}\n\\toprule\n & Stage 1: SFT & Stage 2: GRPO \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\\endgroup\n")


def table_results_skeleton():
    keys = {"Repair": ["success", "first-turn", "exact", "solver calls"], "Design": ["valid", "within 1\\%", "gap (\\%)", "evaluations"],
            "Audit": ["rank", "valid", "gain", "prediction"], "Ledger repair": ["exact", "closure", "all correct", "--"]}
    models = ["Qwen3.5-4B base", "SFT, ground truth only", "SFT, full corpus", "SFT + GRPO", "Teacher model \\todo{name}", "Other open model \\todo{name}"]
    rows = []
    for task, ks in keys.items():
        for k_, m in enumerate(models):
            cells = ["\\todo{pending}" if x != "--" else "--" for x in ks]
            rows.append(f"{task if k_ == 0 else ''} & {m} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    rows.pop()
    head = {"Repair": ks_ for ks_ in []}
    return ("\\begingroup\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}p{1.3cm}p{4.4cm}p{1.9cm}p{1.9cm}p{1.9cm}p{1.9cm}@{}}\n\\toprule\nTask & Model (full inference text) & \\multicolumn{4}{c}{Metrics (per task; see text)} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\\endgroup\n")


def table_results_data(MR):
    """Model results from comparison rows; fall back to the skeleton until the first report exists."""
    rows = (MR or {}).get("rows") or []
    if not rows:
        return table_results_skeleton()
    keys = {"repair": ["success", "first_turn_success", "exact_on_deducible", "mean_solves"],
            "design": ["valid", "within_1pct", "mean_gap_pct", "mean_evals"],
            "audit": ["rank", "valid", "gain", "pred"], "ledger": ["exact", "closure", "all_correct"]}
    labels = {"repair": "Repair", "design": "Design", "audit": "Audit", "ledger": "Ledger repair"}
    out = []
    for row in rows:
        m = row["metrics"]
        cells = [("--" if m.get(k) is None else f"{m[k]:.2f}") for k in keys[row["task"]]]
        arm = f"{row['display_name']} ({row['model_family']}, {row['harness']}, tools {row['tool_access']})"
        out.append(f"{labels[row['task']]} & {tex(arm)} & {row['split'].replace('_', ' ')} & {row['n']} & "
                   + " & ".join(cells) + r" \\")
    return ("\\begingroup\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}p{1.5cm}p{5.2cm}p{1.6cm}rp{1.5cm}p{1.5cm}p{1.5cm}p{1.5cm}@{}}\n\\toprule\n"
            "Task & Model (family, inference text, tools) & Split & $n$ & \\multicolumn{4}{c}{Metrics (per task; see text)} \\\\\n\\midrule\n"
            + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}\\endgroup\n")


def table_comparison(MR):
    """One row per arm/task/split with the comparison axes explicit, for the model-comparison paragraph."""
    rows = (MR or {}).get("rows") or []
    if not rows:
        return ("\\begingroup\\small\n\\begin{tabular}{@{}lllll}\n\\toprule\nArm & Family & Inference text & Tools & Result \\\\\n\\midrule\n"
                "Model reports are pending; see \\texttt{paper/assets/model\\_results\\_manifest.json}. \\\\\n\\bottomrule\n\\end{tabular}\\endgroup\n")
    body = []
    for r in rows:
        metric = ", ".join(f"{k} {v:.2f}" for k, v in r["metrics"].items() if v is not None)
        body.append(f"{tex(r['display_name'])} & {r['model_family']} & {r['harness']} & {r['tool_access']} & "
                    f"{tex(metric)} \\\\")
    return ("\\begingroup\\scriptsize\n\\begin{tabular}{@{}p{3.4cm}p{1.8cm}p{2.2cm}p{1.4cm}p{6.2cm}@{}}\n\\toprule\n"
            "Model & Family & Inference text & Tools & Metrics by task/split \\\\\n\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\\endgroup\n")


def table_ablation_skeleton():
    rows = [("Inference text on the held-out gas-engine diagrams", "none / \\texttt{AGENTS.md} / skills / both / both plus memory (base and trained)", "success"),
            ("Training corpus", "ground truth only / plus architecture corpus / plus teacher traces / plus harness and memory text", "success on held-out diagrams"),
            ("Paraphrase variants per row", "1 / 3 / 20 or more", "success on held-out faults"),
            ("Memory retrieval", "none / retrieved notes / random notes", "success and acceptance rate"),
            ("Reward channels", "full / without site / without minimality / without grounding", "success and valid share"),
            ("SFT against GRPO from the base model", "SFT only / SFT + GRPO / GRPO from base", "success"),
            ("Encoder-decoder variant (optional)", "decoder-only / encoder-decoder", "success"),
            ("ThermoQA before and after training", "base / SFT / SFT + GRPO", "accuracy")]
    body = "\n".join(f"{a} & {b} & {c} & \\todo{{pending}} \\\\" for a, b, c in rows)
    return ("\\begingroup\\small\\renewcommand{\\arraystretch}{1.15}\n\\begin{tabular}{@{}p{4.3cm}p{6.1cm}p{2.9cm}l@{}}\n\\toprule\nAblation & Arms & Metric & Result \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\\endgroup\n")


def table_corpus_skeleton():
    tasks = ["Repair", "Ledger repair", "Audit", "Design"]
    rows = [f"{t} & \\todo{{pending}} & \\todo{{pending}} & \\todo{{pending}} & \\todo{{pending}} \\\\" for t in tasks]
    return ("\\begingroup\\small\\renewcommand{\\arraystretch}{1.15}\n\\begin{tabular}{@{}p{2.3cm}p{2.6cm}p{2.6cm}p{3.1cm}p{3.1cm}@{}}\n\\toprule\nTask & Teacher attempts & Accepted (share) & Rejected, kept as notes & Distinct architectures accepted \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\\endgroup\n")


def table_datasets(S, L):
    order = {"repair": ["train", "val", "test_id", "test_ood_fault", "test_ood_topology"], "design": ["train", "val", "test_id", "test_ood_scale", "test_ood_climate", "test_ood_load"],
             "audit": ["train", "val", "test_id", "test_ood_topology"], "ledger": ["train", "val", "test_id", "test_ood_fault", "test_ood_topology"]}
    splits = ["train", "val", "test_id", "test_ood_fault", "test_ood_topology", "test_ood_scale", "test_ood_climate", "test_ood_load"]
    head = " & ".join(s.replace("test_", "").replace("_", " ") for s in splits)
    rows = []
    for t in ("repair", "design", "audit", "ledger"):
        c = S["counts"][t]
        rows.append(f"{t.capitalize()} & " + " & ".join(str(c[s]) if s in c else "--" for s in splits) + f" & {sum(c.values())} \\\\")
    return ("\\begingroup\\small\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}lrrrrrrrrr@{}}\n\\toprule\nTask & " + head + " & total \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\\endgroup\n")


def table_harness(H):
    marks = {"none": (0, 0), "agents": (1, 0), "skills": (0, 1), "agents+skills": (1, 1)}
    rows = [f"{tt(m)} & {'yes' if a else 'no'} & {'yes' if s else 'no'} \\\\" for m, (a, s) in marks.items() if m in H["modes"]]
    sk = "\n".join(f"{tt(x['name'])} & {', '.join(tt(c) for c in x['components']) or 'task-level'} \\\\" for x in H["skills"])
    return ("\\begingroup\\small\\renewcommand{\\arraystretch}{1.12}\n\\begin{tabular}{@{}lll@{}}\n\\toprule\nMode & \\texttt{AGENTS.md} in prompt & Skill bodies in prompt \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\\endgroup\n"), ("\\begingroup\\small\\renewcommand{\\arraystretch}{1.12}\n\\begin{tabular}{@{}ll@{}}\n\\toprule\nSkill & Components covered \\\\\n\\midrule\n" + sk + "\n\\bottomrule\n\\end{tabular}\\endgroup\n")


def table_agents(H):
    rows = []
    for sec, lines in H["agents_md_sections"].items():
        first = lines[0].strip().replace("`", "")
        first = first if len(first) <= 130 else first[:127] + "..."
        rows.append(f"{tex(sec)} & {len(lines)} & {tt(first)} \\\\")
    return ("\\begingroup\\small\\renewcommand{\\arraystretch}{1.12}\\begin{tabular}{@{}p{3.0cm}rp{9.0cm}@{}}\\toprule\nSection & lines excerpted & First excerpted line \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\\endgroup\n")


def table_ledger_faults(L):
    held = set(L["held_out_fault_types"]); sf = L["by_fault_single_fault_rows"]; al = L["by_fault_all_rows"]
    rows = []
    for layer, fl in L["layers"].items():
        for f in fl:
            a, b = sf[f], al[f]
            rows.append(f"{tex(layer)} & {tt(f)}{'$^*$' if f in held else ''} & {a['n']} & {a['closure']:.2f} & {a['negative_ed']:.2f} & {a['either']:.2f} & {b['n']} & {b['either']:.2f} \\\\")
    return ("\\begingroup\\small\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n & & \\multicolumn{4}{c}{single-fault rows} & \\multicolumn{2}{c}{all rows} \\\\\n\\cmidrule(lr){3-6}\\cmidrule(lr){7-8}\n"
            "Layer & Fault & $n$ & closure & neg. $\\ED$ & either & $n$ & either \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\\endgroup\n")


def table_mech(M):
    rows = []
    for r in M["sweep_f"]:
        pass
    eng = {r["chilled_fraction"]: r for r in M["sweep_f"] if r["prime_mover"] == "engine"}; tur = {r["chilled_fraction"]: r for r in M["sweep_f"] if r["prime_mover"] == "turbine"}
    cell = lambda r: (f"{r['fuel_mw']:.1f} MW, stack {r['stack_K']:.0f} K" if r["valid"] and r["stack_K"] else (f"{r['fuel_mw']:.1f} MW" if r["valid"] else "infeasible"))
    for f in sorted(eng):
        if abs(f * 20 - round(f * 20)) < 1e-9 and round(f * 100) % 10 == 0:
            rows.append(f"{f:.1f} & {cell(eng[f])} & {cell(tur[f])} \\\\")
    return ("\\begingroup\\small\\renewcommand{\\arraystretch}{1.1}\n\\begin{tabular}{@{}rll@{}}\n\\toprule\nChilled fraction & Engine + absorption, wet tower & Turbine + absorption, wet tower \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\\endgroup\n")


def write_values():
    GEN.mkdir(exist_ok=True)
    lines = [f"\\expandafter\\def\\csname val@{k}\\endcsname{{{v['tex']}}}" for k, v in sorted(V.items())]
    (GEN / "values.tex").write_text("\n".join(lines) + "\n")
    summary = {"description": "every measured number quoted in the paper: value, TeX form, source asset, command, seed and sample size", "n_keys": len(V), "values": V}
    (ROOT / "assets_summary.json").write_text(json.dumps(summary, indent=1) + "\n")


def derive():
    E, L, S, H, RP, D, AB, A, PL, RS = derive_values()
    ex = jload("excerpts.json")
    for t in ex:
        if isinstance(ex[t], dict):
            ch = ex[t].get("channels")
            if isinstance(ch, str):
                import ast
                ch = ast.literal_eval(ch)
            ex[t]["channels_parsed"] = ch
    M = jload("mechanisms.json")
    MR = jload("model_results.json") if have("model_results.json") else {"rows": []}
    tabs = {"table_components.tex": table_components(E), "table_params.tex": table_params(E), "table_verifiers.tex": table_verifiers(), "table_tasks.tex": table_tasks(S, ex),
            "table_faults.tex": table_faults(E, L, S), "table_baselines.tex": table_baselines(RP, AB, D), "table_constants.tex": table_constants(E), "table_training.tex": table_training(E),
            "table_results.tex": table_results_data(MR), "table_comparison.tex": table_comparison(MR),
            "table_ablation.tex": table_ablation_skeleton(), "table_corpus_status.tex": table_corpus_skeleton(), "table_datasets.tex": table_datasets(S, L),
            "table_ledger_faults.tex": table_ledger_faults(L), "table_mech.tex": table_mech(M)}
    hm, sk = table_harness(H); tabs["table_harness_modes.tex"] = hm; tabs["table_skills.tex"] = sk; tabs["table_agents.tex"] = table_agents(H)
    GEN.mkdir(exist_ok=True)
    for k, v in tabs.items():
        (GEN / k).write_text(v)
    write_values()
    print(f"values: {len(V)} keys; tables: {len(tabs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None, help="path to the thermo-agent repository")
    ap.add_argument("--scratch", default=str(pathlib.Path(tempfile.gettempdir()) / "thermo_paper_scratch"), help="directory for regenerated datasets")
    ap.add_argument("--stage", nargs="*", default=["all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tests-log", default=None, help="output of `python -m pytest -q`, used for the test count")
    ap.add_argument("--repair-limits", default="test_id=200,test_ood_fault=200,test_ood_topology=200")
    a = ap.parse_args()
    stages = a.stage if a.stage != ["all"] else ["environment", "crosscheck", "example", "mechanisms", "corpus", "repair", "audit", "splits", "derive"]
    project = pathlib.Path(a.project).resolve() if a.project else None
    if project:
        sys.path.insert(0, str(project))
        import os; os.environ["PYTHONPATH"] = str(project) + os.pathsep + os.environ.get("PYTHONPATH", "")
    scratch = pathlib.Path(a.scratch).resolve(); scratch.mkdir(parents=True, exist_ok=True)
    limits = {k: int(v) for k, v in (x.split("=") for x in a.repair_limits.split(","))}
    todo = {"environment": ("environment.json", lambda: study_environment(project, a.tests_log)), "crosscheck": ("crosscheck.json", study_crosscheck), "example": ("example_plant.json", study_example),
            "mechanisms": ("mechanisms.json", study_mechanisms), "corpus": ("corpus.json", lambda: study_corpus(project, scratch)), "repair": ("repair_baselines.json", lambda: study_repair(project, scratch, limits)), "audit": ("audit_baselines.json", lambda: study_audit(project, scratch)),
            "splits": ("splits.json", lambda: study_splits(project, scratch))}
    for s in stages:
        if s == "derive":
            derive(); continue
        name, fn = todo[s]
        if have(name) and not a.force and s != "environment":
            print(f"{s}: assets/{name} exists (use --force to recompute)"); continue
        t = time.time(); fn(); print(f"{s}: done in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
