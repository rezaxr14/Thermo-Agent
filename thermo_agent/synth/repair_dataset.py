"""Generate Repair records: sample a feasible plant, inject a fault, solve and verify the broken plant."""
from __future__ import annotations
import json, random, hashlib, importlib.metadata as md
from pathlib import Path
from ..core.schema import RepairRecord, Site, SCHEMA_VERSION
from ..core.solve import solve
from ..verify.verifiers import run_all
from .topologies import cgam_orc, dc_chp, size_generation
from .faults import INJECTORS, NotApplicable


def _sample_cgam(rng):
    return cgam_orc(n_air=rng.uniform(1500, 4500), PR=rng.uniform(8, 16), eta_c=rng.uniform(0.82, 0.90),
                    eta_t=rng.uniform(0.82, 0.90), T3=rng.uniform(750, 900), T4=rng.uniform(1400, 1600),
                    m_steam=rng.uniform(6, 18), P_steam=rng.uniform(10e5, 30e5), P_orc_high=rng.uniform(8e5, 14e5),
                    T_cond=rng.uniform(308, 320), n_orc=rng.uniform(150, 350), eta_pump=rng.uniform(0.7, 0.8),
                    eta_orct=rng.uniform(0.78, 0.88), dT_cw=rng.uniform(4, 8), name="cgam_orc_sampled")


def _sample_dc(rng, cooling, engine=False, rich=False):
    T_sup = rng.uniform(285, 295); T_hw = rng.uniform(363, 373)
    kw = dict(P_it=rng.uniform(5e6, 60e6), chilled_fraction=rng.uniform(0.3, 0.9), cooling=cooling,
              rejection=rng.choice(["ext", "dry", "wet"]), T_amb=rng.uniform(288, 305), RH=rng.uniform(0.3, 0.8),
              T_supply=T_sup, T_return=T_sup + rng.uniform(8, 12), T_hw=T_hw, T_gen_out=T_hw - rng.uniform(8, 15),
              dT_rej=rng.uniform(6, 10), eps_abs=rng.uniform(0.22, 0.32), eps_ec=rng.uniform(0.4, 0.6))
    if rich:                                       # ORC bottoming and district heat on some plants
        if rng.random() < 0.5: kw.update(orc=True, orc_fluid=rng.choice(["R245fa", "Cyclopentane", "Toluene"]), orc_frac=rng.uniform(0.2, 0.9), P_orc_high=rng.uniform(10e5, 20e5))
        if rng.random() < 0.4 and kw["rejection"] != "ext": kw.update(heat_demand_w=kw["P_it"] * rng.uniform(0.05, 0.25), heat_source=rng.choice(["exhaust", "heat_pump"]))
    if engine:
        kw.update(prime_mover="engine", eta_el=rng.uniform(0.38, 0.46), f_loss=rng.uniform(0.10, 0.20), air_excess=rng.uniform(1.5, 2.2))
    else:
        kw.update(PR=rng.uniform(12, 25), eta_c=rng.uniform(0.82, 0.90), eta_t=rng.uniform(0.84, 0.91), T4=rng.uniform(1400, 1600))
    return size_generation(dc_chp, name=f"dc_{'engine_' if engine else ''}{cooling}_sampled", **kw)


TOPOLOGIES = ["cgam_orc", "dc_absorption", "dc_electric", "dc_engine_absorption", "dc_engine_electric"]
# generation mix: training topologies (turbine plants) are drawn three times as often as the held-out ones
MIX = ["dc_absorption", "dc_electric"] * 3 + ["cgam_orc", "dc_engine_absorption", "dc_engine_electric"]


def sample_plant(rng: random.Random, tries: int = 50, topology: str | None = None, rich: bool = False):
    for _ in range(tries):
        kind = topology or rng.choice(TOPOLOGIES)
        try:
            p = _sample_cgam(rng) if kind == "cgam_orc" else _sample_dc(rng, kind.split("_")[-1], engine="engine" in kind, rich=rich)
        except RuntimeError:                       # sizing run failed: infeasible draw
            continue
        res = solve(p); rep = run_all(p, res)
        if res.ok and rep.valid: return p, res, rep
    raise RuntimeError("could not sample a feasible plant")


def make_records(n: int, seed: int = 0):
    rng = random.Random(seed); kinds = list(INJECTORS); topos = MIX
    for i in range(n):                              # every (topology, fault type) pair appears equally often
        orig, ores, _ = sample_plant(rng, topology=topos[(i // len(kinds)) % len(topos)])
        site = None
        if ores.kpis.get("P_it_w"):
            site = Site(it_load_w=ores.kpis["P_it_w"], cooling_load_w=0.99 * ores.kpis["Q_cooling_w"], grid_import_allowed=False)
        for step in range(len(kinds)):               # fall back to the next fault type if this one cannot be injected here
            try:
                broken, fault = INJECTORS[kinds[(i + step) % len(kinds)]](orig, rng); break
            except NotApplicable:
                continue
        res = solve(broken); rep = run_all(broken, res, site)
        detected = (not res.ok) or (not rep.valid)
        yield RepairRecord(
            id=f"rep-{seed}-{i:05d}", plant_original=orig, plant_broken=broken, site=site, faults=[fault], solve_broken=res,
            verifier_report_broken={**rep.as_dict(), "detected": detected},
            provenance={"seed": str(seed), "schema": SCHEMA_VERSION, "coolprop": md.version("CoolProp")})


def write_jsonl(path: str | Path, n: int, seed: int = 0) -> dict:
    stats = {"n": 0, "detected": 0, "solve_error": 0}
    with open(path, "w") as f:
        for r in make_records(n, seed):
            f.write(r.model_dump_json(by_alias=True) + "\n")
            stats["n"] += 1; stats["detected"] += int(r.verifier_report_broken["detected"]); stats["solve_error"] += int(not r.solve_broken.ok)
    return stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("-n", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); print(write_jsonl(a.out, a.n, a.seed))
