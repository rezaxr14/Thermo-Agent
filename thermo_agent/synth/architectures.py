"""Architecture catalog: labelled plant diagrams for training and evaluation.

The builders in `synth/topologies.py` already cover a matrix of plants, but `repair_dataset` only samples five
named combinations. This module enumerates the matrix explicitly, so data coverage can be measured per
architecture and out-of-distribution splits can hold out whole diagrams rather than single parameters.

A diagram is (prime mover, cooling, heat rejection, ORC bottoming, district heat) plus one steam/HRSG plant:
    turbine | engine
    absorption | electric
    ext (ideal boundary) | dry cooler | wet tower
    no ORC | ORC on the remaining exhaust (R245fa / cyclopentane / toluene)
    no district heat | exhaust heat exchanger | heat pump
That is 64 combinations (heat pump needs dry/wet rejection, so 8 are skipped) plus `cgam_steam_orc`, the
gas-turbine + air-preheater + HRSG (steam export) + ORC plant.

    python -m thermo_agent.synth.architectures --list
    python -m thermo_agent.synth.architectures OUT_DIR -n 400 --seed 0 --workers 8   # writes repair_<split>.jsonl
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import pathlib
import random

from ..core.schema import RepairRecord, Site, SCHEMA_VERSION
from ..core.solve import solve
from ..verify.verifiers import run_all
from .faults import INJECTORS, NotApplicable
from .splits import is_ood_topology
from .meta import write_meta
from .sft_data import to_row
from .topologies import cgam_orc, dc_chp, size_generation

ORC_FLUIDS = ("R245fa", "Cyclopentane", "Toluene")

# Evaporator pressure windows per fluid, chosen so the bubble point sits well below the exhaust temperature
# that reaches the ORC (roughly 450-530 K). Above these windows the pinch check fails at sampling time.
ORC_P_HIGH = {"R245fa": (8e5, 14e5), "Cyclopentane": (5e5, 11e5), "Toluene": (3e5, 8e5)}


def catalog() -> list[dict]:
    """Every labelled diagram this repo can build from the existing component set."""
    out = []
    for prime, cooling, rejection, orc, heat in itertools.product(
        ("turbine", "engine"), ("absorption", "electric"), ("ext", "dry", "wet"),
        (False, True), ("none", "exhaust", "heat_pump"),
    ):
        if heat == "heat_pump" and rejection == "ext":
            continue                       # a heat pump has no rejection loop to draw from
        name = f"{prime}-{cooling}-{rejection}"
        if orc:
            name += "-orc"
        if heat != "none":
            name += f"-dh_{heat}"
        out.append({"name": name, "kind": "dc", "prime": prime, "cooling": cooling,
                    "rejection": rejection, "orc": orc, "heat": heat})
    out.append({"name": "cgam_steam_orc", "kind": "cgam", "prime": "turbine", "cooling": "none",
                "rejection": "ext", "orc": True, "heat": "steam"})
    return out


ARCHITECTURES = catalog()

# Expert-scale cycles, ADDITIVE to the 65 small diagrams above. They are deliberately not
# part of `catalog()` so regenerating the old corpora never silently changes them.
BIG_ARCHITECTURES = [
    {"name": "big_cchp_steam2p_reheat_hp", "kind": "big", "family": "steam2p_reheat_hp"},
]


def all_architectures() -> list[dict]:
    """Small + big diagrams, for builds that explicitly opt into the expert-scale cycles."""
    return ARCHITECTURES + BIG_ARCHITECTURES


def by_name(name: str) -> dict:
    for a in all_architectures():
        if a["name"] == name:
            return a
    raise KeyError(f"unknown architecture {name!r}")
    for a in ARCHITECTURES:
        if a["name"] == name:
            return a
    raise KeyError(f"unknown architecture {name!r}")


def _build_dc(arch: dict, rng: random.Random, p_scale: float = 1.0):
    P_it = rng.uniform(8e6, 55e6)
    T_sup = rng.uniform(285, 295)
    T_hw = rng.uniform(363, 373)
    kw = dict(P_it=P_it, chilled_fraction=rng.uniform(0.3, 0.9), cooling=arch["cooling"],
              rejection=arch["rejection"], T_amb=rng.uniform(288, 305), RH=rng.uniform(0.3, 0.8),
              T_supply=T_sup, T_return=T_sup + rng.uniform(8, 12), T_hw=T_hw,
              T_gen_out=T_hw - rng.uniform(8, 15), dT_rej=rng.uniform(6, 10),
              eps_abs=rng.uniform(0.22, 0.32), eps_ec=rng.uniform(0.4, 0.6))
    if arch["prime"] == "engine":
        kw.update(prime_mover="engine", eta_el=rng.uniform(0.38, 0.46), f_loss=rng.uniform(0.10, 0.20),
                  air_excess=rng.uniform(1.5, 2.2))
    else:
        kw.update(PR=rng.uniform(12, 25), eta_c=rng.uniform(0.82, 0.90), eta_t=rng.uniform(0.84, 0.91),
                  T4=rng.uniform(1400, 1600))
    if arch["orc"]:
        fluid = rng.choice(ORC_FLUIDS)
        lo, hi = ORC_P_HIGH[fluid]
        kw.update(orc=True, orc_fluid=fluid, orc_frac=rng.uniform(0.2, 0.9), P_orc_high=rng.uniform(lo, hi) * p_scale)
    if arch["heat"] != "none":
        kw.update(heat_demand_w=P_it * rng.uniform(0.05, 0.25), heat_source=arch["heat"])
    return size_generation(dc_chp, name=arch["name"], **kw)


def _build_big(arch: dict, rng: random.Random):
    """Expert-scale cycle, sized to the sampled IT load; steam flows scale with air flow."""
    from .big_topologies import BIG_BUILDERS, STEAM_PER_AIR_HP, STEAM_PER_AIR_LP
    from .topologies import size_generation

    builder = BIG_BUILDERS.get(arch.get("family")) or BIG_BUILDERS.get(arch["name"])
    if builder is None:
        raise KeyError(f"no big builder for {arch['family']}")
    hp_ratio, lp_ratio = STEAM_PER_AIR_HP, STEAM_PER_AIR_LP
    P_it = rng.uniform(60e6, 120e6)
    common = dict(
        P_it=P_it, chilled_fraction=rng.uniform(0.35, 0.75),
        T_reheat=rng.uniform(460, 530), P_hp=rng.uniform(30e5, 55e5),
        P_lp=rng.uniform(6e5, 12e5), T_cond=rng.uniform(308, 318),
        T_amb=rng.uniform(288, 305),
        name=arch["name"],
    )

    def sized(n_air: float, **kw):
        return builder(n_air=n_air, m_hp=n_air * hp_ratio, m_lp=n_air * lp_ratio, **kw)

    return size_generation(sized, **common)


def _build_cgam(arch: dict, rng: random.Random):
    return cgam_orc(n_air=rng.uniform(1500, 4500), PR=rng.uniform(8, 16), eta_c=rng.uniform(0.82, 0.90),
                    eta_t=rng.uniform(0.82, 0.90), T3=rng.uniform(750, 900), T4=rng.uniform(1400, 1600),
                    m_steam=rng.uniform(6, 18), P_steam=rng.uniform(10e5, 30e5),
                    P_orc_high=rng.uniform(8e5, 14e5), T_cond=rng.uniform(308, 320),
                    n_orc=rng.uniform(150, 350), eta_pump=rng.uniform(0.7, 0.8),
                    eta_orct=rng.uniform(0.78, 0.88), dT_cw=rng.uniform(4, 8), name=arch["name"])


def sample_plant(arch: dict, rng: random.Random, tries: int = 6):
    """A feasible, verifier-clean plant for this diagram, or RuntimeError.

    ORC pinch failures are the common rejection, and they are a pressure problem, so each retry lowers the
    evaporator pressure a little before redrawing the other parameters.
    """
    last = None
    p_scale = 1.0
    for _ in range(tries):
        try:
            if arch["kind"] == "big":
                p = _build_big(arch, rng)
            else:
                p = _build_cgam(arch, rng) if arch["kind"] == "cgam" else _build_dc(arch, rng, p_scale)
        except (RuntimeError, ValueError) as exc:
            last = exc
            p_scale *= 0.75
            continue
        res = solve(p)
        rep = run_all(p, res)
        if res.ok and rep.valid:
            return p, res, rep
        last = RuntimeError(f"{arch['name']}: solver ok={res.ok}, verifier valid={rep.valid}")
        p_scale *= 0.75
    raise RuntimeError(f"could not sample {arch['name']}: {last}")


def make_records(n: int, seed: int = 0, architectures: list[dict] | None = None):
    """Repair records over the architecture catalog, cycling diagrams so coverage is even."""
    rng = random.Random(seed)
    archs = architectures or ARCHITECTURES
    kinds = list(INJECTORS)
    for i in range(n):
        arch = archs[i % len(archs)]
        try:
            plant, ores, _ = sample_plant(arch, rng)
        except RuntimeError:
            continue                       # this draw did not size; the next record uses the next diagram
        site = None
        if ores.kpis.get("P_it_w"):
            site = Site(it_load_w=ores.kpis["P_it_w"], cooling_load_w=0.99 * ores.kpis["Q_cooling_w"],
                        grid_import_allowed=False)
        fault = None
        for step in range(len(kinds)):
            try:
                broken, fault = INJECTORS[kinds[(i + step) % len(kinds)]](plant, rng)
                break
            except NotApplicable:
                continue
        if fault is None:
            continue
        bres = solve(broken)
        brep = run_all(broken, bres, site)
        detected = (not bres.ok) or (not brep.valid)
        yield RepairRecord(
            id=f"arch-{seed}-{i:05d}", plant_original=plant, plant_broken=broken, site=site, faults=[fault],
            solve_broken=bres, verifier_report_broken={**brep.as_dict(), "detected": detected},
            provenance={"seed": str(seed), "schema": SCHEMA_VERSION, "architecture": arch["name"],
                        "topology": plant.name})


def split_for(rec: RepairRecord, architecture: str) -> str:
    """Hold out diagram *combinations*, not families: the steam/HRSG export plant and engine-with-ideal-rejection."""
    if is_ood_topology(architecture):
        return "test_ood_topology"
    if rec.faults[0].type in ("sign_flip", "swapped_ports"):
        return "test_ood_fault"
    h = int(hashlib.md5(rec.id.encode()).hexdigest(), 16) % 10
    return "val" if h == 0 else ("test_id" if h == 1 else "train")


CHUNK = 24


def _chunk(args):
    seed, n, names = args
    archs = [by_name(x) for x in names]
    rows = []
    for rec in make_records(n, seed, archs):
        row = to_row(rec)
        row["split"] = split_for(rec, row["topology"])
        if row["detected"]:
            rows.append(row)
    return rows


def build(out_dir, n: int, seed: int = 0, workers: int = 1, names: list[str] | None = None) -> dict:
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    names = names or [a["name"] for a in ARCHITECTURES]
    # rotate the catalog per chunk so every chunk covers different diagrams
    jobs = []
    for c in range((n + CHUNK - 1) // CHUNK):
        take = min(CHUNK, n - c * CHUNK)
        rotated = names[(c * CHUNK) % len(names):] + names[:(c * CHUNK) % len(names)]
        jobs.append((seed * 100000 + c, take, rotated))
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            chunks = pool.map(_chunk, jobs)
    else:
        chunks = [_chunk(j) for j in jobs]
    counts = collections.Counter()
    files = {}
    for rows in chunks:
        for row in rows:
            sp = row["split"]
            if sp not in files:
                files[sp] = open(out / f"repair_{sp}.jsonl", "a")
            files[sp].write(json.dumps(row) + "\n")
            counts[sp] += 1
    for f in files.values():
        f.close()
    write_meta(out, "repair_arch", n=n, seed=seed, workers=workers, architectures=len(names), counts=dict(counts))
    return dict(counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("-n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="architecture names (default: the whole catalog)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.out:
        for arch in ARCHITECTURES:
            print(arch["name"])
        print(f"{len(ARCHITECTURES)} architectures")
    else:
        print(build(a.out, a.n, a.seed, a.workers, a.only))
