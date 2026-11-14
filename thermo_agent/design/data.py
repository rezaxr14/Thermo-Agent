"""Design dataset: sampled sites, reference designs (enumeration + polish), expert completions, structural splits.
    python -m thermo_agent.design.data OUT_DIR -n 600 --seed 0 [--workers 4]  -> design_<split>.jsonl"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, pathlib, random
from ..synth.meta import write_meta
from ..core.schema import Site, Ambient
from .spec import format_design, default_spec, DISCRETE
from .build import evaluate_design
from .prompts import design_messages
from .search import reference_design

_WHY = {"too hot": "its exhaust is too hot for that ORC fluid at this site", "V4a_no_temperature_cross": "its exhaust cannot supply the chilled-water duty at this site","V8e_water_limit": "it exceeds the site water limit", "V8b_electricity_balance": "it cannot cover the site electricity",
        "sizing": "its exhaust cannot supply the required heat duty", "E_TEMP_CROSS": "its exhaust cannot supply the required heat duty"}


def sample_site(rng: random.Random) -> Site:
    P = math.exp(rng.uniform(math.log(5e6), math.log(60e6))); T_sup = rng.uniform(288, 296)
    site = Site(it_load_w=P, cooling_load_w=P * rng.uniform(0.3, 0.95), grid_import_allowed=False,
                ambient=Ambient(T0=rng.uniform(283, 310), RH=rng.uniform(0.2, 0.9)), t_supply_k=T_sup, t_return_k=T_sup + rng.uniform(7, 12))
    if rng.random() < 0.4: site = site.model_copy(update={"heat_demand_w": P * rng.uniform(0.05, 0.3)})
    u = rng.random()
    if u > 0.6:                                              # water-limited sites: none at all, or half of a tower's use
        probe = evaluate_design(site, default_spec("turbine", "electric", "wet"))
        if probe.result is not None and probe.result.ok:
            w = probe.result.kpis["water_use_kg_s"]; site = site.model_copy(update={"max_water_kg_s": 0.0 if u > 0.8 else 0.5 * w})
    return site


def assign_split(site: Site, sid: str) -> str:
    if site.it_load_w >= 45e6: return "test_ood_scale"
    if site.ambient.T0 >= 305.0: return "test_ood_climate"
    if site.cooling_load_w / site.it_load_w >= 0.85: return "test_ood_load"
    h = int(hashlib.md5(sid.encode()).hexdigest(), 16) % 10
    return "val" if h == 0 else ("test_id" if h == 1 else "train")


def _why(reason: str) -> str:
    return next((v for k, v in _WHY.items() if k in reason), "it is not valid at this site")


def rationale(spec: dict, ev, table) -> str:
    """Grounded in the chosen design's own solved numbers; comparisons are qualitative or in whole percent words."""
    k = ev.result.kpis; stack = next(s for s in ev.plant.streams if s.dst == "ext:stack").id
    T = next(p.T for p in ev.result.states if p.stream == stack)
    notes = []
    keys = [k for k in DISCRETE if k in spec] + (["heat_source"] if "heat_source" in spec else [])
    for key in keys:
        alts = [(f, r, s) for s, f, r in table if all(s.get(o) == spec[o] for o in keys if o != key) and s.get(key) != spec[key]]
        if not alts: continue
        f, r, s = alts[0]
        label = f"{s[key]} {key.replace('_', ' ')}"
        if f is None: notes.append(f"{label} is rejected because {_why(r)}")
        elif f > ev.fuel_w: notes.append(f"{label} would burn about {round(100 * (f / ev.fuel_w - 1))} percent more fuel")
    head = f"{spec['prime_mover']} with {spec['cooling']} cooling and a {spec['rejection']} heat-rejection loop"
    head += (f", {spec['bottoming'].replace('orc_', 'an ORC on ')} bottoming" if spec["bottoming"] != "none" else ", no bottoming cycle")
    head += (f" and district heat from {spec['heat_source'].replace('_', ' ')}." if "heat_source" in spec else ".")
    tail = f"Expected fuel input {k['Q_fuel_w']/1e6:.1f} MW, net electricity {k['W_net_w']/1e6:.1f} MW and stack temperature {T:.0f} K."
    return " ".join([head, ("; ".join(notes) + ".") if notes else "", tail]).replace("  ", " ")


def make_row(sid: str, site: Site, max_evals: int = 60) -> dict | None:
    ref = reference_design(site, max_evals=max_evals)
    if ref is None: return None
    ev = evaluate_design(site, ref["spec"])
    comp = f"<rationale>{rationale(ref['spec'], ev, ref['table'])}</rationale>\n{format_design(ref['spec'])}"
    return {"id": sid, "split": assign_split(site, sid), "prompt": design_messages(site), "completion": [{"role": "assistant", "content": comp}],
            "site": site.model_dump_json(), "ref_fuel_w": ref["fuel_w"], "ref_spec": json.dumps(ref["spec"]),
            "n_valid": sum(1 for _, f, _ in ref["table"] if f is not None), "ref_evals": ref["evals"]}


def _job(args):
    sid, site_js, me = args
    return make_row(sid, Site.model_validate_json(site_js), me)


def build(out_dir, n: int, seed: int = 0, workers: int = 1, max_evals: int = 60) -> dict:
    rng = random.Random(seed); sites = [(f"des-{seed}-{i:05d}", sample_site(rng).model_dump_json(), max_evals) for i in range(n)]
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool: rows = pool.map(_job, sites, chunksize=4)
    else: rows = [_job(s) for s in sites]
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True); counts = collections.Counter(); files = {}
    for r in rows:
        if r is None: continue
        sp = r["split"]
        if sp not in files: files[sp] = open(out / f"design_{sp}.jsonl", "w")
        files[sp].write(json.dumps(r) + "\n"); counts[sp] += 1
    for f in files.values(): f.close()
    write_meta(out, "design", n=n, seed=seed, workers=workers, max_evals=max_evals, counts=dict(counts))
    return dict(counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("-n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-evals", type=int, default=60, help="polish budget for reference designs; use >= 60 for training data")
    a = ap.parse_args(); print(build(a.out, a.n, a.seed, a.workers, a.max_evals))
