"""Audit dataset. python -m thermo_agent.audit.data OUT_DIR -n 200 --seed 0 [--workers 4] [--architectures] -> audit_<split>.jsonl

`--architectures` cycles the 65-diagram catalog in `synth/architectures.py` (ORC bottoming, gas engines, the
steam/HRSG plant, district heat) instead of the five-plant `MIX`, so audit trains on the same diagram families as
repair. Only the held-out combinations (`synth/splits.py`) go to `test_ood_topology`.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, pathlib, random
from ..synth.meta import write_meta
from ..synth.splits import is_ood_topology
from ..synth import architectures
from ..core.schema import Site
from ..synth.repair_dataset import sample_plant, MIX
from .prompts import audit_messages
from .truth import truth
from . import changes as ch


def site_of(plant, res) -> Site | None:
    k = res.kpis
    if not k.get("P_it_w"): return None
    return Site(it_load_w=k["P_it_w"], cooling_load_w=0.99 * k["Q_cooling_w"], heat_demand_w=0.99 * k.get("Q_heat_w", 0.0), grid_import_allowed=False)


def split_of(plant, sid: str) -> str:
    topo = plant.name.replace("_sampled", "")
    if is_ood_topology(topo): return "test_ood_topology"
    h = int(hashlib.md5(sid.encode()).hexdigest(), 16) % 10
    return "val" if h == 0 else ("test_id" if h == 1 else "train")


def expert_completion(plant, T) -> str:
    top = T["ranking"][:4]; fuel = T["fuel_w"] / 1e6
    ed = ", ".join(f"{c} ({v/1e6:.3g} MW)" for c, v in top[:3])
    edit, delta = T["best"]
    analysis = f"Exergy is destroyed mostly in {ed}, with a fuel input of {fuel:.3g} MW. The change {edit} should lower fuel input for the same services."
    return (f"<analysis>{analysis}</analysis>\n<ranking>{' > '.join(c for c, _ in top)}</ranking>\n<change>{edit}</change>\n"
            f"<prediction>fuel_change_mw = {delta/1e6:.4g}</prediction>")


def make_row(sid: str, plant) -> dict | None:
    from ..core.solve import solve
    res = solve(plant); site = site_of(plant, res); T = truth(plant, site)
    if T is None or T["best"] is None or T["best_gain_w"] <= 0: return None
    return {"id": sid, "split": split_of(plant, sid), "topology": plant.name.replace("_sampled", ""),
            "prompt": audit_messages(plant, site, T["result"]),
            "completion": [{"role": "assistant", "content": expert_completion(plant, T)}],
            "plant": plant.model_dump_json(by_alias=True), "site": site.model_dump_json() if site else "",
            "ed_json": json.dumps({c: v for c, v in T["ranking"]}), "fuel_w": T["fuel_w"], "net_w": T["net_w"],
            "best_gain_w": T["best_gain_w"], "n_candidates": len(T["table"])}


def _job(args):
    sid, seed, topo = args
    rng = random.Random(f"{seed}-{sid}")
    plant, _, _ = sample_plant(rng, topology=topo, rich=True)
    return make_row(sid, plant)


def _job_arch(args):
    """One audit row on a catalog diagram: the plant is already solved and verifier-clean."""
    sid, seed, name = args
    rng = random.Random(f"{seed}-{sid}")
    try:
        plant, _, _ = architectures.sample_plant(architectures.by_name(name), rng)
    except RuntimeError:
        return None
    return make_row(sid, plant)


def build(out_dir, n: int, seed: int = 0, workers: int = 1, arch: bool = False,
          names: list[str] | None = None) -> dict:
    catalog: list[str] = []
    if arch:
        catalog = names or [a["name"] for a in architectures.ARCHITECTURES]
        jobs = [(f"aud-arch-{seed}-{i:05d}", seed, catalog[i % len(catalog)]) for i in range(n)]
        job = _job_arch
    else:
        jobs = [(f"aud-{seed}-{i:05d}", seed, MIX[i % len(MIX)]) for i in range(n)]
        job = _job
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool: rows = pool.map(job, jobs, chunksize=4)
    else: rows = [job(j) for j in jobs]
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True); counts = collections.Counter(); files = {}
    for r in rows:
        if r is None: continue
        sp = r["split"]
        if sp not in files: files[sp] = open(out / f"audit_{sp}.jsonl", "w")
        files[sp].write(json.dumps(r) + "\n"); counts[sp] += 1
    for f in files.values(): f.close()
    write_meta(out, "audit_arch" if arch else "audit", n=n, seed=seed, workers=workers,
               architectures=len(catalog) if arch else len(MIX), counts=dict(counts))
    return dict(counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--architectures", action="store_true", help="cycle the 65-diagram catalog instead of MIX")
    ap.add_argument("--only", nargs="*", help="architecture names (with --architectures)")
    a = ap.parse_args(); print(build(a.out, a.n, a.seed, a.workers, a.architectures, a.only))
