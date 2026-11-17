"""Ledger-repair dataset. python -m thermo_agent.ledger.data OUT_DIR -n 1500 --seed 0 [--workers 4] [--architectures] -> ledger_<split>.jsonl

`--architectures` cycles the 65-diagram catalog (ORC bottoming, gas engines, the steam/HRSG plant, district heat)
instead of the five-plant `MIX`; only the held-out combinations go to `test_ood_topology` (`synth/splits.py`).
"""
from __future__ import annotations
import argparse, collections, hashlib, json, pathlib, random
from ..synth.meta import write_meta
from ..synth.splits import is_ood_topology, topology_label
from ..synth import architectures
from ..synth.repair_dataset import sample_plant, MIX
from ..core.solve import solve
from . import program as P, faults as F
from .prompts import ledger_messages, numbers_shown, _g


def diagnose(rec: dict, data: dict, result: dict) -> str:
    d, k, t = rec["detail"], rec["key"], rec["type"]
    if k == "T0": return f"T0 = {rec['buggy']} is not the ambient temperature in kelvin shown under AMBIENT; every stream exergy line uses T0."
    if k.startswith("E["):
        x = d["x"]; why = {"missing_m": f"E[{x}] lacks the flow factor m[{x}] that the other stream lines have",
                           "missing_ech": f"E[{x}] lacks the chemical exergy term ech[{x}] that the other stream lines have",
                           "unit_slip": f"E[{x}] divides by 1e3 where the other stream lines divide by 1e6",
                           "sign_T0": f"E[{x}] adds T0*s[{x}] where the other stream lines subtract it",
                           "t0_literal": f"E[{x}] uses the number 25 where the other stream lines use T0",
                           "wrong_ref": f"E[{x}] uses g0[{d.get('y')}], the reference of another stream, instead of g0[{x}]"}[t]
        return f"{why}; it evaluates to {_g(result['E'][x])} MW."
    c = k[3:-1]; v = _g(result["ED"][c])
    why = {"sign_flip": f"the term for {d.get('ident')} has the wrong sign in ED[{c}]",
           "dropped_term": f"the term for stream {d.get('ident')} is missing from ED[{c}]",
           "dropped_work": f"the work term W[{c}] is missing from ED[{c}]",
           "unit_slip": f"the term for {d.get('ident')} in ED[{c}] is multiplied by {d.get('factor', 1):g}, which no balance has",
           "wrong_factor": f"the term for {d.get('ident')} in ED[{c}] is multiplied by {d.get('factor', 1):g}, which no balance has",
           "wrong_stream": f"ED[{c}] uses E[{d.get('wrong')}] but {c} is connected to E[{d.get('right')}]"}[t]
    return f"{why}; ED[{c}] evaluates to {v} MW, but destruction is inlet exergy minus outlet exergy minus work."


def expert_completion(recs: list[dict], data: dict, result: dict) -> str:
    diag = " ".join(diagnose(r, data, result) for r in recs)
    return f"<diagnosis>{diag}</diagnosis>\n<fix>\n" + "\n".join(f"FIX {r['key']} = {r['correct']}" for r in recs) + "\n</fix>"


def split_of(topology: str, types: set, sid: str) -> str:
    if is_ood_topology(topology): return "test_ood_topology"
    if types & F.HELD_OUT: return "test_ood_fault"
    h = int(hashlib.md5(sid.encode()).hexdigest(), 16) % 10
    return "val" if h == 0 else ("test_id" if h == 1 else "train")


def make_row(sid: str, plant, rng: random.Random) -> dict | None:
    res = solve(plant)
    if not res.ok: return None
    data = P.build(plant, res)
    for _ in range(5):
        inj = F.inject(rng, data, 2 if rng.random() < 0.3 else 1)
        if inj: break
    else: return None
    lines, recs = inj; result = P.run(lines, data)
    if result["errors"]: return None
    topo = topology_label(plant)
    ledger = {"data": data, "lines": lines, "faulty": [r["key"] for r in recs], "numbers": numbers_shown(data, result)}
    return {"id": sid, "split": split_of(topo, {r["type"] for r in recs}, sid), "topology": topo, "fault_types": [r["type"] for r in recs],
            "prompt": ledger_messages(data, lines, result), "completion": [{"role": "assistant", "content": expert_completion(recs, data, result)}],
            "ledger": json.dumps(ledger), "min_edits": len(recs)}


def _job(args):
    sid, seed, topo = args
    rng = random.Random(f"{seed}-{sid}")
    try: plant, _, _ = sample_plant(rng, topology=topo, rich=True)
    except RuntimeError: return None
    return make_row(sid, plant, rng)


def _job_arch(args):
    """One ledger row on a catalog diagram: the plant is already solved and verifier-clean."""
    sid, seed, name = args
    rng = random.Random(f"{seed}-{sid}")
    try: plant, _, _ = architectures.sample_plant(architectures.by_name(name), rng)
    except RuntimeError: return None
    return make_row(sid, plant, rng)


def build(out_dir, n: int, seed: int = 0, workers: int = 1, arch: bool = False,
          names: list[str] | None = None) -> dict:
    catalog: list[str] = []
    if arch:
        catalog = names or [a["name"] for a in architectures.ARCHITECTURES]
        jobs = [(f"led-arch-{seed}-{i:05d}", seed, catalog[i % len(catalog)]) for i in range(n)]
        job = _job_arch
    else:
        jobs = [(f"led-{seed}-{i:05d}", seed, MIX[i % len(MIX)]) for i in range(n)]
        job = _job
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool: rows = pool.map(job, jobs, chunksize=8)
    else: rows = [job(j) for j in jobs]
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True); counts = collections.Counter(); files = {}
    for r in rows:
        if r is None: continue
        sp = r["split"]
        if sp not in files: files[sp] = open(out / f"ledger_{sp}.jsonl", "w")
        files[sp].write(json.dumps(r) + "\n"); counts[sp] += 1
    for f in files.values(): f.close()
    write_meta(out, "ledger_arch" if arch else "ledger", n=n, seed=seed, workers=workers,
               architectures=len(catalog) if arch else len(MIX), counts=dict(counts))
    return dict(counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("-n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--architectures", action="store_true", help="cycle the 65-diagram catalog instead of MIX")
    ap.add_argument("--only", nargs="*", help="architecture names (with --architectures)")
    a = ap.parse_args(); print(build(a.out, a.n, a.seed, a.workers, a.architectures, a.only))
