"""python -m thermo_agent.synth.build_repair_data OUT_DIR -n 3000 --seed 0 [--workers 4] : writes repair_<split>.jsonl
Records are generated in chunks of 90 (two full cycles of the topology x fault mix), each with its own seed, so workers are independent."""
from __future__ import annotations
import argparse, collections, json, pathlib
from .meta import write_meta
from .repair_dataset import make_records
from .sft_data import to_row

CHUNK = 90


def _chunk(args):
    seed, n = args
    rows = []
    for rec in make_records(n, seed):
        row = to_row(rec)
        if row["detected"]: rows.append(row)               # undetected faults carry no verifier signal
    return rows


def build(out_dir, n: int, seed: int = 0, workers: int = 1) -> dict:
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    jobs = [(seed * 100000 + c, min(CHUNK, n - c * CHUNK)) for c in range((n + CHUNK - 1) // CHUNK)]
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool: chunks = pool.map(_chunk, jobs)
    else: chunks = [_chunk(j) for j in jobs]
    files, counts = {}, collections.Counter()
    for rows in chunks:
        for row in rows:
            sp = row["split"]
            if sp not in files: files[sp] = open(out / f"repair_{sp}.jsonl", "w")
            files[sp].write(json.dumps(row) + "\n"); counts[sp] += 1
    for f in files.values(): f.close()
    write_meta(out, "repair", n=n, seed=seed, workers=workers, chunk=CHUNK, counts=dict(counts))
    return dict(counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("-n", type=int, default=3000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(); print(build(a.out, a.n, a.seed, a.workers))
