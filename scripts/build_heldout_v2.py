"""Build ``data/heldout_v2`` — the frozen test set for the paper (DECISIONS F1 / plan 0.6).

Additive-only: this script never writes to ``data/heldout`` (the dev set).  It composes the
same CPU generators that produced the dev set, with a fresh seed, into test splits only.
Every row is stamped with its ``gen_seed`` and, after generation, with a **real** difficulty
``tier`` re-derived from the documented per-task size measure in
``thermo_agent/synth/tiers.py`` (Reza directive §4.2).  Generation is target-driven: short
cells get top-up batches from a real generator, never synthetic rows.  If a cell still cannot
reach the target it is merged with its documented sibling cell.

Usage (from the repo root):
    .venv/Scripts/python.exe scripts/build_heldout_v2.py \
        --seed 20261119 --out data/heldout_v2 --workers 12 --max-evals 60 --target 85

Runtime is dominated by the design generator (~CPU-hours); run this in the background.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

from thermo_agent.synth.tiers import TIERS, assign_tiers, tier_ranges

ROOT = pathlib.Path(__file__).resolve().parents[1]
STAGE_PARENT = pathlib.Path(os.environ.get("TMPDIR", r"C:\Users\rezax\AppData\Local\Temp\opencode"))

# Eval reports exactly these split families per task (docs/EVAL_PROTOCOL.md).
CELLS: dict[str, tuple[str, ...]] = {
    "repair": ("test_id", "test_ood_fault", "test_ood_topology"),
    "audit": ("test_id", "test_ood_topology"),
    "ledger": ("test_id", "test_ood_fault", "test_ood_topology"),
    "design": ("test_id", "test_ood_climate", "test_ood_load", "test_ood_scale"),
}

# The 9 diagram combinations reserved whole for test_ood_topology (synth/splits.py).
OOD_ARCHS = [
    "engine-absorption-ext", "engine-absorption-ext-dh_exhaust",
    "engine-absorption-ext-orc", "engine-absorption-ext-orc-dh_exhaust",
    "engine-electric-ext", "engine-electric-ext-dh_exhaust",
    "engine-electric-ext-orc", "engine-electric-ext-orc-dh_exhaust",
    "cgam_steam_orc",
]
BIG_ARCH = "big_cchp_steam2p_reheat_hp"

IDENTITY_FIELDS = ("plant", "plant_broken", "plant_original", "site", "ledger")

# The documented per-task size measure (thermo_agent/synth/tiers.py).  Recorded in the
# manifest so a reader knows exactly how easy/medium/hard were derived.
MEASURE = {
    "repair": "number of components in plant_broken",
    "audit": "number of components in plant",
    "ledger": "number of components in ledger.data.topo",
    "design": "number of components of the reference design built from site + ref_spec",
}


@dataclass
class Source:
    key: str
    task: str
    tier: str                      # generator's own label (provenance only; final tier is re-derived)
    module: str
    n: int
    seed_offset: int
    topup_n: int
    extra: tuple[str, ...] = ()
    cells: tuple[tuple[str, str], ...] = ()   # cells this source is preferred for when topping up

    def command(self, seed: int, n: int, workers: int, outdir: pathlib.Path, max_evals: int) -> list[str]:
        cmd = [sys.executable, "-m", self.module, str(outdir), "-n", str(n),
               "--seed", str(seed), "--workers", str(workers)]
        if self.module == "thermo_agent.design.data":
            cmd += ["--max-evals", str(max_evals)]
        cmd += list(self.extra)
        return cmd


def source_plan() -> list[Source]:
    return [
        Source("repair_easy", "repair", "easy", "thermo_agent.synth.build_repair_data",
               n=1300, seed_offset=0, topup_n=300,
               cells=(("repair", "test_id"), ("repair", "test_ood_fault"))),
        Source("repair_medium", "repair", "medium", "thermo_agent.synth.architectures",
               n=800, seed_offset=1, topup_n=200),
        Source("repair_medium_ood", "repair", "medium", "thermo_agent.synth.architectures",
               n=180, seed_offset=2, topup_n=150, extra=("--only", *OOD_ARCHS),
               cells=(("repair", "test_ood_topology"),)),
        Source("repair_hard", "repair", "hard", "thermo_agent.synth.architectures",
               n=340, seed_offset=3, topup_n=200, extra=("--only", BIG_ARCH)),
        Source("audit_easy", "audit", "easy", "thermo_agent.audit.data",
               n=2200, seed_offset=4, topup_n=600,
               cells=(("audit", "test_id"),)),
        Source("audit_medium", "audit", "medium", "thermo_agent.audit.data",
               n=700, seed_offset=5, topup_n=250, extra=("--architectures",)),
        Source("audit_medium_ood", "audit", "medium", "thermo_agent.audit.data",
               n=180, seed_offset=6, topup_n=150,
               extra=("--architectures", "--only", *OOD_ARCHS),
               cells=(("audit", "test_ood_topology"),)),
        Source("audit_hard", "audit", "hard", "thermo_agent.audit.data",
               n=340, seed_offset=7, topup_n=200,
               extra=("--architectures", "--only", BIG_ARCH)),
        Source("ledger_easy", "ledger", "easy", "thermo_agent.ledger.data",
               n=1500, seed_offset=8, topup_n=350,
               cells=(("ledger", "test_id"), ("ledger", "test_ood_fault"))),
        Source("ledger_medium", "ledger", "medium", "thermo_agent.ledger.data",
               n=800, seed_offset=9, topup_n=250, extra=("--architectures",)),
        Source("ledger_medium_ood", "ledger", "medium", "thermo_agent.ledger.data",
               n=180, seed_offset=10, topup_n=150,
               extra=("--architectures", "--only", *OOD_ARCHS),
               cells=(("ledger", "test_ood_topology"),)),
        Source("design_easy", "design", "easy", "thermo_agent.design.data",
               n=1800, seed_offset=11, topup_n=450,
               cells=(("design", "test_id"), ("design", "test_ood_climate"),
                      ("design", "test_ood_load"), ("design", "test_ood_scale"))),
    ]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def run_source(src: Source, seed: int, n: int, workers: int, max_evals: int,
               stage: pathlib.Path, runs: list[dict]) -> list[dict]:
    outdir = stage / f"{src.key}_s{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = src.command(seed, n, workers, outdir, max_evals)
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {src.key}: n={n} seed={seed}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"generator failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-2000:]}")
    rows: list[dict] = []
    per_file: dict[str, int] = {}
    for path in sorted(outdir.glob("*.jsonl")):
        count = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["gen_seed"] = seed
                row["_gen_tier"] = src.tier       # provenance only; overwritten by assign_tiers
                rows.append(row)
                count += 1
        if count:
            per_file[path.name] = count
    runs.append({"source": src.key, "generator_tier": src.tier, "task": src.task,
                 "command": " ".join(cmd), "seed": seed, "n_requested": n,
                 "n_rows": len(rows), "per_file": per_file,
                 "seconds": round(elapsed, 2)})
    print(f"    -> {len(rows)} rows in {elapsed:.1f}s {per_file}", flush=True)
    return rows


def bucket(rows: list[dict], task: str, pool: dict[tuple[str, str], list[dict]],
           dropped: collections.Counter) -> None:
    """Keep only the documented *test* cells; never write train/val into the test set."""
    for row in rows:
        split = row.get("split", "")
        if split in CELLS[task]:
            pool.setdefault((task, split), []).append(row)
        else:
            dropped[f"{task}/{split}"] += 1


# ---------------------------------------------------------------------------
# Real difficulty tiers — Reza directive §4.2
# ---------------------------------------------------------------------------
def retier(pool: dict[tuple[str, str], list[dict]]) -> dict[str, dict]:
    """Re-tier the FINAL rows of each task with the documented size measure.

    ``assign_tiers`` ranks by ``complexity(task, row)``, breaks ties by the sha256 of the
    row id, and cuts into floor(n/3)/floor(n/3)/remainder (remainder to hard).  This is
    never random and uses no assistant-chosen threshold.
    """
    info: dict[str, dict] = {}
    for task in CELLS:
        rows = [r for (t, _s), rs in pool.items() if t == task for r in rs]
        ids = [str(r.get("record_id") or r.get("id")) for r in rows]
        dupes = [k for k, n in collections.Counter(ids).items() if n > 1]
        if dupes:
            raise RuntimeError(f"{task}: duplicate ids would corrupt tier assignment: {dupes[:5]}")
        mapping = assign_tiers(task, rows)
        unscored = len(rows) - len(mapping)
        if unscored:
            raise RuntimeError(
                f"{task}: {unscored} rows have no measurable artefact and cannot be tiered "
                f"(the test set must have zero unscored rows)"
            )
        for r in rows:
            r["tier"] = mapping[str(r.get("record_id") or r.get("id"))]
        counts = collections.Counter(mapping.values())
        info[task] = {
            "n": len(rows),
            "measure": MEASURE[task],
            "counts": {t: counts.get(t, 0) for t in TIERS},
            "ranges": tier_ranges(task, rows, mapping),
        }
        print(f"tiers {task}: {info[task]['counts']} ranges={info[task]['ranges']}", flush=True)
    return info


# ---------------------------------------------------------------------------
# Contamination scan (same signature definition as tests/test_contamination.py)
# ---------------------------------------------------------------------------
def _parse(value):
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    if isinstance(value, (dict, list)) and value:
        return value
    return None


def _digest(value) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def signature_set(row: dict) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for name in IDENTITY_FIELDS:
        obj = _parse(row.get(name))
        if obj is not None:
            out.add((name, _digest(obj)))
    site, spec = _parse(row.get("site")), _parse(row.get("ref_spec"))
    if site is not None and spec is not None:
        out.add(("site+ref_spec", _digest({"site": site, "ref_spec": spec})))
    rid = row.get("id")
    if isinstance(rid, str) and rid:
        out.add(("id", _digest(rid)))
    return out


def target_files() -> list[pathlib.Path]:
    """EVERY training / prompt / export artifact under data/, except the two held-out sets.

    This deliberately covers sft_tools*, grpo_tools*, a_slice, sft_mix*, sft_all,
    grpo_all, processed*, every frontier export, teacher_runs and memory.  ``data/heldout``
    is the dev set, not a training dir, and is excluded by design; ``data/heldout_v2`` is
    the set being built and is excluded to avoid scanning its own staged output.
    """
    found = {p for p in (ROOT / "data").rglob("*.jsonl")
             if p.is_file()
             and "heldout_v2" not in p.parts
             and "heldout" not in p.parts}
    return sorted(found)


def scan_files(paths: list[pathlib.Path], index: set[tuple[str, str]]) -> tuple[list[dict], int]:
    leaks: list[dict] = []
    lines = 0
    next_report = 500_000
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line_no, raw in enumerate(fh, 1):
                if not raw.strip():
                    continue
                lines += 1
                try:
                    row = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if not isinstance(row, dict):
                    continue
                kinds = sorted(k for k, h in signature_set(row) if (k, h) in index)
                if kinds:
                    try:
                        label = str(path.relative_to(ROOT))
                    except ValueError:
                        label = str(path)
                    leaks.append({"file": label, "line": line_no,
                                  "record_id": row.get("record_id", row.get("id")),
                                  "kinds": kinds})
        if lines >= next_report:
            print(f"    scan: {path.name}: {lines} lines, {len(leaks)} leaks", flush=True)
            next_report += 500_000
    return leaks, lines


def contamination_scan(pool: dict[tuple[str, str], list[dict]]) -> dict:
    index: set[tuple[str, str]] = set()
    n_rows = 0
    for rows in pool.values():
        for row in rows:
            index |= signature_set(row)
            n_rows += 1
    files = target_files()
    t0 = time.perf_counter()
    leaks, lines = scan_files(files, index)

    # Self-check: a planted v2 row MUST be detected by the very same scanner (proves the
    # scanner can fail).  The temp file lives under the approved opencode temp directory.
    STAGE_PARENT.mkdir(parents=True, exist_ok=True)
    probe = next(iter(next(iter(pool.values()))))
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8",
                                     dir=STAGE_PARENT) as fh:
        fh.write(json.dumps(probe, ensure_ascii=False) + "\n")
        probe_path = pathlib.Path(fh.name)
    planted, _ = scan_files([probe_path], index)
    probe_path.unlink(missing_ok=True)
    self_check = bool(planted)
    return {
        "passed": not leaks and self_check,
        "self_check_detects_planted_row": self_check,
        "heldout_v2_rows_indexed": n_rows,
        "signatures": len(index),
        "target_dirs": sorted({str(p.parent.relative_to(ROOT)) for p in files}),
        "files_scanned": len(files),
        "lines_scanned": lines,
        "overlap": len(leaks),
        "leaks": leaks[:20],
        "seconds": round(time.perf_counter() - t0, 2),
    }


# ---------------------------------------------------------------------------
# Freeze / manifest
# ---------------------------------------------------------------------------
def tier_counts(rows: list[dict]) -> dict[str, int]:
    c = collections.Counter(r.get("tier", "unlabelled") for r in rows)
    return {t: c.get(t, 0) for t in TIERS if c.get(t, 0)}


def merge_if_needed(pool: dict[tuple[str, str], list[dict]], target: int) -> list[dict]:
    """Only merge same-task structural-OOD siblings; never id with OOD, never across tasks."""
    merges: list[dict] = []
    rules = {
        "repair": (("test_ood_fault", "test_ood_topology"), "repair structural-OOD"),
        "ledger": (("test_ood_fault", "test_ood_topology"), "ledger structural-OOD"),
        "design": (("test_ood_climate", "test_ood_load", "test_ood_scale"), "design OOD-tail"),
    }
    for task, (members, label) in rules.items():
        short = [s for s in members if len(pool.get((task, s), [])) < target]
        if not short:
            continue
        merged: list[dict] = []
        for s in members:
            merged.extend(pool.get((task, s), []))
        if len(merged) < target:
            continue   # merging does not rescue it; leave cells visible and report the shortfall
        for s in members:
            pool.pop((task, s), None)
        pool[(task, label)] = merged
        merges.append({"task": task, "members": list(members), "label": label,
                       "rows": len(merged),
                       "reason": f"generator produced < {target} rows in at least one member cell "
                                 f"({', '.join(short)}); same task and same success metric"})
    return merges


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_hashes(out: pathlib.Path) -> tuple[dict, dict, str]:
    """Freeze file hashes and the set hash; never includes MANIFEST.json itself."""
    files: dict[str, dict] = {}
    h = hashlib.sha256()
    for path in sorted(out.glob("*")):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        digest = sha256_file(path)
        files[path.name] = {"sha256": digest, "bytes": path.stat().st_size}
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    jsonl = {name: v["sha256"] for name, v in files.items() if name.endswith(".jsonl")}
    return files, jsonl, h.hexdigest()


SET_HASH_RULE = ("sha256 over sorted(file_name_bytes + file_bytes) for every file in "
                 "data/heldout_v2 except MANIFEST.json")


def write_set(out: pathlib.Path, pool: dict[tuple[str, str], list[dict]],
              merges: list[dict], record: dict) -> dict:
    for (task, split), rows in sorted(pool.items()):
        rows.sort(key=lambda r: str(r["id"]))
        path = out / f"{task}_{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        record["cells"][f"{task}/{split}"] = {"rows": len(rows), "tiers": tier_counts(rows)}
        (out / f"meta_{task}.json").write_text(
            json.dumps({"task": task, "seed": record["seed"],
                        "splits": {s: len(pool.get((task, s), []))
                                   for s in CELLS[task] if (task, s) in pool}},
                       indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
    files, jsonl, digest = compute_hashes(out)
    record["files"] = files
    record["jsonl_sha256"] = jsonl
    record["set_hash"] = digest
    record["set_hash_rule"] = SET_HASH_RULE
    record["merged_cells"] = merges
    (out / "MANIFEST.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    return record


def retier_existing(out: pathlib.Path) -> None:
    """Re-tier an ALREADY GENERATED set in place — no generator is run and no design row is remade.

    Additive guarantee: only the ``tier`` column is rewritten.  Every other field and the row
    order of every file are checked equal before and after, task content is never touched.
    """
    t0 = time.perf_counter()
    mpath = out / "MANIFEST.json"
    m = json.loads(mpath.read_text(encoding="utf-8"))

    def content(row: dict) -> str:
        return json.dumps({k: v for k, v in row.items() if k != "tier"},
                          sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    pool: dict[tuple[str, str], list[dict]] = {}
    before: dict[str, list[str]] = {}
    for path in sorted(out.glob("*_test_*.jsonl")):
        task = path.name.split("_", 1)[0]
        split = path.name[len(task) + 1:-len(".jsonl")]
        with path.open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        if not rows:
            raise RuntimeError(f"empty cell: {path.name}")
        pool[(task, split)] = rows
        before[path.name] = [content(r) for r in rows]

    assert set(pool) == {tuple(k.split("/", 1)) for k in m["cells"]}, "cell set changed"
    short = {f"{t}/{s}": len(rs) for (t, s), rs in pool.items()
             if len(rs) < m["target_rows_per_cell"]}
    if short:
        raise RuntimeError(f"cells below the {m['target_rows_per_cell']}-row target: {short}")

    tiers = retier(pool)   # row-level assign_tiers; sets row["tier"] in place

    for (task, split), rows in pool.items():
        name = f"{task}_{split}.jsonl"
        assert [content(r) for r in rows] == before[name], f"task content changed in {name}"

    # Rewrite each file with identical row order and values; only `tier` differs.
    for (task, split), rows in pool.items():
        path = out / f"{task}_{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    for key, cell in m["cells"].items():
        task, split = key.split("/", 1)
        cell["tiers"] = tier_counts(pool[(task, split)])

    print("contamination scan (every training dir, no line cap)...", flush=True)
    contamination = contamination_scan(pool)
    print("contamination:", {k: v for k, v in contamination.items() if k != "leaks"}, flush=True)
    if contamination["overlap"]:
        print(f"OVERLAP FOUND: {contamination['leaks'][:20]}", flush=True)
        raise SystemExit("heldout_v2 overlaps a training artifact — not frozen; "
                         "regenerate with a NEW seed, never delete training data")

    command = [sys.executable] + sys.argv
    m["generation_command"] = [sys.executable, "scripts/build_heldout_v2.py",
                               "--seed", str(m["seed"]), "--out", "data/heldout_v2",
                               "--workers", str(m["workers"]), "--max-evals", str(m["max_evals"]),
                               "--target", str(m["target_rows_per_cell"])]
    m["generation_command_str"] = " ".join(m["generation_command"])
    m["retier_command"] = command
    m["retier_command_str"] = " ".join(command)
    m["tiers"] = tiers
    m["tier_rule"] = ("per task, by ROW: assign_tiers ranks by the documented size measure, breaks "
                      "ties by sha256(id), cuts floor(n/3)/floor(n/3)/remainder (remainder to hard); "
                      "never random, never an assistant-chosen threshold")
    m["tier_support"] = {
        "easy": list(CELLS), "medium": list(CELLS), "hard": list(CELLS),
        "definition": ("real rank tiers from thermo_agent/synth/tiers.py, equal thirds by row "
                       "within EVERY task; hard = largest measured third (the 19-component "
                       "big_cchp_steam2p_reheat_hp cycle for repair/audit where generated)"),
        "unsupported": ("design and ledger generators have no big-cycle source, so their hard "
                        "rows are the largest measured third — no rows were fabricated"),
    }
    m["retier"] = {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "by": "row",
        "function": "thermo_agent.synth.tiers.assign_tiers",
        "rule": m["tier_rule"],
        "only_column_rewritten": "tier",
        "content_preserved": True,
        "note": ("generator source labels in generator_runs are provenance only; the final tier "
                 "column is the measured per-task rank tier"),
    }
    m["retiered_from_existing"] = True
    m["contamination"] = contamination
    m["notes"] = [
        "Test splits only: no train/val row is written into heldout_v2.",
        "Rows were generated ONCE with seed 20261119; design rows were never regenerated.",
        "The tier column was re-derived by ROW from thermo_agent/synth/tiers.py per task; "
        "all other row fields and row order are unchanged (verified before writing).",
        "Every row carries its real tier and its gen_seed; ids embed the generator seed.",
        "Hard = largest measured third per task; repair/audit include the 19-component big cycle.",
    ]

    files, jsonl, digest = compute_hashes(out)
    m["files"] = files
    m["jsonl_sha256"] = jsonl
    m["set_hash"] = digest
    m["set_hash_rule"] = SET_HASH_RULE
    m["retier_seconds"] = round(time.perf_counter() - t0, 2)
    mpath.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"RETIERED set_hash={digest} rows={sum(v['rows'] for v in m['cells'].values())} "
          f"tiers={ {t: m['tiers'][t]['counts'] for t in CELLS} } "
          f"overlap={contamination['overlap']} seconds={m['retier_seconds']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retier-existing", action="store_true",
                    help="re-tier an already generated set in place; run no generator")
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--out", default="data/heldout_v2")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-evals", type=int, default=60)
    ap.add_argument("--target", type=int, default=85)
    ap.add_argument("--max-topups", type=int, default=8)
    a = ap.parse_args()

    out = ROOT / a.out
    if a.retier_existing:
        retier_existing(out)
        return
    out.mkdir(parents=True, exist_ok=True)
    STAGE_PARENT.mkdir(parents=True, exist_ok=True)
    stage = pathlib.Path(tempfile.mkdtemp(prefix="heldout_v2_stage_", dir=STAGE_PARENT))
    print(f"staging in {stage}", flush=True)
    t_start = time.perf_counter()
    sources = source_plan()
    by_key = {s.key: s for s in sources}
    preferred = {cell: s for s in sources for cell in s.cells}
    runs: list[dict] = []
    pool: dict[tuple[str, str], list[dict]] = {}
    dropped: collections.Counter = collections.Counter()

    for src in sources:
        rows = run_source(src, a.seed + src.seed_offset, src.n, a.workers, a.max_evals, stage, runs)
        bucket(rows, src.task, pool, dropped)

    before = {f"{t}/{s}": len(pool.get((t, s), [])) for t in CELLS for s in CELLS[t]}
    print("cells after first pass:", before, flush=True)

    for attempt in range(1, a.max_topups + 1):
        missing = [(t, s) for t in CELLS for s in CELLS[t]
                   if len(pool.get((t, s), [])) < a.target]
        if not missing:
            break
        grouped: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
        for cell in missing:
            grouped[preferred[cell].key].append(cell)
        print(f"top-up {attempt}: {grouped}", flush=True)
        for key, cells in grouped.items():
            src = by_key[key]
            seed = a.seed + src.seed_offset + 1000 * attempt
            rows = run_source(src, seed, src.topup_n, a.workers, a.max_evals, stage, runs)
            bucket(rows, src.task, pool, dropped)

    after_gen = {f"{t}/{s}": len(pool.get((t, s), [])) for t in CELLS for s in CELLS[t]}
    merges = merge_if_needed(pool, a.target)
    final = {f"{t}/{s}": len(pool.get((t, s), [])) for t, s in sorted(pool)}

    print("cells after generation:", after_gen, flush=True)
    print("merges:", merges, flush=True)

    # Real tiers — derived from the same documented measure as training, per task.
    tiers = retier(pool)

    for rows in pool.values():
        for row in rows:
            row.pop("_task", None)
            row.pop("_gen_tier", None)

    print("contamination scan...", flush=True)
    contamination = contamination_scan(pool)
    print("contamination:", {k: v for k, v in contamination.items() if k != "leaks"}, flush=True)
    if contamination["overlap"]:
        print(f"OVERLAP FOUND: {contamination['leaks'][:20]}", flush=True)
        shutil.rmtree(stage, ignore_errors=True)
        raise SystemExit(
            "heldout_v2 overlaps a training artifact. NOT frozen. "
            "Regenerate heldout_v2 with a NEW seed (--seed <other>, never delete training data)."
        )

    record = {
        "eval_set": "heldout_v2",
        "role": "frozen test set for the paper (F1); data/heldout (214 ids) is the dev set",
        "seed": a.seed,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "generation_command": [sys.executable] + sys.argv,
        "generation_command_str": " ".join([sys.executable] + sys.argv),
        "target_rows_per_cell": a.target,
        "workers": a.workers,
        "max_evals": a.max_evals,
        "tiers": tiers,
        "tier_rule": ("per task: assign_tiers ranks by the documented size measure, breaks ties by "
                      "sha256(id), cuts floor(n/3)/floor(n/3)/remainder (remainder to hard); "
                      "never random, never an assistant-chosen threshold"),
        "cells_before_topup": before,
        "cells_after_generation": after_gen,
        "cells": {},
        "generator_runs": runs,
        "dropped_non_test_rows": dict(sorted(dropped.items())),
        "scoring": {"limit": 0, "rule": "score every row of every test file; no truncation, no sampling"},
        "contamination": contamination,
        "notes": [
            "Test splits only: no train/val row is written into heldout_v2 (all non-test generator "
            "rows are dropped and counted in dropped_non_test_rows).",
            "Every row carries a real tier derived from thermo_agent/synth/tiers.py and its gen_seed; "
            "ids embed the generator seed, so batches never collide.",
            "Hard = the largest third by the measured size for each task. For repair/audit this is "
            "the 19-component big_cchp_steam2p_reheat_hp cycle where the generators support it; "
            "design/ledger have no big generator, so their hard tier is the largest available size.",
            "OOD topology cells are filled from the 9 held-out diagram combinations; no rows are fabricated.",
        ],
    }
    write_set(out, pool, merges, record)
    record["generation_seconds"] = round(time.perf_counter() - t_start, 1)
    # Rewrite the manifest now that total runtime is known (files/hashes unchanged).
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest["generation_seconds"] = record["generation_seconds"]
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    print(f"DONE set_hash={manifest['set_hash']} files={len(manifest['files'])} "
          f"rows={sum(v['rows'] for v in manifest['cells'].values())} "
          f"seconds={record['generation_seconds']}", flush=True)
    shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
