"""End-to-end, local-only provenance chain audit (no network, no models, no writes).

The audit proves "what came from where" across four hand-offs:

    cohort rows -> prompt shards -> harness prompt -> mix rows
                          memory notes -------------^

and proves that answer-bearing / bookkeeping fields never cross those boundaries.
Every check returns a :class:`CheckResult` so the same code powers both the CLI
report and the unit tests; the CLI prints a compact table and a final verdict.

    python scripts/audit_provenance_chain.py --strict
    python scripts/audit_provenance_chain.py --json /tmp/provenance_report.json

Exit status is 1 only with ``--strict`` and at least one failing check.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_agent.synth import tiers as tier_mod  # noqa: E402

# --------------------------------------------------------------------------- #
# Policy constants (mirrors docs/PROVENANCE.md and scripts/check_datagen_complete.py)
# --------------------------------------------------------------------------- #
BUDGET_MODELS = ("gpt-6-luna", "claude-haiku-5-5")
PREMIUM_MODELS = (
    "gpt-6.2-sol", "claude-sonnet-5-5", "claude-opus-5-5-default", "gemini-4-argon",
)
DETERMINISTIC_SOURCES = ("ground_truth", "deterministic_tools")
CANONICAL_FRONTIER_SOURCE = "frontier_tools"

# Legacy accepted-trace stores from the superseded early campaign. These are the
# frontier directories covered by ``scripts.check_datagen_complete.FRONTIER_DIR_GLOBS``,
# minus the current luna/haiku cohort (traced separately) and the prompt-shard dir
# (no trace rows). A small set of gemini-3.8-flash traces was legitimately carried
# forward from here into the mix; the 32 cleaned traces in
# ``data/frontier_tools_clean`` are explicitly expected by the datagen gate and
# were never to be re-run (docs/PROVENANCE.md, docs/DECISIONS.md §6).
# Every row in these stores names its model explicitly in ``model``/``source``.
LEGACY_FRONTIER_DIRS = (
    "data/frontier_tools",
    "data/frontier_tools_clean",
    "data/frontier_tools_clean",
    "data/frontier_tools_pilot",
    "data/frontier_tools_accepted",
)

ANSWER_FIELDS = ("ref_spec", "ref_evals", "ref_fuel_w", "completion", "sft_completion")
SHARD_FORBIDDEN_FIELDS = ANSWER_FIELDS + (
    "provenance", "model", "run_id", "cost", "tokens", "latency",
)

CHECK_NAMES = (
    "cohort_provenance_fields",
    "frontier_source_mapping",
    "tiers_recomputed",
    "prompt_shard_leakage",
    "harness_prompt_leakage",
    "memory_outcomes_and_serving",
    "mix_propagation",
    "mix_leakage",
)

# Bookkeeping patterns used by both the harness-prompt and served-memory checks.
_TIMESTAMP = r"\b20\d{2}-\d{2}-\d{2}(?:[T ][0-9:.+-]+)?\b"
_KV_BOOKKEEPING = r"\b(?:ts|cost|tokens?|latency|run[_-]?id|record[_-]?id)\s*="
_JOURNAL_PATH = r"(?:data|teacher_runs|kaggle)/[\w./-]+\.(?:jsonl|json|md)\b"
_GENERIC_MODEL = (
    r"\b(?:gpt-6(?:[.-][a-z0-9]+)*|claude-(?:haiku|sonnet|opus)[a-z0-9.-]*"
    r"|gemini-[a-z0-9.-]+)\b"
)


@dataclass
class CheckResult:
    """One named audit check."""

    name: str
    ok: bool
    detail: str
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "name": self.name, "ok": bool(self.ok), "detail": self.detail,
            "problems": list(self.problems),
        }


# --------------------------------------------------------------------------- #
# small io / scanning helpers
# --------------------------------------------------------------------------- #
def _resolve(value: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(value)
    return path if path.is_absolute() else (ROOT / path)


def _jsonl_files(directory: pathlib.Path, recursive: bool = False) -> list[pathlib.Path]:
    if not directory.is_dir():
        return []
    return sorted(directory.rglob("*.jsonl") if recursive else directory.glob("*.jsonl"))


def _iter_jsonl(files: list[pathlib.Path]):
    """Yield (path, line_number, row) for well-formed JSON object lines."""
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(row, dict):
                    yield path, lineno, row


def _deep_keys(obj) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        keys.update(str(k) for k in obj)
        for value in obj.values():
            keys.update(_deep_keys(value))
    elif isinstance(obj, list):
        for value in obj:
            keys.update(_deep_keys(value))
    return keys


def _prompt_text(row: dict) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n".join(
            str(msg.get("content", "")) for msg in prompt if isinstance(msg, dict)
        )
    return ""


def _row_id(row: dict) -> str:
    return str(row.get("record_id") or row.get("id") or "")


def _prov(row: dict) -> dict:
    value = row.get("provenance")
    return value if isinstance(value, dict) else {}


def _field(row: dict, prov: dict, name: str):
    """Read a provenance field from the stamp, falling back to the top level."""
    return prov.get(name, row.get(name))


def _model_of(row: dict) -> str:
    prov = _prov(row)
    return str(_field(row, prov, "model") or row.get("source_model") or "")


def _is_ai(row: dict) -> bool:
    prov = _prov(row)
    source = str(row.get("source") or "")
    return (
        prov.get("source_family") == "ai_generated"
        or source.startswith("frontier:")
        or source == CANONICAL_FRONTIER_SOURCE
        or bool(_model_of(row))
    )


def _bookkeeping_hits(text: str, literals: tuple[str, ...] = ()) -> list[str]:
    """Return the distinct kinds of bookkeeping that leaked into ``text``."""
    hits: list[str] = []
    lowered = text.lower()
    for literal in literals:
        if literal and literal.lower() in lowered:
            hits.append(f"literal:{literal}")
    if re.search(_TIMESTAMP, text):
        hits.append("timestamp")
    if re.search(_KV_BOOKKEEPING, text, re.I):
        hits.append("key=value bookkeeping")
    if re.search(_JOURNAL_PATH, text, re.I):
        hits.append("journal path")
    if re.search(_GENERIC_MODEL, text, re.I):
        hits.append("model id")
    if "ref_spec" in text:
        hits.append("ref_spec")
    return sorted(set(hits))


def _sample(items: list, limit: int) -> list:
    return items[: max(0, limit)]


# --------------------------------------------------------------------------- #
# check 1 -- per-row provenance fields
# --------------------------------------------------------------------------- #
def check_provenance_fields(cohort: pathlib.Path) -> CheckResult:
    name = CHECK_NAMES[0]
    rows = [row for _, _, row in _iter_jsonl(_jsonl_files(cohort, recursive=True))]
    problems: list[str] = []
    if not rows:
        problems.append("no accepted frontier rows found in cohort")

    for path, lineno, row in _iter_jsonl(_jsonl_files(cohort, recursive=True)):
        prov = _prov(row)
        where = f"{path.name}:{lineno}"
        model = _field(row, prov, "model")
        provider = _field(row, prov, "provider")
        run_id = _field(row, prov, "run_id")
        source = _field(row, prov, "source")

        if not model:
            problems.append(f"{where}: missing model")
        if not source:
            problems.append(f"{where}: missing source")
        if not run_id:
            problems.append(f"{where}: missing run_id")
        explicit_exception = bool(
            prov.get("provider_exception") or row.get("provider_exception")
        )
        if not provider and not explicit_exception:
            problems.append(f"{where}: missing provider (no documented exception)")

        top_model = row.get("model")
        if top_model and prov.get("model") and str(top_model) != str(prov["model"]):
            problems.append(
                f"{where}: top-level model {top_model!r} disagrees with "
                f"provenance.model {prov['model']!r}"
            )

        text = str(model or "")
        is_budget = any(text == allowed or allowed in text for allowed in BUDGET_MODELS)
        is_premium = any(text == p or p in text for p in PREMIUM_MODELS)
        marked_premium = bool(
            prov.get("premium") or row.get("premium")
            or prov.get("source_family") == "premium"
        )
        if text and not is_budget and not (is_premium and marked_premium):
            problems.append(
                f"{where}: model {text!r} is neither an authorized budget model "
                "nor a clearly-marked premium row"
            )

    detail = (f"{len(rows)} cohort rows; {len(problems)} field problem(s)"
              + ("; no data" if not rows else ""))
    return CheckResult(name, not problems, detail, problems)


# --------------------------------------------------------------------------- #
# check 2 -- source-family mapping
# --------------------------------------------------------------------------- #
def check_source_mapping(cohort: pathlib.Path) -> CheckResult:
    name = CHECK_NAMES[1]
    problems: list[str] = []
    families: dict[str, set[str]] = {}
    n_ai = n_det = 0

    for path, lineno, row in _iter_jsonl(_jsonl_files(cohort, recursive=True)):
        prov = _prov(row)
        where = f"{path.name}:{lineno}"
        source = str(row.get("source") or "")
        detail = str(row.get("source_detail") or prov.get("source_detail") or "")
        model = _model_of(row)

        if source in DETERMINISTIC_SOURCES:
            n_det += 1
            if model:
                problems.append(f"{where}: deterministic row carries a model ({model!r})")
            families.setdefault(source, set()).add("none")
            continue

        n_ai += 1
        if source.startswith("frontier:"):
            suffix = source.split(":", 1)[1]
            if not model or suffix != model:
                problems.append(
                    f"{where}: source {source!r} does not name exact model {model!r}"
                )
            if detail and detail != CANONICAL_FRONTIER_SOURCE:
                problems.append(
                    f"{where}: source_detail {detail!r} should be "
                    f"{CANONICAL_FRONTIER_SOURCE!r}"
                )
            families.setdefault("frontier:<model>", set()).add(detail or "<missing>")
        elif source == CANONICAL_FRONTIER_SOURCE:
            if not model:
                problems.append(f"{where}: canonical frontier_tools row has no model")
            families.setdefault(CANONICAL_FRONTIER_SOURCE, set()).add(model or "<missing>")
        else:
            problems.append(f"{where}: unexpected source family {source!r}")

    # A family must map to exactly one canonical detail / model convention.
    for family, values in families.items():
        if family == CANONICAL_FRONTIER_SOURCE:
            continue  # many models legitimately share the canonical label
        if len(values) > 1:
            problems.append(f"source family {family!r} maps to inconsistent details {sorted(values)}")

    detail = f"{n_ai} AI rows, {n_det} deterministic rows; {len(problems)} mapping problem(s)"
    return CheckResult(name, not problems, detail, problems)


# --------------------------------------------------------------------------- #
# check 3 -- tiers are recomputed, never trusted
# --------------------------------------------------------------------------- #
def check_tiers(cohort: pathlib.Path) -> CheckResult:
    name = CHECK_NAMES[2]
    rows = [row for _, _, row in _iter_jsonl(_jsonl_files(cohort, recursive=True))]
    problems: list[str] = []
    n_derivable = n_no_spec = n_no_stored = n_duplicate_key = 0

    for task in tier_mod.TASKS:
        derivable = [
            row for row in rows
            if str(row.get("task") or "") == task
            and tier_mod.complexity(task, row) is not None
        ]
        recomputed = tier_mod.assign_tiers(task, derivable)
        for row in derivable:
            n_derivable += 1
            key = _row_id(row)
            expected = recomputed.get(key)
            if expected is None:
                # Duplicate record ids share one assigned tier; nothing to compare.
                n_duplicate_key += 1
                continue
            stored = row.get("tier")
            if stored is None:
                n_no_stored += 1
                continue
            if str(stored) != str(expected):
                problems.append(
                    f"{task}/{key}: stored tier {stored!r} != recomputed {expected!r}"
                )

    n_no_spec = len(rows) - n_derivable
    detail = (
        f"{n_derivable} rows with derivable spec; {n_no_stored} without stored tier "
        f"(not a failure); {n_duplicate_key} shared-id rows skip comparison; "
        f"{n_no_spec} without derivable spec (ignored); {len(problems)} disagreement(s)"
    )
    return CheckResult(name, not problems, detail, problems)


# --------------------------------------------------------------------------- #
# check 4 -- prompt shards
# --------------------------------------------------------------------------- #
def check_prompt_shards(shards_dir: pathlib.Path) -> CheckResult:
    name = CHECK_NAMES[3]
    files = _jsonl_files(shards_dir)
    problems: list[str] = []
    if not files:
        problems.append("no prompt shard jsonl files found")
    n_rows = 0
    for path, lineno, row in _iter_jsonl(files):
        n_rows += 1
        leaked = _deep_keys(row) & set(SHARD_FORBIDDEN_FIELDS)
        if leaked:
            problems.append(f"{path.name}:{lineno} contains {sorted(leaked)}")
    detail = f"{len(files)} shards, {n_rows} rows; {len(problems)} leak(s)"
    return CheckResult(name, not problems, detail, problems)


# --------------------------------------------------------------------------- #
# check 5 -- harness prompt leakage
# --------------------------------------------------------------------------- #
def check_harness_prompt_leakage(
    cohort: pathlib.Path, sample: int = 25, rebuild: bool = True,
) -> CheckResult:
    name = CHECK_NAMES[4]
    all_rows = list(_iter_jsonl(_jsonl_files(cohort, recursive=True)))
    picked = _sample(all_rows, sample)
    run_ids = tuple(sorted({
        str(_field(row, _prov(row), "run_id") or "")
        for _, _, row in all_rows if _field(row, _prov(row), "run_id")
    }))
    record_ids = tuple(sorted({
        _row_id(row) for _, _, row in all_rows if _row_id(row)
    }))
    literals = BUDGET_MODELS + PREMIUM_MODELS + run_ids + record_ids

    problems: list[str] = []
    for path, lineno, row in picked:
        texts = [_prompt_text(row)]
        if rebuild:
            try:
                from thermo_agent.agent.tool_prompt import build_compact_harness_messages
                rebuilt = build_compact_harness_messages(
                    row.get("prompt") or [], str(row.get("task") or "repair"),
                    row.get("plant_broken") or row.get("plant"),
                )
                texts.append("\n".join(str(m.get("content", "")) for m in rebuilt))
            except Exception:
                # A fixture without a real plant/prompt falls back to the stored text.
                pass
        hits = _bookkeeping_hits("\n".join(texts), literals)
        if hits:
            problems.append(f"{path.name}:{lineno} harness prompt leaks {hits}")

    detail = (
        f"inspected {len(picked)}/{len(all_rows)} rows "
        f"(rebuild={'on' if rebuild else 'off'}); {len(problems)} leak(s)"
    )
    return CheckResult(name, not problems, detail, problems)


# --------------------------------------------------------------------------- #
# check 6 -- memory records both outcomes and serves clean text
# --------------------------------------------------------------------------- #
def check_memory(memory_dir: pathlib.Path) -> CheckResult:
    name = CHECK_NAMES[5]
    files = _jsonl_files(memory_dir)
    problems: list[str] = []
    if not files:
        problems.append("no memory jsonl files found")

    n_accepted = n_rejected = 0
    all_rows: list[dict] = []
    for path, lineno, row in _iter_jsonl(files):
        all_rows.append(row)
        flag = row.get("accepted")
        if not isinstance(flag, bool):
            problems.append(f"{path.name}:{lineno}: accepted is not a boolean ({flag!r})")
        elif flag:
            n_accepted += 1
        else:
            n_rejected += 1
    if n_accepted == 0:
        problems.append("no accepted memory notes")
    if n_rejected == 0:
        problems.append("no rejected memory notes")

    # Served text is compact_notes output; it must carry no bookkeeping at all.
    try:
        from thermo_agent.teacher import memory as mem
        from thermo_agent.teacher.compact_memory import compact_notes

        notes = []
        for path in files:
            try:
                notes.extend(mem.load(path))
            except Exception:
                continue
        served = compact_notes(notes, k=len(notes))
        literals = tuple(sorted({
            str(n.source or "") for n in notes if n.source
        })) + tuple(sorted({str(n.id or "") for n in notes if n.id}))
        hits = _bookkeeping_hits(served, literals)
        if hits:
            problems.append(f"served compact_notes leaks {hits}")
    except Exception as exc:  # pragma: no cover - defensive, corpus only
        problems.append(f"could not render compact_notes: {exc}")

    detail = (
        f"{len(all_rows)} notes ({n_accepted} accepted / {n_rejected} rejected); "
        f"{len(problems)} problem(s)"
    )
    return CheckResult(name, not problems, detail, problems)


# --------------------------------------------------------------------------- #
# checks 7 & 8 -- mix propagation and leakage
# --------------------------------------------------------------------------- #
def _read_mix(sft_dir: pathlib.Path, grpo_dir: pathlib.Path):
    rows = []
    for stream, directory in (("sft", sft_dir), ("grpo", grpo_dir)):
        for path, lineno, row in _iter_jsonl(_jsonl_files(directory)):
            rows.append((stream, path, lineno, row))
    return rows


def _load_manifest(sft_dir: pathlib.Path, grpo_dir: pathlib.Path):
    for directory in (sft_dir, grpo_dir):
        path = directory / "MANIFEST.json"
        if path.is_file():
            try:
                return path, json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                return path, None
    return None, None


def _legacy_frontier_index(dirs) -> dict[str, set[str]]:
    """Map legacy-trace id / record_id to the set of explicit model ids on those rows.

    The mix builder adds a ``-ft-tool_cot-<hash>`` suffix to the row ``id`` while
    keeping the base ``record_id``, which equals the legacy store's ``id`` (see
    docs/PROVENANCE.md on canonical ids). Indexing both keys on each side makes
    that carry-forward traceable without trusting the suffix.
    """
    index: dict[str, set[str]] = collections.defaultdict(set)
    for directory in dirs:
        directory = _resolve(directory)
        for _, _, row in _iter_jsonl(_jsonl_files(directory, recursive=True)):
            model = _model_of(row)
            for key in (str(row.get("id") or ""), _row_id(row)):
                if key:
                    index[key].add(model)
    return index


def check_mix_propagation(
    sft_dir: pathlib.Path, grpo_dir: pathlib.Path, cohort: pathlib.Path,
    legacy_dirs=LEGACY_FRONTIER_DIRS,
) -> CheckResult:
    name = CHECK_NAMES[6]
    cohort_rows = [row for _, _, row in _iter_jsonl(_jsonl_files(cohort, recursive=True))]
    cohort_ids = {str(row.get("id") or "") for row in cohort_rows}
    cohort_record_ids = {_row_id(row) for row in cohort_rows}
    legacy_index = _legacy_frontier_index(legacy_dirs)
    problems: list[str] = []
    if not cohort_rows:
        problems.append("no cohort rows to trace against")

    mix_rows = _read_mix(sft_dir, grpo_dir)
    n_frontier = n_cohort = n_legacy = n_deterministic = 0
    for stream, path, lineno, row in mix_rows:
        where = f"{stream}/{path.name}:{lineno}"
        source = str(row.get("source") or "")
        if _is_ai(row) or source == CANONICAL_FRONTIER_SOURCE:
            n_frontier += 1
            rid = str(row.get("id") or "")
            rec = _row_id(row)
            if rid in cohort_ids or rec in cohort_record_ids:
                n_cohort += 1
                continue

            # Not in the current luna/haiku cohort: it may be a legacy accepted
            # trace carried forward from a documented early-campaign store. Trace
            # it through those stores, but stay strict about identity: both sides
            # must name the same model explicitly, never accept an unlabelled row.
            store_models: set[str] = set()
            for key in (rid, rec):
                if key:
                    store_models |= legacy_index.get(key, set())
            if not store_models:
                problems.append(
                    f"{where}: frontier row {rid or rec!r} has no cohort match "
                    "and no legacy store match")
                continue
            prov = _prov(row)
            declared = _model_of(row)
            explicitly_named = bool(prov.get("model")) or source.startswith("frontier:")
            if not explicitly_named or not declared:
                problems.append(
                    f"{where}: legacy frontier row {rid or rec!r} has no explicit "
                    "model in provenance/source")
                continue
            if len(store_models) != 1 or declared not in store_models:
                problems.append(
                    f"{where}: legacy frontier row model {declared!r} disagrees with "
                    f"legacy store model(s) {sorted(store_models)}")
                continue
            n_legacy += 1
        else:
            n_deterministic += 1
            if source not in DETERMINISTIC_SOURCES:
                problems.append(f"{where}: deterministic row has source {source!r}")
            family = _prov(row).get("source_family")
            if family and family != "deterministic":
                problems.append(f"{where}: deterministic row has source_family {family!r}")

    manifest_path, manifest = _load_manifest(sft_dir, grpo_dir)
    if manifest is None:
        problems.append(
            "no readable MANIFEST.json in mix dirs"
            + (f" (found {manifest_path})" if manifest_path else "")
        )
    else:
        for stream in ("sft", "grpo"):
            actual_by_task: dict[str, int] = {}
            for st, _, _, row in mix_rows:
                if st == stream:
                    task = str(row.get("task") or "")
                    actual_by_task[task] = actual_by_task.get(task, 0) + 1
            tasks = (manifest.get("counts") or {}).get(stream, {})
            for task, expected in tasks.items():
                want = int((expected or {}).get("total", 0))
                got = actual_by_task.get(task, 0)
                if got != want:
                    problems.append(
                        f"manifest {stream}/{task}: {want} rows but files have {got}"
                    )
            totals = (manifest.get("totals") or {}).get(stream)
            if totals is not None and int(totals) != sum(actual_by_task.values()):
                problems.append(
                    f"manifest totals.{stream}: {totals} rows but files have "
                    f"{sum(actual_by_task.values())}"
                )

    detail = (
        f"{n_frontier} frontier rows traced ({n_cohort} cohort, {n_legacy} legacy); "
        f"{n_deterministic} deterministic rows; {len(problems)} problem(s)"
    )
    return CheckResult(name, not problems, detail, problems)


def check_mix_leakage(sft_dir: pathlib.Path, grpo_dir: pathlib.Path) -> CheckResult:
    name = CHECK_NAMES[7]
    problems: list[str] = []
    mix_rows = _read_mix(sft_dir, grpo_dir)
    if not mix_rows:
        problems.append("no mix rows found")
    n_rows = 0
    for stream, path, lineno, row in mix_rows:
        n_rows += 1
        if "ref_spec" in _deep_keys(row):
            problems.append(f"{stream}/{path.name}:{lineno} carries ref_spec")
        elif "ref_spec" in _prompt_text(row):
            problems.append(f"{stream}/{path.name}:{lineno} prompt mentions ref_spec")
    detail = f"{n_rows} mix rows inspected; {len(problems)} ref_spec leak(s)"
    return CheckResult(name, not problems, detail, problems)


# --------------------------------------------------------------------------- #
# aggregate + CLI
# --------------------------------------------------------------------------- #
def run_audit(
    cohort: pathlib.Path | str,
    sft: pathlib.Path | str,
    grpo: pathlib.Path | str,
    memory: pathlib.Path | str,
    prompts: pathlib.Path | str | None = None,
    sample: int = 25,
    rebuild: bool = True,
    legacy_dirs=LEGACY_FRONTIER_DIRS,
) -> list[CheckResult]:
    """Run all eight checks. Paths may be absolute or repo-relative."""
    cohort = _resolve(cohort)
    sft = _resolve(sft)
    grpo = _resolve(grpo)
    memory = _resolve(memory)
    if prompts is None:
        prompts = ROOT / "data" / "frontier_tools_cohort_prompts"
    prompts = _resolve(prompts)
    return [
        check_provenance_fields(cohort),
        check_source_mapping(cohort),
        check_tiers(cohort),
        check_prompt_shards(prompts),
        check_harness_prompt_leakage(cohort, sample=sample, rebuild=rebuild),
        check_memory(memory),
        check_mix_propagation(sft, grpo, cohort, legacy_dirs=legacy_dirs),
        check_mix_leakage(sft, grpo),
    ]


def _print_report(results: list[CheckResult]) -> None:
    width = max(len(r.name) for r in results) + 2
    print(f"{'CHECK':<{width}} {'STATUS':<7} DETAIL")
    for result in results:
        print(f"{result.name:<{width}} {'OK' if result.ok else 'FAIL':<7} {result.detail}")
    print()
    print("PROVENANCE OK" if all(r.ok for r in results) else "PROVENANCE FAILED")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", default="data/frontier_tools_cohort")
    ap.add_argument("--prompts", default="data/frontier_tools_cohort_prompts")
    ap.add_argument("--sft", default="data/sft_mix_v3")
    ap.add_argument("--grpo", default="data/grpo_mix_v3")
    ap.add_argument("--memory", default="data/memory")
    ap.add_argument("--legacy-frontier", nargs="*", default=list(LEGACY_FRONTIER_DIRS),
                    help="legacy accepted-trace stores to trace mix frontier rows "
                         "through when they are not in the current cohort")
    ap.add_argument("--json", dest="json_path", default=None,
                    help="write a machine-readable report to this path")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any check fails")
    ap.add_argument("--sample", type=int, default=25,
                    help="number of cohort rows whose harness prompt is inspected")
    args = ap.parse_args(argv)

    results = run_audit(
        args.cohort, args.sft, args.grpo, args.memory, prompts=args.prompts,
        sample=args.sample, legacy_dirs=args.legacy_frontier,
    )
    _print_report(results)
    report = {
        "passed": all(r.ok for r in results),
        "checks": [r.as_dict() for r in results],
    }
    if args.json_path:
        out = pathlib.Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    return 1 if args.strict and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
