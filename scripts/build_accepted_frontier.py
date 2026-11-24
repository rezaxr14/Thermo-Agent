"""Flatten accepted luna/haiku traces into a mix-ready frontier directory.

The mix script discovers `frontier_<task>_*.jsonl` at a single level; the cohort
ingest writes per-model subdirectories. This script copies the accepted rows into
`data/frontier_tools_accepted/` without rewriting their content.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_agent.agent import tool_protocol  # noqa: E402
SRC = ROOT / "data/frontier_tools_cohort"
DST = ROOT / "data/frontier_tools_accepted"
TASKS = ("repair", "design", "audit", "ledger")


def main() -> int:
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for model_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
        for path in sorted(model_dir.glob("frontier_*.jsonl")):
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                row = json.loads(line)
                # One protocol only: rewrite provider wrappers into the native
                # structure, and drop any row that still uses the retired
                # `<tool_call name=...>` text. No compatibility corpus.
                row = tool_protocol.normalize_train_row(row)
                if row is None:
                    continue
                # transfer.ingest writes `source="frontier:<model>"`. The mix and
                # provenance audit expect the canonical `frontier_tools` family.
                original_source = str(row.get("source") or "")
                model = (row.get("model") or (row.get("provenance") or {}).get("model")
                         or (original_source.split(":", 1)[1] if ":" in original_source else "unknown"))
                prov = dict(row.get("provenance") or {})
                prov["original_source"] = original_source
                prov["generator"] = "frontier_tools"
                prov["source_family"] = "ai_generated"
                prov["model"] = model
                row["source"] = "frontier_tools"
                row["source_model"] = model
                row["provenance"] = prov
                # Dedupe by the model-tagged row id, not the shared record_id: two
                # models that solved the same plant are two distinct pilot traces and
                # both belong in the per-model report.
                rid = str(row.get("id") or row.get("record_id") or "")
                if rid in seen:
                    continue
                seen.add(rid)
                task = str(row.get("task") or "repair")
                if task not in TASKS:
                    continue
                out = DST / f"frontier_{task}_train.jsonl"
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[task] = counts.get(task, 0) + 1
    (DST / "MANIFEST.json").write_text(
        json.dumps({"generator": "build_accepted_frontier", "rows": len(seen), "by_task": counts},
                   indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"rows": len(seen), "by_task": counts, "dir": str(DST.relative_to(ROOT))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
