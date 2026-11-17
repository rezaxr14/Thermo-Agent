"""The baseline's per-fault breakdown must score the rows it names.

`run_eval.run(..., by_fault=True)` used to re-read the split once per fault and evaluate the *first* `len(sub)`
rows instead of the rows of that fault, so every per-fault row of the paper's baseline table was actually another
fault's sample. This test pins the breakdown to the individual rows.
"""
import json

from thermo_agent.eval.run_eval import run

from _rows import make_row


def test_by_fault_scores_the_rows_of_that_fault(tmp_path):
    rows = [make_row("sign_flip"), make_row("efficiency_gt_1")]
    d = tmp_path / "data"; d.mkdir()
    (d / "repair_test_id.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    bf = run(d, "test_id", None, by_fault=True)
    assert set(bf) == {r["fault_type"] for r in rows}
    for r in rows:
        one = tmp_path / r["fault_type"]; one.mkdir()
        (one / "repair_test_id.jsonl").write_text(json.dumps(r) + "\n", encoding="utf-8")
        assert bf[r["fault_type"]] == run(one, "test_id", None)
    assert bf["sign_flip"]["n"] == bf["efficiency_gt_1"]["n"] == 1
