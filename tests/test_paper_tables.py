import json
from thermo_agent.eval import paper_tables as PT

REP = {"meta": {"adapter": "runs/sft", "harness": "none"},
       "repair": {"test_id": {"n": 20, "success": 0.8, "success_ci95": [0.58, 0.92], "first_turn_success": 0.7, "exact_on_deducible": 0.5, "mean_solves": 1.3},
                  "test_ood_fault": {"n": 20, "success": 0.4, "success_ci95": [0.22, 0.61], "first_turn_success": 0.3, "exact_on_deducible": None, "mean_solves": 2.0}},
       "ledger": {"test_id": {"n": 10, "exact": 0.9, "closure": 1.0, "all_correct": 0.7, "all_correct_ci95": [0.4, 0.89]}}}

def test_latex_has_intervals_missing_values_and_escaped_names():
    t = PT.latex({"reports/sft_run.json": REP})
    assert "0.80 (0.58--0.92)" in t and "--" in t and r"sft\_run" in t and "\\toprule" in t and "ledger results" in t and "audit results" not in t

def test_cli_writes_tables_and_figures(tmp_path):
    p = tmp_path / "a.json"; p.write_text(json.dumps(REP)); out = tmp_path / "gen"
    reports = {str(p): REP}; (out).mkdir(); (out / "tables.tex").write_text(PT.latex(reports)); files = PT.plots(reports, out)
    assert files and all(f.exists() and f.stat().st_size > 1000 for f in files) and (out / "repair_success.png").exists()
