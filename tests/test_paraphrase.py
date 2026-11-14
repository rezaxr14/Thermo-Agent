"""Paraphrases must vary in wording and never in content."""
from thermo_agent.agent.dsl import parse_edits
from thermo_agent.synth import paraphrase

from _rows import make_row

FAULTS = ("wrong_fluid", "unit_slip", "sign_flip", "efficiency_gt_1", "swapped_ports")


def test_every_fault_class_has_at_least_twenty_variants():
    for f in FAULTS:
        row = make_row(f)
        n = paraphrase.variant_count(row)
        assert n >= 20, (f, n)
        assert len(set(paraphrase.variants(row))) == n, f"{f} has duplicate renderings"


def test_variants_are_distinct_and_keep_the_ground_truth_edits():
    for f in FAULTS:
        row = make_row(f)
        truth = parse_edits(row["completion"][0]["content"])
        texts = set()
        for i in range(paraphrase.variant_count(row)):
            text = paraphrase.completion(row, i)
            assert parse_edits(text) == truth, f"{f} variant {i} changed the edits"
            assert "<diagnosis>" in text and "</diagnosis>" in text
            texts.add(text.split("</diagnosis>")[0])
        assert len(texts) >= 20, f"{f} produced only {len(texts)} distinct diagnoses"


def test_the_numbers_in_the_diagnosis_come_from_the_fault_record():
    for f in ("unit_slip", "sign_flip", "efficiency_gt_1"):
        row = make_row(f)
        facts = paraphrase.facts(row)
        for text in paraphrase.variants(row):
            assert f"{facts['after']:g}" in text, (f, text)


def test_to_row_changes_only_the_completion_and_keeps_labels():
    row = make_row("unit_slip")
    out = paraphrase.to_row(row, 3)
    assert out["id"] == row["id"] and out["split"] == row["split"]
    assert out["plant_broken"] == row["plant_broken"]
    assert out["completion"][0]["content"] != row["completion"][0]["content"]
    assert out["paraphrase"]["n"] == paraphrase.variant_count(row)


def test_variant_index_is_deterministic_and_spreads_rows():
    picks = {paraphrase.variant_index(f"rep-0-{i:05d}", 80) for i in range(200)}
    assert len(picks) >= 40, "hash should spread rows across most templates"
    assert paraphrase.variant_index("rep-0-00001", 80) == paraphrase.variant_index("rep-0-00001", 80)
