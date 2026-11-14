"""The validator must accept the ground truth and reject plausible wrong answers, with real reasons."""
import pytest

from thermo_agent.teacher import validate

from _rows import make_row


@pytest.mark.parametrize("fault", ["wrong_fluid", "unit_slip", "sign_flip", "efficiency_gt_1", "swapped_ports"])
def test_ground_truth_is_accepted(fault):
    row = make_row(fault)
    v = validate.validate("repair", row["completion"][0]["content"], row)
    assert v["accepted"], (fault, v["reasons"])
    assert v["exact"] is True, "the expert answer restores the original plant exactly"


def test_no_edit_block_is_rejected_with_a_format_reason():
    row = make_row("wrong_fluid")
    v = validate.validate("repair", "<diagnosis>nothing</diagnosis>", row)
    assert not v["accepted"] and "format" in v["reasons"][0]


def test_echoing_the_broken_value_is_rejected():
    row = make_row("wrong_fluid")
    cands = dict(validate.adversarial_candidates(row))
    v = validate.validate("repair", cands["echo_broken"], row)
    assert not v["accepted"]


def test_a_third_fluid_can_be_valid_but_is_not_exact():
    """The tool suite judges validity, not identity: an alternate fluid that still solves is a valid repair.

    That is the right acceptance rule for RL (the reward is physical validity), and it is why `exact` is
    reported separately. The memory miner only records rejections, so this candidate produces no note.
    """
    row = make_row("wrong_fluid")
    cands = dict(validate.adversarial_candidates(row))
    v = validate.validate("repair", cands["third_fluid"], row)
    if v["accepted"]:
        assert v["exact"] is False
    else:
        assert v["reasons"]


def test_a_nonexistent_port_is_rejected_before_solving():
    row = make_row("swapped_ports")
    cands = dict(validate.adversarial_candidates(row))
    v = validate.validate("repair", cands["bad_port"], row)
    assert not v["accepted"] and any("edit" in r or "port" in r for r in v["reasons"])


def test_teacher_row_refuses_rejected_answers_and_keeps_task_columns():
    row = make_row("unit_slip")
    good = validate.validate("repair", row["completion"][0]["content"], row)
    out = validate.teacher_row("repair", row["completion"][0]["content"], row, good)
    assert out["source"] == "frontier" and out["plant_broken"] == row["plant_broken"]
    assert out["split"] == row["split"]
    with pytest.raises(ValueError):
        validate.teacher_row("repair", "nope", row, {"accepted": False, "reward": 0.0, "reasons": ["x"]})


def test_rejection_note_carries_the_class_lesson_and_the_pattern():
    row = make_row("wrong_fluid")
    cands = dict(validate.adversarial_candidates(row))
    v = validate.validate("repair", cands["echo_broken"], row)
    n = validate.rejection_note("repair", row, v)
    assert n.task == "repair" and n.fault_type == "wrong_fluid"
    assert "destination component fixes the fluid class" in n.lesson
    assert n.fix.startswith("FLUID") or n.fix.startswith("SET") or n.fix.startswith("REWIRE")
    assert n.record_id == row["id"]


def test_mine_failures_produces_distinct_lessons():
    notes = validate.mine_failures([make_row("wrong_fluid")], limit=1)
    assert notes, "at least one adversarial candidate must be rejected"
    assert len({n.id for n in notes}) == len(notes)
    assert all(not n.accepted for n in notes)
