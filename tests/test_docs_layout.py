"""Anti-litter rule (docs/DECISIONS.md 2026-11-20 item 1): docs/ holds exactly these files.

To record something new, add a dated section to the file that owns the topic. Do not add a file; do not edit this
allow-list without Reza's explicit yes. Generated logs belong in reports/, not docs/.
"""
import pathlib

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"
ALLOWED = {
    "PROJECT_OVERVIEW.md", "STATUS.md", "DECISIONS.md", "USER_PROMPTS_VERBATIM.md", "PROVENANCE.md",
    "EVAL_PROTOCOL.md", "TEACHER_CAMPAIGN.md", "AUDITS.md", "RUNBOOK.md", "IMPLEMENTATION_PLAN.md",
}


def test_docs_dir_has_only_the_owned_documents():
    found = {p.name for p in DOCS.iterdir() if p.is_file()}
    assert found == ALLOWED, (f"extra: {sorted(found - ALLOWED)}, missing: {sorted(ALLOWED - found)}. "
                              "Extend the owning doc with a dated section instead of adding a file.")


def test_no_decision_or_directive_text_outside_decisions_doc():
    stray = [p.name for p in DOCS.iterdir() if p.is_file() and p.name not in ALLOWED]
    assert not stray
