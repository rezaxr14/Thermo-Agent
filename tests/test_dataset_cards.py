import pathlib, random, re
from thermo_agent.synth.repair_dataset import make_records, sample_plant
from thermo_agent.synth.sft_data import to_row
from thermo_agent.design.data import make_row as design_row, sample_site
from thermo_agent.audit.data import make_row as audit_row
from thermo_agent.ledger.data import make_row as ledger_row

CARD = (pathlib.Path(__file__).parent.parent / "docs" / "PROVENANCE.md").read_text()

def _documented(section):
    # dataset cards now live as "### <Task>" sections inside docs/PROVENANCE.md (merged 2026-11-20)
    m = re.search(rf"(?m)^###? {re.escape(section)}\s*$", CARD)
    body = CARD[m.end():]
    body = re.split(r"(?m)^#{2,3} ", body)[0]
    return set(re.findall(r"`(\w+)`", re.search(r"Columns: (.*)", body).group(1)))

def test_documented_columns_equal_actual_columns():
    rep = to_row(next(make_records(5, seed=2)))
    des = design_row("d", sample_site(random.Random(0)), max_evals=5)
    aud = audit_row("a", sample_plant(random.Random(41), topology="dc_absorption", rich=True)[0])
    lrng = random.Random(5); led = ledger_row("l", sample_plant(lrng, topology="dc_electric")[0], lrng)
    assert aud is not None and des is not None and led is not None and _documented("Ledger") == set(led)
    assert _documented("Repair") == set(rep) and _documented("Design") == set(des) and _documented("Audit") == set(aud)
