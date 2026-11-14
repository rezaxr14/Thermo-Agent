import json, pytest
from pydantic import ValidationError
from thermo_agent.core.schema import Plant, Component, Stream, SolveResult, ErrorCode

def _mini():
    return Plant(name="mini", components=[Component(id="AC", type="compressor", params={"eta_s": 0.86, "PR": 10})],
                 streams=[Stream(id="s1", **{"from": "ext:air", "to": "AC.in"}, fluid="CGAM_AIR"),
                          Stream(id="s2", **{"from": "AC.out", "to": "ext:out"}, fluid="CGAM_AIR")])

def test_roundtrip():
    p = _mini(); q = Plant.model_validate_json(p.model_dump_json(by_alias=True))
    assert p == q

def test_unknown_endpoint_rejected():
    with pytest.raises(ValidationError):
        Plant(name="bad", components=[], streams=[Stream(id="s", **{"from": "X.out", "to": "ext:y"}, fluid="Water")])

def test_error_codes_are_strings():
    assert SolveResult(ok=False, error=ErrorCode.TEMP_CROSS).model_dump(mode="json")["error"] == "E_TEMP_CROSS"
