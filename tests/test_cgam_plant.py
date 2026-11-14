"""Reference-plant checks against published CGAM-plus-ORC results.
Gated at 1 % (efficiencies, main destructions) and 3 % (ORC side). ORC superheat and HRSG pinch are
sensitive to property-data differences and are not gated."""
import pytest
from thermo_agent.core.cgam_plant import solve

R = solve()
rel = lambda a, b: abs(a - b) / abs(b)

@pytest.mark.parametrize("key,ref", [("eta_I", 0.8389), ("eta_II", 0.5157), ("eta_I_gt", 0.3651),
        ("eta_I_cogen", 0.8237), ("eta_II_cogen", 0.5028), ("W_GT", 59.661), ("W_AC", 29.661), ("Q_in", 82.176)])
def test_headline(key, ref): assert rel(R[key], ref) < 0.01

@pytest.mark.parametrize("c,ref", [("CC", 25.61), ("AP", 2.556), ("AC", 2.101), ("GT", 2.994), ("HRSG", 6.454)])
def test_exergy_destruction_main(c, ref): assert rel(R["ED"][c], ref) < 0.01

@pytest.mark.parametrize("c,ref", [("EVA", 0.5899), ("ORCT", 0.1969)])
def test_exergy_destruction_orc(c, ref): assert rel(R["ED"][c], ref) < 0.03

def test_orc_power(): assert rel(R["W_orct"], 1.286) < 0.03 and rel(R["W_pump"], 0.0353) < 0.01
def test_dew_point(): assert abs(R["T_DP_C"] - 41.98) < 0.1
def test_stream_exergies():
    for i, ref in {2: 27.56, 3: 41.27, 4: 100.7, 5: 38.07, 6: 21.8, 9: 12.83, 10: 85.06}.items():
        assert rel(R["e"][i], ref) < 0.01
def test_exergy_closure(): assert abs(R["closure_residual"]) < 1e-6
