"""Plant-spec schema: the single contract between simulator, verifiers, trace generator, and LLM.

Design rules
  * SI base units everywhere (K, Pa, kg/s, W, J/kg). Reporting layers convert to MW / bar / degC.
  * A plant is a typed graph: components (nodes) + streams (edges). Boundary streams use "ext:<name>".
  * Everything an LLM may be asked to emit is JSON-serialisable and validated by pydantic *before*
    the solver runs, so format errors are separated from physics errors (distinct ErrorCode values).
  * Repair records carry the exact inverse of the injected fault so reward can be exact-match
    as well as physics-based.
"""
from __future__ import annotations
from enum import Enum
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "0.1.0"

ComponentType = Literal[
    "compressor", "combustor", "turbine", "recuperator", "hrsg", "orc_evaporator", "orc_turbine",
    "pump", "condenser", "absorption_chiller", "heat_pump", "electric_chiller", "dc_load",
    "mixer", "splitter", "valve", "heat_rejector", "gas_engine",
]
Scalar = Union[float, int, str, bool]


class ErrorCode(str, Enum):
    SCHEMA = "E_SCHEMA"                    # pydantic validation failed
    UNKNOWN_FLUID = "E_UNKNOWN_FLUID"
    UNKNOWN_COMPONENT = "E_UNKNOWN_COMPONENT"
    FLUID_MISMATCH = "E_FLUID_MISMATCH"    # stream fluid class (gas / liquid) does not fit the port it is connected to
    DISCONNECTED = "E_DISCONNECTED"        # dangling port / unreachable node
    OVER_SPECIFIED = "E_OVER_SPECIFIED"
    UNDER_SPECIFIED = "E_UNDER_SPECIFIED"
    PROP_RANGE = "E_PROP_RANGE"            # CoolProp/NASA out of validity range
    NOT_CONVERGED = "E_NOT_CONVERGED"
    MASS_IMBALANCE = "E_MASS_IMBALANCE"
    ENERGY_IMBALANCE = "E_ENERGY_IMBALANCE"
    NEG_EXERGY_DESTRUCTION = "E_NEG_EXERGY_DESTRUCTION"
    TEMP_CROSS = "E_TEMP_CROSS"            # pinch violated / second-law-infeasible heat transfer
    CARNOT_VIOLATION = "E_CARNOT_VIOLATION"
    UNIT_RANGE = "E_UNIT_RANGE"            # value outside physically plausible range for its unit


class FixedState(BaseModel):
    """Boundary / specified thermodynamic state of a stream. Any two of (T, P, x, h) fix the state."""
    T: Optional[float] = Field(None, description="K")
    P: Optional[float] = Field(None, description="Pa")
    x: Optional[float] = Field(None, ge=0, le=1, description="vapour quality")
    h: Optional[float] = Field(None, description="J/kg")


class Stream(BaseModel):
    id: str
    src: str = Field(alias="from", description="'<component>.<port>' or 'ext:<name>'")
    dst: str = Field(alias="to")
    fluid: str = Field(description="CoolProp name, or 'CGAM_AIR' / 'FLUE' for the ideal-gas mixtures")
    m_dot: Optional[float] = Field(None, gt=0, description="kg/s; None = solved")
    fixed: Optional[FixedState] = None
    composition: Optional[dict[str, float]] = Field(None, description="mole fractions for ideal-gas streams without a preset fluid name")
    model_config = {"populate_by_name": True}


class Component(BaseModel):
    id: str
    type: ComponentType
    params: dict[str, Scalar] = Field(default_factory=dict,
                                      description="efficiencies [-], pressure ratios [-], dP fractions [-], pinch [K], ...")


class Ambient(BaseModel):
    T0: float = 298.15
    P0: float = 101325.0
    RH: float = 0.6


class Site(BaseModel):
    """Design-task input: what the customer wants, nothing about the plant."""
    it_load_w: float = Field(gt=0)
    cooling_load_w: Optional[float] = None
    hourly_load_profile_id: Optional[str] = None
    ambient: Ambient = Ambient()
    fuel: Literal["CH4"] = "CH4"
    grid_import_allowed: bool = True
    heat_demand_w: float = 0.0
    max_water_kg_s: Optional[float] = Field(None, description="cap on evaporative make-up water; None = unconstrained")
    t_dh_supply_k: Optional[float] = Field(None, description="district-heat supply temperature")
    t_dh_return_k: Optional[float] = Field(None, description="district-heat return temperature")
    t_supply_k: Optional[float] = Field(None, description="IT coolant supply temperature")
    t_return_k: Optional[float] = Field(None, description="IT coolant return temperature")


class Plant(BaseModel):
    schema_version: str = SCHEMA_VERSION
    name: str
    ambient: Ambient = Ambient()
    components: list[Component]
    streams: list[Stream]
    notes: str = ""

    @model_validator(mode="after")
    def _unique_and_connected(self):
        cids = [c.id for c in self.components]
        if len(set(cids)) != len(cids):
            raise ValueError("duplicate component id")
        sids = [s.id for s in self.streams]
        if len(set(sids)) != len(sids):
            raise ValueError("duplicate stream id")
        known = set(cids)
        for s in self.streams:
            for end in (s.src, s.dst):
                if not end.startswith("ext:") and end.split(".")[0] not in known:
                    raise ValueError(f"stream {s.id}: unknown endpoint {end}")
        return self


# ---------- solver output ----------
class StatePoint(BaseModel):
    stream: str
    T: float; P: float; h: float; s: float; m_dot: float
    x: Optional[float] = None
    e_ph: float = Field(description="physical specific exergy J/kg")
    e_ch: float = Field(0.0, description="chemical specific exergy J/kg")


class ComponentBalance(BaseModel):
    component: str
    Q_w: float = 0.0; W_w: float = 0.0
    Q_loss_w: float = Field(0.0, description="heat dissipated to ambient (first-law loss)")
    mass_residual: float = 0.0; energy_residual: float = 0.0
    exergy_destruction_w: float = 0.0
    min_approach_K: Optional[float] = None


class SolveResult(BaseModel):
    ok: bool
    error: Optional[ErrorCode] = None
    message: str = ""
    states: list[StatePoint] = []
    balances: list[ComponentBalance] = []
    kpis: dict[str, float] = Field(default_factory=dict,
                                   description="eta_I, eta_II, W_net_w, Q_useful_w, ED_total_w, ...")


# ---------- Repair task record (JSONL line) ----------
FaultType = Literal["unit_slip", "sign_flip", "dropped_term", "wrong_fluid", "swapped_ports", "efficiency_gt_1",
                    "missing_stream", "wrong_pinch", "mixed_basis", "wrong_ref_state"]


class Fault(BaseModel):
    type: FaultType
    component: Optional[str] = None
    path: str = Field(description="JSON-pointer into the Plant, e.g. /components/3/params/eta_s")
    before: Scalar | None = None
    after: Scalar | None = None
    inverse_patch: list[dict] = Field(description="RFC-6902 JSON-patch that restores the original plant")
    deducible: bool = Field(True, description="whether the original value can be recovered from the broken plant and verifier output alone")


class RepairRecord(BaseModel):
    id: str
    split: Literal["train", "val", "test_id", "test_ood_fault", "test_ood_topology"] = "train"
    plant_original: Plant
    plant_broken: Plant
    site: Optional[Site] = Field(None, description="requirements the repaired plant must still meet (data-centre plants)")
    faults: list[Fault]
    solve_broken: SolveResult
    verifier_report_broken: dict[str, float | bool | str]
    provenance: dict[str, str] = Field(default_factory=dict, description="generator seed, schema/version hashes, CoolProp version")
