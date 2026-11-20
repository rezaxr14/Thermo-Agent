---
name: thermo-properties
description: Use for any fluid-state question: CoolProp phase, saturation, quality x and unit traps when reading pressures, temperatures and specific energies.
---
# Thermodynamic properties

Objective: get the phase and state right before using a number, so that a unit or phase mistake does not masquerade as a solver fault.

## Variants
| variant | when it fits | key limits |
| --- | --- | --- |
| single-phase | compressed liquid or superheated gas | T and P are independent |
| saturated | boiling or condensing | T and P are linked |
| two-phase | quality between 0 and 1 | x in [0, 1], not percent |

## Physics and equations
- A single-phase state needs two independent properties: `state_TP(T, P)` (T in K, P in Pa) or `state_hP(h, P)` with h in J/kg and s in J/(kg K).
- Saturation fixes `T_sat = Tsat_P(P)`; the two-phase state is `state_sat(P, x)` with quality `x = m_vapour/(m_vapour + m_liquid)`, where x = 0 is saturated liquid and x = 1 saturated vapour.
- Flow exergy uses `h - h0 - T0*(s - s0)` in J/kg; the solver stores h and s in J/kg but tools report kJ/kg, a factor of 1000.
- Gas mixtures use ideal-gas composition; every other fluid is a CoolProp fluid. A phase guess that conflicts with the port class gives E_FLUID_MISMATCH.

## Parameters and constraints
- Pressure is Pa, never bar: a 20 Pa or 20 bar value where Pa is expected fails V6/E_PROP_RANGE.
- Temperature is absolute K, never an offset scale: an ambient near 298 K must not be entered as 25.
- Energy is J in the solver and kJ in tool reports; efficiencies and pressure-drop fractions are fractions, never percent.

## Failure modes
- E_PROP_RANGE -> unit slip such as bar instead of Pa or an offset-scale temperature -> convert to SI.
- E_FLUID_MISMATCH -> a gas-class fluid on a liquid port -> choose the right fluid class.
- Negative exergy after a state lookup -> wrong dead-state reference -> use the fluid's own g0.

## Tool checks
- `fluid_state`: pass a JSON object string in the `candidate` slot, e.g. `{"fluid": "Water", "T_K": 300, "P_kPa": 101.325}`; read `phase`, `quality`, `h_kJ_kg`, `s_kJ_kgK`.
- `saturation_state`: pass `fluid` and exactly one of `P_kPa` or `T_K`, plus optional `quality`; read `T_sat_K`, `P_sat_kPa` and the saturated enthalpies.

## Answer habits
- Convert to SI before calling a tool, and quote the tool's units back.
- Use quality as a fraction, never a percentage.
- Call a property tool when a number is uncertain instead of guessing a phase.
