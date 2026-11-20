---
name: exergy-accounting
description: Use for second-law reasoning: physical and chemical exergy, ambient T0, flow exergy, destruction, closure and avoiding negative destruction.
---
# Exergy accounting

Objective: account for exergy so that every stream and component has a consistent exergy value and destruction is non-negative and closes to the boundary balance.

## Variants
| variant | when it fits | key limits |
| --- | --- | --- |
| physical_exergy | temperature and pressure above ambient | vanishes at T0 and P0 |
| chemical_exergy | fuels and non-ambient mixtures | zero for ambient air and liquids |
| destruction | any real component | T0 times entropy generation, never negative |

## Physics and equations
- Specific flow exergy `e = (h - h0) - T0*(s - s0) + e_ch` in J/kg, with the dead state at ambient `T0`, `P0`; chemical exergy `e_ch` matters for fuel and exhaust, and is zero for water and ambient air.
- Stream exergy rate `E = m_dot*e` in W; component destruction `ED = sum(inlet E) - sum(outlet E) - W` in W, so a work-consuming component has a negative W term.
- Closure: total destruction equals boundary exergy in minus boundary exergy out minus net work; internal streams cancel with opposite signs.
- Destruction is `T0 * S_gen >= 0`. A negative value means a wrong line, sign or reference, not a better machine.

## Parameters and constraints
- `T0` is the ambient temperature from the plant in K; it is not 298.15 by default and never 25.
- Efficiencies and second-law fractions such as `eps` are fractions, never percent.
- Exergy destruction is reported in MW by the tools, while residuals are in W.

## Failure modes
- Negative destruction -> wrong sign in a component line or a stream's e_ch -> fix the expression, not the plant.
- Closure residual -> a boundary stream or work term is wrong -> recheck the in/out list.
- Large destruction in the condenser -> normal: heat leaves near ambient with little exergy.

## Tool checks
- `exergy_stream`: pass a JSON object string in the `candidate` slot, e.g. `{"fluid": "FLUE", "T_K": 800, "P_kPa": 101.325, "m_dot_kg_s": 1, "T0_K": 300, "composition": {"N2": 0.75, "O2": 0.06, "CO2": 0.12, "H2O": 0.07}}`; read physical and chemical exergy.
- `second_law_check` -> per-component destruction (MW), closure residual (W) and eta_II; `solve_plant` -> e_ph and e_ch per stream.

## Answer habits
- State T0 from the plant before writing any exergy line.
- Report destruction in MW and residuals in W, with the sign convention explicit.
- Never fix a plant to hide negative destruction: fix the ledger expression.
