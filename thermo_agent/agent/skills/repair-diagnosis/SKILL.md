---
name: repair-diagnosis
description: Use for Repair tasks: inspect the broken plant first, classify the fault, make the fewest edits, then re-solve and re-check.
task: repair
---
# Repair diagnosis

Objective: restore a valid plant with the smallest possible change by identifying exactly which value, sign, port or fluid the failure points at.

## Variants
| variant | when it fits | key limits |
| --- | --- | --- |
| unit_or_sign | a W/kW/MW slip or a flipped drop | one number or sign |
| port_or_tear | swapped endpoints or an unconverged tear | rewire or matching fixed T |
| range_or_fluid | a parameter out of range or wrong fluid class | one value or fluid |

## Physics and equations
- Mass and energy are conserved at each component; a residual points at the stream or parameter carrying the error, not at the component that reports it.
- A tear stream's fixed T and P must equal the producing component's outlet; changing one side alone raises E_NOT_CONVERGED.
- Heat flows from hot to cold: the minimum approach in K must stay positive, so a duty or temperature set-point beyond the hot stream's ability gives E_TEMP_CROSS.
- Exergy destruction is non-negative; a negative value signals a wrong ledger expression or an efficiency above 1.

## Parameters and constraints
- Efficiencies and pressure-drop fractions are fractions, never percent; temperatures are K and pressures Pa; duties and powers W.
- Fault classes are unit, sign, port, fluid, range and tear; name the class before editing.
- Do not add or remove components, and do not change a solved flow such as engine air, chiller generator flow or IT coolant flow.

## Failure modes
- Compressor/turbine/pump: `eta_s` <= 0 or a wrong `PR`/`P_out` -> restore a positive fraction, Pa pressure and the loop pressure.
- Combustor: `T_out` unreachable or `q_loss`/`dp` too large -> lower the outlet temperature or loss.
- Exchangers/HRSG/ORC: pinch cross -> lower the hot-side outlet set-point or the pinch; check `useful_heat`.
- Chillers/heat pump/rejector: sink or rejection temperature inverted -> re-order the loops and check the tear's fixed T.

## Tool checks
- `inspect_plant` first, with no candidate, to read the as-given solver status and failed checks.
- `first_law_check` for residuals, `validity_check` for units and approach, `solve_plant` for the offending state, `second_law_check` for destruction.

## Answer habits
- Emit only `<diagnosis>` citing numbers the tool showed, then `<edits>` with `SET`, `REWIRE` or `FLUID` lines.
- Change one thing, re-solve, and stop when every check passes.
- Never invent a number: quote the input or the solved plant.
