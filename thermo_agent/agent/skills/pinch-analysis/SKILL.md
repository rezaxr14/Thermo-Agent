---
name: pinch-analysis
description: Use when matching hot and cold streams through heat exchangers: minimum approach, temperature crosses, utility targets and why an infeasible exchanger fails.
---
# Pinch analysis

Objective: match hot and cold streams so every exchanger satisfies the minimum approach and the required duty, and identify which set-point makes a match infeasible.

## Variants
| variant | when it fits | key limits |
| --- | --- | --- |
| counterflow matching | sensible heat exchange | approach at both ends |
| phase change | boiling or condensing side | pinch at bubble point |
| utility targeting | heating and cooling needs | smallest hot and cold utility |

## Physics and equations
- For a counterflow exchanger the two end approaches are `T_hot_in - T_cold_out` and `T_hot_out - T_cold_in`; the minimum of the two is the approach in K and must stay positive.
- A temperature cross occurs when a cold outlet exceeds the hot inlet, or when the specified duty exceeds what the hot stream can give; the solver raises E_TEMP_CROSS.
- At a boiling or condensing plateau the controlling difference is the bubble-point pinch; for an HRSG it is `T_gas_at_bubble - T_sat >= econ_approach_K`, and for an ORC `T_gas_in >= T_bubble + pinch_K`.
- Utility targets follow from the largest feasible recovery: shrinking the approach recovers more heat but raises area and irreversibility.

## Parameters and constraints
- Approaches are K, not fractions: `pinch_K` default 3.0, `econ_approach_K` default 15.0, `approach_K` on chillers and pumps default 5.0.
- Temperatures are K and duties W; a smaller approach is not automatically better.
- A hot outlet below the cold inlet is infeasible regardless of area.

## Failure modes
- V4 min approach below 0 K -> crossed streams -> swap the inlets or reduce the duty.
- E_TEMP_CROSS on an HRSG or ORC -> pinch above the available gas temperature -> lower the steam/evaporation pressure or the pinch.
- E_TEMP_CROSS at a chiller -> rejection warmer than evaporation -> cool the rejection loop.

## Tool checks
- `validity_check` -> `min_approach_K` per component and the verdict.
- `heat_exchanger`: pass a JSON object string in the `candidate` slot with the outlet or duty; read both outlet temperatures and the approach.

## Answer habits
- Name the exact end where the approach fails, with both temperatures in K.
- Fix the set-point that controls the cross rather than adding area or a new component.
- Quote the pinch from the tool, never from a design rule of thumb.
