---
name: thermo-tools
description: Use the continuous thermodynamics tools: how to pass a JSON object string in the candidate slot and which number each tool returns.
---
# Thermodynamics tools

Objective: call the right physics tool with a well-formed JSON object and read the number it returns, without expecting an answer oracle.

## Variants
| variant | when it fits | key limits |
| --- | --- | --- |
| property | a state or saturation is uncertain | SI inputs only |
| component | size or check one machine | one component per call |
| plant | check a full candidate | needs a parsed plant or design |

## Physics and equations
- Every tool has signature `fn(candidate: str) -> str`. The shared parser calls `fn(payload.get("candidate", ""))`, so the JSON object goes in the `candidate` slot as a string, e.g. `{"candidate": "{\"fluid\": \"Water\", \"T_K\": 300, \"P_kPa\": 101.325}"}`.
- Tool inputs use K, kPa, kJ/kg, kg/s and W; the solver itself uses Pa and J/kg. Fractions are never percent.
- Tools return JSON with `ok` and computed numbers plus units; there is no verdict or acceptance tool, so read and revise.

## Parameters and constraints
- `fluid_state`: fluid, T/P or h/P -> phase, `h_kJ_kg`, `s_kJ_kgK`.
- `saturation_state`: fluid plus P or T -> `T_sat_K`, quality x and saturated enthalpies.
- `turbomachine`: kind plus `PR` or `P_out` and `eta_s` -> outlet state and shaft power.
- `heat_exchanger`: inlet states and an outlet or duty -> both outlets, duty and approach.
- `combustor`: `T_out`, `dp`, `q_loss` -> fuel flow and outlet state.

## Failure modes
- Malformed JSON or a missing key -> tool error -> fix the object before re-calling.
- A bar pressure or an offset-scale temperature -> E_PROP_RANGE -> convert to SI and retry.
- Treating a tool number as a verdict -> wrong final answer -> cite only the numbers shown.

## Tool checks
- `cycle_performance`: cycle inputs -> efficiency, work and heat; `flow_network`: nodes and streams -> flows and balances.
- `psychrometrics`: dry-bulb, pressure and relative humidity -> wet-bulb and humidity ratio; `exergy_stream`: state and `T0` -> physical and chemical exergy.
- Use the plant-level tools (`inspect_plant`, `first_law_check`, `second_law_check`, `validity_check`, `solve_plant`) to confirm any change.

## Answer habits
- Pass one JSON object per call and read the returned field names exactly.
- Convert units on the way in and quote the tool's units on the way out.
- Stop calling tools once residuals are near zero and every check passes.
