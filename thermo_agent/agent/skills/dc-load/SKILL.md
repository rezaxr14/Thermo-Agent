---
name: dc-load
description: Use when a plant contains a dc_load: the IT electrical demand, the fraction of its heat sent to coolant and the required coolant return temperature.
components: [dc_load]
variants: [air_cooled, liquid_cooled, two_phase]
---
# Data-centre IT load

Objective: serve the IT electrical demand and route its heat to the coolant at a return temperature the cooling plant can actually handle.

## Variants
| variant | when it fits | key limits |
| --- | --- | --- |
| air_cooled | legacy halls | high supply temperature, large flow |
| liquid_cooled | high-density racks | coolant ΔT and return limit |
| two_phase | highest density | tight return-temperature control |

## Physics and equations
- `W = -P_it` in W is the IT electricity demand; heat to coolant `Q = chilled_fraction * P_it`, and the remainder `(1 - chilled_fraction)*P_it` leaves directly to ambient on the air side.
- Coolant flow is solved: `m_dot = Q/(h(T_return) - h(T_supply))`; the return temperature is `T_return` and must exceed the supply temperature.
- PUE is a site metric, not a solver parameter: a lower facility overhead means less grid import for the same served IT load.
- A higher return temperature helps heat recovery but may exceed equipment limits; a lower one raises flow and pump power.

## Parameters and constraints
- `P_it`: IT electrical power (W), required, no default; a kW/MW slip fails unit and balance checks.
- `chilled_fraction`: share of heat to the chilled loop (-), default 1.0, in (0, 1]; `T_return`: coolant return temperature (K), required and above the inlet.
- The coolant is a liquid CoolProp fluid; the flow is solved, so a fixed `m_dot` raises E_OVER_SPECIFIED.

## Failure modes
- E_PROP_RANGE -> `T_return` not above supply, or `chilled_fraction` outside (0, 1] -> correct the value.
- E_OVER_SPECIFIED -> coolant flow given -> remove it.
- V8 IT load not served or grid import -> `P_it` unit slip or an undersized prime mover -> check W and resize.

## Tool checks
- `first_law_check` -> `P_IT_W`, `Q_cooling_W` and per-component balances.
- `solve_plant` -> coolant inlet/outlet `T_K` and solved flow; `validity_check` -> site service of the IT load.

## Answer habits
- State `P_it` in W and temperatures in K.
- Never set the coolant flow: it is solved from the load and `T_return`.
- Quote the return temperature from the plant, not from a rack datasheet.
