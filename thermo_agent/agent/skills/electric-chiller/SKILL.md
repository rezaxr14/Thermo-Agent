---
name: electric-chiller
description: Use when a plant contains an electric_chiller: vapour-compression cooling whose part-load COP depends on temperatures, approach and fouling.
components: [electric_chiller]
variants: [centrifugal, screw, scroll]
---
# Electric chiller

Objective: deliver the chilled-water duty with the least electricity, keeping the heat-rejection temperature low enough for a useful COP and no temperature cross.

## Variants
| variant | when it fits | key limits |
| --- | --- | --- |
| centrifugal | large flow, high part-load | surge at low load |
| screw | medium load, stable part-load | oil and leakage losses |
| scroll | small load | fixed displacement, low ceiling |

## Physics and equations
- `COP = eps * (Te - approach_K) / ((Tr + approach_K) - (Te - approach_K))`, with `Te` and `Tr` the log-mean chilled and rejection stream temperatures; electricity `W = Q_cooling/COP` in W.
- Chilled duty `Q_cooling = m_chw*(h_evap_in - h_out)` with `T_chw_out` setting the outlet; rejection flow is solved as `(Q_cooling + W)/(h_rej_out - h_rej_in)`.
- A hotter rejection loop lowers COP; the plant pays for it as extra parasitic electricity and more fuel at the prime mover.
- Part-load behaviour is captured by `eps`; fouling raises the effective approach and lowers COP, which is not a separate parameter.

## Parameters and constraints
- `T_chw_out`: chilled-water outlet (K), required, no default; it must be below the inlet.
- `dT_rej`: rejection water rise (K), required and positive; `eps`: fraction of reversible COP (-), default 0.5, at or below 1.
- `approach_K`: evaporator/condenser approach (K), default 5.0; larger values mean worse heat transfer and lower COP.

## Failure modes
- E_TEMP_CROSS -> rejection temperature not above evaporator temperature -> lower `dT_rej` or cool the rejection loop.
- E_PROP_RANGE -> `T_chw_out` above inlet, `dT_rej` <= 0, or `eps` above 1 -> correct the values and units (K).
- V8 electricity shortfall -> a hot rejection loop collapsed COP -> improve rejection before resizing the plant.

## Tool checks
- `cycle_performance`: pass a JSON object string in the `candidate` slot, e.g. `{"kind": "chiller", "T_hot_K": 305, "T_cold_K": 280, "eta_second_law": 0.5}`; read `real_cop` and the Carnot limit.
- `second_law_check` -> COP versus the reversible limit; `solve_plant` -> chilled and rejection outlet temperatures.

## Answer habits
- Quote COP, duty and electricity only as the tools report them.
- Do not specify the rejection flow: it is solved from the duty and COP.
- Fix a hot rejection loop before adding electrical capacity.
