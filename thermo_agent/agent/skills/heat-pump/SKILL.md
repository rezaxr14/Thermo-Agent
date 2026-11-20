---
name: heat-pump
description: Use when a plant contains a heat_pump: heat lifted from a source stream to a hotter sink, limited by Lorenz COP, sink temperature, approach and lift.
components: [heat_pump]
variants: [air_source, water_source, ground_source]
---
# Heat pump

Objective: deliver the required sink heat with the least electricity while keeping the sink hotter than the source and the lift inside a feasible COP.

## Variants
| variant | when it fits | key limits |
| --- | --- | --- |
| air_source | no water loop | defrost and low ambient lift |
| water_source | chiller rejection loop available | source cooling must stay above freeze |
| ground_source | stable source temperature | pumping and ground-loop limits |

## Physics and equations
- `COP = eps * (Ts + approach_K) / ((Ts + approach_K) - (Tc - approach_K))`, where `Ts` and `Tc` are log-mean sink and source temperatures; electricity `W = Q_sink - Q_source` in W.
- With `Q_sink` the solver sizes the source cooling and iterates the source outlet temperature; with `dT_src` it fixes the source drop and derives the duty. The sink flow is solved in both cases.
- Lorenz COP is the reversible limit; the actual COP is a fraction `eps` of it and must exceed 1. A larger lift (hotter sink or colder source) reduces COP and raises electricity.
- Defrost and freeze protection are operational limits, not parameters: keep the source outlet above the fluid freeze point, especially for air-source units.

## Parameters and constraints
- `T_sink_out`: sink outlet (K), required and above the sink inlet; `eps`: fraction of Lorenz COP (-), default 0.5; `approach_K`: heat-exchanger approach (K), default 5.0.
- `Q_sink`: heat duty (W) or `dT_src`: source-side drop (K); one of the two is required, and if both are given `Q_sink` is used.
- The sink flow is solved; giving a sink `m_dot` raises E_OVER_SPECIFIED.

## Failure modes
- E_TEMP_CROSS -> sink not hotter than source -> raise `T_sink_out` or warm the source.
- E_PROP_RANGE -> COP <= 1 or `dT_src` <= 0 -> reduce the lift or correct the parameter.
- E_OVER_SPECIFIED -> fixed sink flow -> remove it; E_NOT_CONVERGED -> source temperature iteration failed -> relax `Q_sink`.

## Tool checks
- `cycle_performance`: pass a JSON object string in the `candidate` slot, e.g. `{"kind": "heat_pump", "T_hot_K": 340, "T_cold_K": 290, "eta_second_law": 0.5}`; read `real_cop` and `lorenz_cop`.
- `second_law_check` -> COP versus reversible limit; `solve_plant` -> source and sink outlet temperatures.

## Answer habits
- Choose exactly one of `Q_sink` and `dT_src` unless you intend `Q_sink` to win.
- Treat electricity as a parasitic load that raises fuel elsewhere in the plant.
- Quote sink temperature in K and duties in W.
