---
name: district-heat
description: Use when a plant supplies district heat through an exhaust heat exchanger flagged useful or a heat pump lifting chiller rejection heat to the network.
components: [heat_pump, recuperator]
variants: [exhaust_hx, heat_pump, storage]
---
# District heat

Objective: meet the network supply temperature and heat demand with the least fuel, using exhaust heat first and a heat pump only when the exhaust is committed elsewhere.

## Variants
| variant | when it fits | key limits |
| --- | --- | --- |
| exhaust_hx | heat remains after other recoveries | pinch and exhaust stack floor |
| heat_pump | exhaust needed for an ORC | lift sets COP and electricity |
| storage | heat and demand are not coincident | model as a duty buffer, not a component |

## Physics and equations
- Exhaust route: a `recuperator` with `useful_heat = true` heats network water; `Q = m_water*(h_out - h_in)` in W, with the outlet set by `T_cold_out` and the gas cooled by the same duty.
- Heat-pump route: `Q_sink` is delivered to the network and `W = Q_sink - Q_source` is parasitic electricity; sink flow is solved and `T_sink_out` is the network supply temperature.
- Network supply/return temperatures set the water flow; losses are a site-level deduction, not a solver parameter. Storage shifts duty in time but adds no new component type.
- Pinch on the exhaust exchanger and COP on the heat pump decide which route is cheaper in fuel; heat rejected near ambient carries little exergy even at large duty.

## Parameters and constraints
- `useful_heat`: flag, default False; must be true on the exhaust exchanger for its duty to count as delivered heat.
- `T_cold_out`: exchanger cold outlet (K), required; `dp_cold`, `dp_hot`: drop fractions (-), default 0.0.
- `T_sink_out`: heat-pump sink outlet (K), required and above its inlet; `eps` (-) default 0.5, `approach_K` (K) default 5.0, and either `Q_sink` or `dT_src`.

## Failure modes
- V8 heat shortfall -> `useful_heat` left False or a pinch cross -> set the flag and lower `T_cold_out`.
- E_TEMP_CROSS on the pump -> sink not hotter than source -> raise `T_sink_out`.
- E_OVER_SPECIFIED -> network water flow fixed on a solved sink -> remove it.

## Tool checks
- `heat_exchanger` and `cycle_performance`: pass JSON object strings in the `candidate` slot; read delivered duty and electricity.
- `second_law_check` -> destruction on the two routes; `validity_check` -> site heat demand satisfied.

## Answer habits
- Prefer exhaust heat when it is genuinely left over after cooling and bottoming.
- Quote delivered heat in W and supply temperatures in K.
- Verify the `useful_heat` flag before changing any temperature set-point.
