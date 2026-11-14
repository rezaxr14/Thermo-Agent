---
name: heat-pump-district-heat
description: Use when a plant delivers district heat, with a heat_pump on the chiller heat-rejection loop or a heat exchanger flagged useful_heat on the exhaust.
components: [heat_pump, recuperator]
---
# District heat: heat pump or exhaust

<!-- AUTO:components -->
### heat_pump
Ports: src_in (liquid), src_out (liquid), sink_in (liquid), sink_out (liquid)
Params: `T_sink_out`: sink outlet temperature (K), required; `eps`: fraction of the Lorenz COP (-), 0.5; `approach_K`: evaporator and condenser approach (K), 5.0; `Q_sink`: heat duty to deliver (W); sizes the source cooling, one of Q_sink or dT_src; `dT_src`: source-side temperature drop (K), one of Q_sink or dT_src

### recuperator
Ports: cold_in (any), cold_out (any), hot_in (any), hot_out (any)
Params: `T_cold_out`: cold-side outlet temperature (K), required; `dp_cold`: cold-side pressure-drop fraction (-), 0.0; `dp_hot`: hot-side pressure-drop fraction (-), 0.0; `useful_heat`: flag: duty counts as delivered heat, False
<!-- /AUTO -->

## Behaviour
- Exhaust route: a `recuperator` with `useful_heat = true` heats district water on the exhaust, after the chiller's own heat exchanger. Its duty counts as delivered heat. The district-water flow is fixed by the demand and the supply/return temperatures.
- Heat-pump route: the heat pump sits in series on the chiller rejection water (before the heat rejector). With `Q_sink` it delivers exactly that heat and cools the source as far as needed; the sink flow is solved. The heat pump uses electricity: `W = Q_sink - Q_source`, a parasitic load that raises fuel use.
- COP = `eps` x Lorenz COP on refrigerant-side temperatures (stream log-mean temperatures moved outward by `approach_K`). It needs a sink hotter than the source and COP above 1.
- The heat pump needs a dry or wet rejection loop (not an ideal boundary). If it cools the rejection water below what the rejector can reach, the rejector raises E_TEMP_CROSS.
- The exhaust route is cheaper in fuel when the exhaust still has heat after the other recoveries; the heat pump route leaves the exhaust free for an ORC.

## Typical faults
- Sink outlet not hotter than sink inlet: E_PROP_RANGE.
- Neither `Q_sink` nor `dT_src` given: E_UNDER_SPECIFIED (if both are given, `Q_sink` is used).
- District flow specified on a heat-pump sink (it is solved): E_OVER_SPECIFIED.
