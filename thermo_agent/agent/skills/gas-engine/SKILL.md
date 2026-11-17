---
name: gas-engine
description: Use when a plant contains a gas_engine component: a reciprocating gas engine sized by rated electrical output, with air, fuel and exhaust ports.
components: [gas_engine]
---
# Gas engine

<!-- AUTO:components -->
### gas_engine
Ports: air_in (gas), fuel_in (gas), exh_out (gas)
Params: `P_el`: rated electrical output (W), required; `eta_el`: electrical efficiency on LHV (-), required; `f_loss`: jacket and radiation heat, fraction of fuel energy, rejected to ambient (-), required; `air_excess`: air flow over stoichiometric (-), 1.8; `P_exh`: exhaust pressure (Pa), 110000.0
<!-- /AUTO -->

## Behaviour
- Sized by `P_el`. Fuel flow is `P_el / (eta_el * LHV)`; air flow is `air_excess` times stoichiometric. Both flows are solved: do not give `m_dot` on the `air` or `fuel` streams.
- Energy split: electricity `P_el`, jacket and radiation heat `f_loss` of the fuel energy (rejected to ambient, not recovered), and the rest leaves in the exhaust. The exhaust temperature follows from that balance; it is not an input.
- `eta_el + f_loss` must be below 1. Typical values in these plants: `eta_el` near 0.43, `f_loss` 0.10 to 0.20, `air_excess` 1.5 to 2.2.
- Only exhaust heat is recovered, so an engine plant can serve less chilled load by absorption cooling than a gas turbine of the same electricity. When the exhaust cannot supply the hot-water duty, expect E_TEMP_CROSS on the heat exchanger or a failed V4 check.
- Exhaust leaves at `P_exh` (default 110 kPa), above ambient, so downstream heat exchangers have back-pressure.

## Typical faults
- `P_el` off by a power of ten (W entered as MW or kW) fails V6 or breaks the electricity balance of the site.
- Specifying `m_dot` on `air` or `fuel` raises E_OVER_SPECIFIED.
- `eta_el` above 1, or `eta_el + f_loss` of 1 or more, is invalid.

## State: constant versus changing
- Constant: fuel composition and LHV, air composition, `air_excess`, `f_loss`, `P_el`, `P_exh`. Solved (never specified): fuel and air mass flows.
- Changing: air and fuel enter at ambient T/P and leave as exhaust at higher T; chemical exergy becomes shaft work plus exhaust enthalpy plus jacket/radiation loss. Exhaust temperature is an output of the energy balance, not a set-point.
- Only recoverable heat is the exhaust above ambient plus `T_stack_min`; jacket heat is lost. More air excess lowers exhaust temperature but does not create new energy.
