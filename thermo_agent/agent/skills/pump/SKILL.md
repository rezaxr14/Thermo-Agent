---
name: pump
description: Use when a plant contains a pump: liquid pressurisation by outlet pressure and isentropic efficiency, with NPSH, cavitation and two-phase-inlet limits.
components: [pump]
variants: [centrifugal, positive_displacement, canned]
---
# Pump

Objective: raise the liquid to the required pressure with the smallest shaft work while keeping the inlet single-phase and above its saturation pressure.

## Variants
| variant | when it fits | key limits |
| --- | --- | --- |
| centrifugal | high flow, moderate head | NPSH available must exceed required |
| positive_displacement | low flow, high head | pulsation; relief needed |
| canned | leak-tight service | smaller efficiency, limited head |

## Physics and equations
- `Pout` is the outlet pressure in Pa; isentropic target `h2s = h(s1, Pout)`; `h2 = h1 + (h2s - h1)/eta_s`; shaft power `W = -m_dot*(h2 - h1)` in W.
- For an incompressible liquid the rise is small, about `v*(Pout - P1)/eta_s` in J/kg, so pump work is far smaller than compressor or turbine work.
- Cavitation starts when the inlet static pressure falls to the saturation pressure at the inlet temperature; NPSH available must stay above NPSH required. This is a design constraint, not a solver parameter.
- A two-phase inlet is dangerous: the isentropic state lookup assumes a compressed liquid, so flashing at the suction raises E_PROP_RANGE or a fluid-class error.

## Parameters and constraints
- `P_out`: outlet pressure (Pa), required, no default; it must exceed the inlet pressure and, in an ORC loop, exceed the evaporating pressure.
- `eta_s`: isentropic efficiency (-), required, no default; `eta_s` <= 0 raises E_PROP_RANGE; keep below 1.
- In an ORC the evaporator sets the mass flow; do not add a conflicting fixed `m_dot` on the pump inlet.

## Failure modes
- E_NOT_CONVERGED in an ORC loop -> pump `P_out` below evaporating pressure -> set it to the high-side pressure.
- E_PROP_RANGE at suction -> inlet near saturation or two-phase -> raise inlet pressure or subcool the liquid.
- V3 negative exergy destruction -> efficiency above 1 -> reduce `eta_s`.

## Tool checks
- `turbomachine`: pass a JSON object string in the `candidate` slot, e.g. `{"kind": "pump", "fluid": "Water", "T_in_K": 300, "P_in_kPa": 101.325, "P_out_kPa": 1000, "eta_s": 0.75}`; read `power_W` and `outlet.T_K`.
- `solve_plant` -> inlet phase and outlet `h_kJ_kg`; `first_law_check` -> the small pump work and mass residual.

## Answer habits
- Keep pressure in Pa and efficiency as a fraction; never quote a default for either parameter.
- Check the upstream loop first: the pump usually inherits its flow from the evaporator or condenser.
- Quote pump work as a small number next to turbine work, not as a heat duty.
