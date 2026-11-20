---
name: compressor
description: Use when a plant contains a compressor: gas compression by pressure ratio and isentropic efficiency, with surge and discharge-temperature limits.
components: [compressor]
variants: [centrifugal, reciprocating, screw, scroll]
---
# Compressor

Objective: set a pressure ratio and efficiency that reach the outlet pressure without surge, an excessive discharge temperature, or wasted shaft work.

## Variants
| variant | when it fits | key limits |
| --- | --- | --- |
| centrifugal | high flow, moderate PR | surge at low flow; PR per stage bounded |
| reciprocating | low flow, high PR | valve losses and pulsation |
| screw | moderate flow, oil-flooded | leakage grows with PR |
| scroll | small flow, oil-free | fixed built-in volume ratio |

## Physics and equations
- `P2 = P1 * PR`; isentropic target `h2s = h(s1, P2)`; `h2 = h1 + (h2s - h1)/eta_s`; shaft power `W = -m_dot*(h2 - h1)` in W.
- Discharge temperature comes from `h2`; it is an output, never an input. Both ports take a gas-class fluid (`CGAM_AIR`, `FLUE`, `CH4`).
- Intercooling is a plant-level choice: split the total PR over several `compressor` components with coolers between them. There is no per-component intercooler parameter.
- Surge is a low-flow/high-PR instability, not a solver parameter; avoid it by lowering PR or adding flow, and keep the discharge below the machine temperature limit.

## Parameters and constraints
- `PR`: pressure ratio (-), required, no default; a value below 1 inverts the work sign and the machine is no longer a compressor.
- `eta_s`: isentropic efficiency (-), required, no default; `eta_s` <= 0 raises E_PROP_RANGE, and values above 1 break the second law.
- Inlet/outlet mass flows are solved by the plant; do not over-specify a stream flow that contradicts the upstream machine.

## Failure modes
- E_PROP_RANGE pointing at `eta_s` -> non-positive efficiency -> set a positive fraction below 1.
- V3 negative exergy destruction -> efficiency above 1 or a physically impossible PR -> lower `eta_s` or PR.
- E_FLUID_MISMATCH -> liquid on a gas port -> use a gas-class fluid, or a `pump` for liquid.

## Tool checks
- `turbomachine`: pass a JSON object string in the `candidate` slot, e.g. `{"kind": "compressor", "fluid": "CGAM_AIR", "T_in_K": 300, "P_in_kPa": 101.325, "pressure_ratio": 10, "eta_s": 0.85}`; read `outlet.T_K` and `power_W`.
- `solve_plant` -> outlet `T_K` and `h_kJ_kg`; `second_law_check` -> exergy destruction (MW) for this compression.

## Answer habits
- Quote PR, efficiency, discharge temperature and power only as the tools report them.
- For repair, change one parameter; for design, trade PR against discharge temperature and staging.
- Never invent a default: both `PR` and `eta_s` are required.
