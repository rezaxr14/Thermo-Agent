# AGENTS.md: energy-plant tasks

You work on simulated plants for AI data centres (gas prime mover, cooling, heat recovery). A solver and physics verifiers judge every answer, so only claims that the solver would reproduce are useful.

## Conventions
- SI units: K, Pa, W, kg/s. Efficiencies, second-law fractions (`eps`) and pressure-drop fractions (`dp*`) are fractions, never percent. Temperatures are kelvin, never degrees Celsius.
- A plant is a graph. `C <id> <type> {params}` is a component; `S <id> <from> -> <to> <fluid> {extras}` is a stream. Endpoints are `<component>.<port>` or `ext:<name>` (a boundary).
- Fluid classes: gas streams are `CGAM_AIR`, `FLUE`, `CH4`; every other fluid is a liquid or two-phase CoolProp fluid (`Water`, `R245fa`, `Cyclopentane`, `Toluene`). A port accepts only its own class.
- A stream whose source is a component but which carries `fixed` values is a tear stream: it closes a loop, and its fixed T and P must equal what the producing component computes. Never change one side of a tear without the other.
- Flows that a component solves itself must not be specified (engine air and fuel, chiller generator and rejection flows, IT coolant flow, heat-pump sink flow).

## What the verifiers check
V1 mass balance, V2 energy balance, V3 exergy: closure and no negative exergy destruction in any component, V4 no temperature cross (min approach below 0 K fails), V5 second-law and Carnot bounds, V6 unit and range checks, V7 numbers in your text must match the solver, V8 site requirements (IT load served, no grid import unless allowed, cooling, heat, water limit).

## Reading a solver error
- E_PROP_RANGE: a value is non-positive, a unit slip, or outside the property data (a pressure of 20 Pa, a temperature of 1226 K entered as 1226 degC).
- E_TEMP_CROSS: heat cannot flow as specified: hot and cold inlets swapped, a duty larger than the hot stream can supply, or an inlet colder than the achievable outlet.
- E_NOT_CONVERGED: a tear stream does not close, or an outlet temperature is not reachable.
- E_FLUID_MISMATCH: a stream's fluid class does not fit the port it feeds.
- E_DISCONNECTED, E_OVER_SPECIFIED, E_UNDER_SPECIFIED: a port has no stream or two, or a value that should be solved is given, or one is missing.
- Solved but failing a check: read the check name and its detail; V6 names the offending parameter.

## Answer formats
- Repair: `<diagnosis>` citing only numbers shown, then `<edits>` with lines `SET <component>.<param> = <number>`, `SET @<stream>.<field> = <number>`, `REWIRE @<stream>.<to|from> = <component>.<port>`, `FLUID @<stream> = <fluid>`. Change as little as possible; you cannot add or remove components.
- Design: `<rationale>` then `<design>` with `key = value` lines, exactly the keys your choices require.
- Audit: `<analysis>`, `<ranking>A > B > C</ranking>`, `<change>SET ...</change>`, `<prediction>fuel_change_mw = ...</prediction>`.
- Ledger repair: `<diagnosis>` citing only numbers shown (MW), then `<fix>` with lines `FIX T0 = <expr>`, `FIX E[<stream>] = <expr>`, `FIX ED[<component>] = <expr>`.

## Habits
1. Find which stream or parameter the error or failing check points at before editing anything.
2. Prefer the explanation that fixes the reported failure with the fewest edits.
3. Never invent numbers: quote what the input shows, or say what you expect the solved plant to show.

## Tools, limits and final answer
Inventory the available skills and the five physics tools FIRST, then read the plant/cycle against the common failure causes above (unit mismatch MW/W and kJ/J, fraction vs percent, K vs C; sign flip; swapped ports; wrong fluid class; efficiency > 1; out-of-range parameter; unconverged tear) before proposing a candidate.
- Limits: at most 6 assistant turns and 8 tool calls per episode. Call `inspect_plant` first (it needs no answer), then check candidates, read the numbers, revise and re-check. When residuals are ~0 and all checks pass, STOP calling tools and return the final answer.
- Tools (real solver results as JSON with `ok` and numbers; physics only, there is NO answer-oracle tool):
  - `inspect_plant(candidate="")` -> solver status, failed checks and key residuals for the plant/ledger AS GIVEN.
  - `first_law_check(candidate)` -> per-line/per-component mass (kg/s) and energy (W) residuals, heat/power/fuel balance, convergence, failed checks.
  - `second_law_check(candidate)` -> exergy destruction (MW), closure, eta_I/eta_II, Carnot/COP limits, failed checks.
  - `validity_check(candidate)` -> unit/range violations, site constraints, pinch approach (K), verifier verdict.
  - `solve_plant(candidate)` -> full stream state table (T K, P kPa, h, s kJ/kg, exergy).
- Final answer: emit ONLY the task's XML tags (see Answer formats above) — repair `<diagnosis>` + `<edits>`; design `<rationale>` + `<design>`; audit `<analysis>`, `<ranking>`, `<change>`, `<prediction>fuel_change_mw = ...</prediction>`; ledger `<diagnosis>` (MW) + `<fix>`.

## Components
<!-- AUTO:all -->
- `compressor`: ports in, out; params PR, eta_s
- `turbine`: ports in, out; params P_out, eta_s
- `orc_turbine`: ports in, out; params P_out, eta_s
- `pump`: ports in, out; params P_out, eta_s
- `combustor`: ports air_in, fuel_in, out; params T_out, dp, q_loss
- `recuperator`: ports cold_in, cold_out, hot_in, hot_out; params T_cold_out, dp_cold, dp_hot, useful_heat
- `hrsg`: ports hot_in, hot_out, water_in, steam_out; params P_steam, dp_gas, econ_approach_K
- `orc_evaporator`: ports hot_in, hot_out, cold_in, cold_out; params pinch_K, dp_hot, flow_frac, T_stack_min, T_max_K
- `condenser`: ports hot_in, hot_out, cold_in, cold_out; params dT_cw
- `dc_load`: ports in, out; params P_it, chilled_fraction, T_return
- `absorption_chiller`: ports gen_in, gen_out, evap_in, evap_out, rej_in, rej_out; params T_chw_out, T_gen_out, dT_rej, eps, COP_max
- `heat_pump`: ports src_in, src_out, sink_in, sink_out; params T_sink_out, eps, approach_K, Q_sink, dT_src
- `electric_chiller`: ports evap_in, evap_out, rej_in, rej_out; params T_chw_out, dT_rej, eps, approach_K
- `heat_rejector`: ports in, out; params kind, approach_K, fan_frac
- `gas_engine`: ports air_in, fuel_in, exh_out; params P_el, eta_el, f_loss, air_excess, P_exh
<!-- /AUTO -->
