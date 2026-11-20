"""One shared multi-turn environment for the real physics tools (Gate 2).

A :class:`ThermoEnv` holds a single dataset row and exposes the five physics tools
from :mod:`thermo_agent.teacher.toolkit` as bound methods. The same class serves the
three drivers that need tool calling:

* TRL GRPO through ``environment_factory`` (one instance per rollout),
* the kbench frontier harness (its tool list),
* the local/HF/OpenAI-compatible multi-turn generator.

There is deliberately **no** ``check_answer`` oracle: acceptance and reward live only
in :func:`thermo_agent.teacher.validate.validate`, called by :meth:`ThermoEnv.step`.

The explicit signatures and Google-style docstrings below are the source TRL uses to
build the JSON tool schema, so they are kept on the public methods rather than
delegating straight to the toolkit closures' signatures.

Gate 8 wiring note: TRL 1.18.1 probes ``environment_factory()`` with no arguments,
then treats every public function on the *class* as a tool except ``reset`` and
``get_reward`` (``inspect.getmembers(type(instance), predicate=inspect.isfunction)``).
So ``step`` and ``observation_block`` must not be handed to GRPO as environment
methods: give ``environment_factory`` a thin composition wrapper (holding a
ThermoEnv) whose only public callables are the five tool methods, or pass an
explicit ``tools=`` list and filter the probed methods. They are public here
because the kbench and local/HF drivers call them directly.
"""
from __future__ import annotations

from typing import Callable

from ..teacher import toolkit
from ..teacher.validate import validate

# Final-answer XML contracts mirror agent/AGENTS.md ("Answer formats") and
# train/prompts.SYSTEM. They live here so env.py is self-contained and importable.
_FINAL_XML: dict[str, str] = {
    "repair": "<diagnosis>...</diagnosis> then <edits> with SET/REWIRE/FLUID lines.",
    "design": "<rationale>...</rationale> then <design> with key = value lines.",
    "audit": ("<analysis>...</analysis><ranking>A > B > C</ranking>"
              "<change>SET ...</change><prediction>fuel_change_mw = ...</prediction>."),
    "ledger": "<diagnosis>...</diagnosis> (MW) then <fix> with FIX T0/E[...]/ED[...] lines.",
}
_UNKNOWN_XML = "the task's XML tags exactly as specified in the prompt."

# Literal header of the tool contract emitted by ``observation_block`` and by the shared
# ``thermo_agent.agent.tool_prompt`` builder. Used to detect rows that already carry it.
_TOOL_PROMPT_MARKER = "MULTI-TURN TOOL ENVIRONMENT"

# Fail fast at import if the toolkit's tool list changes underneath this class.
assert tuple(toolkit.tool_names()) == ("inspect_plant", "first_law_check", "second_law_check",
                                       "validity_check", "solve_plant")


class ThermoEnv:
    """A single-row, multi-turn episode with real thermodynamics tools.

    Attributes:
        row: The dataset row the episode is scored against.
        task: One of ``repair``, ``design``, ``audit``, ``ledger``.
        max_turns: Configured assistant-turn budget (reported, not enforced here).
        max_tool_calls: Hard cap on tool invocations per episode.
        turn: Number of actions taken so far (tool calls plus the final step).
        tool_calls: Number of tool invocations made so far.
        history: Structured record of tool calls and the final answer.
        done: True after :meth:`step` has finalized the episode.
        final_answer: The answer passed to :meth:`step`, or ``None`` before it.
        reward: The verifier reward from :meth:`step`, or ``None`` before it.
    """

    def __init__(self, row: dict, task: str, max_turns: int = 6, max_tool_calls: int = 8,
                 include_dynamic: bool = False) -> None:
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        # OFF by default: all corpora and accepted traces keep the five-tool contract.
        # The hosted frontier harness opts in for the nine continuous-input tools.
        self.include_dynamic = bool(include_dynamic)
        self._tools_by_name: dict[str, Callable[..., str]] = {}
        self._bind(row, task)
        # TRL/kbench consume `tools` order; it must equal the toolkit's canonical order.
        assert [fn.__name__ for fn in self.tools] == toolkit.tool_names(
            include_dynamic=self.include_dynamic)
        self.reset()

    # ------------------------------------------------------------------ #
    # construction / state
    # ------------------------------------------------------------------ #
    def _bind(self, row: dict, task: str) -> None:
        """Bind the toolkit's closures to ``row``/``task`` and verify their order."""
        self.row = row
        self.task = task
        self._tools_by_name = {
            fn.__name__: fn for fn in toolkit.make_tools(
                task, row, include_dynamic=self.include_dynamic)
        }
        assert list(self._tools_by_name) == toolkit.tool_names(
            include_dynamic=self.include_dynamic)

    @staticmethod
    def _prompt_text(row: dict) -> str:
        """The task prompt: the non-system message contents of ``row['prompt']``."""
        prompt = row.get("prompt", "")
        if isinstance(prompt, str):
            return prompt
        return "\n\n".join(m["content"] for m in prompt
                           if m.get("role") != "system" and m.get("content"))

    def reset(self, row: dict | None = None, task: str | None = None) -> str:
        """Rebind an optional new row/task and zero all episode state.

        Args:
            row: Optional replacement dataset row. Keeps the current row when omitted.
            task: Optional replacement task name. Keeps the current task when omitted.

        Returns:
            str: The initial observation — the task prompt from the bound row.
        """
        if row is not None or task is not None:
            self._bind(row if row is not None else self.row,
                       task if task is not None else self.task)
        self.turn = 0
        self.tool_calls = 0
        self.history: list[dict] = []
        self.done = False
        self.final_answer: str | None = None
        self.reward: float | None = None
        return self._prompt_text(self.row)

    @property
    def tools(self) -> list[Callable[..., str]]:
        """The physics tools in ``toolkit.tool_names()`` order.

        The five plant tools are bound methods; the nine continuous-input component
        tools (when ``include_dynamic`` is on) are plain ``(candidate) -> str`` callables
        from :mod:`thermo_agent.teacher.thermo_tools`, executed through the same
        ``_call``/parser path as the plant tools.
        """
        base = [self.inspect_plant, self.first_law_check, self.second_law_check,
                self.validity_check, self.solve_plant]
        if self.include_dynamic:
            # Nine continuous-input component tools + the two on-demand skill tools.
            return base + toolkit.dynamic_tools() + toolkit.skill_tools()
        return base

    # ------------------------------------------------------------------ #
    # tools
    # ------------------------------------------------------------------ #
    def _guard(self) -> None:
        if self.tool_calls >= self.max_tool_calls:
            raise RuntimeError(
                f"tool-call limit exceeded: at most {self.max_tool_calls} tool calls per episode "
                f"(task={self.task!r})"
            )
        self.tool_calls += 1
        self.turn += 1

    def _call(self, name: str, candidate: str) -> str:
        """Enforce the cap, run the bound toolkit closure, and record the turn."""
        self._guard()
        result = self._tools_by_name[name](candidate)
        self.history.append({"tool": name, "candidate": candidate, "result": result})
        return result

    def inspect_plant(self, candidate: str = "") -> str:
        """Solve and verify the plant or ledger as given, before any candidate exists.

        This is the only tool the student can call first: it needs no answer. The
        ``candidate`` argument is ignored so every tool shares one JSON schema.

        Args:
            candidate (str): Ignored; pass an empty string. Defaults to "".

        Returns:
            str: JSON with the solver status, failed physics checks and key residuals
            (mass in kg/s, energy and exergy in W) for the row's plant/ledger.
        """
        return self._call("inspect_plant", candidate)

    def first_law_check(self, candidate: str) -> str:
        """Check the candidate against the first law using the real solver.

        Args:
            candidate (str): The candidate answer in the task's final XML format.

        Returns:
            str: JSON with per-line/per-component mass residuals (kg/s), energy
            residuals (W), heat/power/fuel balance, convergence, tear residual,
            unit/range flags and the list of failed checks.
        """
        return self._call("first_law_check", candidate)

    def second_law_check(self, candidate: str) -> str:
        """Check the candidate against the second law using the real solver.

        Args:
            candidate (str): The candidate answer in the task's final XML format.

        Returns:
            str: JSON with exergy destruction (MW), exergy closure (MW), eta_I and
            eta_II, Carnot and COP-vs-reversible limits, and the failed checks.
        """
        return self._call("second_law_check", candidate)

    def validity_check(self, candidate: str) -> str:
        """Check the candidate's units, ranges, site constraints and hard verifiers.

        Args:
            candidate (str): The candidate answer in the task's final XML format.

        Returns:
            str: JSON with unit/range violations, site constraints, pinch approach
            (K), the verifier verdict/score and the failed checks.
        """
        return self._call("validity_check", candidate)

    def solve_plant(self, candidate: str) -> str:
        """Solve the candidate plant and return the full stream state table.

        Args:
            candidate (str): The candidate answer in the task's final XML format.

        Returns:
            str: JSON with one entry per stream (T in K, P in kPa, h and s in
            kJ/kg and kJ/kg-K, exergy in kJ/kg) so states can be reasoned about.
        """
        return self._call("solve_plant", candidate)

    # ------------------------------------------------------------------ #
    # finalization / prompt text
    # ------------------------------------------------------------------ #
    def step(self, answer: str) -> float:
        """Finalize the episode with the student's final answer and score it.

        Args:
            answer (str): The final assistant message, carrying the task's XML tags.

        Returns:
            float: The verifier reward from ``teacher.validate.validate`` — the
            exact value GRPO uses, computed only here and never as a tool.
        """
        self.turn += 1
        self.done = True
        self.final_answer = answer
        self.history.append({"final": answer})
        verdict = validate(self.task, answer, self.row)
        self.reward = float(verdict["reward"])
        return self.reward

    def observation_block(self) -> str:
        """Return the compact, config-driven environment block for the student.

        Contains the turn limit, the tool list with one-line return descriptions,
        the inventory-then-examine habit, and the exact final-answer XML tags. It
        carries no model ids, timestamps, cost or token accounting (D11).

        Returns:
            str: The observation block text.
        """
        contract = _FINAL_XML.get(self.task, _UNKNOWN_XML)
        lines = [
            _TOOL_PROMPT_MARKER,
            "Inventory the available skills and the physics tools first; then examine the "
            "plant/cycle against known failure modes before proposing a candidate.",
            f"Limits: at most {self.max_turns} assistant turns and {self.max_tool_calls} tool calls.",
            "Tools return JSON with real numbers (physics only; there is no answer-oracle tool):",
            "- inspect_plant(candidate=\"\") -> as-given solver status, failed checks and key residuals.",
            "- first_law_check(candidate) -> mass/energy residuals, heat-power-fuel balance, failed checks.",
            "- second_law_check(candidate) -> exergy destruction, eta_I/eta_II and reversibility limits.",
            "- validity_check(candidate) -> unit/range, site and pinch violations, verifier verdict.",
            "- solve_plant(candidate) -> full stream state table (T, P, h, s, exergy).",
        ]
        if self.include_dynamic:
            # Short one-liners only, shared with the compact contract (no skill-text duplication).
            from .tool_prompt import _COMPACT_TOOL_LINE
            lines.append("Component/skill tools take a JSON object string as candidate:")
            for name in toolkit.DYNAMIC_TOOL_NAMES + toolkit.SKILL_TOOL_NAMES:
                lines.append("- " + _COMPACT_TOOL_LINE[name])
        lines += [
            "Read the numbers, revise and re-check; when residuals are ~0 and all checks pass, stop.",
            "Final answer (stop calling tools and emit exactly these tags): " + contract,
        ]
        return "\n".join(lines)


def row_has_harness_tools(row: dict) -> bool:
    """Whether ``row`` already carries the full multi-turn tool contract in its system prompt.

    Rows built with ``prompt_variant="harness_tools"`` contain the same tool contract as the
    ``ours_harness`` eval arm, so GRPO must not append a second copy of
    :meth:`ThermoEnv.observation_block`. Detection accepts either the explicit row metadata
    or the literal marker in the system message(s); the latter keeps working for data loaders
    that drop the ``prompt_variant`` column.

    Args:
        row: One dataset row (the columns TRL forwards to ``environment.reset``).

    Returns:
        bool: True when the row already has the harness tool prompt.
    """
    if row.get("prompt_variant") == "harness_tools":
        return True
    prompt = row.get("prompt", "")
    if isinstance(prompt, str):
        return _TOOL_PROMPT_MARKER in prompt
    if isinstance(prompt, (list, tuple)):
        return any(
            isinstance(m, dict) and m.get("role") == "system"
            and _TOOL_PROMPT_MARKER in (m.get("content") or "")
            for m in prompt
        )
    return False


__all__ = ["ThermoEnv", "row_has_harness_tools"]
