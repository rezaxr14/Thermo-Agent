"""Regenerate the AUTO sections of AGENTS.md and the skills from the component registry (ports, fluid classes, parameters).
    python -m thermo_agent.agent.build_harness          # rewrite files
    python -m thermo_agent.agent.build_harness --check  # exit 1 if any file is out of date"""
from __future__ import annotations
import re, sys
from pathlib import Path
from ..core.components import REQUIRED_PORTS, PARAMS, PORT_CLASS

ROOT = Path(__file__).parent
_BLOCK = re.compile(r"(<!-- AUTO:\w+ -->\n)(.*?)(<!-- /AUTO -->)", re.S)


def _fmt_default(d) -> str:
    return d if isinstance(d, str) else repr(d)


def component_block(ctype: str) -> str:
    ports = ", ".join(f"{p} ({PORT_CLASS.get((ctype, p), 'any')})" for p in REQUIRED_PORTS[ctype])
    params = "; ".join(f"`{k}`: {meaning}, {_fmt_default(d)}" for k, (meaning, d) in PARAMS[ctype].items())
    return f"### {ctype}\nPorts: {ports}\nParams: {params}\n"


def all_block() -> str:
    lines = []
    for t in REQUIRED_PORTS:
        ports = ", ".join(REQUIRED_PORTS[t]); params = ", ".join(PARAMS[t])
        lines.append(f"- `{t}`: ports {ports}; params {params}")
    return "\n".join(lines) + "\n"


def frontmatter(text: str) -> dict:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    out = {}
    for line in (m.group(1) if m else "").splitlines():
        k, _, v = line.partition(":"); out[k.strip()] = v.strip()
    if "components" in out: out["components"] = [x.strip() for x in out["components"].strip("[]").split(",") if x.strip()]
    return out


def render(path: Path) -> str:
    text = path.read_text(); fm = frontmatter(text)
    def repl(m):
        kind = re.search(r"AUTO:(\w+)", m.group(1)).group(1)
        body = all_block() if kind == "all" else "\n".join(component_block(c) for c in fm.get("components", []))
        return m.group(1) + body + m.group(3)
    return _BLOCK.sub(repl, text)


def files() -> list[Path]:
    return [ROOT / "AGENTS.md", *sorted((ROOT / "skills").glob("*/SKILL.md"))]


def main(check: bool = False) -> int:
    stale = []
    for p in files():
        new = render(p)
        if new != p.read_text():
            stale.append(p.name)
            if not check: p.write_text(new)
    if stale: print(("out of date: " if check else "rewrote: ") + ", ".join(stale))
    return 1 if (check and stale) else 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
