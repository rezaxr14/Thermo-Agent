"""Gate 2 check: does the student tokenizer render our tool trajectory correctly?

Run with the tokenizer available locally (no network needed when the HF cache is warm):

    python scripts/check_qwen_tool_template.py --model Qwen/Qwen3.5-4B

It answers the questions the training/eval path depends on:

1. Does the chat template accept the OpenAI-style ``tools=`` schema and a ``role="tool"``
   message, i.e. does the tokenizer declare tool-calling support?
2. Does the trajectory format we actually train on survive rendering? We store the whole
   agent turn as ONE assistant message containing ``<think>``, ``<tool_call>`` and
   ``<tool_result>`` blocks, so the tags must come back verbatim.
3. Is the generation prompt a true prefix of the full conversation in the CHOSEN mode
   (``train.render.TRAIN_ON_REASONING``)? ``completion_only_loss`` masks labels up to the
   prompt boundary, so a template that rewrites the assistant prefix would silently train
   on the wrong tokens.
4. Can the shared parser read BOTH the native Qwen tool-call block and our XML block?

Exit code is 0 only when every check passes. Nothing is written and no model is downloaded
unless ``--allow-download`` is passed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# This file is a script (sys.path[0] == scripts/); make the repo importable for the
# shared render/toolcall helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermo_agent.agent.toolcall import parse_tool_calls
from thermo_agent.train.render import CHAT_KW, TRAIN_ON_REASONING

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_plant",
            "description": "Solve and verify the plant/ledger exactly as given.",
            "parameters": {
                "type": "object",
                "properties": {"candidate": {"type": "string"}},
                "required": [],
            },
        },
    }
]

TRAJECTORY = (
    "<think>\nInventory the plant, then check the hot-side fluid.\n</think>\n"
    '<tool_call name="inspect_plant">{"candidate": ""}</tool_call>\n'
    '<tool_result>{"ok": true, "solved": true, "failed_checks": ["fluid mismatch"]}</tool_result>\n'
    "<diagnosis>The hot-side fluid is wrong.</diagnosis>\n"
    "<edits>\nFLUID CH.hot_in = water\n</edits>"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--json", action="store_true", help="print a machine-readable summary")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    template = tok.chat_template or ""
    results: dict = {"model": args.model, "chosen_mode": CHAT_KW, "checks": {}, "ok": True}

    def check(name: str, passed: bool, detail: str = "") -> None:
        results["checks"][name] = {"passed": bool(passed), "detail": detail}
        results["ok"] = results["ok"] and bool(passed)

    # 0. The chosen training mode must be the reasoning-preserving one.
    check(
        "chosen_mode_is_reasoning",
        TRAIN_ON_REASONING is True and CHAT_KW["enable_thinking"] is True,
        f"TRAIN_ON_REASONING={TRAIN_ON_REASONING}, CHAT_KW={CHAT_KW}",
    )

    # 1. The template must know about tool calling at all.
    check(
        "template_mentions_tools",
        ("tool_call" in template) or ("tools" in template),
        "chat_template contains tool_call/tools" if template else "no chat_template found",
    )

    # 2. Native tool-calling render must not raise. NOTE: the template applies
    #    `arguments | items`, so `arguments` MUST be a mapping, not a JSON string.
    native = [
        {"role": "system", "content": "You repair plants."},
        {"role": "user", "content": "Repair this plant."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "inspect_plant", "arguments": {}},
                }
            ],
        },
        {"role": "tool", "name": "inspect_plant", "content": '{"ok": true}'},
        {"role": "assistant", "content": "<diagnosis>done</diagnosis>"},
    ]
    try:
        native_text = tok.apply_chat_template(
            native, tools=TOOLS, add_generation_prompt=False, tokenize=False,
            **CHAT_KW,
        )
        check("native_tool_render", True, "%d chars" % len(native_text))
    except Exception as exc:  # noqa: BLE001 - the error text is the evidence
        check("native_tool_render", False, "%s: %s" % (type(exc).__name__, exc))
        native_text = ""

    # 3. Our stored one-message trajectory must survive rendering verbatim.
    stored = [
        {"role": "system", "content": "You repair plants."},
        {"role": "user", "content": "Repair this plant."},
        {"role": "assistant", "content": TRAJECTORY},
    ]
    rendered = tok.apply_chat_template(stored, add_generation_prompt=False, tokenize=False, **CHAT_KW)
    check("xml_tool_call_survives", "<tool_call name=" in rendered, "verbatim <tool_call>")
    check("xml_tool_result_survives", "<tool_result>" in rendered, "verbatim <tool_result>")
    check("think_block_survives", "<think>" in rendered, "verbatim <think>")

    # 4. Generation prompt must be a prefix of the full conversation in the CHOSEN mode
    #    (completion_only_loss). With TRAIN_ON_REASONING this MUST hold.
    prompt = tok.apply_chat_template(stored[:2], add_generation_prompt=True, tokenize=False, **CHAT_KW)
    check(
        "generation_prompt_is_prefix",
        rendered.startswith(prompt),
        "prompt %d chars, full %d chars" % (len(prompt), len(rendered)),
    )

    # 5. The shared parser must accept BOTH native and our XML tool-call formats.
    native_calls = parse_tool_calls(native_text)
    check(
        "native_tool_call_parseable",
        any(c.name == "inspect_plant" and c.format == "qwen-native" for c in native_calls),
        "parsed %d native block(s): %s" % (len(native_calls), [c.name for c in native_calls]),
    )
    xml_calls = parse_tool_calls(TRAJECTORY)
    check(
        "xml_tool_call_parseable",
        any(c.name == "inspect_plant" and c.format == "xml-json" and c.arguments.get("candidate") == ""
            for c in xml_calls),
        "parsed %d xml block(s): %s" % (len(xml_calls), [c.name for c in xml_calls]),
    )

    # 6. A tokenizer must mark the boundary tokens it will be asked to mask.
    ids = tok(rendered, add_special_tokens=False)["input_ids"]
    results["rendered_chars"] = len(rendered)
    results["rendered_tokens"] = len(ids)

    if args.json:
        print(json.dumps(results, indent=1))
    else:
        for name, item in results["checks"].items():
            print(("PASS " if item["passed"] else "FAIL ") + name + "  " + item["detail"])
        print("rendered: %d chars / %d tokens" % (len(rendered), len(ids)))
        print("VERDICT:", "PASS" if results["ok"] else "FAIL")
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
