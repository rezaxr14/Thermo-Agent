"""D10/D11: the AI gets a token-efficient failure memory with behavior guidance, not verbose logs."""
from thermo_agent.teacher import memory as mem
from thermo_agent.teacher.compact_memory import (
    agent_guidance_block, compact_notes, COMMON_FAILURE_CAUSES,
)
from thermo_agent.teacher.agent_graph import build_context


def _note():
    return mem.Note(id='n1', task='repair', fault_type='unit_slip', architecture='dc_absorption',
                    symptoms='verifier: invalid plant', lesson='Attempt rejected. Correct the power of ten.',
                    fix='SET CH.T_gen_out = 355.8',
                    source='frontier:google/gemini-3.1-flash-lite-preview ts=1 cost=45 tokens=9',
                    accepted=False, record_id='x')


def test_compact_memory_strips_non_reasoning_metadata():
    out = compact_notes([_note()])
    assert 'unit_slip' in out and '355.8' in out
    for leak in ('gemini', 'cost', 'ts=', 'tokens=', 'frontier:'):
        assert leak not in out, f"leaked {leak}"


def test_guidance_orders_inventory_then_memory_and_lists_unit_mismatch():
    g = agent_guidance_block()
    assert 'FIRST inventory' in g and 'THEN read the plant/cycle' in g
    assert 'UNIT MISMATCH' in g and 'MW vs W' in g


def test_build_context_uses_compact_not_verbose_memory():
    import json
    row = json.loads(open('data/processed/repair_train.jsonl', encoding='utf-8').readline())
    sys = build_context({'row': row, 'task': 'repair', 'notes': [_note()],
                         'k_memory': 3, 'mode': 'agents+skills'})['system_prompt']
    assert 'FAILURE MEMORY (what failed' in sys
    assert 'KNOWN FAILURE MODES' not in sys
    assert 'FIRST inventory' in sys
    assert 'gemini' not in sys
