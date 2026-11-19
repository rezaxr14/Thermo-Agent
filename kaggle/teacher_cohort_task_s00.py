"""Hosted LangGraph + REAL-tools teacher pass over one compact prompt shard.

Self-contained: bootstraps the thermo-agent source bundle using the STANDARD LIBRARY
ONLY, then reuses our own agent_graph / kbench_adapter so the hosted model sees
AGENTS.md + component skills + compact failure memory + the real tool contract.

No secrets, no ground truth. One JSON line is appended (and flushed) per row, so a
dead worker loses at most one answer; re-running skips ids already in the journal.
The model id on every record is the REAL ``kbench.llm.name`` -- never an invented one.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys
import time
import zipfile

SPEC = {
    "task_name": "thermo-teacher-cohort-s00",
    "dataset": "rezanadimi/thermo-teacher-prompts-big",
    "code_dataset": "rezanadimi/thermo-agent-code",
    "shard": "prompts_s00.jsonl",
    "limit": 0,           # 0 = every row in the shard (no assistant-invented row cap)
    "max_cost_usd": 0.0,  # 0 = no internal cost stop (account-level quotas still apply)
    # "rounds" is filled from agent_graph.DEFAULT_MAX_ROUNDS after the bootstrap import,
    # so the hosted prompt, adapter and graph never disagree on the round budget.
    "k_memory": 3,
    "deadline_seconds": 0,  # 0 = no internal wall-clock stop (the Kaggle session limit still applies)
}

OUT = pathlib.Path("/kaggle/working")
JOURNAL = OUT / "harness_completions.jsonl"
MANIFEST = OUT / "manifest.json"

# Hosted slugs we expect to see. This is ONLY a warning guard: an unrecognised name
# is still recorded verbatim from kbench.llm, never replaced with a guessed slug.
EXPECTED_MODELS = (
    "gpt-6-luna", "claude-haiku-5-5", "gemini-3.8-flash",
    "gpt-6.2-sol", "claude-sonnet-5-5", "claude-opus-5-5", "gemini-4-argon",
)

TASKS = ("repair", "ledger", "audit", "design")

# Embedded from scripts/frontier_cohort_assign.py plans (global one-model-per-diagram ledger).
MODEL_IDS = {'claude-haiku-5-5': ['aud-arch-0-01316', 'aud-0-00522', 'aud-0-02406', 'aud-0-01346', 'aud-0-02873', 'aud-0-00347', 'aud-0-01328', 'aud-0-02408', 'aud-0-02840', 'aud-arch-0-00759', 'aud-0-01224', 'aud-arch-0-00471', 'aud-arch-0-00313', 'aud-0-02444', 'aud-0-01441', 'aud-0-01306', 'aud-0-01911', 'aud-0-00180', 'aud-0-02258', 'aud-0-01855', 'aud-arch-0-00769', 'aud-0-00631', 'aud-0-00448', 'aud-0-00142', 'aud-0-02111', 'aud-0-01558', 'aud-arch-0-00253', 'aud-arch-0-01082', 'aud-arch-0-00897', 'aud-arch-0-00775', 'aud-arch-0-01157', 'aud-0-02253', 'aud-arch-0-01771', 'aud-arch-0-00627', 'aud-0-01708', 'aud-0-01376', 'aud-0-01121', 'aud-0-00565', 'aud-arch-0-01484', 'aud-0-01270', 'aud-0-00415', 'aud-arch-0-01424', 'aud-0-01681', 'aud-arch-0-01927', 'aud-0-00705', 'aud-0-02993', 'aud-0-02154', 'aud-0-00046', 'aud-0-02042', 'aud-0-00059', 'aud-arch-0-00249', 'aud-0-02424', 'aud-arch-0-00320', 'aud-0-02357', 'aud-arch-0-00767', 'aud-arch-0-00319', 'aud-0-01556', 'aud-0-01934', 'aud-0-02503', 'aud-0-00656', 'aud-arch-0-00932', 'aud-0-02898', 'aud-0-02480', 'aud-arch-0-00239', 'aud-arch-0-00743', 'aud-0-00974', 'aud-0-01846', 'aud-arch-0-01451', 'aud-arch-0-01505', 'aud-arch-0-00677', 'aud-0-02962', 'aud-0-00865', 'aud-arch-0-00277', 'aud-0-00817', 'aud-arch-0-01001', 'aud-0-02280', 'aud-0-01498', 'aud-0-02262', 'aud-0-02929', 'aud-0-01994', 'aud-0-00749', 'aud-0-00273', 'aud-0-00801', 'aud-0-00093', 'aud-0-01018', 'aud-arch-0-00628', 'aud-0-02298', 'aud-arch-0-01127', 'aud-arch-0-01322', 'aud-0-00921', 'aud-0-01105', 'aud-arch-0-00520', 'aud-0-00270', 'aud-arch-0-00091', 'aud-0-02552', 'aud-0-01134', 'aud-0-01054', 'aud-0-01109', 'aud-0-02263', 'aud-0-01093', 'aud-0-00725', 'aud-arch-0-01950', 'aud-arch-0-01830', 'aud-0-02100', 'aud-0-02939', 'aud-0-02117', 'aud-arch-0-01603', 'aud-0-00972', 'aud-arch-0-00232', 'aud-arch-0-01521', 'aud-arch-0-01077', 'aud-0-01854', 'aud-0-01094', 'aud-0-02172', 'aud-0-01777', 'aud-0-02944', 'aud-0-02883', 'aud-arch-0-01993', 'aud-arch-0-01512', 'aud-arch-0-00390', 'led-0-04878', 'led-arch-0-01142', 'led-0-04123', 'led-0-07324', 'led-arch-0-02140', 'led-0-08426', 'led-0-06753', 'led-arch-0-02723', 'led-arch-0-02847', 'led-0-02107', 'led-0-05993', 'led-0-07407', 'led-0-05201', 'led-0-05151', 'led-0-04231', 'led-0-06832', 'led-0-03308', 'led-arch-0-00797', 'led-0-03593', 'led-0-06209', 'led-0-03124', 'led-0-02831', 'led-arch-0-00773', 'led-0-03873', 'led-0-03915', 'led-0-07721', 'led-0-07052', 'led-0-05454', 'led-0-02762', 'led-arch-0-01092', 'led-0-06175', 'led-0-02879', 'led-arch-0-00904', 'led-0-04570', 'led-0-05358', 'led-arch-0-00995', 'led-0-03968', 'led-arch-0-00903', 'led-0-08026', 'led-0-08206', 'led-0-02811', 'led-0-07484', 'led-0-00883', 'led-0-01066', 'led-0-07186', 'led-0-03992', 'led-arch-0-02675', 'led-0-00929', 'led-0-04709', 'led-0-07004', 'led-0-01234', 'led-arch-0-00790', 'led-0-07634', 'led-0-01245', 'led-0-07992', 'led-0-01405', 'led-0-05873', 'led-0-00982', 'led-0-05535', 'led-0-06924', 'led-0-03406', 'led-0-00590', 'led-0-05340', 'led-0-01162', 'led-0-00488', 'led-0-00961', 'led-0-07939', 'led-0-07402', 'led-0-03314', 'led-0-01063', 'led-0-02991', 'led-0-08974', 'led-0-01348', 'led-0-04224', 'led-0-04086', 'led-0-07959', 'led-arch-0-02312', 'led-0-07825', 'led-0-06610', 'led-0-03908'], 'gpt-6-luna': ['aud-0-01715', 'aud-0-02307', 'aud-arch-0-00702', 'aud-0-02478', 'aud-0-02008', 'aud-0-02728', 'aud-0-00512', 'aud-0-02843', 'aud-0-00795', 'aud-arch-0-00574', 'aud-0-01025', 'aud-0-01472', 'aud-arch-0-00905', 'aud-0-02957', 'aud-0-00595', 'aud-0-00507', 'aud-0-01199', 'aud-arch-0-01683', 'aud-arch-0-01538', 'aud-arch-0-00975', 'aud-0-00661', 'aud-arch-0-00493', 'aud-0-01927', 'aud-arch-0-01289', 'aud-arch-0-01040', 'aud-0-02694', 'aud-0-02419', 'aud-0-01996', 'aud-0-02531', 'aud-0-01569', 'aud-0-00873', 'aud-0-01678', 'aud-0-02146', 'aud-arch-0-01966', 'aud-0-01285', 'aud-arch-0-00579', 'aud-arch-0-01772', 'aud-0-02837', 'aud-0-00392', 'aud-0-01321', 'aud-arch-0-00666', 'aud-0-02914', 'aud-0-02641', 'aud-0-02969', 'aud-0-02176', 'aud-0-01371', 'aud-0-00224', 'aud-arch-0-00182', 'aud-arch-0-00221', 'aud-0-01523', 'aud-arch-0-01467', 'aud-arch-0-01316', 'aud-0-02051', 'aud-0-00522', 'aud-arch-0-01352', 'aud-0-01844', 'aud-arch-0-01710', 'aud-0-01625', 'aud-arch-0-00995', 'aud-0-02277', 'aud-arch-0-00335', 'aud-0-00399', 'aud-0-01333', 'aud-0-01742', 'aud-0-02214', 'aud-0-01667', 'aud-arch-0-00937', 'aud-0-00343', 'aud-0-00985', 'aud-arch-0-00530', 'aud-arch-0-01003', 'aud-0-00981', 'aud-0-02547', 'aud-0-00100', 'aud-0-02511', 'aud-0-02827', 'aud-arch-0-01387', 'aud-0-02628', 'aud-0-00082', 'aud-0-02399', 'aud-0-01253', 'aud-arch-0-01106', 'aud-arch-0-01895', 'aud-0-00237', 'aud-0-01884', 'aud-0-00101', 'aud-0-00950', 'aud-0-02012', 'aud-0-01209', 'aud-0-02515', 'aud-0-02551', 'aud-arch-0-01653', 'aud-arch-0-00173', 'aud-0-00518', 'aud-0-01605', 'aud-0-02066', 'aud-0-02550', 'aud-0-02332', 'aud-0-01335', 'aud-0-00723', 'aud-arch-0-01198', 'aud-0-01354', 'aud-0-02812', 'aud-0-01164', 'aud-arch-0-00342', 'aud-0-01082', 'aud-0-01823', 'aud-arch-0-01450', 'aud-arch-0-01902', 'aud-arch-0-01338', 'aud-arch-0-00151', 'aud-0-02575', 'aud-0-02502', 'aud-0-00364', 'aud-arch-0-01970', 'aud-0-01838', 'aud-arch-0-01115', 'aud-arch-0-00260', 'aud-0-02134', 'aud-0-01469', 'aud-0-01104', 'aud-arch-0-01145', 'aud-arch-0-00112', 'aud-0-01849', 'aud-arch-0-01670', 'aud-0-00582', 'aud-arch-0-01405', 'aud-0-01869', 'aud-arch-0-01432', 'aud-arch-0-01503', 'aud-0-00434', 'aud-0-02139', 'aud-arch-0-00847', 'aud-0-01269', 'aud-0-01284', 'aud-arch-0-00405', 'aud-0-02895', 'aud-0-00663', 'aud-0-02675', 'aud-0-00844', 'aud-0-01591', 'aud-0-01068', 'aud-0-02965', 'aud-0-02371', 'aud-0-02022', 'aud-0-00825', 'aud-0-02382', 'aud-0-02518', 'aud-0-02103', 'aud-arch-0-01368', 'aud-0-02436', 'aud-0-01366', 'aud-0-01638', 'aud-0-02904', 'aud-arch-0-00653', 'aud-arch-0-01509', 'aud-0-01557', 'aud-arch-0-00566', 'aud-arch-0-00923', 'aud-arch-0-01438', 'aud-arch-0-00691', 'aud-arch-0-00659', 'aud-0-02328', 'aud-0-02805', 'aud-arch-0-01081', 'aud-0-02596', 'aud-arch-0-00981', 'aud-0-00913', 'aud-0-02265', 'aud-0-02916', 'aud-0-02243', 'aud-arch-0-00002', 'aud-arch-0-00522', 'aud-0-00540', 'aud-0-01417', 'aud-0-00247', 'aud-arch-0-00785', 'aud-arch-0-01630', 'aud-arch-0-00561', 'aud-arch-0-00853']}
ROW_CELL_IDS = {'aud-arch-0-01316': 'audit/easy', 'aud-0-00522': 'audit/easy', 'aud-0-02406': 'audit/easy', 'aud-0-01346': 'audit/easy', 'aud-0-02873': 'audit/easy', 'aud-0-00347': 'audit/easy', 'aud-0-01328': 'audit/easy', 'aud-0-02408': 'audit/easy', 'aud-0-02840': 'audit/easy', 'aud-arch-0-00759': 'audit/easy', 'aud-0-01224': 'audit/easy', 'aud-arch-0-00471': 'audit/easy', 'aud-arch-0-00313': 'audit/easy', 'aud-0-02444': 'audit/easy', 'aud-0-01441': 'audit/easy', 'aud-0-01306': 'audit/easy', 'aud-0-01911': 'audit/easy', 'aud-0-00180': 'audit/easy', 'aud-0-02258': 'audit/easy', 'aud-0-01855': 'audit/easy', 'aud-arch-0-00769': 'audit/easy', 'aud-0-00631': 'audit/easy', 'aud-0-00448': 'audit/easy', 'aud-0-00142': 'audit/easy', 'aud-0-02111': 'audit/easy', 'aud-0-01558': 'audit/easy', 'aud-arch-0-00253': 'audit/easy', 'aud-arch-0-01082': 'audit/easy', 'aud-arch-0-00897': 'audit/easy', 'aud-arch-0-00775': 'audit/easy', 'aud-arch-0-01157': 'audit/easy', 'aud-0-02253': 'audit/easy', 'aud-arch-0-01771': 'audit/easy', 'aud-arch-0-00627': 'audit/easy', 'aud-0-01708': 'audit/easy', 'aud-0-01376': 'audit/easy', 'aud-0-01121': 'audit/easy', 'aud-0-00565': 'audit/easy', 'aud-arch-0-01484': 'audit/easy', 'aud-0-01270': 'audit/easy', 'aud-0-00415': 'audit/easy', 'aud-arch-0-01424': 'audit/easy', 'aud-0-01681': 'audit/easy', 'aud-arch-0-01927': 'audit/easy', 'aud-0-00705': 'audit/easy', 'aud-0-02993': 'audit/easy', 'aud-0-02154': 'audit/easy', 'aud-0-00046': 'audit/easy', 'aud-0-02042': 'audit/easy', 'aud-0-00059': 'audit/easy', 'aud-arch-0-00249': 'audit/easy', 'aud-0-02424': 'audit/easy', 'aud-arch-0-00320': 'audit/easy', 'aud-0-02357': 'audit/easy', 'aud-arch-0-00767': 'audit/easy', 'aud-arch-0-00319': 'audit/easy', 'aud-0-01556': 'audit/easy', 'aud-0-01934': 'audit/easy', 'aud-0-02503': 'audit/easy', 'aud-0-00656': 'audit/easy', 'aud-arch-0-00932': 'audit/medium', 'aud-0-02898': 'audit/medium', 'aud-0-02480': 'audit/medium', 'aud-arch-0-00239': 'audit/medium', 'aud-arch-0-00743': 'audit/medium', 'aud-0-00974': 'audit/medium', 'aud-0-01846': 'audit/medium', 'aud-arch-0-01451': 'audit/medium', 'aud-arch-0-01505': 'audit/medium', 'aud-arch-0-00677': 'audit/medium', 'aud-0-02962': 'audit/medium', 'aud-0-00865': 'audit/medium', 'aud-arch-0-00277': 'audit/medium', 'aud-0-00817': 'audit/medium', 'aud-arch-0-01001': 'audit/medium', 'aud-0-02280': 'audit/medium', 'aud-0-01498': 'audit/medium', 'aud-0-02262': 'audit/medium', 'aud-0-02929': 'audit/medium', 'aud-0-01994': 'audit/medium', 'aud-0-00749': 'audit/medium', 'aud-0-00273': 'audit/medium', 'aud-0-00801': 'audit/medium', 'aud-0-00093': 'audit/medium', 'aud-0-01018': 'audit/medium', 'aud-arch-0-00628': 'audit/medium', 'aud-0-02298': 'audit/medium', 'aud-arch-0-01127': 'audit/medium', 'aud-arch-0-01322': 'audit/medium', 'aud-0-00921': 'audit/medium', 'aud-0-01105': 'audit/medium', 'aud-arch-0-00520': 'audit/medium', 'aud-0-00270': 'audit/medium', 'aud-arch-0-00091': 'audit/medium', 'aud-0-02552': 'audit/medium', 'aud-0-01134': 'audit/medium', 'aud-0-01054': 'audit/medium', 'aud-0-01109': 'audit/medium', 'aud-0-02263': 'audit/medium', 'aud-0-01093': 'audit/medium', 'aud-0-00725': 'audit/medium', 'aud-arch-0-01950': 'audit/medium', 'aud-arch-0-01830': 'audit/medium', 'aud-0-02100': 'audit/medium', 'aud-0-02939': 'audit/medium', 'aud-0-02117': 'audit/medium', 'aud-arch-0-01603': 'audit/medium', 'aud-0-00972': 'audit/medium', 'aud-arch-0-00232': 'audit/medium', 'aud-arch-0-01521': 'audit/medium', 'aud-arch-0-01077': 'audit/medium', 'aud-0-01854': 'audit/medium', 'aud-0-01094': 'audit/medium', 'aud-0-02172': 'audit/medium', 'aud-0-01777': 'audit/medium', 'aud-0-02944': 'audit/medium', 'aud-0-02883': 'audit/medium', 'aud-arch-0-01993': 'audit/medium', 'aud-arch-0-01512': 'audit/medium', 'aud-arch-0-00390': 'audit/medium', 'led-0-04878': 'ledger/easy', 'led-arch-0-01142': 'ledger/easy', 'led-0-04123': 'ledger/easy', 'led-0-07324': 'ledger/easy', 'led-arch-0-02140': 'ledger/easy', 'led-0-08426': 'ledger/easy', 'led-0-06753': 'ledger/easy', 'led-arch-0-02723': 'ledger/easy', 'led-arch-0-02847': 'ledger/easy', 'led-0-02107': 'ledger/easy', 'led-0-05993': 'ledger/easy', 'led-0-07407': 'ledger/easy', 'led-0-05201': 'ledger/easy', 'led-0-05151': 'ledger/easy', 'led-0-04231': 'ledger/easy', 'led-0-06832': 'ledger/easy', 'led-0-03308': 'ledger/easy', 'led-arch-0-00797': 'ledger/easy', 'led-0-03593': 'ledger/easy', 'led-0-06209': 'ledger/easy', 'led-0-03124': 'ledger/easy', 'led-0-02831': 'ledger/easy', 'led-arch-0-00773': 'ledger/easy', 'led-0-03873': 'ledger/easy', 'led-0-03915': 'ledger/easy', 'led-0-07721': 'ledger/easy', 'led-0-07052': 'ledger/easy', 'led-0-05454': 'ledger/easy', 'led-0-02762': 'ledger/easy', 'led-arch-0-01092': 'ledger/easy', 'led-0-06175': 'ledger/easy', 'led-0-02879': 'ledger/easy', 'led-arch-0-00904': 'ledger/easy', 'led-0-04570': 'ledger/easy', 'led-0-05358': 'ledger/easy', 'led-arch-0-00995': 'ledger/easy', 'led-0-03968': 'ledger/easy', 'led-arch-0-00903': 'ledger/easy', 'led-0-08026': 'ledger/easy', 'led-0-08206': 'ledger/easy', 'led-0-02811': 'ledger/medium', 'led-0-07484': 'ledger/medium', 'led-0-00883': 'ledger/medium', 'led-0-01066': 'ledger/medium', 'led-0-07186': 'ledger/medium', 'led-0-03992': 'ledger/medium', 'led-arch-0-02675': 'ledger/medium', 'led-0-00929': 'ledger/medium', 'led-0-04709': 'ledger/medium', 'led-0-07004': 'ledger/medium', 'led-0-01234': 'ledger/medium', 'led-arch-0-00790': 'ledger/medium', 'led-0-07634': 'ledger/medium', 'led-0-01245': 'ledger/medium', 'led-0-07992': 'ledger/medium', 'led-0-01405': 'ledger/medium', 'led-0-05873': 'ledger/medium', 'led-0-00982': 'ledger/medium', 'led-0-05535': 'ledger/medium', 'led-0-06924': 'ledger/medium', 'led-0-03406': 'ledger/medium', 'led-0-00590': 'ledger/medium', 'led-0-05340': 'ledger/medium', 'led-0-01162': 'ledger/medium', 'led-0-00488': 'ledger/medium', 'led-0-00961': 'ledger/medium', 'led-0-07939': 'ledger/medium', 'led-0-07402': 'ledger/medium', 'led-0-03314': 'ledger/medium', 'led-0-01063': 'ledger/medium', 'led-0-02991': 'ledger/medium', 'led-0-08974': 'ledger/medium', 'led-0-01348': 'ledger/medium', 'led-0-04224': 'ledger/medium', 'led-0-04086': 'ledger/medium', 'led-0-07959': 'ledger/medium', 'led-arch-0-02312': 'ledger/medium', 'led-0-07825': 'ledger/medium', 'led-0-06610': 'ledger/medium', 'led-0-03908': 'ledger/medium', 'aud-0-01715': 'audit/easy', 'aud-0-02307': 'audit/easy', 'aud-arch-0-00702': 'audit/easy', 'aud-0-02478': 'audit/easy', 'aud-0-02008': 'audit/easy', 'aud-0-02728': 'audit/easy', 'aud-0-00512': 'audit/easy', 'aud-0-02843': 'audit/easy', 'aud-0-00795': 'audit/easy', 'aud-arch-0-00574': 'audit/easy', 'aud-0-01025': 'audit/easy', 'aud-0-01472': 'audit/easy', 'aud-arch-0-00905': 'audit/easy', 'aud-0-02957': 'audit/easy', 'aud-0-00595': 'audit/easy', 'aud-0-00507': 'audit/easy', 'aud-0-01199': 'audit/easy', 'aud-arch-0-01683': 'audit/easy', 'aud-arch-0-01538': 'audit/easy', 'aud-arch-0-00975': 'audit/easy', 'aud-0-00661': 'audit/easy', 'aud-arch-0-00493': 'audit/easy', 'aud-0-01927': 'audit/easy', 'aud-arch-0-01289': 'audit/easy', 'aud-arch-0-01040': 'audit/easy', 'aud-0-02694': 'audit/easy', 'aud-0-02419': 'audit/easy', 'aud-0-01996': 'audit/easy', 'aud-0-02531': 'audit/easy', 'aud-0-01569': 'audit/easy', 'aud-0-00873': 'audit/easy', 'aud-0-01678': 'audit/easy', 'aud-0-02146': 'audit/easy', 'aud-arch-0-01966': 'audit/easy', 'aud-0-01285': 'audit/easy', 'aud-arch-0-00579': 'audit/easy', 'aud-arch-0-01772': 'audit/easy', 'aud-0-02837': 'audit/easy', 'aud-0-00392': 'audit/easy', 'aud-0-01321': 'audit/easy', 'aud-arch-0-00666': 'audit/easy', 'aud-0-02914': 'audit/easy', 'aud-0-02641': 'audit/easy', 'aud-0-02969': 'audit/easy', 'aud-0-02176': 'audit/easy', 'aud-0-01371': 'audit/easy', 'aud-0-00224': 'audit/easy', 'aud-arch-0-00182': 'audit/easy', 'aud-arch-0-00221': 'audit/easy', 'aud-0-01523': 'audit/easy', 'aud-arch-0-01467': 'audit/easy', 'aud-0-02051': 'audit/easy', 'aud-arch-0-01352': 'audit/easy', 'aud-0-01844': 'audit/easy', 'aud-arch-0-01710': 'audit/easy', 'aud-0-01625': 'audit/easy', 'aud-arch-0-00995': 'audit/easy', 'aud-0-02277': 'audit/easy', 'aud-arch-0-00335': 'audit/medium', 'aud-0-00399': 'audit/medium', 'aud-0-01333': 'audit/medium', 'aud-0-01742': 'audit/medium', 'aud-0-02214': 'audit/medium', 'aud-0-01667': 'audit/medium', 'aud-arch-0-00937': 'audit/medium', 'aud-0-00343': 'audit/medium', 'aud-0-00985': 'audit/medium', 'aud-arch-0-00530': 'audit/medium', 'aud-arch-0-01003': 'audit/medium', 'aud-0-00981': 'audit/medium', 'aud-0-02547': 'audit/medium', 'aud-0-00100': 'audit/medium', 'aud-0-02511': 'audit/medium', 'aud-0-02827': 'audit/medium', 'aud-arch-0-01387': 'audit/medium', 'aud-0-02628': 'audit/medium', 'aud-0-00082': 'audit/medium', 'aud-0-02399': 'audit/medium', 'aud-0-01253': 'audit/medium', 'aud-arch-0-01106': 'audit/medium', 'aud-arch-0-01895': 'audit/medium', 'aud-0-00237': 'audit/medium', 'aud-0-01884': 'audit/medium', 'aud-0-00101': 'audit/medium', 'aud-0-00950': 'audit/medium', 'aud-0-02012': 'audit/medium', 'aud-0-01209': 'audit/medium', 'aud-0-02515': 'audit/medium', 'aud-0-02551': 'audit/medium', 'aud-arch-0-01653': 'audit/medium', 'aud-arch-0-00173': 'audit/medium', 'aud-0-00518': 'audit/medium', 'aud-0-01605': 'audit/medium', 'aud-0-02066': 'audit/medium', 'aud-0-02550': 'audit/medium', 'aud-0-02332': 'audit/medium', 'aud-0-01335': 'audit/medium', 'aud-0-00723': 'audit/medium', 'aud-arch-0-01198': 'audit/medium', 'aud-0-01354': 'audit/medium', 'aud-0-02812': 'audit/medium', 'aud-0-01164': 'audit/medium', 'aud-arch-0-00342': 'audit/medium', 'aud-0-01082': 'audit/medium', 'aud-0-01823': 'audit/medium', 'aud-arch-0-01450': 'audit/medium', 'aud-arch-0-01902': 'audit/medium', 'aud-arch-0-01338': 'audit/medium', 'aud-arch-0-00151': 'audit/medium', 'aud-0-02575': 'audit/medium', 'aud-0-02502': 'audit/medium', 'aud-0-00364': 'audit/medium', 'aud-arch-0-01970': 'audit/medium', 'aud-0-01838': 'audit/medium', 'aud-arch-0-01115': 'audit/medium', 'aud-arch-0-00260': 'audit/medium', 'aud-0-02134': 'audit/medium', 'aud-0-01469': 'audit/medium', 'aud-0-01104': 'audit/hard', 'aud-arch-0-01145': 'audit/hard', 'aud-arch-0-00112': 'audit/hard', 'aud-0-01849': 'audit/hard', 'aud-arch-0-01670': 'audit/hard', 'aud-0-00582': 'audit/hard', 'aud-arch-0-01405': 'audit/hard', 'aud-0-01869': 'audit/hard', 'aud-arch-0-01432': 'audit/hard', 'aud-arch-0-01503': 'audit/hard', 'aud-0-00434': 'audit/hard', 'aud-0-02139': 'audit/hard', 'aud-arch-0-00847': 'audit/hard', 'aud-0-01269': 'audit/hard', 'aud-0-01284': 'audit/hard', 'aud-arch-0-00405': 'audit/hard', 'aud-0-02895': 'audit/hard', 'aud-0-00663': 'audit/hard', 'aud-0-02675': 'audit/hard', 'aud-0-00844': 'audit/hard', 'aud-0-01591': 'audit/hard', 'aud-0-01068': 'audit/hard', 'aud-0-02965': 'audit/hard', 'aud-0-02371': 'audit/hard', 'aud-0-02022': 'audit/hard', 'aud-0-00825': 'audit/hard', 'aud-0-02382': 'audit/hard', 'aud-0-02518': 'audit/hard', 'aud-0-02103': 'audit/hard', 'aud-arch-0-01368': 'audit/hard', 'aud-0-02436': 'audit/hard', 'aud-0-01366': 'audit/hard', 'aud-0-01638': 'audit/hard', 'aud-0-02904': 'audit/hard', 'aud-arch-0-00653': 'audit/hard', 'aud-arch-0-01509': 'audit/hard', 'aud-0-01557': 'audit/hard', 'aud-arch-0-00566': 'audit/hard', 'aud-arch-0-00923': 'audit/hard', 'aud-arch-0-01438': 'audit/hard', 'aud-arch-0-00691': 'audit/hard', 'aud-arch-0-00659': 'audit/hard', 'aud-0-02328': 'audit/hard', 'aud-0-02805': 'audit/hard', 'aud-arch-0-01081': 'audit/hard', 'aud-0-02596': 'audit/hard', 'aud-arch-0-00981': 'audit/hard', 'aud-0-00913': 'audit/hard', 'aud-0-02265': 'audit/hard', 'aud-0-02916': 'audit/hard', 'aud-0-02243': 'audit/hard', 'aud-arch-0-00002': 'audit/hard', 'aud-arch-0-00522': 'audit/hard', 'aud-0-00540': 'audit/hard', 'aud-0-01417': 'audit/hard', 'aud-0-00247': 'audit/hard', 'aud-arch-0-00785': 'audit/hard', 'aud-arch-0-01630': 'audit/hard', 'aud-arch-0-00561': 'audit/hard', 'aud-arch-0-00853': 'audit/hard'}


# A verdict is NEVER fabricated on the hosted side. Compact shard rows have no plant/spec
# columns, so their only acceptance source is the local transfer.ingest re-grade by id.
DEFERRED_VERDICT = {"accepted": None, "graded": False, "reason": "deferred_to_ingest"}
NOT_RUN_VERDICT = {"accepted": None, "graded": False, "reason": "not_run"}

# Columns each in-task validator needs (mirrors thermo_agent.teacher.validate).
GRADABLE_KEYS = {
    "repair": ("plant_broken", "plant_original"),
    "ledger": ("ledger",),
    "audit": ("plant", "ed_json", "fuel_w", "net_w"),
    "design": ("site",),
}


# --------------------------------------------------------------------------- #
# bootstrap (standard library only)
# --------------------------------------------------------------------------- #
def _bundle_from_input_dir(inp: pathlib.Path):
    """An attached code dataset may already contain the extracted source tree."""
    if not inp.exists():
        return None
    for marker in inp.rglob("agent_graph.py"):
        if marker.parent.name != "teacher":
            continue
        root = marker.parents[2]
        if (root / "thermo_agent" / "teacher" / "agent_graph.py").exists():
            return root
    return None


def _bundle_from_zip(inp: pathlib.Path, work: pathlib.Path):
    """Otherwise unpack kaggle_bundle.zip from the attached code dataset."""
    if not inp.exists():
        return None
    zips = sorted(inp.rglob("kaggle_bundle.zip"))
    if not zips:
        return None
    work.mkdir(parents=True, exist_ok=True)
    marker = work / "thermo_agent" / "teacher" / "agent_graph.py"
    if not marker.exists():
        dest = work.resolve()
        with zipfile.ZipFile(zips[0]) as zf:
            for info in zf.infolist():
                target = (work / info.filename).resolve()
                # zip-slip guard: never write outside the extraction root.
                if target != dest and dest not in target.parents:
                    continue
                zf.extract(info, work)
    return work if marker.exists() else None


def _bootstrap() -> pathlib.Path:
    inp = pathlib.Path("/kaggle/input")
    work = OUT / "thermo-agent"
    root = _bundle_from_input_dir(inp)
    if root is None:
        root = _bundle_from_zip(inp, work)
    if root is None:
        # Local development / py_compile convenience: the repo is one level up.
        local = pathlib.Path(__file__).resolve().parents[1]
        if (local / "thermo_agent" / "teacher" / "agent_graph.py").exists():
            root = local
    if root is None:
        raise SystemExit(
            "thermo-agent source not found: attach -d %s (or set up /kaggle/working/thermo-agent)"
            % SPEC["code_dataset"]
        )
    return root


BUNDLE_ROOT = _bootstrap()
sys.path.insert(0, str(BUNDLE_ROOT))


def _ensure_optional_deps() -> None:
    """The benchmarks venv is minimal (no scipy, sometimes no CoolProp).

    scipy is handled by thermo_agent.core._roots (pure-Python Brent fallback), so the
    graph imports without it. CoolProp cannot be reimplemented: install the wheel into a
    writable directory when the network allows, and let gas-only plants run when it does
    not. Never fail the whole task because a real-fluid backend is missing.
    """
    try:
        import CoolProp  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    target = OUT / "pylibs"
    target.mkdir(parents=True, exist_ok=True)
    for cmd in ([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                 "--target", str(target), "CoolProp==6.8.0"],):
        try:
            import subprocess
            subprocess.run(cmd, check=True, timeout=420, capture_output=True)
            sys.path.insert(0, str(target))
            import CoolProp  # noqa: F401
            print("[deps] CoolProp installed into", target, flush=True)
            return
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[deps] CoolProp unavailable ({type(exc).__name__}); gas-only plants will run", flush=True)


_ensure_optional_deps()

# Repo imports come AFTER the stdlib-only bootstrap.
import kaggle_benchmarks as kbench  # noqa: E402
from kaggle_benchmarks import actors  # noqa: E402

from thermo_agent.teacher import memory as mem  # noqa: E402
from thermo_agent.teacher.agent_graph import (  # noqa: E402
    DEFAULT_MAX_ROUNDS, build_context, run_direct,
)
from thermo_agent.teacher.kbench_adapter import make_respond, reasoning_for_model, usage_of  # noqa: E402
from thermo_agent.teacher.toolkit import tool_names  # noqa: E402

SPEC["rounds"] = DEFAULT_MAX_ROUNDS
SPEC_SHA = hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()[:16]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


SHARD = os.environ.get("HARNESS_SHARD") or SPEC["shard"]
LIMIT = _env_int("HARNESS_LIMIT", SPEC["limit"])
# Default to the project constant, not a local copy of the number.
ROUNDS = _env_int("HARNESS_ROUNDS", DEFAULT_MAX_ROUNDS)
K_MEMORY = _env_int("HARNESS_K_MEMORY", SPEC["k_memory"])
DEADLINE = _env_int("HARNESS_DEADLINE", SPEC["deadline_seconds"])
MAX_COST_USD = float(os.environ.get("HARNESS_MAX_COST_USD", SPEC["max_cost_usd"]))

# Never journal a credential, even if a provider error happens to echo one.
_SECRET_PAT = re.compile(
    r"(?i)(kaggle[_-]?api[_-]?token|hf[_-]?token|api[_-]?key|authorization|bearer\s+\S+|sk-[A-Za-z0-9_\-]{8,})"
)


def _redact(value) -> str:
    return _SECRET_PAT.sub(lambda m: m.group(1).split()[0] + "=***", str(value))


# --------------------------------------------------------------------------- #
# row loading
# --------------------------------------------------------------------------- #
def _find_shard(root: pathlib.Path) -> pathlib.Path:
    candidates = []
    for base in (pathlib.Path("/kaggle/input"), root, pathlib.Path(".")):
        if base.exists():
            candidates += [p.parent for p in base.rglob(SHARD)]
    candidates.append(root / "data" / "kaggle_datasets" / "teacher_big_prompts")
    seen = set()
    for cand in candidates:
        key = str(cand)
        if key in seen or not (cand / SHARD).exists():
            continue
        seen.add(key)
        return cand / SHARD
    raise SystemExit(
        "prompt shard %s not found; attach -d %s or set HARNESS_SHARD"
        % (SHARD, SPEC["dataset"])
    )


def _find_export_dir(root: pathlib.Path):
    """Full export rows (prompt + plant/spec columns), if any dataset carries them."""
    for base in (pathlib.Path("/kaggle/input"), root):
        if not base.exists():
            continue
        for marker in base.rglob("repair_train.jsonl"):
            if all((marker.parent / "%s_train.jsonl" % t).exists() for t in TASKS):
                return marker.parent
    local = root / "data" / "frontier_export"
    if (local / "repair_train.jsonl").exists():
        return local
    return None


def _load_exports_callable(root: pathlib.Path):
    """Prefer the driver's own load_exports; fall back to the transfer function it imports."""
    cli = root / "scripts" / "kaggle_harness_cli.py"
    if cli.exists():
        try:
            spec = importlib.util.spec_from_file_location("_harness_cli_for_exports", cli)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if callable(getattr(module, "load_exports", None)):
                return module.load_exports
        except Exception:
            pass
    try:
        from thermo_agent.teacher.transfer import load_exports
        return load_exports
    except Exception:
        return None


def _load_exports_replica(export_dir, tasks=TASKS, split: str = "train") -> dict:
    """Exact replicate of transfer.load_exports (id-keyed, one row per id)."""
    rows = {}
    for task in tasks:
        path = pathlib.Path(export_dir) / ("%s_%s.jsonl" % (task, split))
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def _adapt_row(row: dict) -> dict:
    """Give compact shard rows the shape agent_graph expects, keeping every source column."""
    out = dict(row)
    if not isinstance(out.get("prompt"), list):
        messages = []
        if out.get("system"):
            messages.append({"role": "system", "content": out["system"]})
        if out.get("user"):
            messages.append({"role": "user", "content": out["user"]})
        out["prompt"] = messages
    out.setdefault("task", "repair")
    out.setdefault("split", "train")
    return out


def _load_rows(root: pathlib.Path, model_slug: str):
    """Load ONLY the diagrams the global ledger assigned to this model.

    Rows are the unmodified tool-ready inputs from the pilot's
    ``thermo-teacher-prompts-big`` dataset.  The id list is produced by
    ``scripts/frontier_cohort_assign.py``; no bookkeeping id, tier label or
    manifest text is added to the model-visible prompt.  Per-model journal
    dedupe (below) still handles resume.
    """
    wanted = list(MODEL_IDS.get(model_slug, ()))
    target = set(wanted)
    found: dict[str, dict] = {}
    for base in (pathlib.Path("/kaggle/input"), root, pathlib.Path(".")):
        if not base.exists():
            continue
        for path in base.rglob("prompts_s*.jsonl"):
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                rid = str(row.get("id", ""))
                if rid in target and rid not in found:
                    found[rid] = _adapt_row(row)
    ordered = [found[rid] for rid in wanted if rid in found]
    print("[cohort] model=%s wanted=%d found=%d"
          % (model_slug, len(wanted), len(ordered)), flush=True)
    return ordered


def _memory_dir_candidates(root: pathlib.Path):
    """Every location retrieval memory could live, in priority order (existing or not)."""
    out = []
    if os.environ.get("HARNESS_MEMORY_DIR"):
        out.append(pathlib.Path(os.environ["HARNESS_MEMORY_DIR"]))
    # The attached code bundle is built with Compress-Archive, which stripped the `data`
    # parent: the notes land at BUNDLE_ROOT/memory. Check that FIRST, then keep the legacy
    # data/memory locations and the /kaggle/input rglob fallbacks below.
    out += [root / "memory", root / "data" / "memory", OUT / "thermo-agent" / "data" / "memory"]
    inp = pathlib.Path("/kaggle/input")
    if inp.exists():
        out += list(inp.rglob("data/memory"))[:5]
        out += list(inp.rglob("memory"))[:5]
    return out


def _memory_dirs(root: pathlib.Path):
    return [d for d in _memory_dir_candidates(root) if d.exists()]


def _fresh_notes(root: pathlib.Path):
    notes, seen = [], set()
    searched = _memory_dirs(root)
    for directory in searched:
        for path in sorted(directory.glob("*.jsonl")):
            if path in seen:
                continue
            seen.add(path)
            for line in path.open(encoding="utf-8"):
                try:
                    notes.append(mem.Note(**json.loads(line)))
                except Exception:
                    pass
    try:
        notes = mem.dedupe(notes)
    except Exception:
        pass
    if not notes:
        # Never hide a silent loss of all retrieval memory: the manifest records the count.
        print("[warn] no retrieval memory found; searched: %s"
              % ", ".join(str(d) for d in _memory_dir_candidates(root)), flush=True)
    return notes


def _row_is_gradable(task: str, row: dict) -> bool:
    """True only when the row carries the plant/spec columns the real validator reads."""
    keys = GRADABLE_KEYS.get(task)
    if not keys:
        return False
    return all(row.get(k) not in (None, "", {}, []) for k in keys)


def _install_safe_validator():
    """Let finalize finish on compact rows WITHOUT claiming any acceptance.

    On a missing-column failure the wrapper returns ``DEFERRED_VERDICT`` (accepted=None,
    graded=False). It never returns accepted=True or accepted=False: local
    ``transfer.ingest`` re-grades against the full export by id and is the ONLY source
    of acceptance for those rows. Full rows are unaffected -- the real validator runs
    first and its verdict is used as-is.
    """
    import thermo_agent.teacher.validate as V

    if getattr(V.validate, "_harness_safe", False):
        return
    original = V.validate

    def safe(task, completion, row):
        try:
            return original(task, completion, row)
        except (KeyError, ValueError, TypeError, IndexError):
            return dict(DEFERRED_VERDICT)

    safe._harness_safe = True
    V.validate = safe


# --------------------------------------------------------------------------- #
# journal / manifest
# --------------------------------------------------------------------------- #
COUNTS = {"attempted": 0, "answered": 0, "error": 0, "skipped": 0,
          "graded": 0, "deferred": 0}
MODEL_SLUG = "unknown"
MODEL_OK = False
ERROR = None
ROWS_SOURCE = ""
ROWS_DETAIL = ""
ROWS_TOTAL = 0
MEMORY_NOTES_LOADED = 0  # set from _fresh_notes so a silent zero-note loss is recorded
REASONING_LEVEL = None  # "none" when the model rejects tools + reasoning_effort
ROW_CELL: dict[str, str] = {}  # id -> task/tier, journal bookkeeping only


def _manifest(status: str, stage: str, **kw):
    MANIFEST.write_text(json.dumps({
        "status": status, "stage": stage, "error": ERROR,
        "task": SPEC["task_name"], "harness": "tools",
        "model": MODEL_SLUG, "model_recognized": MODEL_OK,
        "shard": SHARD, "rows_source": ROWS_SOURCE,
        "rows_detail": ROWS_DETAIL, "rows_total": ROWS_TOTAL,
        "memory_notes_loaded": MEMORY_NOTES_LOADED,
        "reasoning": REASONING_LEVEL or "backend-default",
        "limits": {"limit": LIMIT, "rounds": ROUNDS, "k_memory": K_MEMORY,
                   "deadline_seconds": DEADLINE},
        "counts": COUNTS, "tools": tool_names(), "spec_sha": SPEC_SHA, **kw,
    }, indent=1, ensure_ascii=False), encoding="utf-8")


@kbench.task(name=SPEC["task_name"],
             description="Thermo teacher: LangGraph + real physics tools over a prompt shard")
def thermo_teacher_cohort_bulk_s00(llm):
    global MODEL_SLUG, MODEL_OK, ERROR, ROWS_SOURCE, ROWS_DETAIL, ROWS_TOTAL
    global MEMORY_NOTES_LOADED, REASONING_LEVEL, ROW_CELL

    llm = llm or kbench.llm
    MODEL_SLUG = getattr(llm, "name", None) or getattr(llm, "model", None) or "unknown"
    MODEL_OK = any(s in str(MODEL_SLUG).lower() for s in EXPECTED_MODELS)
    if not MODEL_OK:
        print("[warn] model %r not in the expected hosted list; recording its real name"
              % MODEL_SLUG, flush=True)

    COUNTS.update(attempted=0, answered=0, error=0, skipped=0, graded=0, deferred=0)
    ERROR = None
    OUT.mkdir(parents=True, exist_ok=True)
    _manifest("running", "start")

    rows = _load_rows(BUNDLE_ROOT, MODEL_SLUG)
    ROWS_SOURCE, ROWS_DETAIL = "cohort-ledger", SPEC["task_name"]
    if LIMIT > 0:
        rows = rows[:LIMIT]
    ROWS_TOTAL = len(rows)
    _install_safe_validator()
    notes = _fresh_notes(BUNDLE_ROOT)
    MEMORY_NOTES_LOADED = len(notes)
    REASONING = reasoning_for_model(MODEL_SLUG)
    REASONING_LEVEL = REASONING
    print("[harness] rows=%d source=%s model=%s tools=%s rounds=%d notes=%d reasoning=%s"
          % (ROWS_TOTAL, ROWS_SOURCE, MODEL_SLUG, ",".join(tool_names()), ROUNDS,
             len(notes), REASONING or "backend-default"), flush=True)

    done = set()
    if JOURNAL.exists():
        for line in JOURNAL.open(encoding="utf-8"):
            try:
                done.add(str(json.loads(line)["id"]))
            except Exception:
                pass

    started, written, spent_nanodollars = time.time(), 0, 0
    with JOURNAL.open("a", encoding="utf-8") as fh:
        for index, row in enumerate(rows):
            rid = str(row.get("id", ""))
            if rid in done:
                COUNTS["skipped"] += 1
                continue
            if DEADLINE > 0 and time.time() - started > DEADLINE:
                ERROR = "deadline"
                break
            task = row.get("task") or "repair"
            gradable = _row_is_gradable(task, row)
            t0 = time.time()
            rec = {
                "id": rid, "task": task, "model": MODEL_SLUG, "harness": "tools",
                "tools": tool_names(), "completion": "", "sft_completion": "",
                "verdict": dict(DEFERRED_VERDICT) if not gradable else dict(NOT_RUN_VERDICT),
                "graded": False,
                "trajectory": {"turns": [], "notes_used": [],
                               "tools_available": tool_names(), "final": ""},
                "usage": {}, "error": None,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "shard": SHARD, "row_source": ROWS_SOURCE, "spec_sha": SPEC_SHA,
                "cell": ROW_CELL.get(rid, ""),
            }
            try:
                ctx = build_context({"row": row, "task": task, "notes": notes,
                                     "k_memory": K_MEMORY, "mode": "agents+skills"})
                # gpt-6 chat-completions rejects tools + reasoning_effort. We know the slug
                # up front, but keep a one-shot retry so a new model slug cannot kill a run.
                for attempt in (0, 1):
                    usage_holder = {}
                    respond = make_respond(
                        llm, kbench, actors,
                        system_prompt=ctx["system_prompt"],
                        max_tool_rounds=ROUNDS, usage_holder=usage_holder,
                        reasoning=REASONING,
                    )
                    try:
                        state = run_direct(row, task, respond, notes=notes, max_rounds=ROUNDS,
                                           k_memory=K_MEMORY, mode="agents+skills")
                        break
                    except Exception as exc:
                        retryable = ("reasoning_effort" in str(exc) and REASONING is None
                                     and attempt == 0)
                        if not retryable:
                            raise
                        REASONING = "none"
                        REASONING_LEVEL = "none"
                        print("[harness] tools+reasoning rejected by the API; retrying %s "
                              "with reasoning=none" % rid, flush=True)
                final = state.get("final") or ""
                rec["reasoning"] = REASONING or "backend-default"
                rec["completion"] = final
                rec["sft_completion"] = state.get("sft_completion") or ""
                verdict = {k: v for k, v in (state.get("verdict") or {}).items()
                           if k != "failed"}
                if not gradable:
                    # Compact shard: transfer.ingest is the ONLY grading authority.
                    verdict = dict(DEFERRED_VERDICT)
                elif not verdict:
                    verdict = dict(NOT_RUN_VERDICT)
                rec["verdict"] = verdict
                rec["graded"] = bool(
                    gradable and verdict.get("accepted") is not None
                    and verdict.get("reason") != "deferred_to_ingest"
                )
                rec["trajectory"] = {
                    "turns": state.get("turns") or [],
                    "notes_used": state.get("notes_used") or [],
                    "tools_available": tool_names(),
                    "final": final,
                }
                rec["usage"] = usage_of(usage_holder["chat"]) if "chat" in usage_holder else {}
                rec["usage"]["seconds"] = round(time.time() - t0, 2)
                COUNTS["answered"] += 1
            except Exception as exc:
                message = "%s: %s" % (type(exc).__name__, exc)
                rec["error"] = _redact(message)[:500]
                COUNTS["error"] += 1
                ERROR = rec["error"]
            COUNTS["graded" if rec["graded"] else "deferred"] += 1
            COUNTS["attempted"] += 1
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
            fh.flush()
            done.add(rid)
            written += 1
            spent_nanodollars += int((rec.get("usage") or {}).get("total_cost_nanodollars") or 0)
            if written % 10 == 0:
                _manifest("running", "rows", next_index=index + 1,
                          spent_usd=round(spent_nanodollars / 1e9, 6))
            if MAX_COST_USD > 0 and spent_nanodollars / 1e9 >= MAX_COST_USD:
                ERROR = "cost_cap"
                _manifest("paused", "cost_cap", spent_usd=round(spent_nanodollars / 1e9, 6))
                break
            if COUNTS["error"] >= 5 and COUNTS["answered"] == 0:
                _manifest("paused", "too_many_errors")
                break

    _manifest("completed" if ERROR is None else "partial", "done")


thermo_teacher_cohort_bulk_s00.run(kbench.llm)
