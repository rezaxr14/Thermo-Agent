# User prompts — verbatim

Exact text of Reza's prompts in this session, in order. Kept for handing to another model for direction. Do not edit wording.

---

## 1

```
fuking fix this shity project before spending ai shit and runing my quota, everythingle time you are finding a new shitty bug or non functional shit idiot
```

## 2

```
the fuking per turn text that we get from these ai frontier kaggle runs are our resoning dont fuking tell me again that our resoning is fuking "" non exisiting you retard idiot moron trash, dont fuking send the fuking resoning true to the fuking models that you already know dont fuking accept them you shit head idiot
```

## 3

```
fuck head we need to send the resoning back to the fuking model so we can get th e next turns tools and skills and resoning text moton , this shit is multi turn agentics fuking themo model
```

## 4

```
shit head
```

## 5

```
we alredy have multi turn shit head idiot, you are just duplicating work mron retrad
```

## 6

```
fix everything before tuching any ai shit
```

## 7

```
garbage trash,
```

## 8

```
commit everything and my prompts verbatuim in a file i will ask cladue for a direction
```

## 9

```
and proccess the kaggle runs and also inculde new subagent works in the commit and push i will get feedback and after that we are continue to work , include this prompt in the verbaitm as well, we will get a detailed instructions and fix our project  and the harness and how we do stuff here. and we will tidy up the docs so we dont have stale of docs filling my docs file we will combine the stuff to eachitehr , i dont want to have 10 user directions and decitions and shit like that, you are incredible irrisponsable and litter all around, and we need to decide how to treat already captured Cot's my inturations is too keep them and add these new stuff on top of them and make our tools backward compatible, or make cople of subagents to go and check and modyfy them one by one proeprly sub agents are free so we dont loose anything but then need to do it reraly one by one no bulshit grep and replace nonsense same for the detreministic traning data we have (multi turn and other wise) and the ground trughts dataset, realy anything inculde this in the commit nad push so i can get a feed back and move a head , commit some samples from everywhre so claude can know what in the hell is going on here, and we need a anti litter rule so we dont just make a documnet if exist a elready made document which we can reuse (add,delete,modify,add new date), we need to load the skill on demand, we kinda need to show a disctiption and load the full skill when the models reach for them like a proper harness, do this and commit and push so i can get a feed back so ew can continue with the project
```

## 10

```
fuck off and commit and push i did not told you to dod these i told you to put this shit in a file and pus hso cladue can figure your stupid ass out stop you stupid agents
```

## 11

```
do do fuking this and put this in a file and push idiot
```

---

## Earlier standing instructions (from the session checkpoint preamble, paraphrased by the harness — not verbatim)

- Generate thousands of equal-third, 4-task CoT traces from budget frontier models (`gpt-6-luna`, `claude-haiku-5-5`) with tools + memory + skills; ingest every batch into memory and the training mix; then SFT → GRPO k4/8/16 → full arm evaluation.
- "make the project to have proper skills for everything and make our tools dynamic so we don't return hard constants for a given task… make actual continuous varied input tools and make quite a lot of skills… thermodynamic components… multiple variations of the same component so our tools and skill need to handle them."
- Only budget models; never gemini 3.8/3.5; premium only with explicit permission. Equal thirds easy/medium/hard per task, no cap. Every batch: gate at 8,192 tokens → ingest with provenance/model/run id → memory recap → refresh mixes → HF sync → progress report. Provenance must say deterministic vs AI-generated plus exact model id. Memory/RAG records successes and failures. Harness prompts carry only task/artefact/skills/tools/short memory — no bookkeeping ids/manifests.
- Credentials only from repo-root `.env`; never print tokens. Kaggle traffic only via `socks5h://127.0.0.1:10808`; no proxy for other traffic. Data flow: local → private HF → Kaggle kernel → ONE Kaggle dataset; no local data upload to Kaggle. One reusable hosted task `thermo-teacher-cotproof-s00`.
- Use free Hy4 flash subagents aggressively; never DeepSeek subagents.
- **Correction from prompt 2/3:** the per-turn assistant text from frontier runs IS the reasoning. It must be captured and replayed to the model on later turns (multi-turn agentic loop), and `reasoning_effort` is NOT to be sent to models that reject it. Older trace-audit language claiming "reasoning missing/empty" is wrong and was corrected.
- **Correction from prompt 5:** multi-turn was already implemented; do not duplicate it.
- **Correction from prompt 6:** fix everything locally before any further AI/Kaggle quota spend.
