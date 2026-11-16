# WS1: Harness & collection (manager brief, phase A)

**Worker:** `codex` (primary).

**Goal of phase A:** DESIGN_V2 phases P0, P1 and P2. Then **STOP and report**. Do not start P3 collection; the orchestrator must approve the prompts and the budget first.

## P0: Foundations (no inference except smoke tests)
1. **`v2/domains/scan.py`**
   - A deterministic SCAN interpreter (command → actions) and a structured parser producing the gold plan used by S3 (clauses + repeat) and S4 (full structure).
   - Test it: the interpreter must reproduce **every** line of both `tasks_train_length.txt` and `tasks_test_length.txt` exactly. Reading test *targets* for this code check is allowed; sending them to a model is not.
2. **`v2/domains/chainops.py`**
   - A seeded generator per DESIGN_V2 §3 D2.
   - A Python executor and NL rendering with ≥2 templates per operation.
   - Unambiguous wording. Write tests that every template round-trips through gold-plan execution.
   - Keep values bounded.
3. **`v2/build_data.py`**
   - Writes `data/v2/<domain>/<split>.jsonl` for: `dev`, `router_train` (600), `router_val` (200), `holdout` (400), and OOD (300).
   - SCAN splits are prefixes of the existing partitions. `holdout = audit[32:432]`. `ood_length` is a seeded sample of the official test, written to disk but locked.
   - Writes `data/v2/manifest.json` with hashes.
   - Verifies disjointness, including text-level disjointness for ChainOps.
4. **`v2/harness.py`**: the arms S0–S4 and G1/G2 for both domains, per §4.
   - JSON-schema outputs.
   - Logprobs captured on every call (Ollama `logprobs: true`; verified working on this server).
   - Thinking tokens recorded separately for S2. `eval_count` includes thinking; find a defensible way to separate them and document it.
   - Records follow the §5 contract.
   - Resumable per `(id, model, strategy)`. A config change requires a new run name.
   - Hold `gpu_lock` for the whole run.
   - Refuses OOD splits without `--final-eval` and the `freeze-v2` tag; refuses `holdout` without `--collect-holdout`.
   - Transport errors halt; format errors count as incorrect.
   - Model-major ordering.
5. **`v2/build_labels.py`**: raw runs → `results/v2/labels/<domain>.csv`.
6. **Tests:** `v2/tests/`, runnable with `.venv/Scripts/python.exe -m unittest discover -s v2/tests -t .`, including:
   - a gold plan + code leaves = 100% check on both domains;
   - schema validation;
   - lock refusal for OOD;
   - resume idempotency, using a mocked Ollama.

## P1: Prompt development (`dev` split only)
- Write prompts for S0–S4 for both domains.
- The few-shot exemplars come from `dev` and are excluded from dev scoring. The exemplars are the same for every model size.
- Iterate **at most 5 logged iterations per strategy per domain**, scored on the remaining dev items with `qwen3:1.7b` and `qwen3:8b`.
- Fairness rule: put comparable effort into the direct prompts (S0/S1) and the decomposition prompts.
- Log each iteration's prompt hash and dev score in `results/v2/prompt_dev/log.jsonl`.
- Choose the final prompts by the dev mean across those two models. Do not tune per model.

## P2: Timing pilot
- Run 20 dev items × all 4 models × all arms (S2 included).
- Report per-arm mean/median seconds and tokens, the truncation rate (especially S2 thinking at a 2048 cap), and the format-failure rate.
- From these, compute the GPU-hours the §7 plan needs for both domains, and propose sizes if it exceeds about 20 GPU-hours total.

## Stop and report
Write `managers/WS1_STATUS.md` per COMMON.md. It must include:
- the dev accuracy table (model × arm × domain);
- the timing table;
- the final prompts' locations and hashes;
- anything in the spec that proved ambiguous or infeasible.
