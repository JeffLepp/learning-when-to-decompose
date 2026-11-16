# Design v2: What can small local models do with decomposition, and can we learn when to use it?

Status: **design, authored by the orchestrator (Claude) on 2026-11-15**. Nothing here has been run unless a results file says so. The v1 pilot (`results/*-v1`, `CHECKPOINT_3.md`) is historical evidence and stays unchanged.

## 1. Motivation from current literature

- **Routers often fail to beat the best single model.** LLMRouterBench (arXiv 2601.07206, Jan 2026) and "The Routing Plateau" (arXiv 2606.07587, 2026) report that many routers do not beat the best single model, and that oracle gaps of 10–30 pp remain. *Consequence:* every routing claim here is made against the **best single arm selected on validation** and against the oracle, never just against direct-only.
- **Models' pre-generation activations predict their own success.** "LLMs Encode Their Failures" (Lugoloobi et al., ICLR 2026, arXiv 2602.09924) shows linear probes on these activations beat surface and TF-IDF features at predicting success, and enable cost-saving routing on MATH and code. *Our test:* does this hold for **small (0.6B–8B) models on compositional tasks**, and for predicting which **strategy** succeeds, not only which model?
- **Escalate only when the cheap answer looks unreliable.** Cascades with consistency-based escalation (Yue et al., ICLR 2024, arXiv 2310.03094) and as-needed decomposition (ADaPT, Prasad et al., NAACL Findings 2024, arXiv 2311.05772) both do this. *Our test:* a post-hoc cascade compared with pre-generation routing.
- **Prior decomposition work on SCAN.** Least-to-most prompting (Zhou et al., 2022) already decomposes SCAN with large models. Our contribution is not "decomposition works on SCAN". It is (a) how its benefit scales across a **small-model ladder**, (b) a **planning-vs-execution** factorization of failures, and (c) **learned, cost-sensitive selection** among model × strategy arms.

## 2. Research questions and pre-registered hypotheses

- **RQ1 (capability).** How does exact-match accuracy change with model size (Qwen3 0.6B / 1.7B / 4B / 8B) and decomposition granularity?
  - H1: the benefit of decomposition is non-monotonic in size. The tiniest models fail at planning, the mid-size models gain most, and the largest gain less on short inputs but more on long ones.
- **RQ2 (diagnosis).** Which component fails: planning or leaf execution?
  - H2: planning errors dominate for the smallest models; leaf-execution errors dominate for the larger ones.
- **RQ3 (ML: predictability).** How well can each arm's success be predicted *before generation*?
  - H3: AUROC ordering is activation probe > TF-IDF > structural features. The gap shrinks under distribution shift.
- **RQ4 (ML: routing).** Does learned routing reach cost–accuracy points beyond the fixed-arm Pareto frontier?
  - H4: yes in-distribution, with the gain shrinking under length/depth shift.
  - H5: a post-hoc confidence cascade dominates pre-generation routing at matched cost.

Each hypothesis may be refuted. A refutation is a reportable result and is not to be "fixed" by tuning on holdout data.

## 3. Domains

### D1: SCAN (existing, pinned)
- **Items:** the `data/*.json` partitions. Their hashes are frozen in `data/manifest.json` and must not change.
- **v2 splits** (prefixes of the existing shuffled partitions, so they are disjoint by construction):
  - `dev`: `development` (128). Prompt development and few-shot exemplars come from here only.
  - `router_train`: `controller_train[:N_train]`.
  - `router_val`: `validation[:N_val]`.
  - `holdout`: `audit[32:32+N_hold]`. Fresh: v1 inspected only `audit[:32]`.
  - `ood_length`: a seeded random sample of the official length test. It is **locked until the design freeze** (§8).
- **Gold structure:** a deterministic SCAN interpreter/parser in code. It must reproduce every official target exactly; this is tested on all training and test lines. It provides gold plans for the diagnostics and structure metadata (depth, clause count, output length).

### D2: ChainOps (new, synthetic, contamination-free)
- **Task:** natural-language chains of list operations applied to a short integer list. Example:
  > Start with [3, 8, 1, 5]. Reverse the list. Then remove every odd number. Then add 2 to each number.

  The answer is the final list, checked by exact match.
- **Operation inventory:** about 12 operations, each a pure Python function with unambiguous wording. Candidates: reverse, sort ascending/descending, remove first k, keep first k, remove all x, add k to each, multiply each by k, keep only even/odd, rotate left by k, swap first and last, append x, remove duplicates. Values stay small (0–20 input, bounded growth) so that arithmetic is not the bottleneck.
- **Generator:** seeded, with ≥2 surface templates per operation. Disjoint item IDs across splits; no item text appears in two splits.
- **In-distribution:** chain depth 1–4, list length 3–6.
- **`ood_depth`:** depth 6–8 and list length 6–8. Locked until freeze, like `ood_length`.
- **Splits:** `dev`, `router_train`, `router_val`, `holdout`, `ood_depth`, with sizes matching D1 where possible.
- **Gold plan:** the operation list. A code executor computes the target.

## 4. Arms: model × strategy

**Models:** Ollama `qwen3:0.6b`, `qwen3:1.7b`, `qwen3:4b`, `qwen3:8b` (Q4_K_M). Record digests.

**Decoding:** temperature 0, seed 437, JSON-schema output, `think:false` unless the strategy says otherwise, `num_ctx` 4096. `num_predict` is 512 for non-thinking calls and 2048 for thinking calls, with truncation recorded. Request `logprobs` on every call and save them.

| Code | Strategy | Who does what | Deployable? |
| --- | --- | --- | --- |
| S0 | direct zero-shot | model does everything in one call | yes |
| S1 | direct few-shot (k fixed dev exemplars, same for all models) | model does everything | yes |
| S2 | direct + thinking (`think:true`) | model, extra test-time compute; the compute-matched control | yes; run on a subset (§7) |
| S3 | coarse decomposition. SCAN: clause-level plan (and/after), code repeats twice/thrice and concatenates (v1-style, but with few-shot plan exemplars). ChainOps: plan into chunks of ≤2 operations, model executes chunk by chunk, code passes state | model plans and executes chunks; code does repetition, concatenation, state passing | yes |
| S4 | fine decomposition. SCAN: model emits a full structured parse (actions, direction, opposite/around, repeat); code expands the structure, and the model solves only atomic leaves. ChainOps: one operation per step | model parses and does atomic leaves; code does all structure | yes |
| G1 | gold plan + model leaves (S4 structure) | isolates leaf-execution error | **diagnostic only** |
| G2 | model plan (S4) + code leaves | isolates planning error | **diagnostic only** |

Gold plan + code leaves must score 100%; this is a unit-test sanity check, not an arm. Every arm logs a `code_responsibility` description string. Format and validation failures count as incorrect. Transport/server errors halt the run; they are never recorded as labels. No retries, repair, or reference access during execution.

## 5. Data contracts (shared by all workstreams)

- **Items:** `data/v2/<domain>/<split>.jsonl`. One JSON object per line:
  ```
  {id, domain, split, input, target, gold_plan, meta:{depth, n_clauses/ops, target_len, ...}}
  ```
  Hashes go in `data/v2/manifest.json`.
- **Raw runs:** `results/v2/runs/<run_name>/`, holding `config.json` (model, digest, strategy, prompts, schemas, code sha256, items sha256), `environment.json`, `records.jsonl`, and `summary.json`. Each record contains:
  ```
  {id, domain, split, model, strategy, prediction, correct, error,
   prompt_tokens, completion_tokens, thinking_tokens, total_tokens,
   wall_seconds, n_calls, plan,
   logprob_stats:{mean, min, sum, n},
   final_call_logprob_stats, calls:[...full Ollama responses incl. logprobs...]}
  ```
- **Label table:** `results/v2/labels/<domain>.csv`, built by a script from the raw runs and never edited by hand. Columns:
  ```
  id, split, model, strategy, correct, total_tokens, prompt_tokens, completion_tokens,
  wall_seconds, n_calls, error, mean_logprob, min_logprob, answer_sha1
  ```
- **Activation features:** `results/v2/features/<domain>/<model>/<layer>.npy` plus `index.json` (item-ID order, prompt template, HF model ID, revision, dtype, pooling).
- **Router outputs:** `results/v2/router/<experiment_name>/` with `report.json`, per-item decisions, and figures.

## 6. ML component: the course's core learning problem

**Targets:** for each item *i* and deployable arm *a* (model × S0–S4), the success `y[i,a]` (binary) and the cost `c[i,a]` (total tokens; seconds as a secondary measure).

**Feature families**, compared under identical splits and learners:
- **F1 structural:** v1 features plus domain analogues (operation counts, list length, depth).
- **F2 lexical:** TF-IDF word 1–3-grams.
- **F3 activation probe:** hidden states of the HF Qwen3 model *of the arm's own size*, run on the S0 prompt before any generation. Last-token and mean-pooled states at several layers. Choose the layer on `router_val` only. Use the HF base weights matching the Ollama model family (bf16 for ≤4B, 4-bit for 8B). Report the quantization mismatch with Ollama as a caveat.
- **F4 post-hoc signals (cascade only):** from cheap arms already executed, the token logprob statistics, answer agreement between arms (Yue et al.), validation/format failures, and output-length signals. **No domain verifier that recomputes the gold answer.**

**Learners:**
- L2 logistic regression (probe-style), HistGradientBoosting, and a small MLP.
- One model per arm, plus a shared model with arm one-hot features. The shared model pools data across arms.
- Calibrate on `router_val` (Platt or isotonic) and report Brier, ECE, and AUROC per arm.

**Policies**, each tuned on `router_val` only:
- each fixed arm; best single arm by validation; cheapest arm meeting the target;
- a hand-written heuristic (decompose if structural complexity > k; pick model size by input length);
- predict-then-route: `argmax p̂ − λ·cost`, sweeping λ to trace a curve, plus the v1-style threshold rule;
- cascade: an ordered ladder of arms, escalating when post-hoc confidence < τ; the cost includes every executed arm;
- oracle: the cheapest correct arm (upper bound only).

**Metrics:**
- accuracy–cost curves and their normalized area;
- accuracy at the cost of the best single arm, and cost at its accuracy;
- the fraction of the best-single-to-oracle gap that each policy closes;
- learning curves over the number of training labels (e.g. 50, 100, 200, 400, all);
- paired bootstrap 95% CIs over items for every comparison, and McNemar tests at chosen operating points;
- routing overhead counted explicitly. A probe's forward pass is charged as prompt tokens on that model.

## 7. Budget and ordering (GPU is one RTX 5060 8 GB; everything is serialized through `v2/gpulock.py`)

**Planned sizes.** These are adjusted after the timing pilot; the change is recorded in this file with the measured per-arm seconds.
- Per domain: `router_train` 600, `router_val` 200, `holdout` 400, OOD 300.
- S2 (thinking) runs on a stratified subset: 150 train, 100 val, 150 holdout, 100 OOD.
- G1/G2 diagnostics run on 200 `router_train` items plus all holdout items.

**Collection order:** model-major (all items for one model, then the next) to avoid weight swapping. Within each item, the order of strategies alternates. Collection is resumable and idempotent per `(id, model, strategy)`.

**Phases:**
- **P0 Foundations:** harness, interpreters, generators, tests. No inference except smoke tests.
- **P1 Prompt development:** on `dev` only, at most 5 logged iterations per strategy, with the same prompts for every model size. Freeze prompts by tagging them.
- **P2 Timing pilot:** 20 dev items × all arms, to fix the budget.
- **P3 Collection:** `router_train`, `router_val`, `holdout` for both domains.
- **P4 Router development:** on train and val only. This runs in parallel with the end of P3.
- **P5 Freeze:** git tag `freeze-v2`, then evaluate once on `holdout` and the OOD splits.
- **P6 Analysis and report.**

## 8. Integrity rules (non-negotiable)

1. The official SCAN length test and ChainOps `ood_depth` receive **no inference** until the `freeze-v2` tag exists. The harness refuses those splits unless `--final-eval` is passed and the tag is present.
2. `holdout` is used once, after freeze, for the headline in-distribution claims.
3. Do not modify v1 files, `data/*.json`, `data/manifest.json`, or `results/*-v1`.
4. New run names for new configurations. A changed config with a reused name is an error.
5. Every model call is recorded with its tokens, time, and failures. Diagnostics (G1/G2, oracle) are never presented as deployable.
6. Only the orchestrator commits to git. Agents do not commit, push, or add attribution.
7. Every numeric claim in the report traces to a file under `results/v2/`.

## 9. Workstreams

- **WS1 Harness & collection** (Claude2 manager; Codex worker): §3–5 and §7 P0–P3.
- **WS2 Probe features** (Claude2 manager; Codex2 worker): `.venv-probe`, HF weights, the F3 extraction pipeline, and a first sanity probe on v1 pilot labels.
- **WS3 Router & evaluation** (Claude2 manager; Codex worker): the §6 library, statistics, and figures. Developed first against v1 labels and simulated labels, then against the real v2 labels.
- **Orchestrator (Claude):** design, reviews, decisions recorded in this file, git commits, the freeze, the final analysis, and the report.

## 10. Decision log (orchestrator; binding unless superseded here)

**D1 (2026-11-15). Response to WS3 phase A.**

1. **matplotlib** is installed in `.venv` and recorded in the lock file.
2. **Probe overhead.** Every model whose activations a router reads is charged the item's S0 prompt-token count on that model. Two F3 routing variants are pre-registered and both reported:
   - **F3-own**: each arm's success is predicted from its own model's activations. Overhead is the sum over the probed models. This variant is the RQ3 headline.
   - **F3-small**: every arm is predicted from `qwen3:0.6b` activations. Overhead is one 0.6B prompt.
   - The choice between them is never made on holdout.
3. **F3 index key** is `ids` (WS2 must comply).
4. **`total_tokens`** = prompt + completion, and completion *includes* thinking tokens. `thinking_tokens` is a separate informational column.
5. **Coverage.**
   - The main routing analysis uses arms present for every item (S0, S1, S3, S4 × 4 models = 16 arms).
   - S2 (thinking) is analysed separately on its subset, both as an arm and as a routing extension.
   - G1/G2 are never routing arms.
6. **Normalized AUC** is computed over [cheapest fixed-arm mean cost, most expensive fixed-arm mean cost] on the evaluated split, with the cheapest fixed arm as every curve's left anchor. **Confirmed.**
7. **Pre-registered operating point** for single-point comparisons (McNemar, gap closed, headline table):
   - Take the cheapest validation point whose validation accuracy ≥ the best single arm's validation accuracy.
   - If no such point exists, take the validation point with maximum accuracy, breaking ties by cost.
   - The headline bootstrap uses 10,000 replicates.
