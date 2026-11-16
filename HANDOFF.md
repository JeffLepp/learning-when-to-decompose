# Fresh session handoff

## v2 expansion status (current; updated 2026-11-15)

- The user asked the orchestrator (Claude) to design and drive a larger study. The binding spec is `DESIGN_V2.md`, which covers the RQs, hypotheses, arms, the ML router, integrity rules, and phases.
- The user approved:
  - a multi-agent structure: Claude2 managers directing Codex/Codex2 workers;
  - model and PyTorch downloads;
  - full-auto agents confined to `FinalProject/`;
  - local-only git, with commits made only by the orchestrator;
  - including the second domain, ChainOps, now.
- Done:
  - Backup zip at `../FinalProject-backup-20261115-162041.zip`.
  - Git baseline commit `741c1c3`.
  - `qwen3:0.6b` and `qwen3:4b` pulled.
  - GPU lock `v2/gpulock.py` tested.
- **Phase A launched at 16:24.** WS1 (harness + prompt dev + timing pilot), WS2 (probe environment + feature extraction + v1 feasibility probe), and WS3 (router library). Session outputs are in `workspace_logs/WS*-session-A.json`; reports go to `managers/WS*_STATUS.md`.
- Next obligation: review each status report and its diff, commit accepted work, approve the P3 collection budget, then continue the phases in DESIGN_V2 §7.
- Caveats:
  - The Claude2 sign-in expires around Nov 19–22.
  - Ollama embeddings are disabled on this server; activations come from HF models instead.

## Start here

This is Jefferson Kline's **CS 437 machine-learning final project**, titled **Learning When to Decompose Complex Tasks for Local Language Models**. The user requested a durable handoff so a new session can form its own plan without losing the project's goals, evidence, or decisions.

Canonical working directory: `C:/Users/jeffe/Desktop/fall26/437/FinalProject`.

Read `AGENTS.md`, `CHECKPOINT_3.md`, `RESULTS.md`, and `PROTOCOL.md`, then inspect relevant code. This file is the current state; measured run files remain the numerical authority. Historical runs were collected November 15, 2026. No experimental model collection is currently pending.

## User intent and confirmed decisions

- The user is interested in what smaller local models can accomplish when tasks are decomposed, compared with receiving the full task directly. They also want a rigorous project that supports a strong course grade.
- Keep a supervised ML component: learn a controller predicting success/cost and choosing direct or decomposed execution. Do not turn the project into only a prompting demo.
- **Confirmed ordering: prepare this checkpoint from verified results first, then expand experiments.**
- The user said not to worry about the deadline and specified no format or length constraint. Do not ask again for those details or the rubric already supplied.
- The checkpoint requires (1) a working baseline/initial analysis, (2) a quantitative result or meaningful visualization, (3) what was run and observed, (4) important issues, and (5) next steps. Exact wording is in `references/checkpoint3-user-requirements.txt`.
- The latest request authorized compiling the existing work and preserving context. The checkpoint text is prepared in `CHECKPOINT_3.md`; it has not been submitted or published. There is no request for email, LMS submission, Git commit, or upload.
- Next-session planning should be autonomous within this scope. Proposed expansions below have not been implemented, and model/dataset/budget choices remain to be designed. Ask only if a consequential decision cannot reasonably be resolved from this context.

## Location and document history

- The user originally pasted Milestone 2's full proposal. Its source text is preserved in `references/milestone2-user-provided.txt`; no original final-project PDF was found.
- The user later pointed to `E:/RebootStuff/fall26/415/MILESTONE2`. Inspection confirmed its Word report is the **separate CS 415 YouTube/Neo4j project with Justus Jones**, not this project. Do not import its database milestones, team membership, grading requirements, or instructions into CS 437.
- A neighboring `E:/RebootStuff/fall26/437` copy contained studio coursework when inspected; the active new project remains at the C: path above. No move or synchronization was requested.
- Before this work the C: course directory contained Studio 1–4 and was not a Git repository. Existing coursework was not changed. New root `../AGENTS.md` directs future project sessions here.

## Completed implementation

`experiment.py` downloads the official SCAN length split at a pinned commit, hashes files, partitions data with seed 437, runs both strategies, and logs JSONL raw responses plus timing, token counts, exact-match correctness, failures, and model settings.

- Full official training-pool allocation: development 128; controller training 10,000; validation 3,000; audit 3,862. These are pool sizes, not counts actually evaluated.
- Actual evaluated samples: development 24 commands with both 1.7B and 8B; controller training 96, validation 32, audit 32 with 8B. There are 416 strategy records across the five execution runs, corresponding to 184 distinct commands. Each development command was evaluated with both models.
- Official length test: 3,920 commands. Downloaded and used for integrity/disjointness checks; **no model inference on it**.
- SCAN source commit: `c4b756cbc010d75c912f16c42c8f15dc6b7e6c8f`; manifest includes SHA-256 hashes.
- Decoding: temperature 0, seed 437, thinking false, context 4096, maximum 512 generated tokens per call, JSON-schema output. One excluded warmup per run. Strategy order alternates by command.
- Direct returns actions in one call. Decomposed makes a one/two-clause plan, solves each clause using the same model, and uses code for repetition/concatenation. The model retains responsibility for plan order, turn direction, and around/opposite semantics. No semantic repair, retries, or reference answers in execution.
- Format/truncation failures count as incorrect. Transport failures stop collection rather than silently becoming wrong-answer labels.
- `controller.py` fits logistic regression and random forests to success labels, and a random forest to total token cost. Features are lengths and command-word counts; embeddings are not implemented.
- Routing chooses the cheapest predicted strategy passing a probability threshold; if neither qualifies, choose higher predicted success. Threshold/family are selected using validation only. Target is 80% empirical validation accuracy; if infeasible, maximize validation accuracy then minimize token cost. Grid: 0, .25, .5, .6, .7, .8, .9, .95.
- `results/controller-v1/controller.joblib` saves the selected fitted classifier pair, vectorizer, cost model, and threshold. Both classifier families were fitted/evaluated, but only the selected fitted family is persisted as a model artifact; candidate metrics for both are saved.

## Verified findings

| 8B audit strategy | Correct / 32 | Accuracy | Mean total tokens | Mean recorded/replayed seconds |
| --- | ---: | ---: | ---: | ---: |
| Direct | 3 | 9.375% | 257.3125 | 1.0859091875 |
| Decomposed | 16 | 50% | 671.40625 | 0.8760366125 |
| Selected controller | 15 | 46.875% | 625.09375 | 0.8371605375 |

Selection: logistic regression, threshold .5. Validation: 10/32 correct, 629.9375 mean tokens. **The 80% target was not met.** Controller audit used direct on four commands and decomposed on 28. Against always decomposing, switches gained one correct answer and lost two: 6.8978% fewer tokens, one fewer correct answer overall. No superiority or competitive/optimal claim is justified.

Audit oracle ceiling from choosing whichever recorded answer is correct: 18/32 (56.25%). This is an outcome-aware diagnostic bound, not an implemented route.

Development: 1.7B direct 1/24, decomposed 0/24 (17 invalid plans); 8B direct 1/24, decomposed 13/24 (one invalid plan). Training 8B: direct 8/96, decomposed 40/96, either correct 42/96. Full uncertainty, example errors, and timing caveats are in `RESULTS.md`.

Verified previously and again for handoff: six tests; source/partition hashes and disjointness; raw-record scoring and token totals; archived executor hashes; saved-model reload reproducing all 32 audit choices. `results/verification.json` records file-readback results. These checks verify implementation/artifacts; they do not establish generalized model improvement.

## Limitations to carry forward

- Weak direct prompting can exaggerate apparent benefit. Improve it fairly using development examples before claiming a capability gain.
- A well-formed plan can have wrong order; a correct plan can have incorrect subtask outputs. Supplied correct plans would help distinguish these failure sources, but that condition has not been built.
- The small sample and imbalanced success labels constrain learning/calibration; one-answer differences are not strong evidence.
- Token totals include input/output and all calls, but are not FLOPs or energy. Controller runtime is replay, excludes full routing overhead, and is not live deployment latency. Prompt caching/order and long generations affect timing; direct median was faster even though its mean was slower.
- Original 1.7B pilot used `localhost` with roughly 2 seconds connection overhead per request; later 8B runs use `127.0.0.1`. Never claim the raw timing difference is model acceleration.
- The audit has now been inspected and used in planning. Keep its original results as historical pilot evidence; use a fresh holdout for redesign claims. Do not tune on the official test.
- SCAN length testing is not a substitute for a separate composition-generalization split. Public-pretraining exposure is unknown. SCAN does not establish general agent capabilities.
- Prior least-to-most work already studies decomposition on SCAN. Intended contribution: learned cost-sensitive local-model routing, plus a planning-versus-execution diagnosis. References are in `PROTOCOL.md`.

## Proposed expansion, not yet run

The user responded positively to discussing expansion, then explicitly prioritized the current checkpoint. The previous recommendation was:

1. Strengthen direct and decomposition prompts on development data only; include fair few-shot examples and a comparable-compute control (and consider a thinking-mode ablation).
2. Compare direct solving, self-generated decomposition, and correct supplied plans on the existing 1.7B and 8B models. Treat supplied plans as diagnostic assistance, never evidence of autonomous planning or a deployable baseline. Record how much work code performs.
3. Keep learned strategy selection as the ML core. Add a simple routing heuristic, more labels/learning curves, and calibration/error analysis. Measure full live routing costs and label-generation cost.
4. Freeze the revised design before a fresh holdout; reserve official length and a separate composition split for final evaluation without cross-split leakage.
5. Optional stretch: one automatically checked multistep arithmetic task family with template-disjoint partitions. No second dataset has been chosen, downloaded, implemented, or approved as mandatory scope.

The new agent should choose a proportional plan and sample budget after inspecting the code and local availability. Do not manufacture positive results, start broad unrelated work, or imply the supplied-plan study already exists.

## Reproduction and safe entry

Windows PowerShell; use the project-local `.venv/Scripts/python.exe` (Python 3.14, locked dependencies in `requirements-lock.txt`). The parent course `.venv` points to a removed Python 3.13 and is unusable. Hardware at collection: RTX 5060, 8151 MiB, driver 576.88. Ollama version was 0.40.13. Exact model digests/settings are in each run's config and environment files; do not assume live availability or identical versions in a future session.

Read-only/model-free validation from this directory:

```powershell
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe experiment.py verify
.\.venv\Scripts\python.exe verify_results.py
```

The last command writes a verification summary; it does not generate new model responses. `.\reproduce.ps1` creates/syncs the environment when needed, tests, prepares data, resumes/reuses existing 8B runs, and refits the selected controller. It requires a running Ollama API at `http://127.0.0.1:11434` and the named model; it is **not necessary just to understand this handoff**. Existing records are reused, so it is not a fresh repeated benchmark.

New model/config/count experiments need new run names. Do not overwrite old result directories or frozen partitions. Executor hashes are part of configs; archived `results/*/experiment.py` files are historical evidence, not current entry points. If using the portable ZIP elsewhere, recreate the virtual environment and obtain the same model weights separately; neither is bundled. Do not load untrusted joblib files; the bundled one was generated locally in this project.

## Delivery status and next action

- Prepared: `CHECKPOINT_3.md`, structured around the user's five requirements, with quantitative table and honest limitations. No visualization is required because a quantitative result satisfies the wording.
- Prepared: this handoff, entry-point instructions, exact user requirements, updated project docs, and a portable snapshot with file hashes.
- Not done/not authorized as part of this handoff: LMS submission, publishing, emailing, Git operations, moving to E:, new experiments, or final generalization evaluation.
- No unanswered user clarification remains for the checkpoint. No future session has actually been launched/tested; files and entry points were independently read back instead.
- Suggested first response in a fresh session: briefly state you read the current checkpoint and verified evidence, then make a plan for fair baseline improvements and the supplied-plan diagnostic. Do not restart the already completed pilot from scratch.
