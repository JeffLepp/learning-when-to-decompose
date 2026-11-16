# WS3 status — Router & evaluation library, phase A (stop point reached)

Date: 2026-11-15. Manager: Claude2 (WS3). Worker: `codex` (3 serial tasks: A library core, B evaluation/simulation/plots, C review fixes).
Prompts and worker final messages: `workspace_logs/ws3-task{A,B,C}.prompt.md` / `.last.md`.
Everything below marked **measured** was re-run by the manager after the worker finished, not copied from the worker.

## Done (with evidence)

1. **Library `v2/router/`** implementing DESIGN_V2 §6: `data.py`, `models.py`, `policies.py`, `evaluate.py`, `plots.py`, plus `simulate.py`, `readback.py`, `v1_repro.py`, `README.md` (API summary).
2. **Tests (measured):** `.venv/Scripts/python.exe -B -m unittest discover -s v2/tests -p "test_router_*.py" -t .` → `Ran 42 tests in 7.3s, OK (skipped=1)`. The one skip is the plot test (matplotlib absent).
3. **v1 regression (measured):** `.venv/Scripts/python.exe -B -m v2.router.v1_repro` → `match: true`. Threshold .5; validation 10/32, 629.9375 mean tokens; v1 audit **15/32, 625.09375 mean tokens**; all 32 per-item audit choices equal `results/controller-v1/audit_predictions.json`. Built only from the new library on the adapter CSV.
4. **Simulation check (measured from `results/v2/router/sim_check/report.json`; worker ran `python -m v2.router.simulate`, I read the report back):** see numbers below. All five acceptance flags true.
5. **Scope check (measured):** `git status` shows WS3 touched only `v2/router/`, `v2/tests/test_router_*.py`, `results/v2/router/`. No commits. No GPU/Ollama/network. No real `holdout`/`ood_*` file is opened by any router code (grep: the only references are the guard in `evaluate.py`).

## API summary (details in `v2/router/README.md`)

- `data`: `load_labels(csv) -> {split: ArmTable}` (n_items × n_arms matrices: success, cost, prompt/completion tokens, seconds, n_calls, error, mean/min logprob, answer_sha1; NaN where an arm was not run, e.g. S2 subset; `.select()`, `.complete(arms)`). Tolerates WS1's extra columns (`thinking_tokens`, `run_name`). Loud errors on duplicates, cross-split ids, non-binary labels. `v1_label_rows` / `python -m v2.router.data v1-adapter`. Features: F1 `structural_features(inputs, domain, metas)` with `STRUCTURAL` hook (scan == v1 features, tested; chainops aligned to WS1's dev file), F2 `TfidfFeatures` (train-only fit), F3 `load_activations(npy, index.json, ids)` (aligns by id, asserts), F4 `posthoc_features(table, executed)`.
- `models`: `SuccessPredictor(learner∈{logreg,hgb,mlp}, mode∈{per_arm,shared})` with `fit / calibrate(platt|isotonic, val only) / predict_proba`; constant-label arms fall back to the prior; accepts one X or one X per arm (F3 differs by model size); sparse OK. `CostPredictor(mean|rf|rf_joint)`.
- `policies`: `Decision(executed, final)`; `fixed_arm`, `best_single_arm`, `cheapest_arm_meeting_target`, `heuristic(inputs, metas, rule, arms)` + `complexity_rule`, `predict_then_route` + `lambda_grid`, `threshold_route` (v1 rule), `cascade(table, ladder, confidence, tau)` (scalar or per-stage τ; cost = all executed arms), confidences `logprob_confidence`, `agreement_confidence`, `LearnedConfidence`, `oracle`.
- `evaluate`: `point`, `pareto_frontier`, `curve_accuracy_at_cost`, `curve_cost_at_accuracy`, `normalized_auc`, `gap_closed`, `arm_metrics` (AUROC/Brier/ECE/reliability bins), `paired_bootstrap`, `bootstrap_curve_metric`, `mcnemar`, `learning_curve`, and `Experiment` (`select` on router_val only → `freeze()` → `evaluate_on(table, policies, final_eval=False)`; every call logged to `evaluate_calls.jsonl`; `holdout`/`ood_*` need `final_eval=True` and are refused a second time per experiment directory). Optional per-item `overhead` tokens per policy.
- **Cascade availability is structural:** the confidence callback only ever receives `visible(table, executed_arms)`, a copy in which `success` and all columns of non-executed arms are NaN/None.
- **Interpolation (documented in `evaluate.py`):** reduce a sweep to its non-dominated points, interpolate linearly between neighbours (achievable by per-item randomisation between two policies), NaN left of the cheapest point, flat to the right. Normalized AUC = mean accuracy over [cheapest fixed-arm mean cost, most expensive fixed-arm mean cost].

## Measured numbers

`results/v2/router/sim_check/report.json` (seed 437; 1000 val / 2000 sim_test items; 200 bootstrap replicates; 4 synthetic arms, one dominated; independent Bernoulli outcomes). Expected normalized AUC of predict-then-route on sim_test:

| n_train | logreg per-arm | analytic-optimal router | fixed-arm frontier |
| ---: | ---: | ---: | ---: |
| 200 | .716945 | .722271 | .622465 |
| 600 | .720984 | .722271 | .622465 |
| 5000 | .721735 | .722271 | .622465 |

- Gap to optimum shrinks .00533 → .00129 → .00054 (tolerance .02, not loosened). Worker reports all six learner×mode combinations also shrink; I verified only the logreg per-arm row above.
- **Null check** (features carry no information): learned router accuracy .8985 = best single .8985, paired diff 0, CI [0, 0]; oracle .911. No spurious gain.
- **Cascade, informative-feature regime:** pre-generation AUC .7184; logprob cascade .6366; learned-confidence .6394; agreement .5301. All cascades lose here (features reveal difficulty almost exactly, and a cascade always pays for its first arm).
- **Cascade positive control** (uninformative features, informative post-hoc signals; n_train 5000): pre-generation .6226, frontier .5876; logprob cascade .6362, Δ vs pre-generation +.0136 [.0025, .0230], Δ vs frontier +.0486 [.0085, .0556]; learned-confidence cascade .6358, Δ +.0133 [.0028, .0222] and +.0483 [.0094, .0548]. Agreement cascade loses in both regimes (−.0929 vs pre-generation here).
- Availability audit: 63 thresholds / 126 prefix checks passed in each regime.
- `results/v2/router/v1_repro/report.json`: numbers in "Done" item 3.
- `results/v2/router` total size: 5.06 MB (measured).

These are properties of a synthetic generator. They validate the code; they say nothing about H3–H5.

## Files created/changed

- Created: `v2/router/{__init__,data,models,policies,evaluate,simulate,readback,plots,v1_repro}.py`, `v2/router/README.md`, `v2/tests/test_router_{data,models,policies,v1,evaluate,simulate,plots}.py`.
- Created artifacts: `results/v2/router/v1_adapter/{scan_v1_labels.csv,scan_v1_items.jsonl}`, `results/v2/router/v1_repro/report.json`, `results/v2/router/sim_check/**`, and worker review logs `results/v2/router/{review_baseline.json,review_validation.json,review_tests.log,sim_check_run.log}` (safe to leave uncommitted).
- `v2/tests/__init__.py` exists (may have been created by WS1 or by the WS3 worker; one-line docstring). WS3 did not create or edit `v2/__init__.py`.
- Not touched: v1 files, `data/`, `results/*-v1`, other workstreams' files, `HANDOFF.md`, git history.

## Problems / deviations from spec

1. **Figures not produced.** matplotlib is not in `.venv`; per the brief it was not installed. `plots.py` is written (lazy import, Agg) but has never rendered; its test is skipped. **No figure example paths exist yet.**
2. **v1 adapter split name.** v1 `audit[:32]` is exposed as split `v1_audit`, not `holdout`, so it cannot be confused with the v2 holdout or trip the final-eval guard. Strategies are labelled `S0`/`S3` although the v1 prompts differ from the v2 S0/S3 prompts.
3. **v1 labels have no logprobs**, so F4 and the cascade were exercised only on synthetic data, never on real model signals.
4. **Agreement cascade is weak as specified.** One threshold scale does not fit both a logprob (stage 0) and a 0/1 agreement flag (later stages); per-stage τ was added, but a ladder must run two arms before agreement means anything. Its behaviour on real labels is unknown.
5. **Intermediate bloat, fixed.** Task B first wrote 327 MB of per-sweep-point decisions; task C reduced `decisions.csv` to selected policies only and removed superseded preliminary runs. Consequence: `readback.py` can no longer replay individual sweep points, only selected decisions and aggregate curves.
6. Simulation used 200 bootstrap replicates for speed; the library default for real analysis is 10,000.
7. The `Experiment` guard cannot stop a caller from computing decisions with test data inside its own callbacks; it enforces split names, freeze order, and logging only.

## Decisions needed from orchestrator

1. **Approve installing matplotlib into `.venv`** (and adding it to `requirements-lock.txt`)? Needed for every figure in the brief.
2. **Probe overhead accounting.** DESIGN_V2 charges a probe forward pass "as prompt tokens on that model". Proposed: overhead[i] = S0 prompt-token count of item i for each model whose activations the router reads; a per-arm router that probes all four sizes pays all four. Alternatives: probe only the smallest model, or share one probe model across arms (cheaper, changes F3's "own size" definition). Which one is the headline?
3. **F3 `index.json` key.** The loader expects the id order under `ids` (alias `item_ids`). Confirm with WS2, or tell me their key.
4. **Cost for S2 (thinking).** Is `total_tokens` in the label CSV inclusive of `thinking_tokens`? The loader uses `total_tokens` as cost as-is.
5. **Arms with partial coverage (S2 subset).** Proposed: main routing analysis over arms present for all items (`table.complete(arms)`), S2 analysed separately on its subset. OK?
6. **Normalized-AUC range and left anchor.** Currently [cheapest, most expensive fixed arm] on the evaluated split, with the cheapest fixed arm added to every policy curve as the left anchor. Confirm, since this defines a headline metric.
7. **Operating-point rule for single-point comparisons** (McNemar, gap closed). The simulation used max of accuracy − .001·tokens on val; the real study needs a pre-registered rule (e.g. cheapest val point with accuracy ≥ best single arm's).

## Proposed next steps (not started)

1. After decision 1: render the four figure types on the simulation and the v1 adapter data; un-skip the plot test.
2. When WS1's `results/v2/labels/<domain>.csv` has `router_train`/`router_val` rows: run F1/F2 × three learners × per-arm/shared, calibration, learning curves, heuristic and cascade ladders on train/val only (P4). No holdout.
3. When WS2 delivers features: wire F3 with layer selection on `router_val` via `Experiment.select`, and add the agreed probe overhead.
4. Write the single frozen-evaluation driver script for P5 so the orchestrator runs holdout/OOD exactly once after `freeze-v2`.
