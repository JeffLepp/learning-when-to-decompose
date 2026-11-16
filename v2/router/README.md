# Router and evaluation library

CPU-only library for supervised arm selection and replay evaluation. Real final
splits remain reserved for the orchestrator after the project freeze.

## Modules

- `data.py`: `ArmTable` aligns item IDs and arm columns, supports selection and
  complete-case subsets. Label/item loading, label writing, historical v1 adapter,
  SCAN/ChainOps structural features, train-fitted TF-IDF, aligned activation loading,
  and post-hoc features restricted to executed arms. ChainOps recognizes generated
  operation metadata and `list_len`.
- `models.py`: `SuccessPredictor(learner, mode, **params)` supports logreg, HGB,
  and MLP in per-arm/shared modes; `fit`, validation-only `calibrate` (Platt or
  isotonic), and `predict_proba`. Constant labels use priors. `CostPredictor`
  supports mean, per-arm RF, or joint RF costs through `fit`/`predict`.
- `policies.py`: `Decision(executed, final)`, fixed/best-single/target baselines,
  structural heuristics using explicit table arm order, lambda and threshold
  routing, and an outcome oracle (diagnostic only). `cascade(table, ladder,
  confidence, tau)` accepts scalar or per-stage thresholds and charges every
  executed arm. Confidence callbacks see only executed-arm signals and no labels.
  Logprob, agreement, and train-only `LearnedConfidence` callbacks are included.
- `evaluate.py`: point metrics, Pareto curves, linear interpolation, normalized
  AUC, arm AUROC/Brier/ECE/reliability, paired item bootstrap, McNemar, learning
  subsamples, and `Experiment` (below). Undefined metrics serialize as JSON null.
- `simulate.py`: deterministic independent Bernoulli synthetic labels, known
  expected-probability benchmarks, six learner/mode combinations, calibration,
  null leakage check, cascades, and `cascade_positive_control`. Every cascade
  threshold receives label/unavailable-signal perturbation audits.
- `readback.py`: replays selected CSV decisions against regenerated synthetic
  labels and cumulative costs; checks JSON points, integrates aggregate curves,
  reconciles summary AUCs, and verifies completion logs. No refitting or real
  holdout/OOD access. It cannot replay per-item sweeps, expected learned AUCs, or
  bootstrap CIs once only aggregate curve points are saved. Writes hashed
  `sim_check/readback.json` and fails on inconsistencies.
- `plots.py`: optional cost-accuracy, reliability, learning-curve, and AUROC PNGs;
  imports matplotlib lazily with Agg. Matplotlib is absent; rendering remains
  unverified and its unit test is skipped. No installation is performed.
- `v1_repro.py`: historical v1 regression using the v2 API, including validation
  selection and exact audit-choice/cost comparisons; writes adapter/reproduction
  artifacts under `results/v2/router/`. This is not a v2 final evaluation.

## Frozen evaluation

`Experiment(name, val_table, n_boot=10000, seed=437)` requires `router_val` and
writes under `results/v2/router/<name>`. `select(name, candidates, score_fn)`
maximizes `score_fn(candidate, val_table)` and records the selected key. After all
selections, call `freeze()`, then `evaluate_on(table, policies)`:

```python
policies = {
    'router': {
        'decision': selected_decision,
        'curve': frozen_sweep_decisions,
        'overhead': per_item_prompt_token_charge,  # optional
    }
}
```

Policy names must match the frozen selections. The best single arm is selected
on validation. The API cannot inspect external data captured by callbacks or
prove how caller-owned decisions were produced. Callers own split discipline and
the orchestrator freeze. Every attempt appends `started` or `refused` to
`evaluate_calls.jsonl`; successful artifact writes append `completed`. Splits
`holdout`, `ood_length`, and `ood_depth` require `final_eval=True` and refuse a
second completed evaluation of the same split in that experiment directory,
including from a fresh Experiment object. Interrupted attempts can be retried.

`decisions.csv` contains exactly one row per item for `best_single`, `oracle`,
and each selected policy. Sweep points remain as aggregate `curve` entries in
`report.json`; there are no `policy@k` CSV rows. Routing overhead is explicit;
fixed-arm and oracle references have zero routing overhead.

Curves remove dominated points and interpolate linearly, without a concave hull.
Include the cheapest fixed arm as a left anchor. AUC is mean accuracy between the
cheapest and most expensive fixed-arm mean token costs, using the same range for
all comparisons; it is not divided by oracle area. Bootstrap indices are paired
across policies, curves, and references, with no refitting or reselection.
Undefined target-reaching replicates yield an undefined CI and a defined-count
field instead of being dropped. Differences are named `left - right`.

## Simulation protocol and artifacts

`sim_check/report.json` references the regenerated `*_independent` evaluations.
Sizes are 200, 600, and 5000 training items with fixed validation/test streams
(1000/2000 items), seed 437, and 200 bootstrap replicates (library default 10000).
The analytic benchmark uses known probabilities and a fixed lambda grid; it is
not a claim of global optimality. Expected-probability AUCs and sampled-label
cascade AUCs are distinct. Synthetic routing overhead is zero. Calibration is
reported before/after Platt and isotonic regardless of improvement.

The ordinary regime exposes noisy difficulty before generation. The cascade
positive control uses `null=True`: pre-generation features are independent noise,
while existing post-hoc logprob signals remain informative. It changes neither
signal strength nor costs to obtain a win. Models fit on router_train; operating
points are selected on router_val; sim_test evaluation uses `evaluate_on`.
Logprob and learned-confidence cascades pass only if paired-bootstrap 95% CIs of
AUC differences against both pre-generation routing and the fixed frontier are
strictly positive. Availability is audited for every tau in both regimes.
The separate null check compares the validation-selected maximum-accuracy router
with the best single arm, and verifies that leaked outcome knowledge could help.

The simulate runner passes MLP `max_iter=1500`; library defaults and warning
filters are unchanged. This may change MLP results relative to the previous
500-iteration run; `review_baseline.json` preserves the pre-review summary.
The superseded nested-outcome artifacts were removed. `sim_check_run.log` now
records the independent run. Cleanup uses explicit file deletion followed by
empty-directory removal because recursive shell deletion was policy-blocked.

## Commands

Run from the project root with the project interpreter:

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s v2/tests -p "test_router_*.py" -t .
.venv/Scripts/python.exe -B -m v2.router.simulate
.venv/Scripts/python.exe -B -m v2.router.readback
.venv/Scripts/python.exe -B -m v2.router.v1_repro
```

`simulate --plots` additionally requests example figures when matplotlib exists.
Tests use automatically removed temporary directories under results/v2/router,
including synthetic tables named holdout for the one-shot guard; they never open
real final-split files. No GPU, Ollama, network, or installs are needed.
