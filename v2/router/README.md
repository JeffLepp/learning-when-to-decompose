# Router and evaluation library

CPU-only library for supervised arm selection and replay evaluation. Real final
splits remain reserved for the orchestrator after the project freeze.

## P4 development and frozen P5 drivers

```powershell
.venv/Scripts/python.exe -B -m v2.router.synth_labels --domain scan --out results/v2/router/synthetic16/labels.csv --synthetic-items-out results/v2/router/synthetic16/items --features-out results/v2/router/synthetic16/features --seed 437
.venv/Scripts/python.exe -B -m v2.router.run_p4 --labels results/v2/router/synthetic16/labels.csv --domain scan --items-dir results/v2/router/synthetic16/items --features-dir results/v2/router/synthetic16/features --out results/v2/router/synthetic16/p4
.venv/Scripts/python.exe -B -m v2.router.run_final --p4-dir results/v2/router/synthetic16/p4 --labels results/v2/router/synthetic16/labels.csv --items-dir results/v2/router/synthetic16/items --features-dir results/v2/router/synthetic16/features --splits holdout,ood_length --synthetic-test
```

Omit `--synthetic-items-out` for Mode A: fake labels on real development items,
opening only `router_train.jsonl` and `router_val.jsonl`. Mode B creates clearly
marked fake items, including fake final splits. Neither mode calls a model.

P4 accepts the WS1 label CSV, development item directory and optional WS2 F3
directory. Use `--expected-arms none` for the historical two-arm adapter:
`--labels results/v2/router/v1_adapter/scan_v1_labels.csv --items-dir data/v2/scan
--features-dir results/v2/features/scan_v1 --out results/v2/router/p4-v1adapter`.
Labels are filtered while reading; P4 never opens final item files. A completed
output directory cannot be reused. `--quick` uses 200 bootstrap replicates, two
learning-curve repeats and smaller policy grids; full defaults are 10,000/20.

Outputs are `report.json`, a summary under 150 lines, selected `decisions.csv`,
four PNGs in `figures/`, and `frozen/{selections.json,objects.joblib}`. S2 is
separate under `s2/`. All validation numbers are selection diagnostics. Calibration
is chosen by five-fold cross-fitted validation Brier; validation routing uses
those out-of-fold probabilities. Saved models calibrate on full validation.
Platt slopes must be positive; nonpositive fits use the fitting fold's constant
prior, with fold/arm and full-validation fallback counts recorded. Isotonic is
explicitly increasing. Different cross-fit mappings can still alter pooled ranks.
`predictor_metrics` reports every frozen predictor's per-arm raw AUROC and
full-validation-calibrated Brier/ECE on both P4 and final splits. P4 is optimistic.
`predictor_contrasts` contains all five H3 family differences per learner/mode,
with shared item-bootstrap indices and counted single-class arm omissions.
`contrasts` separately reports selected overall versus best single/fixed frontier,
each family versus F1, and validation-AUC-selected cascade versus pre-generation
router (AUC, D1.7 accuracy/cost differences and McNemar). Final never reselects.
F3-own/shared zero-pads differing hidden widths; no dimensions are discarded.
Every F3 policy pays probe overhead. Missing historical S0 label columns use
WS2's recorded S0 probe `n_tokens`, explicitly noted in the report.

P5 verifies router code, frozen artifacts, development labels and development
inputs, loads the frozen transforms/models/grids, and writes `final_<split>/`.
It never fits or selects anything. It requires an existing `freeze-v2` tag;
`--synthetic-test` bypasses the tag only for verified Mode-B P4 provenance with
items outside `data/`. A completed final split is refused on subsequent runs.
This driver never creates the tag. Only trusted local frozen artifacts may be
loaded (joblib uses pickle). Full label-file hashes are provenance; development
row hashes permit later final rows without permitting development changes.

The P4/P5 batched bootstrap recomputes Pareto interpolation on every paired
resample; comparisons are against best single, with additional paired curve-AUC
differences against best single and fixed frontier. Only selected per-item
decisions are saved. The zero-overhead cheapest fixed arm anchors each curve;
it is an evaluation reference, not a free activation-based route.

`python -B -m v2.router.readback_p4 --p4-dir <dir>` independently replays saved
selected decisions/costs and integrates aggregate curves. It reads final labels
only for Mode-B synthetic reports, never real final files; it does not refit or
independently reproduce bootstrap CIs. Each readback writes `readback.json`.

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
- `plots.py`: cost-accuracy, reliability, learning-curve, and AUROC PNGs at
  150 dpi; imports matplotlib lazily with Agg. Includes saved-simulation and
  historical v1 train/validation figure commands. No installation is performed.
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

For headline operating points use `operating_point(val_points,
best_single_val_accuracy)` or `Experiment.select_operating_point(name,
val_points, decisions_or_keys, rule='D1.7')`. D1.7 chooses the cheapest validation
point reaching best-single validation accuracy (tolerance 1e-12), then higher
accuracy, then lower index. If none qualifies, maximize accuracy, then minimize
cost, then index. The helper returns an index for a sequence of Decisions, or a
key for a key sequence/mapping in insertion order. It records `choice`, `rule`,
`validation_point`, and `best_single_val_accuracy`; reserved names, duplicate
names, and selection after freeze are refused. Points must include overhead and
come from validation. All three library bootstrap defaults remain 10,000.

Policy names must match the frozen selections except metadata registered with
`select(..., selection_only=True)` (layers/calibration). The best single arm is selected
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

## D1 coverage and activation features

`main_arms(table)` preserves column order for S0/S1/S3/S4. `main_table(table)`
requires 16 arms and complete success/cost coverage for every item; it raises
instead of dropping items. Use `expected_n_arms=None` only for smaller adapters.
`s2_table(table)` requires complete main coverage, then retains the subset with
S2 for all four registered models and returns main plus S2 columns. An absent
S2 model column gives an empty subset. Both helpers exclude G1/G2 diagnostics.
`DEPLOYABLE_MAIN`, `MODELS`, and `DIAGNOSTIC` expose these D1.5 constants.

`model_dir_name('qwen3:0.6b')` returns `qwen3-0.6b`.
`available_layer_keys(features_dir, model)` reads WS2's `index.json['files']`.
`load_f3(features_dir, ids, layer_key, variant, arms)` returns one matrix per arm,
aligned by ID only; `own` reads each arm's model and `small` reads 0.6B for all.
Missing directories, files, layers, or IDs raise. `probe_overhead(table, variant,
arms)` charges each item's S0 prompt tokens once per distinct probed model (own)
or once for 0.6B (small), raising for missing S0 columns or token values. Add this
vector as evaluation `overhead`; F3-own is the pre-registered RQ3 headline and
F3-small a separately reported variant, never selected using final splits.

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
.venv/Scripts/python.exe -B -m v2.router.plots sim
.venv/Scripts/python.exe -B -m v2.router.plots v1
```

`plots sim` reads existing `sim_check` reports without any refit, writing four
PNGs under `sim_check/figures/`. Its learning figure shows expected cost-accuracy
AUC versus training size with the saved 200-replicate bootstrap bands on
sim_test, not repeated validation scores. Historical simulation operating points
retain their original utility rule; rendering does not relabel them as D1.7.

`plots v1` uses only router_train (96) and router_val (32) adapter IDs. It writes
four PNGs and `metrics.json` under `v1_adapter/figures/`: F1 logreg lambda/threshold
curves with D1.7 choice records, per-arm reliability, validation AUROC learning
curves (20 repeats, SD bands), and F1/F2/F3-own by logreg/HGB/MLP AUROC. F3 selects
one layer/pooling per learner by mean validation AUROC over both arms (first key
wins ties); every candidate score and selected prediction is saved. Validation
also supplies the displayed scores, so these are selection diagnostics, not
generalization estimates. HF/Ollama quantization differs. F3 appears only in the
AUROC grid; the cost plot uses F1 with zero probe overhead. The v1 audit remains
exclusive to the historical regression. No model generation is performed.

`simulate --plots` runs the full simulation before requesting example figures;
prefer `plots sim` for rendering existing results.
Tests use automatically removed temporary directories under results/v2/router,
including synthetic tables named holdout for the one-shot guard; they never open
real final-split files. No GPU, Ollama, network, or installs are needed.
When capturing unittest logs in Windows PowerShell, merge stderr inside a Python
subprocess (see `results/v2/router/taskF-run-tests.py`); PowerShell's `2>&1` can
turn successful unittest progress on stderr into `NativeCommandError`.
