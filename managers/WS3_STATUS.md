# WS3 status — phase B (stop point reached)

Date: 2026-11-15. Manager: Claude (fresh WS3 session; predecessor account hit its usage limit). Worker: `codex`, 3 serial tasks:
- D: the D1 rules and the figures.
- E: `run_p4`, `run_final` and synthetic labels.
- F: review fixes for H3–H5 reporting and the Platt calibration fix.

Task prompts and final messages are in `workspace_logs/ws3-task{D,E,F}.prompt.md` and `.last.md`. Everything marked **measured** below I re-ran myself after the worker finished. The numbers quoted from worker artifacts are labelled as such.

## Done (with evidence)

1. **D1 implemented**
   - D1.7 operating point: `evaluate.operating_point(val_points, best_single_val_acc)` is the single rule for every headline single point. `Experiment.select_operating_point` freezes `{choice, rule:'D1.7', validation_point, best_single_val_accuracy}`.
   - D1.5 coverage: `data.main_table(table, expected_n_arms=16)` raises if any item lacks a main arm and never drops items. `s2_table` holds the separate S2 subset. Both assert that G1/G2 are absent.
   - D1.2 probe variants: `data.load_f3(features_dir, ids, layer_key, variant∈{own,small}, arms)` and `probe_overhead(...)`. The overhead is the S0 `prompt_tokens` from the labels once per distinct probed model, so `own` charges four models and `small` charges the 0.6B model only.
   - Bootstrap default: 10,000 replicates is the library default and is asserted in tests.
2. **Figures rendered (all four types, 150 dpi):**
   - `results/v2/router/sim_check/figures/{cost_accuracy,reliability,learning_curve,auroc}.png`, rebuilt from the saved simulation artifacts without refitting.
   - `results/v2/router/v1_adapter/figures/{cost_accuracy,reliability,learning_curve,auroc}.png`, on train/val only. The AUROC grid includes F3-own from WS2's real `scan_v1/qwen3-8b` features. I viewed cost_accuracy and auroc; they render correctly.
   - The P4 drivers also write figures into `<out>/figures/`.
3. **`v2/router/run_p4.py`** runs the full P4 analysis on router_train/router_val only:
   - `load_labels(..., splits=[...])` skips other splits' rows while reading, and only the two dev item files are opened. A test patches `open` to prove this.
   - Feature families F1, F2 and F3-own/F3-small. The layer for each F3 variant is selected on val with `Experiment.select`.
   - Learners logreg, HGB and MLP, each in per-arm and shared mode.
   - Calibration (none/Platt/isotonic) is chosen per predictor by 5-fold cross-fitted Brier on val. Val routing curves use the cross-fitted probabilities, and the frozen model is calibrated on all of val.
   - Per-arm AUROC/Brier/ECE.
   - Learning curves at 50/100/200/400/all training items, with 20 repeats.
   - Policy suite:
     - fixed arms and best single;
     - cheapest arm meeting the target (best, best−.02, best−.05);
     - heuristic grid;
     - λ routing for every family×learner×mode;
     - the v1 threshold rule;
     - cascades over three ladder types (S0 by size, the val frontier, S0→S3 per model), each with logprob, agreement, learned-F4 and learned-F4+F1 confidence;
     - the oracle as a reference only.
   - Every single operating point uses D1.7. One selected router per family plus an overall and a cascade selection, all by val normalized AUC. Then `freeze()` and `evaluate_on(val)`.
   - Outputs: `report.json` (sha256 of the labels, items, features index and every `v2/router/*.py`), `summary.md`, and `frozen/` (objects plus `selections.json`). All val numbers are labelled "validation (used for selection; optimistic)".
4. **`v2/router/run_final.py`** is the one-shot P5 evaluation:
   - It requires the `freeze-v2` tag. The tag is waived only with `--synthetic-test`, which itself requires Mode-B synthetic provenance, items outside `data/` and `synthetic-` ids.
   - It verifies the router code, frozen artifacts, dev label rows and dev items by sha256. It applies the frozen objects without refitting and calls `evaluate_on(final_eval=True)` per split into `<p4>/final_<split>/`.
   - It checks the one-shot log **before** opening any final rows.
   - Final outputs include held-out predictor metrics (H3) and the pre-registered contrasts (H4/H5); the measured numbers are below.
5. **Synthetic 16-arm fixture `v2/router/synth_labels.py`:**
   - Columns are imported from WS1's `build_labels.COLUMNS`, which was read and not edited.
   - Mode A uses the real router_train/router_val items. Mode B generates fake items for all splits, plus a synthetic F3 directory in WS2's index layout.
   - It includes S2 on a subset, G1/G2 rows to prove they are excluded, and `total_tokens` that includes thinking tokens (D1.4).
6. **Review fix (task F): Platt inversion.**
   - Cross-fitted AUROC fell below raw AUROC (e.g. .5788 → .4960) because 376 of 1,920 fold×arm Platt fits had negative slopes, which invert the ranking.
   - Platt now falls back to the constant prior when slope ≤ 0, with a count recorded (70 fallbacks in the synthetic frozen models). Isotonic is fitted increasing, and a unit test covers this.
   - Pooled cross-fit AUROC can still sit below raw AUROC because each fold uses a different calibrator. **Headline AUROC should be raw scores (calibration-invariant).**
7. **Measured verification**
   - Router tests (`.venv/Scripts/python.exe -B -m unittest discover -s v2/tests -p "test_router_*.py" -t .`): `Ran 59 tests in 318.2s, OK`, 0 skipped.
   - `python -m v2.router.v1_repro`: `"match": true`.
   - Repeating `run_final --synthetic-test` on `synthetic16-f` holdout gives `RuntimeError: Final evaluation already completed for holdout`.
   - Real-data `run_final` without the tag gives `RuntimeError: freeze-v2 git tag required; no final data were opened`.
   - No GPU or Ollama use. No real `data/v2/*/holdout|ood_*` file was opened. No git writes.

## Measured numbers (worker artifacts; fixture/dev only)

- **v1 adapter P4** (`results/v2/router/p4-v1adapter-f/summary.md`, val n=32, optimistic):
  - The selected overall router is F1/logreg/per-arm: .3125 accuracy at 601.6 tokens, versus the best single arm (8b|S3) at .3125 / 688.0.
  - Normalized-AUC differences:

    | Contrast | Difference [95% CI] |
    | --- | ---: |
    | Selected overall − fixed frontier | +.0568 [.0014, .1133] |
    | F3-own − F1 | −.1075 [−.1732, −.0449] |
    | F3-small − F1 | −.0872 [−.1521, −.0141] |
    | Agreement cascade − overall | −.1062 [−.1658, −.0438] |

  - Logprob and learned cascades were skipped because the v1 labels have no logprobs.
  - n=32 is far too small to read anything into these numbers. This run only proves the pipeline runs end to end on real F3 features.
- **Synthetic 16-arm** (`results/v2/router/synthetic16-f/p4/final_{holdout,ood_length}/report.json`, logreg/per-arm, 10,000 replicates):

  | Contrast | Holdout | OOD |
  | --- | ---: | ---: |
  | AUROC: F3-own − F2 | .0764 [.0401, .1120] | .1020 [.0350, .1684] |
  | AUROC: F2 − F1 | −.0751 [−.1032, −.0464] | not quoted here |
  | H5, AUC: cascade − pre-generation | .0478 [.0280, .0648] | −.2576 [−.3043, −.2093] |

  - H5 on holdout at D1.7: accuracy +.035 [.0075, .0625], McNemar p=.0201.
  - These properties come from the generator. They validate the code only.
- **Wall times** (task F): adapter P4 54s, synthetic P4 290s, synthetic final 61s (holdout) + 47s (OOD). The real-data P4 should be minutes on CPU.
- **Mode A** (real dev items, synthetic labels, `--quick`): both domains ran; summaries are at `results/v2/router/mode_a_{scan,chainops}/summary.md`.

## Files created/changed

- **Modified:** `v2/router/{README.md,data.py,evaluate.py,models.py,plots.py}`, `v2/tests/test_router_plots.py`.
- **New code:** `v2/router/{run_p4,run_final,workflow,synth_labels,bootstrap,predictor_report,readback_p4}.py`, `v2/tests/test_router_{d1,p4}.py`.
- **New artifacts:** `results/v2/router/{p4-v1adapter-f,synthetic16-f,mode_a_scan,mode_a_chainops,task_d,task_e,sim_check/figures,v1_adapter/figures}`, plus `taskF-*` logs and evidence.
- **Superseded but kept:** `p4-v1adapter/` (14.9 MB) and `synthetic16/` (25.4 MB), which predate the task F fixes. The worker's deletion attempt was policy-blocked.
- **Not touched:** v1 files, `data/`, `results/*-v1`, other workstreams' files, `HANDOFF.md`, `DESIGN_V2.md`, git.

## Problems / deviations

1. **`results/v2/router` is 102 MB**, above the 50 MB guidance. Deleting the two superseded directories would bring it to about 62 MB; `synthetic16-f` alone is 36.5 MB, mostly frozen joblib objects. I recommend not committing the synthetic fixture runs, or committing only their `summary.md`/`report.json`.
2. **Freeze invalidated by any router code edit.** `run_final` hashes every `v2/router/*.py` against the P4 report, so any edit after the real P4 run requires rerunning P4. This is intended.
   - During task E, the worker fixed a metric bug after the synthetic final evaluation and rewrote the synthetic reports' statistics without a new `evaluate_on` call. The pre-fix reports are archived in `synthetic16/.../pre_metric_fix/`.
   - That was acceptable for fixtures. **For real data, no such post-hoc correction path exists or should be used.**
3. **v1-adapter-only fallback.** When an S0 arm for a probed model is absent (the v1 adapter has no 0.6B labels), the F3 overhead uses WS2's `index.json` `n_tokens`. On the real 16-arm tables every S0 arm exists, so the label `prompt_tokens` are used, as D1.2 says.
4. **Speed paths.** `run_p4`/`run_final` use `fast_bootstrap=True` and `comparison_scope='best_single'`, so the long pairwise list compares each policy with the best single arm only. The H3/H4/H5 contrasts are computed separately with shared bootstrap indices, and tests cover their presence.
5. **Small-sample statistics.** Arms with a single class in a bootstrap replicate are skipped and counted. On the v1 val set (32 items) some AUROCs rest on one or two positives.

## Decisions needed from orchestrator

1. **Artifact policy.** Delete `p4-v1adapter/` and `synthetic16/`, and commit only summaries/reports of the synthetic fixture runs (not `frozen/*.joblib`)?
2. **Headline AUROC = raw (uncalibrated) scores**, with Brier/ECE after the frozen full-val calibration. Confirm.
3. **The H5 contrast as pre-registered:** the val-selected best cascade (by val normalized AUC) minus the val-selected best pre-generation router. The comparison uses normalized AUC plus D1.7 accuracy, tokens and McNemar. Confirm this is the H5 test.
4. **Pre-registered H3 contrast list:** F3-own−F2, F3-own−F1, F2−F1, F3-own−F3-small, F3-small−F2 on mean-over-arms raw AUROC, per learner×mode, with logreg/per-arm as the headline. Confirm.

## Proposed next steps (not started)

1. When WS1 publishes `results/v2/labels/<domain>.csv` with router_train/router_val rows, run `python -m v2.router.run_p4 --labels results/v2/labels/<domain>.csv --domain <domain> --items-dir data/v2/<domain> --features-dir results/v2/features/<domain> --out results/v2/router/p4-<domain>-v1`, separately for each domain.
2. Freeze the router code before that real P4 run; any later edit forces a P4 rerun.
3. After the `freeze-v2` tag, the orchestrator runs `run_final --splits holdout,<ood>` once per domain.
