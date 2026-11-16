"""Correct endpoint-roundoff statistics on identical frozen decisions, without a second evaluation call."""
import csv
import json
import time
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from v2.router.bootstrap import evaluate_batched
from v2.router.data import ROOT, load_labels, main_table, s2_table
from v2.router.evaluate import item_vectors, point, write_json
from v2.router.policies import fixed_arm, best_single_arm, oracle
from v2.router.workflow import (apply_state, crossfit_calibration, family_features,
                                inputs_for, sha256, summary, VALIDATION_NOTE)

BASE = ROOT / 'results/v2/router'
DEST = BASE / 'task_e'
archive = DEST / 'pre_metric_fix'
manifest = json.loads((archive / 'manifest.json').read_text())
archive_paths = {str(Path(path).resolve()): archive / f'report_{i}.json.xz' for i, path in enumerate(manifest)}
records = []


def repair(state, report_path, top, *, subset=False, s2=None):
    started = time.perf_counter()
    report = json.loads(report_path.read_text(encoding='utf-8'))
    original_hash = sha256(report_path)
    split = report['evaluation_call']['split']
    raw = load_labels(top['labels_path'], splits=[split])[split]
    table = s2_table(raw) if subset else main_table(raw, expected_n_arms=top['expected_arms'])
    probabilities = None
    if split == 'router_val':
        # Execution-success models are not refit. Recreate only the original
        # five-fold development calibrators from their unchanged raw predictions.
        inputs = inputs_for(table, top['items_dir'], state['domain'])
        matrices, _ = family_features(state, table, inputs, top['features_dir'])
        probabilities = {}
        for name, model in state['predictors'].items():
            raw_p = model._raw(matrices[state['predictor_families'][name]])
            choices, _, _ = crossfit_calibration(raw_p, table.success)
            method = state['selections']['calibration/' + name]['choice']
            probabilities[name] = choices[method]
    policies = apply_state(state, table, top['items_dir'], top['features_dir'], val_probabilities=probabilities)
    # Strong readback: every reconstructed chosen arm/execution mask must equal
    # the originally delivered CSV, including all validation cross-fit decisions.
    id_lookup = {id_: i for i, id_ in enumerate(table.ids)}
    with (report_path.parent / 'decisions.csv').open(newline='', encoding='utf-8') as stream:
        count = 0
        for row in csv.DictReader(stream):
            if row['policy'] not in policies:
                continue
            spec = policies[row['policy']]
            d, i = spec['decision'], id_lookup[row['id']]
            assert table.arms[d.final[i]] == row['final_arm']
            assert [table.arms[a] for a in np.flatnonzero(d.executed[i])] == json.loads(row['executed_arms'])
            count += 1
        assert count == len(table.ids) * len(policies)
    best = fixed_arm(len(table.ids), len(table.arms), best_single_arm(state['val']))
    vectors = {'best_single': item_vectors(best, table)}
    curves, metrics = {}, {}
    for name, spec in policies.items():
        vectors[name] = item_vectors(spec['decision'], table, spec['overhead'])
        curves[name] = [item_vectors(spec['anchor'], table)] + [item_vectors(d, table, spec['overhead']) for d in spec['curve']]
        metrics[name] = dict(point(spec['decision'], table, spec['overhead']), mean_overhead=float(spec['overhead'].mean()))
        for key in ['accuracy', 'mean_cost']:
            assert abs(metrics[name][key] - report['policies'][name][key]) < 1e-10
    comparisons = evaluate_batched(table, vectors, curves, item_vectors(oracle(table), table), metrics,
                                    report['uncertainty']['n_boot'], report['uncertainty']['seed'], 'best_single')
    assert all(np.isfinite(m['normalized_auc']['value']) for m in metrics.values())
    report.update(policies=metrics, comparisons=comparisons)
    if s2 is not None:
        report['s2'] = s2
    correction = dict(reason='Numerical ties at Pareto support/target endpoints now use 1e-12 relative token tolerance and 1e-12 accuracy tolerance.',
                      original_report_sha256=original_hash,
                      original_report_archive=str(archive_paths[str(report_path.resolve())].relative_to(ROOT)),
                      unchanged_selected_decision_rows=count,
                      execution_models_refit=False, policy_selection_repeated=False,
                      evaluation_call_repeated=False, bootstrap_seed_and_replicates_unchanged=True,
                      validation_crossfit_calibrators_recreated=(split == 'router_val'),
                      corrected_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started)
    report['statistics_correction'] = correction
    records.append(dict(report=str(report_path.relative_to(ROOT)), **correction))
    return report


with threadpool_limits(limits=1):
    for directory in [BASE / 'p4-v1adapter', BASE / 'synthetic16/p4']:
        top = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
        state = joblib.load(directory / 'frozen/objects.joblib')
        s2 = repair(state['s2'], directory / 's2/report.json', top, subset=True) if state['s2'] else None
        if s2:
            write_json(directory / 's2/report.json', s2)
        corrected = repair(state, directory / 'report.json', top, s2=s2)
        corrected['fit_code_sha256'] = top['code_sha256']
        corrected['code_sha256'] = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'v2/router').glob('*.py'))}
        write_json(directory / 'report.json', corrected)
        summary(corrected, directory / 'summary.md')
        for split in ['holdout', 'ood_length']:
            destination = directory / f'final_{split}'
            if not (destination / 'report.json').exists():
                continue
            s2 = repair(state['s2'], destination / 's2/report.json', top, subset=True) if state['s2'] else None
            if s2:
                write_json(destination / 's2/report.json', s2)
            final = repair(state, destination / 'report.json', top, s2=s2)
            final['original_p4_report_sha256_at_evaluation'] = final['p4_report_sha256']
            final['p4_report_sha256'] = sha256(directory / 'report.json')
            write_json(destination / 'report.json', final)
            summary(final, destination / 'summary.md', final['note'])
write_json(DEST / 'statistics_correction.json', dict(records=records, script_sha256=sha256(Path(__file__))))
print(json.dumps(records, indent=2))
