"""Independent selected-decision/cost and aggregate-AUC readback, without refits."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from .data import load_labels, main_table, probe_overhead, s2_table
from .evaluate import normalized_auc, write_json, arm_metrics
from .workflow import output_directory, sha256, family_features, inputs_for


def verify(p4_dir):
    p4_dir = output_directory(p4_dir)
    top = json.loads((p4_dir / 'report.json').read_text(encoding='utf-8'))
    for name, digest in top['frozen_sha256'].items():
        if sha256(p4_dir / 'frozen' / name) != digest:
            raise RuntimeError('Frozen artifact hash mismatch')
    state = joblib.load(p4_dir / 'frozen/objects.joblib')
    jobs = [(p4_dir, 'router_val', False)]
    if state['s2'] is not None:
        jobs.append((p4_dir / 's2', 'router_val', True))
    # This command never opens real v2 final splits. Only explicitly Mode-B
    # generated labels may enter final readback; validation works for any P4.
    if (top.get('synthetic_labels') or {}).get('mode') == 'B':
        for split in ['holdout', 'ood_length', 'ood_depth']:
            directory = p4_dir / f'final_{split}'
            if (directory / 'report.json').exists():
                jobs.append((directory, split, False))
                if (directory / 's2/report.json').exists():
                    jobs.append((directory / 's2', split, True))
    checks = []
    for directory, split, is_s2 in jobs:
        report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
        raw = load_labels(top['labels_path'], splits=[split])[split]
        table = s2_table(raw) if is_s2 else main_table(raw, expected_n_arms=top['expected_arms'])
        spec = state['s2'] if is_s2 else state
        ids, arms = {id_: i for i, id_ in enumerate(table.ids)}, {arm: i for i, arm in enumerate(table.arms)}
        overhead = {}
        for name, recipe in spec['recipes'].items():
            if recipe['kind'] == 'alias':
                overhead[name] = overhead[recipe['source']]
            elif recipe.get('family', '').startswith('F3'):
                family = spec['families'][recipe['family']]
                overhead[name] = probe_overhead(table, family['variant'], table.arms, top['features_dir'])
            else:
                overhead[name] = np.zeros(len(table.ids))
        observed = {}
        with (directory / 'decisions.csv').open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                name, i = row['policy'], ids[row['id']]
                values = observed.setdefault(name, {})
                if i in values:
                    raise AssertionError('Duplicate decision row')
                executed = [arms[a] for a in json.loads(row['executed_arms'])]
                final = arms[row['final_arm']]
                assert final in executed
                expected_y = table.success[i, final]
                expected_cost = table.cost[i, executed].sum() + (overhead[name][i] if name in overhead else 0)
                np.testing.assert_allclose([float(row['correct']), float(row['cost'])],
                                           [expected_y, expected_cost], atol=1e-10, rtol=0)
                values[i] = expected_y, expected_cost
        assert set(observed) == {'best_single', 'oracle', *report['policies']}
        for name, result in {**report['policies'], 'best_single': report['best_single'], 'oracle': report['oracle']}.items():
            assert len(observed[name]) == len(table.ids)
            y, c = np.asarray(list(observed[name].values())).T
            np.testing.assert_allclose([result['accuracy'], result['mean_cost']], [y.mean(), c.mean()], atol=1e-10, rtol=0)
            if 'curve' in result:
                np.testing.assert_allclose(normalized_auc(result['curve'], *report['cost_range']),
                                           result['normalized_auc']['value'], atol=1e-10, rtol=0)
        log = [json.loads(line) for line in (directory / 'evaluate_calls.jsonl').read_text().splitlines()]
        for name, contrast in report.get('contrasts', {}).items():
            left, right = name.split(' - ')
            if right in report['policies']:
                np.testing.assert_allclose(contrast['normalized_auc']['value'],
                    report['policies'][left]['normalized_auc']['value'] - report['policies'][right]['normalized_auc']['value'], atol=1e-10)
            if 'accuracy' in contrast:
                lhs, rhs = np.array([observed[left][i] for i in range(len(table.ids))]), np.array([observed[right][i] for i in range(len(table.ids))])
                np.testing.assert_allclose([contrast['accuracy']['value'], contrast['cost']['value']], (lhs - rhs).mean(axis=0), atol=1e-10)
        if report.get('predictor_metrics'):
            matrices, _ = family_features(spec, table, inputs_for(table, top['items_dir'], top['domain']), top['features_dir'])
            for name, metric in report['predictor_metrics'].items():
                model, x = spec['predictors'][name], matrices[spec['predictor_families'][name]]
                raw = arm_metrics(table.success, model._raw(x))
                calibrated = arm_metrics(table.success, model.predict_proba(x))
                for key, expected in [('auroc', raw['auroc']), ('brier', calibrated['brier']), ('ece', calibrated['ece'])]:
                    np.testing.assert_allclose(np.asarray(metric[key], float), expected, atol=1e-12)
            for name, contrast in report['predictor_contrasts'].items():
                left, right = name.split(' - ')
                np.testing.assert_allclose(contrast['value'], report['predictor_metrics'][left]['mean_auroc'] - report['predictor_metrics'][right]['mean_auroc'], atol=1e-12)
        assert sum(r['status'] == 'completed' for r in log) == 1
        checks.append(dict(directory=str(directory), split=split, s2=is_s2,
                           policies=len(observed), items=len(table.ids), passed=True,
                           report_sha256=sha256(directory / 'report.json')))
    result = dict(all_passed=True, checks=checks,
                  limits='Selected decisions/costs, aggregate AUC, contrast estimates and frozen predictor metrics replayed. Bootstrap CIs and model fitting not independently rerun.')
    write_json(p4_dir / 'readback.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p4-dir', type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().p4_dir), indent=2))
