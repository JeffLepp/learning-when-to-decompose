"""Reconcile saved simulation CSV decisions with JSON metrics, without refitting.

Run: .venv/Scripts/python.exe -m v2.router.readback
Only synthetic sim_test artifacts are read; this does not open project holdouts.
"""

import csv
import hashlib
import json

import numpy as np

from .data import ROOT
from .evaluate import decision_cost, normalized_auc, point, write_json
from .policies import Decision, decision_success
from .simulate import simulate


def verify():
    base = ROOT / 'results/v2/router/sim_check'
    summary = json.loads((base / 'report.json').read_text(encoding='utf-8'))
    checks = {}
    jobs = [(f'n{n}_independent', n, False) for n in [200, 600, 5000]]
    jobs += [('cascades_independent', 5000, False), ('null_independent', 5000, True)]
    jobs += [('cascade_positive_control_independent', 5000, True)]
    for folder, n_train, null in jobs:
        tables, _, _ = simulate(n_train, summary['seed'], null=null)
        table = tables['sim_test']
        report = json.loads((base / folder / 'report.json').read_text(encoding='utf-8'))
        n, a = table.cost.shape
        item_index = {id_: i for i, id_ in enumerate(table.ids)}
        arm_index = {arm: j for j, arm in enumerate(table.arms)}
        saved = {}
        rows = 0
        with (base / folder / 'decisions.csv').open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                name = row['policy']
                if name not in saved:
                    saved[name] = dict(final=np.full(n, -1), executed=np.zeros((n, a), bool),
                                       correct=np.full(n, np.nan), cost=np.full(n, np.nan))
                record, i = saved[name], item_index[row['id']]
                assert record['final'][i] == -1, 'Duplicate saved item/policy'
                record['final'][i] = arm_index[row['final_arm']]
                record['executed'][i, [arm_index[arm] for arm in json.loads(row['executed_arms'])]] = True
                record['correct'][i] = float(row['correct'])
                record['cost'][i] = float(row['cost'])
                rows += 1
        decisions = {}
        assert set(saved) == {'best_single', 'oracle', *report['policies']}, 'Unexpected saved policies'
        assert rows == n * (2 + len(report['policies'])), 'Wrong selected-decision row count'
        for name, record in saved.items():
            decision = Decision(record['executed'], record['final'])
            np.testing.assert_array_equal(decision_success(decision, table), record['correct'])
            np.testing.assert_array_equal(decision_cost(decision, table), record['cost'])
            decisions[name] = decision
        for name, result in {**report['policies'], 'best_single': report['best_single'],
                             'oracle': report['oracle']}.items():
            measured = point(decisions[name], table)
            for metric in ['n', 'correct', 'accuracy', 'mean_cost']:
                np.testing.assert_allclose(measured[metric], result[metric], rtol=0, atol=1e-12)
        for name, result in report['policies'].items():
            # Only aggregate curve points are persisted; selected item decisions
            # above are replayed, while sweep AUCs are independently integrated.
            np.testing.assert_allclose(normalized_auc(result['curve'], *report['cost_range']),
                                       result['normalized_auc']['value'], rtol=0, atol=1e-12)
            if folder.startswith('n') and not null:
                assert result['normalized_auc'] == summary['sizes'][str(n_train)]['empirical_auc'][name]
        for k, arm in enumerate(table.arms):
            actual = report['fixed_points'][k]
            assert actual['arm'] == arm
            np.testing.assert_allclose([actual['accuracy'], actual['mean_cost']],
                                       [table.success[:, k].mean(), table.cost[:, k].mean()])
        np.testing.assert_allclose(normalized_auc(report['fixed_points'], *report['cost_range']),
                                   report['fixed_frontier_auc'], rtol=0, atol=1e-12)
        calls = [json.loads(line) for line in (base / folder / 'evaluate_calls.jsonl').read_text().splitlines()]
        assert calls[-1]['status'] == 'completed' and calls[-1]['split'] == 'sim_test'
        assert report['evaluate_calls'] == calls
        checks[folder] = dict(rows=rows, decision_sets=len(saved), passed=True)
    for key in ['cascade', 'cascade_positive_control']:
        result = summary[key]
        report = json.loads((ROOT / result['report']).read_text(encoding='utf-8'))
        assert result['fixed_frontier_auc'] == report['fixed_frontier_auc']
        for name, auc in result['auc'].items():
            assert auc == report['policies'][name]['normalized_auc']
        for name in ['logprob', 'agreement', 'learned']:
            for baseline, auc in [('pregeneration', result['auc']['pregeneration']['value']),
                                  ('fixed_frontier', result['fixed_frontier_auc'])]:
                np.testing.assert_allclose(result[f'auc_minus_{baseline}'][name]['value'],
                                           result['auc'][name]['value'] - auc, atol=1e-12)
            assert result['availability_audit'][name]['passed']
    artifact = dict(checks=checks, all_passed=True,
                    report_sha256=hashlib.sha256((base / 'report.json').read_bytes()).hexdigest(),
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted((ROOT / 'v2/router').glob('*.py'))},
                    note=('Replayed selected decisions against deterministic synthetic outcomes; integrated saved '
                          'aggregate curves. No refitting or per-item sweep/expected-AUC/CI replay.'))
    write_json(base / 'readback.json', artifact)
    return summary, artifact


def main():
    summary, artifact = verify()
    print(json.dumps(artifact['checks'], indent=2))
    for n, result in summary['sizes'].items():
        print(n, {name: {key: comparison[key]['value'] for key in ['learned', 'optimal', 'frontier']}
                  for name, comparison in result['expected'].items()})
    print(json.dumps({key: summary[key] for key in ['null', 'cascade', 'cascade_positive_control', 'acceptance']}, indent=2))


if __name__ == '__main__':
    main()
