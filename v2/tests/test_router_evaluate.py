import csv
import json
import unittest

import numpy as np

from v2.router.evaluate import (
    Experiment, accuracy_at_best_single_cost, arm_metrics, bootstrap_curve_metric,
    cost_at_best_single_accuracy, curve_accuracy_at_cost, curve_cost_at_accuracy,
    decision_cost, gap_closed, learning_curve, mcnemar, normalized_auc,
    paired_bootstrap, pareto_frontier, point,
)
from v2.router.policies import fixed_arm
from v2.tests.test_router_data import TempCase


def p(cost, accuracy):
    return dict(mean_cost=cost, accuracy=accuracy)


class MetricTests(unittest.TestCase):
    def test_frontier_and_interpolation(self):
        curve = [p(30, .9), p(10, .5), p(15, .4), p(10, .4), p(20, .7), p(20, .7), p(40, .9)]
        self.assertEqual(pareto_frontier(curve), [p(10, .5), p(20, .7), p(30, .9)])
        self.assertAlmostEqual(curve_accuracy_at_cost(curve, 15), .6)
        self.assertEqual(curve_accuracy_at_cost(curve, 100), .9)
        self.assertTrue(np.isnan(curve_accuracy_at_cost(curve, 9)))
        self.assertAlmostEqual(curve_cost_at_accuracy(curve, .8), 25)
        self.assertEqual(curve_cost_at_accuracy(curve, .2), 10)
        self.assertTrue(np.isnan(curve_cost_at_accuracy(curve, .95)))
        self.assertTrue(np.isnan(curve_accuracy_at_cost([], 1)))
        self.assertTrue(np.isnan(curve_cost_at_accuracy([], 1)))
        self.assertEqual(accuracy_at_best_single_cost(curve, p(20, .7)), .7)
        self.assertEqual(cost_at_best_single_accuracy(curve, p(20, .7)), 20)

    def test_auc_and_gap(self):
        curve = [p(10, .5), p(30, .9)]
        self.assertAlmostEqual(normalized_auc(curve, 10, 30), .7)
        self.assertAlmostEqual(normalized_auc(curve, 10, 50), .8)
        self.assertTrue(np.isnan(normalized_auc(curve, 0, 30)))
        self.assertAlmostEqual(normalized_auc(curve, 10, 30, curve)['oracle_auc'], .7)
        with self.assertRaises(ValueError):
            normalized_auc(curve, 10, 10)
        self.assertAlmostEqual(gap_closed(.75, .5, 1), .5)
        self.assertTrue(np.isnan(gap_closed(.5, .5, .5)))

    def test_paired_bootstrap(self):
        rng = np.random.default_rng(437)
        a = rng.normal(0, 100, 150)
        identical = paired_bootstrap(a, a, n_boot=200)
        self.assertEqual(identical, dict(diff=0., ci95=[0., 0.], p_two_sided=1.))
        shifted = paired_bootstrap(a + 2, a, n_boot=300)
        np.testing.assert_allclose(shifted['ci95'], [2, 2], atol=1e-12)
        self.assertEqual(shifted, paired_bootstrap(a + 2, a, n_boot=300))
        b = a + rng.normal(2, .1, len(a))
        paired = paired_bootstrap(b, a, n_boot=300)['ci95']
        unpaired = np.array([b[rng.integers(len(b), size=len(b))].mean() -
                             a[rng.integers(len(a), size=len(a))].mean() for _ in range(300)])
        self.assertLess(paired[1] - paired[0], np.ptp(np.quantile(unpaired, [.025, .975])) / 100)
        with self.assertRaises(ValueError):
            paired_bootstrap([1], [1, 2])

    def test_mcnemar_exact(self):
        self.assertEqual(mcnemar([1, 0], [1, 0]), dict(b=0, c=0, p_exact=1.))
        self.assertEqual(mcnemar([1] * 4, [0] * 4), dict(b=4, c=0, p_exact=.125))
        self.assertEqual(mcnemar([1, 1, 1, 0], [0, 0, 0, 1]), dict(b=3, c=1, p_exact=.625))

    def test_arm_metrics(self):
        y = np.array([[0, 1], [1, 1], [np.nan, 1]])
        probabilities = np.array([[0, 1], [1, 1], [np.nan, 1]])
        result = arm_metrics(y, probabilities)
        self.assertEqual(result['auroc'][0], 1)
        self.assertTrue(np.isnan(result['auroc'][1]))
        self.assertEqual(result['brier'], [0, 0])
        self.assertEqual(result['ece'], [0, 0])
        self.assertEqual(result['n'], [2, 3])
        self.assertEqual(result['base_rate'], [.5, 1])
        self.assertEqual(sum(b['n'] for b in result['reliability_bins'][0]), 2)
        # Equal-width bins: (.1, .2) in bin 0; .8 and 1 in bin 1.
        result = arm_metrics([0, 1, 1, 1], [.1, .2, .8, 1], n_bins=2)
        self.assertAlmostEqual(result['ece'][0], .225)

    def test_curve_bootstrap_and_learning_subsamples(self):
        values = np.arange(20.)
        result = bootstrap_curve_metric(lambda idx: values[idx].mean(), 20, n_boot=100)
        self.assertEqual(result, bootstrap_curve_metric(lambda idx: values[idx].mean(), 20, n_boot=100))
        self.assertEqual(result['value'], 9.5)
        self.assertEqual(result['n_defined'], 100)
        seen = []
        def fit_score(idx):
            self.assertEqual(len(set(idx)), len(idx))
            seen.append(idx.copy())
            return {'accuracy': len(idx) / 100}
        result = learning_curve(fit_score, 70, sizes=(50, 100, None), repeats=3)
        self.assertEqual([r['n_train'] for r in result], [50, 70])
        self.assertEqual([len(r['raw']) for r in result], [3, 1])
        self.assertEqual(result[0]['std']['accuracy'], 0)
        self.assertFalse(np.array_equal(seen[0], seen[1]))
        np.testing.assert_array_equal(seen[-1], np.arange(70))


class ExperimentTests(TempCase):
    def test_cost_overhead_and_point(self):
        table = self.table()
        d = fixed_arm(6, 3, 0)
        self.assertEqual(point(d, table)['correct'], 4)
        np.testing.assert_array_equal(decision_cost(d, table, np.ones(6) * 5), table.cost[:, 0] + 5)
        self.assertEqual(point(d, table, np.ones(6) * 5)['mean_cost'], 17.5)
        with self.assertRaises(ValueError):
            decision_cost(d, table, [-1] * 6)

    def test_selection_guards_logging_and_readback(self):
        table = self.table()
        with self.assertRaises(AssertionError):
            Experiment('test', table, experiment_dir=self.path)
        table.split = 'router_val'
        experiment = Experiment('test', table, n_boot=20, experiment_dir=self.path)
        fixed = fixed_arm(6, 3, 0)
        with self.assertRaisesRegex(RuntimeError, 'freeze'):
            experiment.evaluate_on(table, {'fixed': fixed})
        train = table.select()
        train.split = 'router_train'
        with self.assertRaises(AssertionError):
            experiment.select('bad', {'a': fixed}, lambda d, t: point(d, t)['accuracy'], table=train)
        experiment.select('fixed', {'zero': fixed}, lambda d, t: point(d, t)['accuracy'])
        snapshot = experiment.frozen
        snapshot['fixed']['choice'] = 'modified'
        self.assertEqual(experiment.frozen['fixed']['choice'], 'zero')
        experiment.freeze()
        with self.assertRaises(RuntimeError):
            experiment.select('late', {'a': fixed}, lambda d, t: 1)
        for split in ['holdout', 'ood_length', 'ood_depth']:
            synthetic = table.select()
            synthetic.split = split
            with self.assertRaisesRegex(RuntimeError, 'final_eval'):
                experiment.evaluate_on(synthetic, {'fixed': fixed})
        for _ in range(2):
            report = experiment.evaluate_on(table, {'fixed': dict(decision=fixed, curve=[fixed, fixed_arm(6, 3, 1)])})
        log = (self.path / 'evaluate_calls.jsonl').read_text().splitlines()
        self.assertEqual(len(log), 8)  # Four refused attempts, two started/completed pairs.
        self.assertEqual(len(report['evaluate_calls']), 8)
        self.assertEqual([json.loads(line)['status'] for line in log[-4:]],
                         ['started', 'completed', 'started', 'completed'])
        self.assertEqual(report['policies']['fixed']['accuracy'], 4 / 6)
        written = json.loads((self.path / 'report.json').read_text())
        self.assertEqual(written['evaluation_status'], 'completed')
        with (self.path / 'decisions.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        selected = [r for r in rows if r['policy'] == 'fixed']
        self.assertEqual(len(rows), 18)
        self.assertEqual({r['policy'] for r in rows}, {'best_single', 'oracle', 'fixed'})
        self.assertEqual(len(written['policies']['fixed']['curve']), 2)
        self.assertEqual(len(selected), 6)
        self.assertEqual(sum(float(r['correct']) for r in selected), 4)
        synthetic.split = 'holdout'
        experiment.evaluate_on(synthetic, {'fixed': fixed}, final_eval=True)
        self.assertEqual(len((self.path / 'evaluate_calls.jsonl').read_text().splitlines()), 10)

    def test_final_evaluation_once_per_split_persisted(self):
        table = self.table()  # Synthetic fixture only, in an automatically removed temp dir.
        table.split = 'router_val'
        fixed = fixed_arm(6, 3, 0)
        for split in ['holdout', 'ood_length', 'ood_depth']:
            synthetic = table.select()
            synthetic.split = split
            for attempt in range(2):
                # A fresh object must still respect a previous completed evaluation.
                experiment = Experiment('test', table, n_boot=2, experiment_dir=self.path)
                experiment.select('fixed', {'zero': fixed}, lambda d, t: 0.)
                experiment.freeze()
                if attempt == 0:
                    experiment.evaluate_on(synthetic, {'fixed': fixed}, final_eval=True)
                    before = (self.path / 'report.json').read_bytes()
                else:
                    with self.assertRaisesRegex(RuntimeError, 'already completed'):
                        experiment.evaluate_on(synthetic, {'fixed': fixed}, final_eval=True)
                    self.assertEqual((self.path / 'report.json').read_bytes(), before)
        entries = [json.loads(line) for line in (self.path / 'evaluate_calls.jsonl').read_text().splitlines()]
        self.assertEqual([e['status'] for e in entries], ['started', 'completed', 'refused'] * 3)

    def test_interrupted_evaluation_does_not_consume_final_split(self):
        table = self.table()
        table.split = 'router_val'
        fixed = fixed_arm(6, 3, 0)
        experiment = Experiment('test', table, n_boot=2, experiment_dir=self.path)
        experiment.select('fixed', {'zero': fixed}, lambda d, t: 0.)
        experiment.freeze()
        synthetic = table.select()
        synthetic.split = 'holdout'
        with self.assertRaises(ValueError):
            experiment.evaluate_on(synthetic, {'fixed': fixed_arm(1, 3, 0)}, final_eval=True)
        experiment.evaluate_on(synthetic, {'fixed': fixed}, final_eval=True)
        entries = [json.loads(line) for line in (self.path / 'evaluate_calls.jsonl').read_text().splitlines()]
        self.assertEqual([e['status'] for e in entries], ['started', 'started', 'completed'])


if __name__ == '__main__':
    unittest.main()
