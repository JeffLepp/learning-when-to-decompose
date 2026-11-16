import unittest
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from v2.router.data import NUMERIC, load_labels, write_labels
from v2.router.evaluate import paired_bootstrap
from v2.router.models import CostPredictor, SuccessPredictor
from v2.router.policies import (LearnedConfidence, agreement_confidence, cascade,
                                decision_success, fixed_arm, logprob_confidence, oracle)
from v2.router.simulate import (availability_audit, cascade_positive_control, expected_auc, label_rows,
                                routing_curve, simulate)
from v2.tests.test_router_data import TempCase


class SimulationTests(TempCase):
    def setUp(self):
        super().setUp()
        self.enterContext(threadpool_limits(limits=1))
        self.enterContext(warnings.catch_warnings())
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

    def test_label_contract_roundtrip_and_determinism(self):
        tables, x, truth = simulate(dict(router_train=20, router_val=15, sim_test=10))
        write_labels(self.path / 'sim.csv', label_rows(tables))
        loaded = load_labels(self.path / 'sim.csv')
        again, xx, _ = simulate(dict(router_train=20, router_val=15, sim_test=10))
        for split, table in tables.items():
            self.assertEqual(table.ids, loaded[split].ids)
            self.assertEqual(table.arms, loaded[split].arms)
            for name in NUMERIC + ['answer_sha1']:
                np.testing.assert_array_equal(getattr(table, name), getattr(loaded[split], name))
                np.testing.assert_array_equal(getattr(table, name), getattr(again[split], name))
            np.testing.assert_array_equal(x[split], xx[split])
            self.assertEqual(truth['splits'][split]['probabilities'].shape, table.success.shape)

    def test_learned_near_analytic_optimum(self):
        gaps = []
        for count in [200, 600, 5000]:
            tables, x, truth = simulate(count, 437)
            train, test = tables['router_train'], tables['sim_test']
            model = SuccessPredictor('logreg').fit(x['router_train'], train.success)
            cost = CostPredictor('mean').fit(x['router_train'], train.cost).predict(x['sim_test'])
            probabilities = truth['splits']['sim_test']['probabilities']
            true_cost = truth['splits']['sim_test']['mean_cost']
            anchor = fixed_arm(len(test.ids), len(test.arms), 0)
            learned = [anchor] + routing_curve(model.predict_proba(x['sim_test']), cost)
            optimum = [anchor] + routing_curve(probabilities, true_cost)
            gaps.append(expected_auc(optimum, probabilities, true_cost) - expected_auc(learned, probabilities, true_cost))
        self.assertLess(abs(gaps[-1]), .02)  # Prespecified acceptance tolerance, never widened.
        self.assertTrue(np.all(np.diff(gaps) < 0), gaps)

    def test_null_features_no_gain(self):
        tables, x, _ = simulate(dict(router_train=600, router_val=300, sim_test=1000), null=True)
        model = SuccessPredictor('logreg').fit(x['router_train'], tables['router_train'].success)
        costs = CostPredictor('mean').fit(x['router_train'], tables['router_train'].cost)
        val_curve = routing_curve(model.predict_proba(x['router_val']), costs.predict(x['router_val']))
        choice = max(range(len(val_curve)), key=lambda i: decision_success(val_curve[i], tables['router_val']).mean())
        curve = routing_curve(model.predict_proba(x['sim_test']), costs.predict(x['sim_test']))
        best = tables['router_val'].success.mean(axis=0).argmax()
        comparison = paired_bootstrap(decision_success(curve[choice], tables['sim_test']),
                                      tables['sim_test'].success[:, best], n_boot=200)
        self.assertLessEqual(comparison['ci95'][0], 0)
        # Positive control: leaking all outcomes would beat the best arm here,
        # so this null check is capable of exposing a leakage-induced gain.
        leaked = paired_bootstrap(decision_success(oracle(tables['sim_test']), tables['sim_test']),
                                  tables['sim_test'].success[:, best], n_boot=200)
        self.assertGreater(leaked['ci95'][0], 0)

    def test_cascade_availability(self):
        tables, _, _ = simulate(dict(router_train=150, router_val=30, sim_test=60))
        table = tables['sim_test']
        ladder = [table.arms[i] for i in [0, 2, 3]]
        learned = LearnedConfidence('logreg').fit(tables['router_train'], ladder)
        for confidence, taus in [(logprob_confidence, [-np.inf, -1., 0., np.inf]),
                                 (agreement_confidence, [[-np.inf, .5], [-1., .5], [np.inf, .5]]),
                                 (learned, [0., .25, .5, .75, 1.])]:
            audit = availability_audit(table, ladder, confidence, taus)
            self.assertTrue(audit['passed'])
            self.assertEqual(audit['thresholds_checked'], len(taus))
        # Failure sensitivity: deliberately leaking confidence must fail the audit.
        def leaking(t, prefix):
            return t.mean_logprob[:, -1]
        with self.assertRaises(AssertionError):
            availability_audit(table, ladder, leaking, [-1.])

    def test_cascade_positive_control(self):
        result = cascade_positive_control(dict(router_train=600, router_val=400, sim_test=1500),
                                          n_boot=100, seed=437, experiment_dir=self.path)
        for policy in ['logprob', 'learned']:
            for comparison in ['auc_minus_pregeneration', 'auc_minus_fixed_frontier']:
                self.assertGreater(result[comparison][policy]['ci95'][0], 0, result)
        for audit in result['availability_audit'].values():
            self.assertTrue(audit['passed'])
            self.assertEqual(audit['thresholds_checked'], 21)
        self.assertTrue(result['passed'])


if __name__ == '__main__':
    unittest.main()
