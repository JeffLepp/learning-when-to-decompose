import unittest

import numpy as np
from scipy import sparse

from v2.router.data import NUMERIC
from v2.router.policies import (
    LearnedConfidence, agreement_confidence, best_single_arm, cascade,
    cheapest_arm_meeting_target, complexity_rule, decision_cost, decision_success,
    fixed_arm, heuristic, lambda_grid, logprob_confidence, oracle,
    predict_then_route, threshold_route, visible,
)
from v2.tests.test_router_data import TempCase, label_rows


class PolicyTests(TempCase):
    def test_stage_thresholds_and_explicit_heuristic_order(self):
        table = self.table()
        decision = cascade(table, table.arms, agreement_confidence, [-np.inf, .5])
        np.testing.assert_array_equal(decision.final, 0)
        decision = cascade(table, table.arms, agreement_confidence, [np.inf, .5])
        np.testing.assert_array_equal(decision.final, 1)
        with self.assertRaises(ValueError):
            cascade(table, table.arms, agreement_confidence, [.5])
        rule = lambda text, meta: 'm|S1'
        result = heuristic(['x'], None, rule, ['m|S2', 'm|S0', 'm|S1'])
        np.testing.assert_array_equal(result.final, [2])
        self.assertEqual(result.executed.shape, (1, 3))
        with self.assertRaises(ValueError):
            heuristic(['x'], None, rule, ['m|S0'])

    def test_fixed_best_target_and_heuristic(self):
        table = self.table()
        fixed = fixed_arm(6, 3, 1)
        np.testing.assert_array_equal(decision_cost(fixed, table), table.cost[:, 1])
        np.testing.assert_array_equal(decision_success(fixed, table), table.success[:, 1])
        self.assertEqual(best_single_arm(table), 0)  # Equal accuracy; lower cost.
        self.assertEqual(cheapest_arm_meeting_target(table, .5), (0, True))
        self.assertEqual(cheapest_arm_meeting_target(table, 1.), (0, False))
        table.success[:, 1] = 1
        self.assertEqual(best_single_arm(table), 1)
        self.assertEqual(cheapest_arm_meeting_target(table, 1.), (1, True))
        rule = complexity_rule('small', 'large', k=1, length_threshold=4)
        commands = ['walk', 'jump around left twice after walk']
        arms = ['large|S0', 'large|S3', 'small|S0', 'small|S3']
        result = heuristic(commands, None, rule, arms)
        self.assertEqual([arms[a] for a in result.final], ['small|S0', 'large|S3'])
        custom = lambda text, meta: 'm|S0' if meta['simple'] else 'm|S1'
        np.testing.assert_array_equal(heuristic(['a', 'b'],
                                      [{'simple': True}, {'simple': False}], custom,
                                      ['m|S0', 'm|S1']).final,
                                      [0, 1])

    def test_threshold_matches_v1_and_utility_extremes(self):
        from controller import route
        rng = np.random.default_rng(437)
        p, c = rng.random((100, 5)), rng.integers(1, 100, (100, 5)).astype(float)
        p[0, :] = .5  # Exercise exact ties in both eligible and fallback paths.
        for threshold in [0, .25, .5, .95, 1]:
            np.testing.assert_array_equal(threshold_route(p, c, threshold).final,
                                          route(p, c, threshold))
        np.testing.assert_array_equal(predict_then_route(p[1:], c[1:], 0).final,
                                      p[1:].argmax(axis=1))
        routed = predict_then_route(p, c, 1e6).final
        np.testing.assert_array_equal(c[np.arange(len(c)), routed], c.min(axis=1))
        self.assertEqual(predict_then_route([[.5, .5]], [[20, 10]], 0).final[0], 1)
        grid = lambda_grid(c)
        self.assertEqual(len(grid), 25)
        self.assertEqual(grid[0], 0)
        self.assertTrue((np.diff(grid) > 0).all())
        routed = predict_then_route(p, c, grid[-1]).final
        np.testing.assert_array_equal(c[np.arange(len(c)), routed], c.min(axis=1))

    def test_visible_is_structural_and_independent(self):
        table = self.table()
        masked = visible(table, table.arms[:1])
        self.assertTrue(np.isnan(masked.success).all())
        for name in NUMERIC:
            self.assertTrue(np.isnan(getattr(masked, name)[:, 1:]).all())
        self.assertTrue((masked.answer_sha1[:, 1:] == None).all())
        masked.cost[0, 0] = 999
        self.assertNotEqual(table.cost[0, 0], 999)

    def test_all_confidences_ignore_success_and_unexecuted_columns(self):
        table = self.table()
        learned = LearnedConfidence('logreg').fit(table, table.arms)
        rng = np.random.default_rng(437)
        for k in [1, 2]:
            executed = table.arms[:k]
            changed = table.select()
            changed.success[:] = rng.normal(size=changed.success.shape)
            for name in NUMERIC:
                getattr(changed, name)[:, k:] = rng.normal(size=(6, 3 - k))
            changed.answer_sha1[:, k:] = 'unavailable'
            for confidence in [logprob_confidence, agreement_confidence, learned]:
                with self.subTest(stage=k, confidence=type(confidence).__name__):
                    np.testing.assert_array_equal(confidence(table, executed),
                                                  confidence(changed, executed))
        scores = logprob_confidence(table, table.arms[:1])
        self.assertEqual(scores[4], -np.inf)  # Missing logprob.
        self.assertEqual(scores[5], -np.inf)  # Error.
        np.testing.assert_array_equal(agreement_confidence(table, table.arms[:1]), scores)
        np.testing.assert_array_equal(agreement_confidence(table, table.arms[:2]), 1.)
        table.answer_sha1[0, :2] = None
        self.assertEqual(agreement_confidence(table, table.arms[:2])[0], 0.)

    def test_cascade_cost_extremes_and_masked_callback(self):
        table = self.table()
        def checked(view, executed):
            self.assertTrue(np.isnan(view.success).all())
            future = [a for a in range(3) if view.arms[a] not in executed]
            for name in NUMERIC:
                self.assertTrue(np.isnan(getattr(view, name)[:, future]).all())
            self.assertTrue((view.answer_sha1[:, future] == None).all())
            return logprob_confidence(view, executed)
        for tau, counts in [(-np.inf, 1), (np.inf, 3)]:
            result = cascade(table, table.arms, checked, tau)
            np.testing.assert_array_equal(result.executed.sum(axis=1), counts)
            np.testing.assert_array_equal(decision_cost(result, table),
                                          table.cost[:, :counts].sum(axis=1))
        result = cascade(table, table.arms, checked, -.5)
        np.testing.assert_array_equal(result.final, [0, 0, 2, 2, 2, 2])
        np.testing.assert_array_equal(decision_cost(result, table),
                                      np.where(result.executed, table.cost, 0).sum(axis=1))
        table.cost[0, 2] = np.nan
        with self.assertRaisesRegex(ValueError, 'complete'):
            cascade(table, table.arms, checked, -.5)

    def test_later_signals_cannot_change_stopped_items(self):
        table = self.table()
        learned = LearnedConfidence('logreg').fit(table, table.arms)
        rng = np.random.default_rng(438)
        for confidence, tau in [(logprob_confidence, -.5), (agreement_confidence, .5),
                                (learned, .5)]:
            original = cascade(table, table.arms, confidence, tau)
            changed = table.select()
            changed.success[:] = rng.random(changed.success.shape)
            # Change every arm strictly after each item's final executed stage.
            for i, final in enumerate(original.final):
                for name in NUMERIC:
                    getattr(changed, name)[i, final + 1:] = rng.random(2 - final)
                changed.answer_sha1[i, final + 1:] = 'changed'
            again = cascade(changed, changed.arms, confidence, tau)
            np.testing.assert_array_equal(original.final, again.final)
            np.testing.assert_array_equal(original.executed, again.executed)

    def test_learned_confidence_input_features_and_training_guard(self):
        table = self.table()
        x = sparse.csr_matrix(np.arange(12).reshape(6, 2))
        learned = LearnedConfidence('logreg').fit(table, table.arms, X=x)
        decision = cascade(table, table.arms, learned.for_inputs(x), .5)
        self.assertEqual(decision.final.shape, (6,))
        with self.assertRaisesRegex(ValueError, 'dimension'):
            learned(table, table.arms[:1])
        incomplete = table.select()
        incomplete.n_calls[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'complete'):
            learned.predict(incomplete, table.arms[:1], X=x)
        table.split = 'router_val'
        with self.assertRaisesRegex(ValueError, 'router_train'):
            LearnedConfidence('logreg').fit(table, table.arms)

    def test_oracle_bound_and_missing_arm_errors(self):
        table = self.table()
        result = oracle(table)
        for a in range(3):
            self.assertGreaterEqual(decision_success(result, table).mean(),
                                    decision_success(fixed_arm(6, 3, a), table).mean())
        # Enumerate all possible always-correct choices, not only fixed arms.
        from itertools import product
        for choices in product(*[np.flatnonzero(row) for row in table.success]):
            costs = table.cost[np.arange(6), choices]
            self.assertTrue((decision_cost(result, table) <= costs).all())
        table.success[0] = 0
        self.assertEqual(oracle(table).final[0], table.cost[0].argmin())
        missing = self.table(label_rows()[1:])
        with self.assertRaisesRegex(ValueError, 'complete'):
            cascade(missing, missing.arms, logprob_confidence, 0)
        with self.assertRaisesRegex(ValueError, 'missing cost'):
            decision_cost(fixed_arm(6, 3, 0), missing)


if __name__ == '__main__':
    unittest.main()
