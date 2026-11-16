import unittest
import warnings

import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning

from v2.router.models import CostPredictor, SuccessPredictor


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(warnings.catch_warnings())
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        self.x = np.linspace(-2, 2, 40).reshape(-1, 1)
        self.y = np.column_stack([self.x[:, 0] > 0, self.x[:, 0] < -.5]).astype(float)

    def test_constant_fallback_all_learners_and_modes(self):
        y = np.column_stack([np.zeros(40), np.ones(40)])
        y[::3, 0] = np.nan
        for learner in ['logreg', 'hgb', 'mlp']:
            for mode in ['per_arm', 'shared']:
                with self.subTest(learner=learner, mode=mode):
                    model = SuccessPredictor(learner, mode).fit(self.x, y)
                    np.testing.assert_array_equal(model.predict_proba(self.x),
                                                  np.tile([0., 1.], (40, 1)))
                    # A constant arm stays constant when its shared sibling varies.
                    mixed = np.column_stack([np.ones(40), self.y[:, 1]])
                    model = SuccessPredictor(learner, mode).fit(self.x, mixed)
                    np.testing.assert_array_equal(model.predict_proba(self.x)[:, 0], 1.)

    def test_sparse_all_learners_and_modes(self):
        x = sparse.csr_matrix(np.column_stack([self.x, self.x ** 2]))
        for learner in ['logreg', 'hgb', 'mlp']:
            for mode in ['per_arm', 'shared']:
                with self.subTest(learner=learner, mode=mode):
                    model = SuccessPredictor(learner, mode).fit(x, self.y)
                    p = model.predict_proba(x)
                    self.assertEqual(p.shape, self.y.shape)
                    self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_per_arm_features_missing_labels_and_shared_dimension(self):
        xs = [self.x, np.column_stack([self.x, self.x ** 2])]
        y = self.y.copy()
        y[:4, 0] = np.nan
        model = SuccessPredictor('logreg').fit(xs, y)
        self.assertEqual(model.predict_proba(xs).shape, (40, 2))
        separate = SuccessPredictor('logreg').fit(self.x[4:], y[4:, :1])
        np.testing.assert_allclose(model.predict_proba(xs)[:, 0],
                                   separate.predict_proba(self.x)[:, 0])
        with self.assertRaisesRegex(ValueError, 'equal feature dimension'):
            SuccessPredictor('logreg', 'shared').fit(xs, y)
        shared = SuccessPredictor('logreg', 'shared').fit([self.x, -self.x], self.y)
        self.assertEqual(shared.predict_proba([self.x, -self.x]).shape, (40, 2))
        with self.assertRaisesRegex(ValueError, 'no observed'):
            SuccessPredictor('logreg').fit(self.x, np.full((40, 1), np.nan))

    def test_calibration_bounds_ranking_and_constant_validation(self):
        # Distinct validation features with monotonic labels; ties are counted explicitly.
        xv = self.x + .03
        yv = np.column_stack([xv[:, 0] > 0, xv[:, 0] < -.5]).astype(float)
        for method in ['platt', 'isotonic']:
            model = SuccessPredictor('logreg', calibration=method).fit(self.x, self.y)
            raw = model.predict_proba(xv)
            model.calibrate(xv, yv, method)
            calibrated = model.predict_proba(xv)
            self.assertTrue(((calibrated >= 0) & (calibrated <= 1)).all())
            for a in range(2):
                pos, neg = calibrated[yv[:, a] == 1, a], calibrated[yv[:, a] == 0, a]
                ties = (pos[:, None] == neg[None, :]).mean()
                self.assertGreaterEqual(roc_auc_score(yv[:, a], calibrated[:, a]) +
                                        .5 * ties + 1e-12, roc_auc_score(yv[:, a], raw[:, a]))
            model.calibrate(xv, np.ones_like(yv), method)
            np.testing.assert_array_equal(model.predict_proba(xv), raw)

    def test_dense_default_logreg_matches_v1(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = SuccessPredictor('logreg').fit(self.x, self.y)
        expected = np.column_stack([
            make_pipeline(StandardScaler(), LogisticRegression(
                C=1, max_iter=1000, random_state=437
            )).fit(self.x, self.y[:, a]).predict_proba(self.x)[:, 1]
            for a in range(2)
        ])
        np.testing.assert_array_equal(model.predict_proba(self.x), expected)

    def test_cost_models(self):
        c = np.column_stack([self.x[:, 0] ** 2 + 10, self.x[:, 0] ** 2 + 20])
        for kind in ['mean', 'rf', 'rf_joint']:
            predictor = CostPredictor(kind).fit(self.x, c)
            p = predictor.predict(self.x)
            self.assertEqual(p.shape, c.shape)
            self.assertTrue(np.isfinite(p).all())
        from sklearn.ensemble import RandomForestRegressor
        expected = RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                                         random_state=437).fit(self.x, c).predict(self.x)
        np.testing.assert_array_equal(CostPredictor('rf_joint').fit(self.x, c)
                                      .predict(self.x), expected)
        c[:4, 0] = np.nan
        for kind in ['mean', 'rf']:
            self.assertTrue(np.isfinite(CostPredictor(kind).fit(self.x, c)
                                        .predict(self.x)).all())
        with self.assertRaisesRegex(ValueError, 'complete'):
            CostPredictor('rf_joint').fit(self.x, c)


if __name__ == '__main__':
    unittest.main()
