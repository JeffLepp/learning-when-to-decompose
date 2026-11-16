"""Small success and cost predictors with explicit validation calibration."""

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _matrices(x, n_arms):
    matrices = x if isinstance(x, list) else [x] * n_arms
    if len(matrices) != n_arms:
        raise ValueError('Expected one feature matrix per arm')
    if any(m.ndim != 2 or m.shape[0] != matrices[0].shape[0] for m in matrices):
        raise ValueError('Feature matrices must be 2-D with equal row counts')
    return matrices


def _append_arm(x, arm, n_arms):
    onehot = np.zeros((x.shape[0], n_arms))
    onehot[:, arm] = 1
    if sparse.issparse(x):
        return sparse.hstack([x, onehot], format='csr')
    return np.column_stack([x, onehot])


def _logit(p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p)).reshape(-1, 1)


class ConstantCalibration:
    """Training-fold prior when Platt would reverse score ordering."""
    def __init__(self, prior):
        self.prior = float(prior)

    def predict_proba(self, x):
        p = np.full(len(x), self.prior)
        return np.column_stack([1 - p, p])


def fit_platt(scores, labels, seed=437):
    model = LogisticRegression(C=1e6, max_iter=1000, random_state=seed).fit(
        _logit(scores), labels)
    return ConstantCalibration(np.mean(labels)) if model.coef_[0, 0] <= 0 else model


class SuccessPredictor:
    def __init__(self, learner, mode='per_arm', calibration=None, seed=437, **params):
        if learner not in ('logreg', 'hgb', 'mlp'):
            raise ValueError(f'Unknown learner: {learner}')
        if mode not in ('per_arm', 'shared'):
            raise ValueError(f'Unknown mode: {mode}')
        if calibration not in (None, 'platt', 'isotonic'):
            raise ValueError(f'Unknown calibration: {calibration}')
        self.learner, self.mode = learner, mode
        self.calibration, self.seed, self.params = calibration, seed, params

    def _features(self, x):
        return x.toarray() if self.learner == 'hgb' and sparse.issparse(x) else x

    def _model(self, x):
        if self.learner == 'hgb':
            defaults = dict(random_state=self.seed, early_stopping=False)
            return HistGradientBoostingClassifier(**(defaults | self.params))
        if self.learner == 'logreg':
            defaults = dict(C=1., max_iter=1000, random_state=self.seed)
            estimator = LogisticRegression(**(defaults | self.params))
        else:
            defaults = dict(hidden_layer_sizes=(32,), early_stopping=False,
                            max_iter=500, random_state=self.seed)
            estimator = MLPClassifier(**(defaults | self.params))
        return make_pipeline(StandardScaler(with_mean=not sparse.issparse(x)), estimator)

    def fit(self, X, Y):
        y = np.asarray(Y, dtype=float)
        if y.ndim != 2 or y.shape[1] == 0:
            raise ValueError('Y must be an n x A matrix with at least one arm')
        self.n_arms = y.shape[1]
        xs = _matrices(X, self.n_arms)
        if xs[0].shape[0] != len(y):
            raise ValueError('X and Y row counts differ')
        if not np.isin(y[~np.isnan(y)], [0., 1.]).all():
            raise ValueError('Y must contain binary labels or NaN')
        self.constants, self.models = [], []
        self.calibrators = [None] * self.n_arms
        self.platt_nonpositive_count = 0
        masks = [~np.isnan(y[:, a]) for a in range(self.n_arms)]
        for a, mask in enumerate(masks):
            labels = y[mask, a]
            if not len(labels):
                raise ValueError(f'Arm {a} has no observed training labels')
            # With early stopping disabled these learners accept singleton minority
            # classes; only a missing class requires the prior fallback.
            constant = float(labels.mean()) if len(np.unique(labels)) == 1 else None
            self.constants.append(constant)
        if self.mode == 'shared':
            if len({x.shape[1] for x in xs}) != 1:
                raise ValueError('Shared mode requires equal feature dimension across arms')
            blocks = [_append_arm(x[mask], a, self.n_arms)
                      for a, (x, mask) in enumerate(zip(xs, masks))]
            stacked = (sparse.vstack(blocks, format='csr')
                       if any(sparse.issparse(b) for b in blocks) else np.vstack(blocks))
            labels = np.concatenate([y[mask, a] for a, mask in enumerate(masks)])
            self.shared_model = None
            if any(c is None for c in self.constants):
                stacked = self._features(stacked)
                self.shared_model = self._model(stacked).fit(stacked, labels)
        else:
            for a, mask in enumerate(masks):
                x = self._features(xs[a][mask])
                model = None if self.constants[a] is not None else self._model(x).fit(
                    x, y[mask, a]
                )
                self.models.append(model)
        return self

    def _raw(self, X):
        xs = _matrices(X, self.n_arms)
        probabilities = []
        for a, x in enumerate(xs):
            if self.constants[a] is not None:
                probabilities.append(np.full(x.shape[0], self.constants[a]))
                continue
            if self.mode == 'shared':
                model, x = self.shared_model, _append_arm(x, a, self.n_arms)
            else:
                model = self.models[a]
            probabilities.append(model.predict_proba(self._features(x))[:, 1])
        return np.column_stack(probabilities)

    def calibrate(self, X_val, Y_val, method=None):
        """Fit calibrators on validation only; constant/absent val labels use identity."""
        method = method or self.calibration
        if method not in ('platt', 'isotonic'):
            raise ValueError('Calibration must be platt or isotonic')
        raw = self._raw(X_val)
        y = np.asarray(Y_val, dtype=float)
        if y.shape != raw.shape or not np.isin(y[~np.isnan(y)], [0, 1]).all():
            raise ValueError('Validation labels must match predictions and be binary/NaN')
        self.calibrators = [None] * self.n_arms
        self.platt_nonpositive_count = 0
        for a in range(self.n_arms):
            mask = ~np.isnan(y[:, a])
            if len(np.unique(y[mask, a])) < 2:
                continue
            if method == 'platt':
                model = fit_platt(raw[mask, a], y[mask, a], self.seed)
                self.platt_nonpositive_count += isinstance(model, ConstantCalibration)
            else:
                model = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(
                    raw[mask, a], y[mask, a]
                )
            self.calibrators[a] = (method, model)
        return self

    def predict_proba(self, X):
        p = self._raw(X)
        for a, calibrator in enumerate(self.calibrators):
            if calibrator is not None:
                method, model = calibrator
                p[:, a] = (model.predict_proba(_logit(p[:, a]))[:, 1]
                           if method == 'platt' else model.predict(p[:, a]))
        return np.clip(p, 0, 1)


class CostPredictor:
    def __init__(self, kind):
        if kind not in ('mean', 'rf', 'rf_joint'):
            raise ValueError(f'Unknown cost predictor: {kind}')
        self.kind = kind

    @staticmethod
    def _forest():
        return RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                                     random_state=437)

    def fit(self, X, C):
        c = np.asarray(C, dtype=float)
        if c.ndim != 2 or c.shape[0] != X.shape[0]:
            raise ValueError('C must be n x A and aligned to X')
        if np.isinf(c).any() or (c[np.isfinite(c)] < 0).any():
            raise ValueError('Costs must be nonnegative or NaN')
        if np.isnan(c).all(axis=0).any():
            raise ValueError('Every arm needs observed training costs')
        self.n_arms = c.shape[1]
        if self.kind == 'mean':
            self.means = np.nanmean(c, axis=0)
        elif self.kind == 'rf_joint':
            if np.isnan(c).any():
                raise ValueError('rf_joint requires complete cost labels')
            self.model = self._forest().fit(X, c)
        else:
            self.models = []
            for a in range(self.n_arms):
                mask = ~np.isnan(c[:, a])
                self.models.append(self._forest().fit(X[mask], c[mask, a]))
        return self

    def predict(self, X):
        if self.kind == 'mean':
            return np.tile(self.means, (X.shape[0], 1))
        if self.kind == 'rf_joint':
            return self.model.predict(X).reshape(X.shape[0], self.n_arms)
        return np.column_stack([model.predict(X) for model in self.models])
