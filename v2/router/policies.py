"""Routing decisions and replay cascades with structurally masked signals."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .data import ArmTable, NUMERIC, posthoc_features, STRUCTURAL
from .models import SuccessPredictor


@dataclass
class Decision:
    executed: np.ndarray
    final: np.ndarray

    def __post_init__(self):
        self.executed = np.asarray(self.executed, dtype=bool)
        self.final = np.asarray(self.final, dtype=int)
        if self.executed.ndim != 2 or self.final.shape != (len(self.executed),):
            raise ValueError('Decision requires n x A executed and n final indices')
        if ((self.final < 0) | (self.final >= self.executed.shape[1])).any():
            raise ValueError('Final arm index out of range')
        if not self.executed[np.arange(len(self.final)), self.final].all():
            raise ValueError('Final arm must have been executed')


def _decision(final, n_arms):
    final = np.asarray(final, dtype=int)
    executed = np.zeros((len(final), n_arms), dtype=bool)
    executed[np.arange(len(final)), final] = True
    return Decision(executed, final)


def decision_cost(decision, table):
    if decision.executed.shape != table.cost.shape:
        raise ValueError('Decision and table shapes differ')
    if not np.isfinite(table.cost[decision.executed]).all():
        raise ValueError('An executed arm has missing cost')
    return np.where(decision.executed, table.cost, 0.).sum(axis=1)


def decision_success(decision, table):
    if decision.executed.shape != table.success.shape:
        raise ValueError('Decision and table shapes differ')
    success = table.success[np.arange(len(table.ids)), decision.final]
    if not np.isfinite(success).all():
        raise ValueError('A final arm has missing success')
    return success


def fixed_arm(n, n_arms, arm):
    if not 0 <= arm < n_arms:
        raise ValueError('Arm index out of range')
    return _decision(np.full(n, arm), n_arms)


def _validation_summary(table):
    if not len(table.ids) or not np.isfinite(table.success).all() or not np.isfinite(
        table.cost
    ).all():
        raise ValueError('Use a nonempty complete validation table for arm comparison')
    return table.success.mean(axis=0), table.cost.mean(axis=0)


def best_single_arm(val_table):
    accuracy, cost = _validation_summary(val_table)
    return int(np.lexsort((cost, -accuracy))[0])


def cheapest_arm_meeting_target(val_table, target):
    accuracy, cost = _validation_summary(val_table)
    eligible = np.flatnonzero(accuracy >= target)
    if len(eligible):
        return int(eligible[np.argmin(cost[eligible])]), True
    return best_single_arm(val_table), False


def heuristic(inputs, metas, rule, arms):
    """Return indices in the supplied ArmTable column order."""
    inputs = list(inputs)
    metas = [None] * len(inputs) if metas is None else list(metas)
    if len(metas) != len(inputs):
        raise ValueError('inputs and metas must have equal length')
    choices = [rule(text, meta) for text, meta in zip(inputs, metas)]
    arms = list(arms)
    if not arms or len(set(arms)) != len(arms):
        raise ValueError('arms must be nonempty and distinct')
    return _decision([arms.index(arm) for arm in choices], len(arms))


def complexity_rule(small_model, large_model, k=2, length_threshold=10, domain='scan'):
    """Example: structural complexity > k uses S3; long inputs use the large model.

    SCAN complexity counts conjunctions, direction modifiers and repetitions;
    ChainOps complexity is the operation count. Only input metadata is used.
    """
    def rule(text, meta):
        f = STRUCTURAL[domain](text, meta)
        complexity = (sum(f[f'count_{w}'] for w in
                          ['and', 'after', 'opposite', 'around', 'twice', 'thrice'])
                      if domain == 'scan' else f['n_ops'])
        model = large_model if len(text.split()) > length_threshold else small_model
        strategy = 'S3' if complexity > k else 'S0'
        return f'{model}|{strategy}'
    return rule


def _predictions(p_hat, cost_hat):
    p, c = np.asarray(p_hat, dtype=float), np.asarray(cost_hat, dtype=float)
    if p.ndim != 2 or p.shape != c.shape or p.shape[1] == 0:
        raise ValueError('Predictions must be matching n x A matrices')
    if not np.isfinite(p).all() or not np.isfinite(c).all():
        raise ValueError('Predictions must be finite')
    return p, c


def predict_then_route(p_hat, cost_hat, lam):
    p, c = _predictions(p_hat, cost_hat)
    if lam < 0 or not np.isfinite(lam):
        raise ValueError('lambda must be finite and nonnegative')
    score = p - lam * c
    tied = score == score.max(axis=1, keepdims=True)
    return _decision(np.argmin(np.where(tied, c, np.inf), axis=1), p.shape[1])


def lambda_grid(cost_hat, n=25):
    """Zero plus n-1 log points; final lambda dominates any probability gap <= 1."""
    c = np.asarray(cost_hat, dtype=float)
    if c.ndim != 2 or not c.size or not np.isfinite(c).all() or n < 2:
        raise ValueError('Need finite nonempty costs and n >= 2')
    gaps = np.diff(np.sort(c, axis=1), axis=1)
    positive = gaps[gaps > 0]
    if not len(positive):
        return np.r_[0., np.geomspace(1e-4, 1., n - 1)]
    low = 1e-4 / max(float(np.ptp(c, axis=1).max()), 1e-12)
    high = 2. / positive.min()
    return np.r_[0., np.geomspace(low, high, n - 1)]


def threshold_route(p_hat, cost_hat, threshold):
    p, c = _predictions(p_hat, cost_hat)
    eligible = p >= threshold
    choice = np.argmin(np.where(eligible, c, np.inf), axis=1)
    none = ~eligible.any(axis=1)
    choice[none] = p[none].argmax(axis=1)
    return _decision(choice, p.shape[1])


def visible(table, executed_arms):
    """Independent table with no success labels and no unexecuted-arm signals."""
    cols = [table.arms.index(a) for a in executed_arms]
    shape = (len(table.ids), len(table.arms))
    arrays = {name: np.full(shape, np.nan) for name in NUMERIC}
    arrays['answer_sha1'] = np.full(shape, None, dtype=object)
    for name in NUMERIC + ['answer_sha1']:
        if name != 'success':
            arrays[name][:, cols] = getattr(table, name)[:, cols]
    return ArmTable(list(table.ids), list(table.arms), table.split, **arrays)


def _available(table, ladder):
    if not ladder or len(set(ladder)) != len(ladder):
        raise ValueError('Ladder must contain distinct arms')
    cols = [table.arms.index(arm) for arm in ladder]
    # Presence is observable without correctness. Missing logprobs/answers are valid.
    for name in ['cost', 'prompt_tokens', 'completion_tokens', 'seconds', 'n_calls', 'error']:
        if not np.isfinite(getattr(table, name)[:, cols]).all():
            raise ValueError('Missing ladder arm; call table.complete(ladder) first')
    return cols


def cascade(table, ladder, confidence, tau):
    """tau is scalar or one threshold per non-final ladder stage."""
    cols = _available(table, ladder)
    thresholds = np.asarray(tau, dtype=float)
    if thresholds.ndim == 0:
        thresholds = np.full(len(cols) - 1, thresholds)
    if thresholds.shape != (len(cols) - 1,) or np.isnan(thresholds).any():
        raise ValueError('tau needs one threshold per non-final stage (or a scalar)')
    executed = np.zeros(table.cost.shape, dtype=bool)
    final = np.full(len(table.ids), -1, dtype=int)
    active = np.ones(len(table.ids), dtype=bool)
    for stage, arm in enumerate(cols):
        executed[active, arm] = True
        final[active] = arm
        if stage == len(cols) - 1 or not active.any():
            break
        prefix = ladder[:stage + 1]
        scores = np.asarray(confidence(visible(table, prefix), prefix), dtype=float)
        if scores.shape != (len(table.ids),):
            raise ValueError('Confidence must return one score per item')
        # Explicit extremes also cover missing/error confidence (-inf or NaN).
        tau = thresholds[stage]
        stop = np.ones(len(table.ids), dtype=bool) if tau == -np.inf else scores >= tau
        if tau == np.inf:
            stop[:] = False
        active &= ~stop
        if not active.any():
            break
    return Decision(executed, final)


def logprob_confidence(table, executed):
    a = table.arms.index(executed[-1])
    score = table.mean_logprob[:, a].copy()
    score[np.isnan(score) | (table.error[:, a] != 0)] = -np.inf
    return score


def agreement_confidence(table, executed):
    if len(executed) == 1:
        return logprob_confidence(table, executed)
    left, right = [table.arms.index(arm) for arm in executed[-2:]]
    x, y = table.answer_sha1[:, left], table.answer_sha1[:, right]
    return ((x != None) & (y != None) & (x == y)).astype(float)


class LearnedConfidence:
    """One train-only success predictor per prefix, optionally adding input features.

    fit(table, ladder, X=None) requires router_train. For prediction with X, use
    confidence.for_inputs(X) as the cascade callback; its rows align to the table.
    Without input features the fitted object itself is the callback.
    """
    def __init__(self, learner):
        self.learner = learner

    @staticmethod
    def _features(table, executed, X):
        post, _ = posthoc_features(table, executed)
        if X is None:
            return post
        if X.shape[0] != len(table.ids):
            raise ValueError('Input features must align with table rows')
        return (sparse.hstack([X, post], format='csr') if sparse.issparse(X)
                else np.column_stack([X, post]))

    def fit(self, table, ladder, X=None):
        if table.split != 'router_train':
            raise ValueError('LearnedConfidence must fit on router_train')
        cols = _available(table, ladder)
        if not np.isfinite(table.success[:, cols]).all():
            raise ValueError('Training requires observed success for all ladder arms')
        self.ladder, self.models = list(ladder), []
        self.input_dim = None if X is None else X.shape[1]
        for k, a in enumerate(cols):
            prefix = ladder[:k + 1]
            features = self._features(visible(table, prefix), prefix, X)
            self.models.append(SuccessPredictor(self.learner).fit(
                features, table.success[:, [a]]
            ))
        return self

    def predict(self, table, executed, X=None):
        if not executed or list(executed) != self.ladder[:len(executed)]:
            raise ValueError('Executed arms must be a fitted ladder prefix')
        _available(table, executed)
        if (None if X is None else X.shape[1]) != self.input_dim:
            raise ValueError('Prediction input features must match training dimension')
        features = self._features(visible(table, executed), executed, X)
        return self.models[len(executed) - 1].predict_proba(features)[:, 0]

    def __call__(self, table, executed):
        return self.predict(table, executed)

    def for_inputs(self, X):
        return lambda table, executed: self.predict(table, executed, X)


def oracle(table):
    """Outcome-aware, never deployable: cheapest correct, else cheapest wrong arm."""
    _validation_summary(table)
    correct = table.success == 1
    choice = np.argmin(np.where(correct, table.cost, np.inf), axis=1)
    none = ~correct.any(axis=1)
    choice[none] = table.cost[none].argmin(axis=1)
    return _decision(choice, len(table.arms))
