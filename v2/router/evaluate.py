"""Frozen-policy evaluation. Costs are total tokens, including routing overhead.

Curves are Pareto-reduced, then linearly interpolated between adjacent retained
points. Randomising between the corresponding policies per item achieves these
intermediate points in expectation. This is not a concave-hull interpolation.
"""

import copy
import csv
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from .data import ROOT
from .policies import (Decision, best_single_arm, decision_cost as _decision_cost,
                       decision_success, fixed_arm, oracle)


def decision_cost(decision, table, overhead=None):
    """Replay execution tokens plus a finite, nonnegative per-item routing charge."""
    cost = _decision_cost(decision, table)
    if overhead is not None:
        overhead = np.asarray(overhead, dtype=float)
        if overhead.shape != cost.shape or not np.isfinite(overhead).all() or (overhead < 0).any():
            raise ValueError('overhead must be a finite nonnegative per-item vector')
        cost = cost + overhead
    return cost


def item_vectors(decision, table, overhead=None):
    return dict(correct=decision_success(decision, table),
                cost=decision_cost(decision, table, overhead))


def _point(correct, cost):
    return dict(n=len(correct), accuracy=float(np.mean(correct)),
                mean_cost=float(np.mean(cost)), correct=float(np.sum(correct)))


def point(decision, table, overhead=None):
    return _point(**item_vectors(decision, table, overhead))


def operating_point(val_points, best_single_val_accuracy) -> int:
    """D1.7: cheapest point reaching best-single validation accuracy (1e-12).

    Ties prefer higher accuracy, then lower index. If no point meets the target,
    maximize accuracy, then minimize cost, then index. Validation points only.
    """
    points = list(val_points)
    if not points or not np.isfinite(best_single_val_accuracy) or not all(
        np.isfinite([p['accuracy'], p['mean_cost']]).all() for p in points
    ):
        raise ValueError('Need nonempty finite validation points and target')
    eligible = [i for i, p in enumerate(points)
                if p['accuracy'] >= best_single_val_accuracy - 1e-12]
    if eligible:
        return min(eligible, key=lambda i: (points[i]['mean_cost'], -points[i]['accuracy'], i))
    return min(range(len(points)), key=lambda i: (-points[i]['accuracy'], points[i]['mean_cost'], i))


def pareto_frontier(points):
    """Return nondominated point dictionaries, sorted by mean_cost; collapse ties."""
    ordered = sorted((p for p in points if np.isfinite(p['mean_cost']) and
                      np.isfinite(p['accuracy'])),
                     key=lambda p: (p['mean_cost'], -p['accuracy']))
    frontier, best = [], -np.inf
    for p in ordered:
        if p['accuracy'] > best:
            frontier.append(p)
            best = p['accuracy']
    return frontier


def curve_accuracy_at_cost(curve, cost):
    """Linear Pareto interpolation; NaN left of cheapest, flat on the right."""
    frontier = pareto_frontier(curve)
    if not frontier:
        return float('nan')
    return float(np.interp(cost, [p['mean_cost'] for p in frontier],
                           [p['accuracy'] for p in frontier], left=np.nan))


def curve_cost_at_accuracy(curve, accuracy):
    """Minimum interpolated cost reaching accuracy; NaN if never reached.

    Targets below the cheapest retained point's accuracy cost that point's cost.
    """
    frontier = pareto_frontier(curve)
    if not frontier:
        return float('nan')
    return float(np.interp(accuracy, [p['accuracy'] for p in frontier],
                           [p['mean_cost'] for p in frontier], right=np.nan))


def normalized_auc(curve, c_lo, c_hi, oracle_curve=None):
    """Mean interpolated accuracy on [c_lo, c_hi], in accuracy units.

    Callers use the evaluated split's cheapest/most expensive fixed-arm mean
    costs as the default evaluation range (Experiment does this automatically).
    Undefined left support remains NaN: include the cheapest fixed arm explicitly
    as a curve point to extend support. Pass the fixed-arm frontier to this same
    function for the directly comparable baseline. If oracle_curve is provided,
    return {'auc', 'oracle_auc'} using exactly the same range, without division
    by oracle accuracy. The historical argument name can hold any reference curve.
    """
    if not np.isfinite([c_lo, c_hi]).all() or c_hi <= c_lo:
        raise ValueError('Require finite c_lo < c_hi')
    frontier = pareto_frontier(curve)
    xs = sorted({float(c_lo), float(c_hi)} | {
        float(p['mean_cost']) for p in frontier
        if c_lo < p['mean_cost'] < c_hi})
    ys = (np.interp(xs, [p['mean_cost'] for p in frontier],
                    [p['accuracy'] for p in frontier], left=np.nan)
          if frontier else np.full(len(xs), np.nan))
    area = float(np.trapezoid(ys, xs) / (c_hi - c_lo))
    if oracle_curve is not None:
        return dict(auc=area, oracle_auc=normalized_auc(oracle_curve, c_lo, c_hi))
    return area


def accuracy_at_best_single_cost(curve, best_single_point):
    return curve_accuracy_at_cost(curve, best_single_point['mean_cost'])


def cost_at_best_single_accuracy(curve, best_single_point):
    return curve_cost_at_accuracy(curve, best_single_point['accuracy'])


def gap_closed(policy_acc, best_single_acc, oracle_acc):
    return (float('nan') if oracle_acc == best_single_acc else
            float((policy_acc - best_single_acc) / (oracle_acc - best_single_acc)))


def arm_metrics(y, p, n_bins=10):
    """Column-aligned per-arm metric arrays and reliability_bins (one list/arm).

    Ignore NaN labels only. Predictions for observed labels must be probabilities.
    Bins are [left, right), except the last includes 1. Empty bins have NaN means.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.ndim == 1:
        y, p = y[:, None], p[:, None]
    if y.ndim != 2 or p.shape != y.shape or n_bins < 1:
        raise ValueError('Matching n x A arrays and positive n_bins required')
    result = {k: [] for k in ['auroc', 'brier', 'ece', 'n', 'base_rate', 'reliability_bins']}
    for a in range(y.shape[1]):
        mask = ~np.isnan(y[:, a])
        labels, probs = y[mask, a], p[mask, a]
        if not np.isin(labels, [0, 1]).all() or not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
            raise ValueError('Observed labels must be binary; predictions in [0,1]')
        bins = []
        indexes = np.minimum((probs * n_bins).astype(int), n_bins - 1)
        ece = 0.
        for k in range(n_bins):
            selected = indexes == k
            count = int(selected.sum())
            confidence = float(probs[selected].mean()) if count else float('nan')
            accuracy = float(labels[selected].mean()) if count else float('nan')
            bins.append(dict(left=k / n_bins, right=(k + 1) / n_bins,
                             n=count, confidence=confidence, accuracy=accuracy))
            if count:
                ece += count / len(labels) * abs(confidence - accuracy)
        values = dict(n=len(labels), base_rate=float(labels.mean()) if len(labels) else float('nan'),
                      auroc=float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float('nan'),
                      brier=float(np.mean((labels - probs) ** 2)) if len(labels) else float('nan'),
                      ece=ece if len(labels) else float('nan'), reliability_bins=bins)
        for key, value in values.items():
            result[key].append(value)
    return result


def _ci(samples):
    samples = np.asarray(samples, float)
    # Undefined curve targets are not silently dropped, which would condition CIs
    # on reaching a target. Report how many replicates are defined instead.
    return (np.quantile(samples, [.025, .975]).tolist()
            if np.isfinite(samples).all() else [float('nan'), float('nan')])


def paired_bootstrap(a, b, n_boot=10000, seed=437, stat=np.mean):
    """Percentile paired bootstrap; centered bootstrap test of zero difference.

    p_two_sided uses |bootstrap_diff - observed_diff| >= |observed_diff|,
    with a +1 Monte Carlo correction. One index draw is shared by both vectors.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 1 or a.shape != b.shape or not len(a) or n_boot < 1 or not np.isfinite([a, b]).all():
        raise ValueError('Need finite, nonempty paired vectors and positive n_boot')
    rng = np.random.default_rng(seed)
    observed = float(stat(a) - stat(b))
    values = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(len(a), size=len(a))
        values[i] = stat(a[idx]) - stat(b[idx])
    return dict(diff=observed, ci95=_ci(values),
                p_two_sided=float((1 + np.sum(np.abs(values - observed) >= abs(observed))) / (n_boot + 1)))


def bootstrap_curve_metric(fn, n_items, n_boot=10000, seed=437):
    """fn(indices) recomputes a scalar metric using frozen policy decisions.

    Resample the SAME indices for every curve/baseline/oracle in fn. Never refit
    predictors, reselect hyperparameters or regenerate decisions in this callback.
    Pareto reduction and interpolation are recomputed on every resample. Undefined
    replicates produce a NaN CI, with n_defined reported rather than dropped.
    """
    if n_items < 1 or n_boot < 1:
        raise ValueError('Positive n_items and n_boot required')
    rng = np.random.default_rng(seed)
    samples = [fn(rng.integers(n_items, size=n_items)) for _ in range(n_boot)]
    return dict(value=float(fn(np.arange(n_items))), ci95=_ci(samples),
                n_defined=int(np.isfinite(samples).sum()), n_boot=n_boot)


def mcnemar(a_correct, b_correct):
    a, b = np.asarray(a_correct), np.asarray(b_correct)
    if a.ndim != 1 or a.shape != b.shape or not np.isin([a, b], [0, 1]).all():
        raise ValueError('Need paired binary correctness vectors')
    b_count = int(((a == 1) & (b == 0)).sum())
    c_count = int(((a == 0) & (b == 1)).sum())
    p = binomtest(b_count, b_count + c_count).pvalue if b_count + c_count else 1.
    return dict(b=b_count, c=c_count, p_exact=float(p))


def learning_curve(fit_and_score, n_train, sizes=(50, 100, 200, 400, None), repeats=20, seed=437):
    """Subsample training without replacement; callback must score on router_val.

    None uses all training indices once. std is population std across repeats.
    """
    if n_train < 1 or repeats < 1:
        raise ValueError('Positive n_train and repeats required')
    rng, result = np.random.default_rng(seed), []
    for size in sizes:
        n = n_train if size is None else int(size)
        if n > n_train:
            continue
        if n < 1:
            raise ValueError('Training sizes must be positive')
        raw = [fit_and_score(np.arange(n_train) if size is None else
                             rng.choice(n_train, n, replace=False))
               for _ in range(1 if size is None else repeats)]
        keys = raw[0].keys()
        result.append(dict(n_train=n, all_training=size is None, raw=raw,
                           mean={k: float(np.mean([r[k] for r in raw])) for k in keys},
                           std={k: float(np.std([r[k] for r in raw])) for k in keys}))
    return result


def _json_safe(value):
    """Standards-compliant artifacts use null for undefined numerical metrics."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')


class Experiment:
    """Val-only selection and one audited entry point for frozen evaluation.

    select(name, candidates, score_fn) maximises score_fn(candidate, val_table);
    candidates is a mapping of serializable choice keys to candidate objects.
    The constructor and optional select(..., table=...) assert router_val. The
    callback is responsible for using only that supplied validation data.

    policies maps registered names to Decision or {'decision': Decision,
    'curve': [Decision, ...], 'overhead': vector}. A curve's chosen decision must
    be selected on val before freeze; evaluate_on never chooses a test point.
    Selection records are copied on access. Callables/models remain caller-owned.
    """
    def __init__(self, name, val_table, n_boot=10000, seed=437, experiment_dir=None):
        assert val_table.split == 'router_val', 'selection requires router_val'
        if n_boot < 1:
            raise ValueError('n_boot must be positive')
        base = (ROOT / 'results/v2/router').resolve()
        self.directory = (base / name if experiment_dir is None else Path(experiment_dir)).resolve()
        if not self.directory.is_relative_to(base) or self.directory == base:
            raise ValueError('Experiment must be under results/v2/router/<name>')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.val = val_table.select()
        self.best_arm = self.val.arms[best_single_arm(self.val)]
        self._frozen, self._locked = {}, False
        self.selection_only = set()
        self.n_boot, self.seed = n_boot, seed

    @property
    def frozen(self):
        return copy.deepcopy(self._frozen)

    def _selection_guard(self, name):
        if self._locked:
            raise RuntimeError('Cannot select after freeze')
        if name in {'best_single', 'oracle'} or '@' in name:
            raise ValueError('Reserved policy name')
        if name in self._frozen:
            raise ValueError('Need a new policy name')

    def select(self, name, candidates, score_fn, *, table=None, selection_only=False):
        self._selection_guard(name)
        val = self.val if table is None else table
        assert val.split == 'router_val', 'selection requires router_val'
        if not candidates:
            raise ValueError('Need candidates and a new policy name')
        scores = {key: float(score_fn(candidate, val)) for key, candidate in candidates.items()}
        if not np.isfinite(list(scores.values())).all():
            raise ValueError('Selection scores must be finite')
        key = max(scores, key=scores.get)
        self._frozen[name] = dict(choice=key, validation_score=scores[key])
        if selection_only:
            self.selection_only.add(name)
        return key

    def select_operating_point(self, name, val_points, decisions_or_keys, *, rule='D1.7'):
        """Freeze the D1.7 headline choice using this experiment's best single.

        Points align with a sequence of Decisions (returns an index), a sequence
        of serializable keys, or a mapping in insertion order (returns its key).
        Callers must supply points measured on router_val, including overhead.
        """
        self._selection_guard(name)
        if rule != 'D1.7':
            raise ValueError('Unknown operating-point rule; expected D1.7')
        points, choices = list(val_points), list(decisions_or_keys)
        if len(points) != len(choices):
            raise ValueError('Validation points and decisions/keys must align')
        if any(isinstance(c, Decision) for c in choices):
            if not all(isinstance(c, Decision) for c in choices):
                raise ValueError('Do not mix decisions and keys')
            choices = list(range(len(choices)))
        target = float(self.val.success[:, self.val.arms.index(self.best_arm)].mean())
        index = operating_point(points, target)
        choice = choices[index]
        json.dumps(choice)  # Reject nonserializable choice keys before recording.
        self._frozen[name] = copy.deepcopy(dict(
            choice=choice, rule=rule, validation_point=points[index],
            best_single_val_accuracy=target))
        return choice

    def freeze(self):
        self._locked = True

    def evaluate_on(self, split_table, policies, *, final_eval=False,
                    fast_bootstrap=False, comparison_scope='all', contrast_pairs=()):
        if comparison_scope not in {'all', 'best_single'}:
            raise ValueError('Unknown comparison scope')
        table = split_table
        log = self.directory / 'evaluate_calls.jsonl'
        previous = ([json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
                    if log.exists() else [])
        reserved = table.split in {'holdout', 'ood_length', 'ood_depth'}
        reason = ('freeze required' if not self._locked else
                  'final_eval=True required' if reserved and not final_eval else
                  'final evaluation already completed for this split' if reserved and any(
                      entry['split'] == table.split and entry['status'] == 'completed'
                      for entry in previous) else
                  'policy names must match frozen choices' if set(policies) != set(self._frozen) - self.selection_only else None)
        call = dict(split=table.split, time=datetime.now(timezone.utc).isoformat(),
                    n_items=len(table.ids), frozen_choices=self.frozen,
                    best_single_arm=self.best_arm, final_eval=final_eval,
                    status='refused' if reason else 'started', reason=reason)
        with log.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(_json_safe(call), allow_nan=False) + '\n')
        if reason:
            raise RuntimeError(reason)
        if not len(table.ids) or table.arms != self.val.arms:
            raise ValueError('Need nonempty evaluation with validation arm order')
        n, a = table.success.shape
        fixed = [fixed_arm(n, a, k) for k in range(a)]
        fixed_points = [dict(arm=table.arms[k], **point(d, table)) for k, d in enumerate(fixed)]
        best = fixed[table.arms.index(self.best_arm)]
        best_point = point(best, table)
        oracle_decision = oracle(table)
        oracle_vec = item_vectors(oracle_decision, table)  # zero overhead
        vectors = {'best_single': item_vectors(best, table)}
        decisions, curves, report_policies = {}, {}, {}
        for name, spec in policies.items():
            spec = dict(decision=spec) if isinstance(spec, Decision) else spec
            decision = spec['decision']
            overhead = spec.get('overhead')
            vectors[name] = item_vectors(decision, table, overhead)
            decisions[name] = decision
            report_policies[name] = point(decision, table, overhead)
            report_policies[name]['mean_overhead'] = float(np.mean(overhead)) if overhead is not None else 0.
            if 'curve' in spec:
                curves[name] = [item_vectors(d, table, overhead) for d in spec['curve']]
                if 'anchor' in spec:
                    curves[name].insert(0, item_vectors(spec['anchor'], table))
        costs = [p['mean_cost'] for p in fixed_points]
        lo, hi = min(costs), max(costs)
        for name, result in ([] if fast_bootstrap else report_policies.items()):
            vec = vectors[name]
            def gap(idx):
                return gap_closed(vec['correct'][idx].mean(), vectors['best_single']['correct'][idx].mean(),
                                  oracle_vec['correct'][idx].mean())
            result['gap_closed'] = bootstrap_curve_metric(gap, n, self.n_boot, self.seed)
            if name in curves:
                def metrics(idx):
                    curve = [_point(v['correct'][idx], v['cost'][idx]) for v in curves[name]]
                    bp = _point(vectors['best_single']['correct'][idx], vectors['best_single']['cost'][idx])
                    arm_costs = table.cost[idx].mean(axis=0)
                    return dict(normalized_auc=normalized_auc(curve, arm_costs.min(), arm_costs.max()),
                                accuracy_at_best_single_cost=accuracy_at_best_single_cost(curve, bp),
                                cost_at_best_single_accuracy=cost_at_best_single_accuracy(curve, bp))
                result['curve'] = [_point(**v) for v in curves[name]]
                for key in ['normalized_auc', 'accuracy_at_best_single_cost', 'cost_at_best_single_accuracy']:
                    result[key] = bootstrap_curve_metric(lambda idx: metrics(idx)[key], n, self.n_boot, self.seed)
        comparisons = {}
        pairs = (combinations(vectors, 2) if comparison_scope == 'all' else
                 [('best_single', name) for name in report_policies])
        for left, right in ([] if fast_bootstrap else pairs):
            # The direction is left minus right, as named in the report.
            comparisons[f'{left} - {right}'] = dict(
                correctness=paired_bootstrap(vectors[left]['correct'], vectors[right]['correct'], self.n_boot, self.seed),
                cost=paired_bootstrap(vectors[left]['cost'], vectors[right]['cost'], self.n_boot, self.seed),
                mcnemar=mcnemar(vectors[left]['correct'], vectors[right]['correct']))
        contrasts = {}
        if fast_bootstrap or contrast_pairs:
            from .bootstrap import evaluate_batched
            comparisons = evaluate_batched(table, vectors, curves, oracle_vec, report_policies,
                                           self.n_boot, self.seed, comparison_scope, contrast_pairs, contrasts)
        frontier = pareto_frontier(fixed_points)
        report = dict(evaluation_call=call,
                      evaluate_calls=[json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()],
                      fixed_points=fixed_points, fixed_frontier=frontier,
                      fixed_frontier_auc=normalized_auc(frontier, lo, hi),
                      cost_range=[lo, hi], best_single=best_point,
                      oracle=point(oracle_decision, table), policies=report_policies, comparisons=comparisons,
                      contrasts=contrasts,
                      uncertainty=dict(n_boot=self.n_boot, seed=self.seed, refit=False,
                                       note='Percentile paired item bootstrap; null CI means undefined replicates.'))
        with (self.directory / 'decisions.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['id', 'policy', 'final_arm', 'executed_arms', 'correct', 'cost'])
            output_decisions = {'best_single': best, 'oracle': oracle_decision, **decisions}
            output_vectors = {**vectors, 'oracle': oracle_vec}
            # Selected decisions only; aggregate sweep points remain in report.json.
            for name, d in output_decisions.items():
                for i, id_ in enumerate(table.ids):
                    writer.writerow([id_, name, table.arms[d.final[i]],
                                     json.dumps([table.arms[j] for j in np.flatnonzero(d.executed[i])]),
                                     output_vectors[name]['correct'][i], output_vectors[name]['cost'][i]])
        report['evaluation_status'] = 'completed'
        completed = dict(call, status='completed', time=datetime.now(timezone.utc).isoformat())
        report['evaluate_calls'].append(completed)
        write_json(self.directory / 'report.json', report)
        with log.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(_json_safe(completed), allow_nan=False) + '\n')
        return report
