"""Batched paired bootstrap for P4/P5; same linear Pareto definition as evaluate."""
import numpy as np


def curve_metrics(y, c, lo, hi, best_y, best_c):
    """One curve per row, returning AUC, accuracy at cost, cost at accuracy.

    Sorting/dominance/interpolation is recomputed for every bootstrap replicate.
    Columns with duplicate costs retain the highest accuracy. No concave hull.
    """
    order = np.lexsort((-y, c), axis=1)
    c, y = np.take_along_axis(c, order, 1), np.take_along_axis(y, order, 1)
    b = len(y)
    previous_y, previous_c = y[:, 0].copy(), c[:, 0].copy()
    area = np.zeros(b)
    at_cost = np.where(best_c >= previous_c, previous_y, np.nan)
    at_accuracy = np.where(best_y <= previous_y, previous_c, np.nan)
    valid = lo >= previous_c
    for k in range(1, y.shape[1]):
        next_y, next_c = y[:, k], c[:, k]
        keep = next_y > previous_y
        width = next_c - previous_c
        slope = np.divide(next_y - previous_y, width, out=np.zeros(b), where=width > 0)
        left, right = np.maximum(lo, previous_c), np.minimum(hi, next_c)
        span = np.maximum(0, right - left)
        area += np.where(keep, span * (previous_y + slope * ((left + right) / 2 - previous_c)), 0)
        hit = keep & (best_c >= previous_c) & (best_c <= next_c)
        at_cost = np.where(hit, previous_y + slope * (best_c - previous_c), at_cost)
        hit = keep & np.isnan(at_accuracy) & (best_y <= next_y)
        interpolated = previous_c + np.divide(best_y - previous_y, next_y - previous_y,
                                              out=np.zeros(b), where=keep) * width
        at_accuracy = np.where(hit, interpolated, at_accuracy)
        previous_y = np.where(keep, next_y, previous_y)
        previous_c = np.where(keep, next_c, previous_c)
    area += np.maximum(0, hi - np.maximum(lo, previous_c)) * previous_y
    at_cost = np.where(best_c >= previous_c, previous_y, at_cost)
    auc = np.divide(area, hi - lo, out=np.full(b, np.nan), where=(hi > lo) & valid)
    return dict(normalized_auc=auc, accuracy_at_best_single_cost=at_cost,
                cost_at_best_single_accuracy=at_accuracy)


def summarize(observed, samples):
    from .evaluate import _ci
    return dict(value=float(observed), ci95=_ci(samples),
                n_defined=int(np.isfinite(samples).sum()), n_boot=len(samples))


def evaluate_batched(table, vectors, curves, oracle_vec, report_policies, n_boot, seed, scope):
    from .evaluate import _ci, _point, mcnemar
    from itertools import combinations
    n = len(table.ids)
    rng = np.random.default_rng(seed)
    counts = np.empty((n_boot + 1, n), dtype=float)
    counts[0] = 1 / n
    for b in range(1, n_boot + 1):
        counts[b] = np.bincount(rng.integers(n, size=n), minlength=n) / n
    fixed_y, fixed_c = counts @ table.success, counts @ table.cost
    lo, hi = fixed_c.min(axis=1), fixed_c.max(axis=1)
    means = {name: {k: counts @ v for k, v in vec.items()} for name, vec in vectors.items()}
    best = means['best_single']
    oracle_y = counts @ oracle_vec['correct']
    fixed_metrics = curve_metrics(fixed_y, fixed_c, lo, hi, best['correct'], best['cost'])
    best_metrics = curve_metrics(np.column_stack([fixed_y[np.arange(len(lo)), fixed_c.argmin(axis=1)], best['correct']]),
                                np.column_stack([lo, best['cost']]), lo, hi, best['correct'], best['cost'])
    for name, result in report_policies.items():
        values = means[name]
        gap = np.divide(values['correct'] - best['correct'], oracle_y - best['correct'],
                        out=np.full(len(lo), np.nan), where=oracle_y != best['correct'])
        result['gap_closed'] = summarize(gap[0], gap[1:])
        for key in ['correct', 'cost']:
            result['accuracy_ci95' if key == 'correct' else 'mean_cost_ci95'] = _ci(values[key][1:])
        if name in curves:
            raw = curves[name]
            result['curve'] = [_point(**v) for v in raw]
            # Exact duplicate decisions/charges can be removed without altering
            # any replicate's frontier. Aggregate points alone are insufficient.
            unique = list({(v['correct'].tobytes(), v['cost'].tobytes()): v for v in raw}.values())
            ys = counts @ np.column_stack([v['correct'] for v in unique])
            cs = counts @ np.column_stack([v['cost'] for v in unique])
            metrics = curve_metrics(ys, cs, lo, hi, best['correct'], best['cost'])
            for key, value in metrics.items():
                result[key] = summarize(value[0], value[1:])
            for reference, metric in [('fixed_frontier', fixed_metrics), ('best_single', best_metrics)]:
                delta = metrics['normalized_auc'] - metric['normalized_auc']
                result[f'auc_minus_{reference}'] = summarize(delta[0], delta[1:])
    comparisons = {}
    pairs = combinations(vectors, 2) if scope == 'all' else [('best_single', name) for name in report_policies]
    for left, right in pairs:
        entry = dict(mcnemar=mcnemar(vectors[left]['correct'], vectors[right]['correct']))
        for key, output in [('correct', 'correctness'), ('cost', 'cost')]:
            diff = means[left][key] - means[right][key]
            entry[output] = dict(diff=float(diff[0]), ci95=_ci(diff[1:]),
                                 p_two_sided=float((1 + (np.abs(diff[1:] - diff[0]) >= abs(diff[0])).sum()) / (n_boot + 1)))
        comparisons[f'{left} - {right}'] = entry
    return comparisons
