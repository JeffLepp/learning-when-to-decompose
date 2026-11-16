"""Frozen predictor metrics and paired H3 item-bootstrap comparisons."""
import itertools

import numpy as np

from .bootstrap import item_counts, summarize
from .evaluate import arm_metrics

FAMILY_CONTRASTS = [('F3-own', 'F2'), ('F3-own', 'F1'), ('F2', 'F1'),
                    ('F3-own', 'F3-small'), ('F3-small', 'F2')]


def bootstrap_auroc(y, scores, counts):
    """Weighted Mann-Whitney AUROC, including ties, for every item resample.

    Single-class arms are omitted from that replicate's mean and counted.
    Sorting scores once is equivalent to expanding sampled items with duplicates.
    """
    result = np.full((len(counts), y.shape[1]), np.nan)
    for arm in range(y.shape[1]):
        observed = np.flatnonzero(np.isfinite(y[:, arm]))
        order = observed[np.argsort(scores[observed, arm], kind='stable')]
        if not len(order):
            continue
        starts = np.r_[0, 1 + np.flatnonzero(np.diff(scores[order, arm]))]
        for start in range(0, len(counts), 256):
            weights = counts[start:start + 256, order]
            pos = np.add.reduceat(weights * y[order, arm], starts, axis=1)
            neg = np.add.reduceat(weights * (1 - y[order, arm]), starts, axis=1)
            denominator = pos.sum(axis=1) * neg.sum(axis=1)
            numerator = (pos * (np.cumsum(neg, axis=1) - .5 * neg)).sum(axis=1)
            result[start:start + len(weights), arm] = np.divide(
                numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)
    valid = np.isfinite(result).sum(axis=1)
    means = np.divide(np.nansum(result, axis=1), valid,
                      out=np.full(len(counts), np.nan), where=valid > 0)
    return means, dict(single_class_arm_replicates=int(np.isnan(result[1:]).sum()),
                       single_class_by_arm=np.isnan(result[1:]).sum(axis=0).tolist(),
                       all_arms_undefined_replicates=int((valid[1:] == 0).sum()))


def predictor_report(state, table, matrices, n_boot, seed=437):
    """No fit/calibration/selection: identical helper for optimistic P4 and P5."""
    counts = item_counts(len(table.ids), n_boot, seed)
    metrics, samples = {}, {}
    for name, model in state['predictors'].items():
        x = matrices[state['predictor_families'][name]]
        raw = model._raw(x)
        calibrated = model.predict_proba(x)
        raw_metrics = arm_metrics(table.success, raw)
        calibrated_metrics = arm_metrics(table.success, calibrated)
        samples[name], skipped = bootstrap_auroc(table.success, raw, counts)
        metrics[name] = dict(arms=table.arms, raw=raw_metrics, calibrated=calibrated_metrics,
                             auroc=raw_metrics['auroc'], brier=calibrated_metrics['brier'],
                             ece=calibrated_metrics['ece'], mean_auroc=float(samples[name][0]),
                             bootstrap_skipped=skipped,
                             calibration='Frozen full-validation calibrator; AUROC headline uses raw scores')
    contrasts, skipped = {}, []
    for learner, mode in itertools.product(['logreg', 'hgb', 'mlp'], ['per_arm', 'shared']):
        for left, right in FAMILY_CONTRASTS:
            a, b = f'{left}/{learner}/{mode}', f'{right}/{learner}/{mode}'
            name = f'{a} - {b}'
            missing = [k for k in [a, b] if k not in samples]
            if missing:
                skipped.append(f'{name}: absent frozen predictor(s): {", ".join(missing)}')
                continue
            delta = samples[a] - samples[b]
            contrasts[name] = summarize(delta[0], delta[1:])
    return dict(predictor_metrics=metrics, predictor_contrasts=contrasts,
                predictor_contrasts_skipped=skipped,
                predictor_uncertainty=dict(n_boot=n_boot, seed=seed, refit=False,
                    note='Shared item indices across all families and arms; single-class arms skipped and counted. All-arm undefined replicates yield null CI.'))


def policy_contrasts(state):
    pairs = [('selected/overall', 'best_single'), ('selected/overall', 'fixed_frontier')]
    if 'selected/cascade' in state['recipes']:
        pairs.append(('selected/cascade', 'selected/overall'))
    pairs.extend((f'selected/{family}', 'selected/F1') for family in state['families'])
    return pairs
