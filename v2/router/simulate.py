"""CPU-only routing sanity checks; synthetic sim_test is not a project holdout.

Run: .venv/Scripts/python.exe -m v2.router.simulate [--plots]
All design constants are fixed here, before sim_test evaluation. Expected-accuracy
curves integrate known Bernoulli probabilities over the generated items, not their
sampled labels. They are finite-lambda-sweep Bayes benchmarks, not global claims
about real models. Post-hoc cascades have extra information and can exceed them.
"""

import argparse
import hashlib
import importlib.util
import json

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .data import ArmTable, NUMERIC, ROOT
from .evaluate import (Experiment, arm_metrics, bootstrap_curve_metric, normalized_auc,
                       paired_bootstrap, pareto_frontier, point, write_json)
from .models import CostPredictor, SuccessPredictor
from .policies import (LearnedConfidence, agreement_confidence, cascade, decision_success,
                       fixed_arm, logprob_confidence, predict_then_route)


# name, logistic slope, ability threshold, mean total tokens. Arm 1 is dominated.
ARMS = (('sim0|S0', 10., .28, 100.), ('sim1|S0', 10., .28, 220.),
        ('sim2|S0', 10., .62, 350.), ('sim3|S0', 10., .95, 650.))
LAMBDAS = np.r_[0., np.geomspace(1e-5, .02, 60)]


def simulate(n, seed=437, arms=ARMS, *, null=False):
    """Return (split->ArmTable, split->X, truth) with disjoint item IDs.

    n is training count or a mapping of split names to counts. Integer n uses
    1000 validation and 2000 sim_test items. Independent SeedSequence streams
    keep val/test unchanged across training sizes. null replaces observed d with
    independent noise; its Bayes pre-generation probabilities are arm marginals.
    Costs are integer mean_a + symmetric integer noise, independent of outcomes.
    """
    sizes = dict(router_train=n, router_val=1000, sim_test=2000) if isinstance(n, int) else dict(n)
    if set(sizes) != {'router_train', 'router_val', 'sim_test'} or min(sizes.values()) < 1:
        raise ValueError('Provide positive router_train/router_val/sim_test sizes')
    if len(arms) < 4:
        raise ValueError('Simulation requires at least four arms')
    tables, features, truth = {}, {}, dict(null=null, arms=[list(a) for a in arms], splits={})
    for split, stream in zip(['router_train', 'router_val', 'sim_test'],
                             np.random.SeedSequence(seed).spawn(3)):
        rng, count = np.random.default_rng(stream), sizes[split]
        difficulty = rng.random(count)
        p = np.column_stack([expit(k * (theta - difficulty)) for _, k, theta, _ in arms])
        # Independent arm outcomes permit complementarity and a nontrivial oracle.
        # This matters especially for the null leakage guard: if every arm were
        # nested inside the best single arm, even leaking labels could not beat it.
        success = (rng.random(p.shape) < p).astype(float)
        costs = np.array([a[3] for a in arms]) + rng.integers(-8, 9, size=p.shape)
        observed = rng.random(count) if null else difficulty
        x = np.column_stack([observed + rng.normal(0, .015, count), rng.normal(size=(count, 3))])
        logp = -.15 - 1.5 * (1 - success) + rng.normal(0, .4, p.shape)
        hashes = np.empty(p.shape, dtype=object)
        ids = [f'{split}-{i}' for i in range(count)]
        for i in range(count):
            for a in range(len(arms)):
                answer = f'{ids[i]}-correct' if success[i, a] else f'{ids[i]}-wrong-{a}'
                hashes[i, a] = hashlib.sha1(answer.encode()).hexdigest()
        tables[split] = ArmTable(ids, [a[0] for a in arms], split, success, costs.astype(float),
                                costs.astype(float) - 20, np.full(p.shape, 20.), costs / 1000,
                                np.ones(p.shape), np.zeros(p.shape), logp, logp - .5, hashes)
        features[split] = x
        # Exact marginal over d~Uniform(0,1) for the null conditional optimum.
        marginal = np.array([(np.logaddexp(0, k * theta) - np.logaddexp(0, k * (theta - 1))) / k
                             for _, k, theta, _ in arms])
        truth['splits'][split] = dict(difficulty=difficulty, probabilities=p,
                                     bayes_probabilities=np.tile(marginal, (count, 1)) if null else p,
                                     mean_cost=np.tile([a[3] for a in arms], (count, 1)))
    return tables, features, truth


def label_rows(tables):
    """Serialize simulated ArmTables using the real label-table column contract."""
    for table in tables.values():
        for i, id_ in enumerate(table.ids):
            for a, arm in enumerate(table.arms):
                model, strategy = arm.split('|')
                yield dict(id=id_, split=table.split, model=model, strategy=strategy,
                           correct=table.success[i, a], total_tokens=table.cost[i, a],
                           prompt_tokens=table.prompt_tokens[i, a], completion_tokens=table.completion_tokens[i, a],
                           wall_seconds=table.seconds[i, a], n_calls=table.n_calls[i, a],
                           error='' if not table.error[i, a] else 'error',
                           mean_logprob=table.mean_logprob[i, a], min_logprob=table.min_logprob[i, a],
                           answer_sha1=table.answer_sha1[i, a])


def routing_curve(probabilities, costs, lambdas=LAMBDAS):
    return [predict_then_route(probabilities, costs, lam) for lam in lambdas]


def expected_vectors(decisions, probabilities, costs):
    rows = np.arange(len(probabilities))
    return (np.column_stack([probabilities[rows, d.final] for d in decisions]),
            np.column_stack([np.where(d.executed, costs, 0).sum(axis=1) for d in decisions]))


def vector_curve(y, c, indices):
    # Counts preserve paired resampling while avoiding large repeated row copies.
    counts = np.bincount(indices, minlength=len(y))
    return [dict(accuracy=float(a), mean_cost=float(b))
            for a, b in zip(counts @ y / len(indices), counts @ c / len(indices))]


def expected_auc(decisions, probabilities, costs, indices=None):
    y, c = expected_vectors(decisions, probabilities, costs)
    idx = np.arange(len(y)) if indices is None else indices
    means = costs[idx].mean(axis=0)
    return normalized_auc(vector_curve(y, c, idx), means.min(), means.max())


def availability_audit(table, ladder, confidence, thresholds, seed=437):
    """For EVERY tau rerun after changing labels and all unexecuted signals.

    Later arms that an item actually executes must retain their original signals:
    changing them can legitimately change its later stopping decision. This audit
    randomises each item's never-executed arms, and separately compares every
    prefix confidence after randomising all not-yet-executed arms at that prefix.
    """
    rng = np.random.default_rng(seed)
    cols = [table.arms.index(a) for a in ladder]
    for threshold in thresholds:
        decision = cascade(table, ladder, confidence, threshold)
        changed = table.select()
        changed.success[:] = rng.integers(0, 2, changed.success.shape)
        unavailable = ~decision.executed
        for name in NUMERIC:
            if name != 'success':
                getattr(changed, name)[unavailable] = rng.random(unavailable.sum())
        changed.answer_sha1[unavailable] = 'random-unexecuted'
        again = cascade(changed, ladder, confidence, threshold)
        np.testing.assert_array_equal(again.final, decision.final)
        np.testing.assert_array_equal(again.executed, decision.executed)
        for k in range(1, len(ladder)):
            prefix = ladder[:k]
            changed = table.select()
            changed.success[:] = rng.integers(0, 2, changed.success.shape)
            future = [a for a in range(len(table.arms)) if a not in cols[:k]]
            for name in NUMERIC:
                if name != 'success':
                    getattr(changed, name)[:, future] = rng.random((len(table.ids), len(future)))
            changed.answer_sha1[:, future] = 'random-future'
            np.testing.assert_array_equal(confidence(table, prefix), confidence(changed, prefix))
    return dict(thresholds_checked=len(thresholds), prefix_checks=len(thresholds) * (len(ladder) - 1), passed=True)


def _expected_comparison(curve, optimal, fixed, truth, n_boot, seed):
    p, c = truth['bayes_probabilities'], truth['mean_cost']
    y, cost = expected_vectors(curve, p, c)
    yo, co = expected_vectors(optimal, p, c)
    yf, cf = expected_vectors(fixed, p, c)
    lo, hi = c[0].min(), c[0].max()
    def metric(yy, cc, idx):
        return normalized_auc(vector_curve(yy, cc, idx), lo, hi)
    result = {}
    for name, yy, cc in [('learned', y, cost), ('optimal', yo, co), ('frontier', yf, cf)]:
        result[name] = bootstrap_curve_metric(lambda idx: metric(yy, cc, idx), len(p), n_boot, seed)
    result['learned_minus_optimal'] = bootstrap_curve_metric(
        lambda idx: metric(y, cost, idx) - metric(yo, co, idx), len(p), n_boot, seed)
    result['learned_minus_frontier'] = bootstrap_curve_metric(
        lambda idx: metric(y, cost, idx) - metric(yf, cf, idx), len(p), n_boot, seed)
    return result


def run_check(n_boot=200, seed=437):
    """All six learner/mode combinations; val selection precedes each evaluation.

    Uses a prescribed lambda grid, and chooses an operating point on val by
    accuracy - .001 * tokens. Calibration diagnostics are predeclared, not selected
    using sim_test. Models fit only router_train; calibrators fit router_val.
    """
    destination = ROOT / 'results/v2/router/sim_check'
    destination.mkdir(parents=True, exist_ok=True)
    summary = dict(seed=seed, n_boot=n_boot, sizes={}, calibration={},
                   protocol=dict(lambdas=LAMBDAS.tolist(), operating_utility_lambda=.001,
                                 expected_auc='Known probabilities and mean costs, finite frozen lambda sweep',
                                 empirical_auc='Sampled binary labels and realised token costs',
                                 left_anchor='Each routing/cascade curve explicitly includes the cheapest fixed arm',
                                 synthetic_overhead_tokens=0, bootstrap_refit=False,
                                 logreg_acceptance_tolerance=.02,
                                 outcome_coupling='Independent Bernoulli outcomes across arms'))
    for n_train in [200, 600, 5000]:
        tables, x, truth = simulate(n_train, seed)
        train, val, test = [tables[s] for s in ['router_train', 'router_val', 'sim_test']]
        experiment = Experiment(f'sim_check/n{n_train}_independent', val, n_boot=n_boot, seed=seed)
        costs = CostPredictor('mean').fit(x['router_train'], train.cost)
        policies, fitted, selection = {}, {}, {}
        for learner in ['logreg', 'hgb', 'mlp']:
            for mode in ['per_arm', 'shared']:
                name = f'{learner}_{mode}'
                params = {'max_iter': 1500} if learner == 'mlp' else {}
                model = SuccessPredictor(learner, mode, seed=seed, **params).fit(x['router_train'], train.success)
                fitted[name] = model
                val_curve = routing_curve(model.predict_proba(x['router_val']), costs.predict(x['router_val']))
                key = experiment.select(name, dict(enumerate(val_curve)),
                                        lambda d, t: point(d, t)['accuracy'] - .001 * point(d, t)['mean_cost'])
                selection[name] = key
        # Freeze ALL choices before constructing or scoring sim_test decisions.
        experiment.freeze()
        fixed = [fixed_arm(len(test.ids), len(test.arms), a) for a in range(len(test.arms))]
        for name, model in fitted.items():
            curve = routing_curve(model.predict_proba(x['sim_test']), costs.predict(x['sim_test']))
            policies[name] = dict(decision=curve[selection[name]], curve=[fixed[0]] + curve)
        empirical = experiment.evaluate_on(test, policies)
        optimal = [fixed[0]] + routing_curve(truth['splits']['sim_test']['bayes_probabilities'],
                                             truth['splits']['sim_test']['mean_cost'])
        comparisons = {name: _expected_comparison(spec['curve'], optimal, fixed,
                                                 truth['splits']['sim_test'], n_boot, seed)
                       for name, spec in policies.items()}
        summary['sizes'][str(n_train)] = dict(expected=comparisons,
                                              empirical_report=str(experiment.directory.relative_to(ROOT) / 'report.json'),
                                              empirical_auc={k: v['normalized_auc'] for k, v in empirical['policies'].items()},
                                              oracle=empirical['oracle'])
        print(f'n_train={n_train}: ' + ', '.join(
            f"{name}={r['learned']['value']:.6f}" for name, r in comparisons.items()), flush=True)
        if n_train == 5000:
            # Calibration is evaluated after each fit on validation, independently
            # of routing. Before/after use the same trained underlying predictor.
            for name, model in fitted.items():
                before = arm_metrics(test.success, model.predict_proba(x['sim_test']))
                summary['calibration'][name] = {'before': before}
                for method in ['platt', 'isotonic']:
                    model.calibrate(x['router_val'], val.success, method)
                    summary['calibration'][name][method] = arm_metrics(test.success, model.predict_proba(x['sim_test']))
            summary['cascade'] = _cascade_check(train, val, test, policies['logreg_per_arm'], n_boot, seed)
    summary['null'] = _null_check(n_boot, seed)
    summary['cascade_positive_control'] = cascade_positive_control(n_boot=n_boot, seed=seed)
    gaps = [summary['sizes'][str(n)]['expected']['logreg_per_arm']['optimal']['value'] -
            summary['sizes'][str(n)]['expected']['logreg_per_arm']['learned']['value'] for n in [200, 600, 5000]]
    summary['acceptance'] = dict(logreg_gaps=gaps, gap_shrinks=bool(np.all(np.diff(gaps) < 0)),
                                 logreg_within_002=abs(gaps[-1]) <= .02,
                                 null_no_significant_gain=summary['null']['no_significant_gain'])
    summary['acceptance']['cascade_positive_control'] = summary['cascade_positive_control']['passed']
    write_json(destination / 'report.json', summary)
    return summary


def _cascade_check(train, val, test, pregeneration, n_boot, seed, *,
                   name='sim_check/cascades_independent', experiment_dir=None,
                   pregeneration_val=None):
    ladder = [train.arms[i] for i in [0, 2, 3]]  # Drop known dominated synthetic arm.
    learned = LearnedConfidence('logreg').fit(train, ladder)
    specifications = {
        'logprob': (logprob_confidence, list(np.r_[-np.inf, np.linspace(-2.5, .5, 19), np.inf])),
        'agreement': (agreement_confidence, [[float(t), .5] for t in np.r_[-np.inf, np.linspace(-2.5, .5, 19), np.inf]]),
        'learned': (learned, list(np.linspace(0, 1, 21))),
    }
    experiment = Experiment(name, val, n_boot=n_boot, seed=seed, experiment_dir=experiment_dir)
    selection = {}
    for name, (confidence, thresholds) in specifications.items():
        candidates = {i: cascade(val, ladder, confidence, tau) for i, tau in enumerate(thresholds)}
        selection[name] = experiment.select(name, candidates,
                                            lambda d, t: point(d, t)['accuracy'] - .001 * point(d, t)['mean_cost'])
    if pregeneration_val is None:
        experiment.select('pregeneration', {'frozen_logreg_per_arm': None}, lambda _, t: 0.)
    else:
        key = experiment.select('pregeneration', dict(enumerate(pregeneration_val)),
                                lambda d, t: point(d, t)['accuracy'] - .001 * point(d, t)['mean_cost'])
    experiment.freeze()
    policies, audits = {'pregeneration': pregeneration}, {}
    if pregeneration_val is not None:
        # curve[0] is the fixed-arm left anchor, not a lambda candidate.
        policies['pregeneration'] = dict(pregeneration, decision=pregeneration['curve'][key + 1])
    for name, (confidence, thresholds) in specifications.items():
        curve = [cascade(test, ladder, confidence, tau) for tau in thresholds]
        policies[name] = dict(decision=curve[selection[name]],
                              curve=[fixed_arm(len(test.ids), len(test.arms), 0)] + curve)
        audits[name] = availability_audit(test, ladder, confidence, thresholds, seed)
    report = experiment.evaluate_on(test, policies)
    # Paired curve differences retain the same frozen decisions and resample once.
    raw = {}
    for name, spec in policies.items():
        raw[name] = expected_vectors(spec['curve'], test.success, test.cost)
    fixed = [fixed_arm(len(test.ids), len(test.arms), a) for a in range(len(test.arms))]
    raw['fixed_frontier'] = expected_vectors(fixed, test.success, test.cost)
    differences = {}
    for baseline in ['pregeneration', 'fixed_frontier']:
        baseline_y, baseline_c = raw[baseline]
        differences[baseline] = {}
        for policy in specifications:
            y, c = raw[policy]
            def difference(idx):
                means = test.cost[idx].mean(axis=0)
                return (normalized_auc(vector_curve(y, c, idx), means.min(), means.max()) -
                        normalized_auc(vector_curve(baseline_y, baseline_c, idx), means.min(), means.max()))
            differences[baseline][policy] = bootstrap_curve_metric(difference, len(test.ids), n_boot, seed)
    return dict(auc={name: p['normalized_auc'] for name, p in report['policies'].items()},
                chosen_points={name: {k: p[k] for k in ['accuracy', 'mean_cost']} for name, p in report['policies'].items()},
                fixed_frontier_auc=report['fixed_frontier_auc'],
                auc_minus_pregeneration=differences['pregeneration'],
                auc_minus_fixed_frontier=differences['fixed_frontier'], availability_audit=audits,
                report=str(experiment.directory.relative_to(ROOT) / 'report.json'))


def cascade_positive_control(n=5000, n_boot=200, seed=437, *, experiment_dir=None):
    """Same outcomes/costs/post-hoc signal as the main regime, uninformative X.

    No signal-strength tuning: null=True only replaces pre-generation difficulty
    with independent noise. Both cascades must beat routing and the fixed frontier
    with positive paired-bootstrap lower bounds to pass this diagnostic.
    """
    tables, x, _ = simulate(n, seed, null=True)
    train, val, test = [tables[s] for s in ['router_train', 'router_val', 'sim_test']]
    model = SuccessPredictor('logreg').fit(x['router_train'], train.success)
    costs = CostPredictor('mean').fit(x['router_train'], train.cost)
    val_curve = routing_curve(model.predict_proba(x['router_val']), costs.predict(x['router_val']))
    test_curve = routing_curve(model.predict_proba(x['sim_test']), costs.predict(x['sim_test']))
    result = _cascade_check(train, val, test,
                            {'curve': [fixed_arm(len(test.ids), len(test.arms), 0)] + test_curve},
                            n_boot, seed, name='sim_check/cascade_positive_control_independent',
                            experiment_dir=experiment_dir, pregeneration_val=val_curve)
    result.update(n_train=len(train.ids), n_val=len(val.ids), n_test=len(test.ids), null=True,
                  signal='Unchanged logprob = -0.15 - 1.5 * (1 - success) + Normal(0, 0.4)',
                  passed=all(result[comparison][policy]['ci95'][0] > 0
                             for comparison in ['auc_minus_pregeneration', 'auc_minus_fixed_frontier']
                             for policy in ['logprob', 'learned']))
    return result


def _null_check(n_boot, seed):
    tables, x, _ = simulate(5000, seed, null=True)
    train, val, test = [tables[s] for s in ['router_train', 'router_val', 'sim_test']]
    model = SuccessPredictor('logreg').fit(x['router_train'], train.success)
    costs = CostPredictor('mean').fit(x['router_train'], train.cost)
    experiment = Experiment('sim_check/null_independent', val, n_boot=n_boot, seed=seed)
    val_curve = routing_curve(model.predict_proba(x['router_val']), costs.predict(x['router_val']))
    # Max accuracy is the predeclared null hypothesis's operating objective.
    key = experiment.select('logreg', dict(enumerate(val_curve)), lambda d, t: point(d, t)['accuracy'])
    experiment.freeze()
    curve = routing_curve(model.predict_proba(x['sim_test']), costs.predict(x['sim_test']))
    report = experiment.evaluate_on(test, {'logreg': dict(decision=curve[key],
                             curve=[fixed_arm(len(test.ids), len(test.arms), 0)] + curve)})
    best = test.arms.index(experiment.best_arm)
    comparison = paired_bootstrap(decision_success(curve[key], test), test.success[:, best], n_boot, seed)
    return dict(learned_accuracy=report['policies']['logreg']['accuracy'],
                best_single_accuracy=report['best_single']['accuracy'],
                oracle_accuracy=report['oracle']['accuracy'],
                learned_minus_best=comparison, no_significant_gain=comparison['ci95'][0] <= 0,
                report='results/v2/router/sim_check/null_independent/report.json')


def example_plots(summary):
    if importlib.util.find_spec('matplotlib') is None:
        print('matplotlib not installed; skipped figures')
        return
    from .plots import plot_auroc_grid, plot_cost_accuracy, plot_learning_curves, plot_reliability
    directory = ROOT / 'results/v2/router/sim_check/figures'
    report = json.loads((directory.parent / 'cascades_independent/report.json').read_text(encoding='utf-8'))
    plot_cost_accuracy(report['fixed_points'], report['fixed_frontier'],
                       {k: v['curve'] for k, v in report['policies'].items()}, report['oracle'], directory / 'cost_accuracy.png')
    bins = {method: metrics['reliability_bins'][0] for method, metrics in summary['calibration']['logreg_per_arm'].items()}
    plot_reliability(bins, directory / 'reliability.png')
    # These are measured size trends with bootstrap standard error unavailable;
    # don't mislabel CI widths as repeat standard deviations. No learning plot
    # is fabricated here: learning_curve() supplies the real repeated val scores.
    tables, x, _ = simulate(dict(router_train=600, router_val=300, sim_test=10))
    from .evaluate import learning_curve
    def fit_score(idx):
        model = SuccessPredictor('logreg').fit(x['router_train'][idx], tables['router_train'].success[idx])
        return {'brier': float(np.mean((model.predict_proba(x['router_val']) - tables['router_val'].success) ** 2))}
    curves = learning_curve(fit_score, 600, repeats=5)
    write_json(directory.parent / 'learning_curve.json', curves)
    plot_learning_curves({'logreg Brier (val)': curves}, directory / 'learning_curve.png')
    grid = {'observed difficulty': {name: metrics['before']['auroc'] for name, metrics in summary['calibration'].items()}}
    plot_auroc_grid(grid, directory / 'auroc.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    # Prevent small CPU fits from oversubscribing the shared machine.
    with threadpool_limits(limits=1):
        summary = run_check()
        if args.plots:
            example_plots(summary)
    print(json.dumps(summary['acceptance'], indent=2))


if __name__ == '__main__':
    main()
