"""CPU-only PNG figures; `python -m v2.router.plots sim|v1`.

Importing this module never requires matplotlib. The sim command renders saved
artifacts without refitting; v1 fits only adapter router_train/router_val data.
"""

import json
from pathlib import Path

import numpy as np

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000']


def _axes():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
    ax.set_prop_cycle(color=COLORS)
    return plt, fig, ax


def _save(plt, fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, format='png')
    plt.close(fig)
    return path


def plot_cost_accuracy(fixed_points, frontier, policy_curves, oracle_point, path,
                       *, title='Cost and accuracy'):
    from .evaluate import pareto_frontier
    plt, fig, ax = _axes()
    for p in fixed_points:
        ax.scatter(p['mean_cost'], p['accuracy'], color='#666666')
        ax.annotate(p.get('arm', ''), (p['mean_cost'], p['accuracy']), xytext=(4, 5), textcoords='offset points', fontsize=8)
    for name, curve in {'Fixed-arm frontier': frontier, **policy_curves}.items():
        curve = pareto_frontier(curve)
        ax.plot([p['mean_cost'] for p in curve], [p['accuracy'] for p in curve], marker='.', label=name)
    ax.scatter(oracle_point['mean_cost'], oracle_point['accuracy'], marker='*', s=130, color='black', label='Outcome oracle')
    ax.set(title=title, xlabel='Mean total cost (tokens/item)', ylabel='Accuracy (fraction)', ylim=(0, 1.02))
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    return _save(plt, fig, path)


def plot_reliability(bin_tables, path, *, title='Success-probability reliability'):
    """bin_tables maps series name to the reliability_bins list for one arm."""
    plt, fig, ax = _axes()
    ax.plot([0, 1], [0, 1], '--', color='#777777', label='Calibrated')
    for name, bins in bin_tables.items():
        bins = [b for b in bins if b['n']]
        ax.plot([b['confidence'] for b in bins], [b['accuracy'] for b in bins], marker='o', label=name)
    ax.set(title=title, xlabel='Mean predicted success probability', ylabel='Observed success fraction', xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    return _save(plt, fig, path)


def plot_learning_curves(curves, path, *, title='Learning curves',
                         ylabel='Validation score (fraction)'):
    """One line per score key; optional repeat std or explicit bootstrap ci95.

    learning_curve() records supply std; saved simulation size trends supply
    ci95 instead. Missing uncertainty is not fabricated as zero variance.
    """
    plt, fig, ax = _axes()
    for name, records in curves.items():
        for metric in records[0]['mean']:
            x = [r['n_train'] for r in records]
            mean = np.array([r['mean'][metric] for r in records])
            line, = ax.plot(x, mean, marker='o', label=f'{name}: {metric}')
            if all('std' in r for r in records):
                std = np.array([r['std'][metric] for r in records])
                ax.fill_between(x, mean - std, mean + std, alpha=.15, color=line.get_color())
            elif all('ci95' in r for r in records):
                ci = np.array([r['ci95'][metric] for r in records])
                ax.fill_between(x, ci[:, 0], ci[:, 1], alpha=.15, color=line.get_color())
    ax.set(title=title, xlabel='Training items (labels per arm)', ylabel=ylabel)
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    return _save(plt, fig, path)


def plot_auroc_grid(auroc, path, *, title='Success prediction AUROC', arm_labels=None):
    """auroc maps feature family -> learner -> sequence of per-arm AUROCs."""
    plt, fig, ax = _axes()
    labels, rows = [], []
    for family, learners in auroc.items():
        for learner, values in learners.items():
            labels.append(f'{family} / {learner}')
            rows.append(values)
    matrix = np.asarray(rows, float)
    fig.set_size_inches(7, max(4.5, .38 * len(rows) + 1.6))
    plotted = ax.imshow(matrix, aspect='auto', vmin=0, vmax=1, cmap='cividis')
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(matrix.shape[1]), arm_labels if arm_labels is not None
                  else [f'Arm {a}' for a in range(matrix.shape[1])])
    ax.set(title=title, xlabel='Execution arm', ylabel='Feature family / learner')
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f'{value:.2f}' if np.isfinite(value) else 'N/A', ha='center', va='center',
                color='black' if value > .5 else 'white', fontsize=8)
    fig.colorbar(plotted, ax=ax, label='AUROC (fraction)')
    return _save(plt, fig, path)


def render_sim():
    """Render four figures from saved simulation reports; no model fitting."""
    from .data import ROOT
    base = ROOT / 'results/v2/router/sim_check'
    summary = json.loads((base / 'report.json').read_text(encoding='utf-8'))
    report = json.loads((base / 'cascades_independent/report.json').read_text(encoding='utf-8'))
    dest = base / 'figures'
    paths = [plot_cost_accuracy(
        report['fixed_points'], report['fixed_frontier'],
        {k: v['curve'] for k, v in report['policies'].items()}, report['oracle'],
        dest / 'cost_accuracy.png', title='Simulation: cost and accuracy (sim_test)')]
    bins = {method: metrics['reliability_bins'][0]
            for method, metrics in summary['calibration']['logreg_per_arm'].items()}
    paths.append(plot_reliability(bins, dest / 'reliability.png',
                                 title='Simulation: logreg, arm 0 (sim_test)'))
    curves = {}
    for learner in ['logreg', 'hgb', 'mlp']:
        curves[learner] = []
        for n in sorted(summary['sizes'], key=int):
            metric = summary['sizes'][n]['expected'][f'{learner}_per_arm']['learned']
            curves[learner].append(dict(n_train=int(n), mean={'AUC': metric['value']},
                                        ci95={'AUC': metric['ci95']}))
    paths.append(plot_learning_curves(
        curves, dest / 'learning_curve.png',
        title='Simulation: size trends (paired bootstrap 95% bands)',
        ylabel='Expected normalized cost-accuracy AUC (sim_test)'))
    grid = {'Difficulty': {name: metrics['before']['auroc']
                           for name, metrics in summary['calibration'].items()}}
    paths.append(plot_auroc_grid(grid, dest / 'auroc.png',
                                title='Simulation: uncalibrated AUROC (sim_test)',
                                arm_labels=[p['arm'] for p in report['fixed_points']]))
    return paths


def render_v1():
    """Validation illustrations, not final estimates; select F3 by mean val AUROC.

    All preprocessing and learners fit router_train. Layer/pooling selection uses
    router_val per learner, averaging AUROC equally across the two arms. The same
    validation data are displayed, so these are selection diagnostics. No audit
    rows enter feature construction, fitting, selection, or scoring.
    """
    from .data import (ROOT, TfidfFeatures, available_layer_keys, load_f3, load_items,
                       load_labels, main_table, structural_features)
    from .evaluate import (Experiment, arm_metrics, learning_curve, pareto_frontier,
                           point, write_json)
    from .models import CostPredictor, SuccessPredictor
    from .policies import fixed_arm, lambda_grid, oracle, predict_then_route, threshold_route

    base = ROOT / 'results/v2/router/v1_adapter'
    dest = base / 'figures'
    dest.mkdir(parents=True, exist_ok=True)
    tables = load_labels(base / 'scan_v1_labels.csv')
    train, val = [main_table(tables[s], expected_n_arms=None)
                  for s in ['router_train', 'router_val']]
    items = load_items(base / 'scan_v1_items.jsonl')
    texts = [[items[id_]['input'] for id_ in table.ids] for table in [train, val]]
    xt, xv = [structural_features(text, 'scan')[0] for text in texts]
    tfidf = TfidfFeatures().fit(texts[0])
    families = {'F1': (xt, xv), 'F2': tuple(tfidf.transform(t) for t in texts)}
    learners = ['logreg', 'hgb', 'mlp']
    params = {'mlp': {'max_iter': 1500}}
    grid, probabilities = {}, {}
    for family, (xtrain, xval) in families.items():
        grid[family] = {}
        for learner in learners:
            model = SuccessPredictor(learner, **params.get(learner, {})).fit(xtrain, train.success)
            pred = model.predict_proba(xval)
            grid[family][learner] = arm_metrics(val.success, pred)['auroc']
            probabilities[f'{family}/{learner}'] = pred

    f3_dir = ROOT / 'results/v2/features/scan_v1'
    candidates = {learner: [] for learner in learners}
    selected = {}
    for layer in available_layer_keys(f3_dir, 'qwen3:8b'):
        xtrain = load_f3(f3_dir, train.ids, layer, 'own', train.arms)
        xval = load_f3(f3_dir, val.ids, layer, 'own', val.arms)
        for learner in learners:
            model = SuccessPredictor(learner, **params.get(learner, {})).fit(xtrain, train.success)
            pred = model.predict_proba(xval)
            auroc = arm_metrics(val.success, pred)['auroc']
            score = float(np.mean(auroc))
            candidates[learner].append(dict(layer_key=layer, auroc=auroc, mean_auroc=score))
            if learner not in selected or score > selected[learner]['mean_auroc']:
                selected[learner] = candidates[learner][-1]
                probabilities[f'F3-own/{learner}'] = pred
        print(f'v1 F3 validation layer scored: {layer}', flush=True)
    grid['F3-own'] = {learner: selected[learner]['auroc'] for learner in learners}

    pv = probabilities['F1/logreg']
    cv = CostPredictor('rf_joint').fit(xt, train.cost).predict(xv)
    lambdas = lambda_grid(cv)
    thresholds = [0, .25, .5, .6, .7, .8, .9, .95, 1.]
    decisions = {
        'F1 logreg lambda': [predict_then_route(pv, cv, lam) for lam in lambdas],
        'F1 logreg threshold': [threshold_route(pv, cv, tau) for tau in thresholds],
    }
    fixed = [dict(arm=arm, **point(fixed_arm(len(val.ids), len(val.arms), a), val))
             for a, arm in enumerate(val.arms)]
    anchor = min(fixed, key=lambda p: p['mean_cost'])
    curves = {name: [point(d, val) for d in ds] for name, ds in decisions.items()}
    experiment = Experiment('v1_adapter/figures', val)
    for name, ds in decisions.items():
        experiment.select_operating_point(name, curves[name], ds, rule='D1.7')
    experiment.freeze()
    bins = dict(zip(val.arms, arm_metrics(val.success, pv)['reliability_bins']))

    def fit_score(idx):
        model = SuccessPredictor('logreg').fit(xt[idx], train.success[idx])
        auc = arm_metrics(val.success, model.predict_proba(xv))['auroc']
        return dict(zip(val.arms, auc))

    learning = learning_curve(fit_score, len(train.ids), sizes=(16, 32, 50, 72, None))
    metrics = dict(
        note='Historical v1 train/validation diagnostics only; no generalization claim.',
        train_ids=train.ids, validation_ids=val.ids, arms=val.arms,
        seed=437, learner_mode='per_arm', learner_params=params,
        f3_selection_rule='Maximum mean router_val AUROC across arms per learner; first key wins ties.',
        f3_candidates=candidates, f3_selected=selected,
        f3_caveat='HF probe/Ollama quantization mismatch; validation used for layer selection and display.',
        auroc=grid, validation_probabilities=probabilities,
        predicted_cost=cv, lambdas=lambdas, thresholds=thresholds,
        fixed_points=fixed, fixed_frontier=pareto_frontier(fixed),
        curves={name: [anchor] + curve for name, curve in curves.items()},
        oracle=point(oracle(val), val), selections=experiment.frozen,
        reliability=bins, learning_curve=learning,
        cost_note='F1 routing uses no activation probe; F3 is shown only in the AUROC grid.')
    write_json(dest / 'metrics.json', metrics)
    paths = [plot_cost_accuracy(fixed, metrics['fixed_frontier'], metrics['curves'],
                               metrics['oracle'], dest / 'cost_accuracy.png',
                               title='Historical v1: cost and accuracy (router_val)'),
             plot_reliability(bins, dest / 'reliability.png',
                              title='Historical v1: F1 logreg reliability (router_val)'),
             plot_learning_curves({'F1 logreg': learning}, dest / 'learning_curve.png',
                                  title='Historical v1: learning curve (repeat SD bands)',
                                  ylabel='Validation AUROC (fraction)'),
             plot_auroc_grid(grid, dest / 'auroc.png', arm_labels=val.arms,
                             title='Historical v1: AUROC (F3 selected on router_val)')]
    return paths


def main():
    import argparse
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['sim', 'v1'])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        paths = render_sim() if args.dataset == 'sim' else render_v1()
    for path in paths:
        print(path.as_posix())


if __name__ == '__main__':
    main()
