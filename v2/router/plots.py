"""Optional PNG figures. Importing this module never requires matplotlib."""

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


def plot_cost_accuracy(fixed_points, frontier, policy_curves, oracle_point, path):
    from .evaluate import pareto_frontier
    plt, fig, ax = _axes()
    for p in fixed_points:
        ax.scatter(p['mean_cost'], p['accuracy'], color='#666666')
        ax.annotate(p.get('arm', ''), (p['mean_cost'], p['accuracy']), xytext=(4, 5), textcoords='offset points', fontsize=8)
    for name, curve in {'Fixed-arm frontier': frontier, **policy_curves}.items():
        curve = pareto_frontier(curve)
        ax.plot([p['mean_cost'] for p in curve], [p['accuracy'] for p in curve], marker='.', label=name)
    ax.scatter(oracle_point['mean_cost'], oracle_point['accuracy'], marker='*', s=130, color='black', label='Outcome oracle')
    ax.set(xlabel='Mean total cost (tokens/item)', ylabel='Accuracy (fraction)', ylim=(0, 1.02))
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    return _save(plt, fig, path)


def plot_reliability(bin_tables, path):
    """bin_tables maps series name to the reliability_bins list for one arm."""
    plt, fig, ax = _axes()
    ax.plot([0, 1], [0, 1], '--', color='#777777', label='Calibrated')
    for name, bins in bin_tables.items():
        bins = [b for b in bins if b['n']]
        ax.plot([b['confidence'] for b in bins], [b['accuracy'] for b in bins], marker='o', label=name)
    ax.set(xlabel='Mean predicted success probability', ylabel='Observed success fraction', xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    return _save(plt, fig, path)


def plot_learning_curves(curves, path):
    """curves maps name to learning_curve() records; one line per score key."""
    plt, fig, ax = _axes()
    for name, records in curves.items():
        for metric in records[0]['mean']:
            x = [r['n_train'] for r in records]
            mean = np.array([r['mean'][metric] for r in records])
            std = np.array([r['std'][metric] for r in records])
            line, = ax.plot(x, mean, marker='o', label=f'{name}: {metric}')
            ax.fill_between(x, mean - std, mean + std, alpha=.15, color=line.get_color())
    ax.set(xlabel='Training items (labels per arm)', ylabel='Validation score (fraction)')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    return _save(plt, fig, path)


def plot_auroc_grid(auroc, path):
    """auroc maps feature family -> learner -> sequence of per-arm AUROCs."""
    plt, fig, ax = _axes()
    labels, rows = [], []
    for family, learners in auroc.items():
        for learner, values in learners.items():
            labels.append(f'{family} / {learner}')
            rows.append(values)
    matrix = np.asarray(rows, float)
    plotted = ax.imshow(matrix, aspect='auto', vmin=0, vmax=1, cmap='cividis')
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(matrix.shape[1]), [f'Arm {a}' for a in range(matrix.shape[1])])
    ax.set(xlabel='Execution arm', ylabel='Feature family / learner')
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f'{value:.2f}' if np.isfinite(value) else 'N/A', ha='center', va='center',
                color='black' if value > .5 else 'white', fontsize=8)
    fig.colorbar(plotted, ax=ax, label='AUROC (fraction)')
    return _save(plt, fig, path)
