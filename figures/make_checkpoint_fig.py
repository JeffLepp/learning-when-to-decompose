"""Checkpoint 3 (v2) figure + table from dev label CSVs. Dev data only; excludes the
superseded qwen3:4b (Thinking-2507) rows per DESIGN_V2 D2. Run from FinalProject/."""
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
MODELS = [('qwen3:0.6b', 0.6), ('qwen3:1.7b', 1.7), ('qwen3:8b', 8.0)]
ARMS = [('S0', 'Direct', '#2a78d6', '-'), ('S1', 'Few-shot direct', '#eb6834', '-'),
        ('S2', 'Thinking (n=20)', '#1baf7a', '--'), ('S3', 'Coarse decomposition', '#eda100', '-'),
        ('S4', 'Fine decomposition', '#e87ba4', '-'), ('G1', 'Gold plan (diagnostic)', '#898781', ':')]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def load(domain):
    cells = defaultdict(list)
    for r in csv.DictReader(open(ROOT / f'results/v2/labels/{domain}.csv', encoding='utf-8')):
        if r['split'] == 'dev' and r['model'] != 'qwen3:4b':
            cells[(r['model'], r['strategy'])].append((r['correct'] in ('1', 'True', 'true'), float(r['total_tokens'])))
    return cells


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
rows = []
for ax, domain, title in zip(axes, ['scan', 'chainops'], ['SCAN (robot commands)', 'ChainOps (list operations)']):
    cells = load(domain)
    for arm, name, color, ls in ARMS:
        xs, ys = [], []
        for model, size in MODELS:
            v = cells.get((model, arm))
            if not v:
                continue
            k, n = sum(c for c, _ in v), len(v)
            lo, hi = wilson(k, n)
            xs.append(size); ys.append(100 * k / n)
            rows.append((domain, model, arm, k, n, 100 * k / n, 100 * lo, 100 * hi, sum(t for _, t in v) / n))
        ax.plot(xs, ys, ls, color=color, lw=2, marker='o', ms=6, label=name)
    ax.set_xscale('log'); ax.set_xticks([0.6, 1.7, 8]); ax.set_xticklabels(['0.6B', '1.7B', '8B'])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_title(title); ax.set_xlabel('Model size (Qwen3, log scale)'); ax.set_ylim(-3, 103)
    ax.grid(alpha=.3); ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Exact-match accuracy (%)')
axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
fig.suptitle('Development split: accuracy by model size and strategy (112 items per point; thinking n=20)', fontsize=10)
fig.tight_layout()
out = ROOT / 'figures/checkpoint3_dev_accuracy.png'
fig.savefig(out, dpi=160, bbox_inches='tight')
print('wrote', out)
print('domain,model,arm,correct,n,acc,wilson_lo,wilson_hi,mean_tokens')
for r in rows:
    print(','.join(str(round(x, 1)) if isinstance(x, float) else str(x) for x in r))
