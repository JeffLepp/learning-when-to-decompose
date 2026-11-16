"""Independent artifact checks without predictor refits or router helper calls."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[4]
base = ROOT / 'results/v2/router'
metrics_path = base / 'v1_adapter/figures/metrics.json'
metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
with (base / 'v1_adapter/scan_v1_labels.csv').open(newline='', encoding='utf-8') as stream:
    rows = [r for r in csv.DictReader(stream) if r['split'] in {'router_train', 'router_val'}]
for split, key in [('router_train', 'train_ids'), ('router_val', 'validation_ids')]:
    assert metrics[key] == list(dict.fromkeys(r['id'] for r in rows if r['split'] == split))
assert len(metrics['train_ids']) == 96 and len(metrics['validation_ids']) == 32
assert not set(metrics['train_ids']) & set(metrics['validation_ids'])
val_rows = {(r['id'], r['model'] + '|' + r['strategy']): r
            for r in rows if r['split'] == 'router_val'}
y = np.array([[float(val_rows[id_, arm]['correct']) for arm in metrics['arms']]
              for id_ in metrics['validation_ids']])
cost = np.array([[float(val_rows[id_, arm]['total_tokens']) for arm in metrics['arms']]
                 for id_ in metrics['validation_ids']])
for name, values in metrics['validation_probabilities'].items():
    family, learner = name.split('/')
    pred = np.asarray(values)
    np.testing.assert_allclose(metrics['auroc'][family][learner],
                               [roc_auc_score(y[:, a], pred[:, a]) for a in range(2)])
for learner, candidates in metrics['f3_candidates'].items():
    assert len(candidates) == 18
    assert metrics['f3_selected'][learner] == max(candidates, key=lambda c: c['mean_auroc'])
    for candidate in candidates:
        np.testing.assert_allclose(candidate['mean_auroc'], np.mean(candidate['auroc']))


def check_point(choices, saved):
    actual_y = y[np.arange(len(y)), choices]
    actual_cost = cost[np.arange(len(y)), choices]
    np.testing.assert_allclose([saved['n'], saved['correct'], saved['accuracy'], saved['mean_cost']],
                               [len(y), actual_y.sum(), actual_y.mean(), actual_cost.mean()])


for a, fixed in enumerate(metrics['fixed_points']):
    assert fixed['arm'] == metrics['arms'][a]
    check_point(np.full(len(y), a), fixed)
pred = np.array(metrics['validation_probabilities']['F1/logreg'])
pred_cost = np.array(metrics['predicted_cost'])
for name, curve in metrics['curves'].items():
    sweep = curve[1:]  # First entry is the fixed-arm left anchor.
    settings = metrics['lambdas'] if name.endswith('lambda') else metrics['thresholds']
    assert len(sweep) == len(settings)
    for setting, saved in zip(settings, sweep):
        if name.endswith('lambda'):
            score = pred - setting * pred_cost
            ties = score == score.max(axis=1, keepdims=True)
            choices = np.where(ties, pred_cost, np.inf).argmin(axis=1)
        else:
            eligible = pred >= setting
            choices = np.where(eligible, pred_cost, np.inf).argmin(axis=1)
            missing = ~eligible.any(axis=1)
            choices[missing] = pred[missing].argmax(axis=1)
        check_point(choices, saved)
    target = y.mean(axis=0).max()
    eligible = [i for i, p in enumerate(sweep) if p['accuracy'] >= target - 1e-12]
    choice = (min(eligible, key=lambda i: (sweep[i]['mean_cost'], -sweep[i]['accuracy'], i))
              if eligible else min(range(len(sweep)),
                                   key=lambda i: (-sweep[i]['accuracy'], sweep[i]['mean_cost'], i)))
    assert metrics['selections'][name] == dict(choice=choice, rule='D1.7',
                                              validation_point=sweep[choice],
                                              best_single_val_accuracy=target)
correct_cost = np.where(y == 1, cost, np.inf)
choices = correct_cost.argmin(axis=1)
missing = ~(y == 1).any(axis=1)
choices[missing] = cost[missing].argmin(axis=1)
check_point(choices, metrics['oracle'])
for a, arm in enumerate(metrics['arms']):
    bins = metrics['reliability'][arm]
    assignment = np.minimum((pred[:, a] * len(bins)).astype(int), len(bins) - 1)
    for i, bin_ in enumerate(bins):
        mask = assignment == i
        assert bin_['n'] == int(mask.sum())
        if mask.any():
            np.testing.assert_allclose([bin_['accuracy'], bin_['confidence']],
                                       [y[mask, a].mean(), pred[mask, a].mean()])
for record in metrics['learning_curve']:
    assert len(record['raw']) == (1 if record['all_training'] else 20)
    for arm in metrics['arms']:
        values = [r[arm] for r in record['raw']]
        np.testing.assert_allclose([record['mean'][arm], record['std'][arm]],
                                   [np.mean(values), np.std(values)])

images = []
for directory in ['sim_check', 'v1_adapter']:
    for name in ['cost_accuracy', 'reliability', 'learning_curve', 'auroc']:
        path = base / directory / 'figures' / f'{name}.png'
        with Image.open(path) as image:
            dimensions, dpi = image.size, image.info['dpi']
            # PNG stores integer pixels/metre: one unit is 0.0254 dpi.
            assert all(abs(value - 150) <= .0254 / 2 + 1e-9 for value in dpi)
            assert min(dimensions) >= 600
            image.verify()
        images.append(dict(path=path.relative_to(ROOT).as_posix(), dimensions=dimensions,
                           dpi=dpi, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
report = dict(passed=True, n_pngs=len(images), images=images,
              checks=['Train/validation ID alignment', '9 AUROC rows from saved predictions',
                      '18 F3 layer/pooling candidates per learner and selection',
                      'Fixed points, both sweeps, D1.7 records, and oracle replay',
                      'Reliability bins and learning-curve summaries', 'PNG decoding and 150 dpi'],
              limits='No refitting; candidate F3 predictions and individual learning subsamples were not saved.',
              metrics_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest())
(Path(__file__).parent / 'readback.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(passed=True, n_pngs=len(images), checks=report['checks']), indent=2))
