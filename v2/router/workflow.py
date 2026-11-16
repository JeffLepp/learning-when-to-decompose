"""Shared P4/P5 artifact, calibration and frozen-policy machinery (CPU only)."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from .data import (ROOT, load_items, structural_features, load_f3, probe_overhead)
from .evaluate import arm_metrics, normalized_auc, point
from .models import _logit, fit_platt, ConstantCalibration
from .policies import (fixed_arm, complexity_rule, heuristic, cascade,
                       logprob_confidence, agreement_confidence, predict_then_route, threshold_route)

VALIDATION_NOTE = 'validation (used for selection; optimistic)'
BASE = ROOT / 'results/v2/router'


def output_directory(path):
    path = Path(path).resolve()
    if not path.is_relative_to(BASE.resolve()) or path == BASE.resolve():
        raise ValueError('Output must be under results/v2/router/<name>')
    return path


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def label_fingerprint(path, splits=('router_train', 'router_val')):
    """Canonical development-row hash; later final rows may be appended."""
    digest = hashlib.sha256()
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        digest.update(json.dumps(reader.fieldnames).encode())
        for row in reader:
            if row['split'] in splits:
                digest.update(json.dumps(row, sort_keys=True, separators=(',', ':')).encode())
                digest.update(b'\n')
    return digest.hexdigest()


def inputs_for(table, items_dir, domain):
    """Only this explicit split's file is opened; never enumerate item files."""
    path = Path(items_dir) / f'{table.split}.jsonl'
    items = load_items(path)
    selected = [items[id_] for id_ in table.ids]
    if any(i.get('split') != table.split or i.get('domain') != domain for i in selected):
        raise ValueError('Item domain/split mismatch')
    texts, metas = [i['input'] for i in selected], [i.get('meta', {}) for i in selected]
    x, _ = structural_features(texts, domain, metas)
    return texts, metas, x


def fit_calibrators(raw, y, method):
    result = []
    for a in range(y.shape[1]):
        mask = np.isfinite(y[:, a])
        if method == 'none' or len(np.unique(y[mask, a])) < 2:
            result.append(None)
        elif method == 'platt':
            result.append(fit_platt(raw[mask, a], y[mask, a]))
        else:
            result.append(IsotonicRegression(increasing=True, out_of_bounds='clip').fit(raw[mask, a], y[mask, a]))
    return result


def apply_calibrators(raw, calibrators, method):
    result = raw.copy()
    for a, model in enumerate(calibrators):
        if model is not None:
            result[:, a] = (model.predict_proba(_logit(raw[:, a]))[:, 1]
                            if method == 'platt' else model.predict(raw[:, a]))
    return result


def crossfit_calibration(raw, y, seed=437, diagnostics=None):
    if len(y) < 5:
        raise ValueError('Five-fold calibration requires at least five validation items')
    folds = list(KFold(5, shuffle=True, random_state=seed).split(raw))
    predictions = {'none': raw.copy()}
    fallback_counts = []
    for method in ['platt', 'isotonic']:
        p = np.empty_like(raw)
        for train, test in folds:
            calibrators = fit_calibrators(raw[train], y[train], method)
            if method == 'platt':
                fallback_counts.append(sum(isinstance(c, ConstantCalibration) for c in calibrators))
            p[test] = apply_calibrators(raw[test], calibrators, method)
        predictions[method] = p
    scores = {method: float(np.nanmean((p - y) ** 2)) for method, p in predictions.items()}
    if diagnostics is not None:
        diagnostics.update(platt_nonpositive_by_fold=fallback_counts,
                           platt_nonpositive_count=sum(fallback_counts))
    return predictions, scores, [test.tolist() for _, test in folds]


def mean_auroc(y, p):
    values = np.array(arm_metrics(y, p)['auroc'])
    return float(np.nanmean(values)) if np.isfinite(values).any() else .5


def pad_features(xs):
    """Zero padding preserves all native own-model dimensions in shared fits."""
    width = max(x.shape[1] for x in xs)
    return [np.pad(x, ((0, 0), (0, width - x.shape[1]))) for x in xs]


def subset_features(x, idx):
    return [v[idx] for v in x] if isinstance(x, list) else x[idx]


def family_features(state, table, inputs, features_dir):
    texts, _, structural = inputs
    matrices = {'F1': structural, 'F2': state['tfidf'].transform(texts)}
    overheads = {'F1': np.zeros(len(table.ids)), 'F2': np.zeros(len(table.ids))}
    for family, spec in state['families'].items():
        if family.startswith('F3'):
            if features_dir is None:
                raise ValueError(f'{family} requires frozen-compatible feature files')
            matrices[family] = pad_features(load_f3(features_dir, table.ids, spec['layer'], spec['variant'], table.arms))
            overheads[family] = probe_overhead(table, spec['variant'], table.arms, features_dir)
    return matrices, overheads


def heuristic_decision(table, inputs, config, domain):
    rule = complexity_rule(config['small'], config['large'], config['k'], config['length_threshold'], domain)
    def strategy_rule(text, meta):
        arm = rule(text, meta)
        return arm.replace('|S3', '|' + config['strategy'])
    return heuristic(inputs[0], inputs[1], strategy_rule, table.arms)


def apply_recipe(recipe, state, table, inputs, matrices, probabilities, costs):
    kind = recipe['kind']
    if kind == 'fixed':
        return [fixed_arm(len(table.ids), len(table.arms), table.arms.index(recipe['arm']))]
    if kind == 'heuristic':
        return [heuristic_decision(table, inputs, config, state['domain']) for config in recipe['grid']]
    if kind in ['lambda', 'threshold']:
        fn = predict_then_route if kind == 'lambda' else threshold_route
        return [fn(probabilities[recipe['predictor']], costs, value) for value in recipe['grid']]
    if kind == 'cascade':
        confidence = recipe['confidence']
        if confidence == 'logprob':
            fn = logprob_confidence
        elif confidence == 'agreement':
            fn = agreement_confidence
        else:
            learned = state['confidences'][recipe['confidence_key']]
            fn = learned.for_inputs(inputs[2]) if confidence == 'learned_F4_F1' else learned
        return [cascade(table, recipe['ladder'], fn, tau) for tau in recipe['grid']]
    raise ValueError(f'Unknown frozen policy kind: {kind}')


def apply_state(state, table, items_dir, features_dir=None, *, val_probabilities=None):
    inputs = inputs_for(table, items_dir, state['domain'])
    matrices, overheads = family_features(state, table, inputs, features_dir)
    costs = state['cost'].predict(inputs[2])
    probabilities = val_probabilities if val_probabilities is not None else {
        name: model.predict_proba(matrices[state['predictor_families'][name]])
        for name, model in state['predictors'].items()}
    # D1.6 is an evaluated-split reference, not a selected deployable policy.
    anchor = fixed_arm(len(table.ids), len(table.arms), int(table.cost.mean(axis=0).argmin()))
    result = {}
    for name, recipe in state['recipes'].items():
        if recipe['kind'] == 'alias':
            result[name] = result[recipe['source']]
            continue
        curve = apply_recipe(recipe, state, table, inputs, matrices, probabilities, costs)
        result[name] = dict(decision=curve[recipe['chosen']], curve=curve, anchor=anchor,
                            overhead=overheads.get(recipe.get('family'), np.zeros(len(table.ids))))
    return result


def curve_auc(curve, table, overhead=None):
    fixed = [point(fixed_arm(len(table.ids), len(table.arms), a), table) for a in range(len(table.arms))]
    anchor = min(fixed, key=lambda p: p['mean_cost'])
    return normalized_auc([anchor] + [point(d, table, overhead) for d in curve],
                          anchor['mean_cost'], max(p['mean_cost'] for p in fixed))


def restore_experiment(state, directory, n_boot):
    from .evaluate import Experiment
    experiment = Experiment('frozen', state['val'], n_boot=n_boot, experiment_dir=directory)
    experiment._frozen = state['selections']
    experiment.selection_only = set(state['selection_only'])
    experiment.freeze()
    return experiment


def summary(report, path, note=VALIDATION_NOTE):
    """Compact rendering; complete diagnostics live in the machine-readable report."""
    def fmt(value):
        return 'NA' if value is None or not np.isfinite(value) else f'{value:.4f}'
    def interval(metric):
        return f"{fmt(metric['value'])} [{', '.join(fmt(v) for v in metric['ci95'])}]"
    def avg(values):
        valid = [v for v in values if v is not None and np.isfinite(v)]
        return float(np.mean(valid)) if valid else None
    lines = ['# Frozen routing analysis', '', note, '',
             f"Items: {report['best_single']['n']}; bootstrap replicates: {report['uncertainty']['n_boot']}.",
             '', '| Arm | Accuracy | Tokens |', '|---|---:|---:|']
    lines += [f"| {p['arm'].replace('|', '&#124;')} | {fmt(p['accuracy'])} | {fmt(p['mean_cost'])} |" for p in report['fixed_points']]
    grid = report.get('prediction_metrics', {})
    if grid:
        lines += ['', '| Family / learner / mode | Raw AUROC | Cross-fit AUROC | Calibration | Platt fallback folds/full | Frozen Brier / ECE |', '|---|---:|---:|---|---:|---|']
        for name, metrics in grid.items():
            frozen = report['predictor_metrics'][name]
            lines.append(f"| {name} | {fmt(avg(metrics['raw']['auroc']))} | {fmt(avg(metrics['crossfit']['auroc']))} | {metrics['selected_calibration']} | {metrics['calibration_diagnostics']['platt_nonpositive_count']}/{metrics['full_val_platt_nonpositive_count']} | {fmt(avg(frozen['brier']))} / {fmt(avg(frozen['ece']))} |")
    elif report.get('predictor_metrics'):
        lines += ['', '| Family / learner / mode | Raw AUROC | Frozen Brier | Frozen ECE |', '|---|---:|---:|---:|']
        for name, metrics in report['predictor_metrics'].items():
            lines.append(f"| {name} | {fmt(metrics['mean_auroc'])} | {fmt(avg(metrics['brier']))} | {fmt(avg(metrics['ece']))} |")
    if report.get('predictor_contrasts'):
        from .predictor_report import FAMILY_CONTRASTS
        lines += ['', 'H3 mean raw AUROC differences [95% paired CI]; columns are left minus right.',
                  '| Learner / mode | ' + ' | '.join(f'{a} - {b}' for a, b in FAMILY_CONTRASTS) + ' |',
                  '|---|' + '---|' * len(FAMILY_CONTRASTS)]
        for learner in ['logreg', 'hgb', 'mlp']:
            for mode in ['per_arm', 'shared']:
                cells = [interval(report['predictor_contrasts'][key]) if key in report['predictor_contrasts'] else 'absent family' for key in
                         [f'{a}/{learner}/{mode} - {b}/{learner}/{mode}' for a, b in FAMILY_CONTRASTS]]
                lines.append(f'| {learner}/{mode} | ' + ' | '.join(cells) + ' |')
        lines.append('Single-class arm replicates are skipped and counted per predictor in JSON; absent-family reasons are logged there.')
    if report.get('contrasts'):
        lines += ['', '| H4/H5 contrast | Normalized AUC difference [95% CI] |', '|---|---|']
        for name, metric in report['contrasts'].items():
            lines.append(f"| {name} | {interval(metric['normalized_auc'])} |")
        h5 = report['contrasts'].get('selected/cascade - selected/overall')
        if h5:
            lines.append(f"H5 D1.7: accuracy delta {interval(h5['accuracy'])}; tokens delta {interval(h5['cost'])}; McNemar {h5['mcnemar']}.")
    lines += ['', '| Policy | AUC | Accuracy | Tokens | Probe tokens | Accuracy 95% CI |', '|---|---:|---:|---:|---:|---|']
    # All curves/points are in JSON. Headline table includes every cascade and
    # family-selected router; avoids repeating all 24 learner configurations.
    for name, p in report['policies'].items():
        if name.startswith(('fixed/', 'route/')):
            continue
        lines.append(f"| {name} | {fmt(p.get('normalized_auc', {}).get('value'))} | {fmt(p['accuracy'])} | {fmt(p['mean_cost'])} | {fmt(p['mean_overhead'])} | {p.get('accuracy_ci95')} |")
    learning = report.get('learning_curves', {})
    if learning:
        lines += ['', '| Family | Train sizes | Mean AUROC | Mean AUC |', '|---|---|---|---|']
        for family, rows in learning.items():
            lines.append(f"| {family} | {', '.join(str(r['n_train']) for r in rows)} | {', '.join(fmt(r['mean']['auroc']) for r in rows)} | {', '.join(fmt(r['mean']['normalized_auc']) for r in rows)} |")
    lines += ['', 'S2 subset (separate analysis):']
    s2 = report.get('s2')
    if s2 and s2.get('best_single'):
        lines += [f"- n={s2['best_single']['n']}; best-single accuracy={fmt(s2['best_single']['accuracy'])}; main+S2 routing AUC={fmt(s2['policies']['s2/router']['normalized_auc']['value'])}."]
    else:
        lines += ['- No complete S2 subset available.']
    lines += ['', 'Caveats:', '- Synthetic labels are fixtures, not real capability evidence.' if report.get('synthetic_labels') else '- Historical adapter results are exploratory pilot evidence.',
              '- Validation selects layers, calibration, policies, and operating points; cross-fitting does not remove selection optimism.',
              '- Costs are replayed tokens, not live latency, FLOPs, or energy. Every F3 point includes probe overhead; the fixed reference anchor does not.',
              '- HF/Ollama quantization differs. Missing adapter S0 rows use WS2 tokenized S0 probe lengths.',
              '- Bootstrap conditions on fitted models and frozen choices; no refitting or reselection.',
              '- Full per-arm metrics, paired CIs, McNemar tests, grids and provenance: report.json.']
    lines += ['- ' + s for s in report.get('skipped', [])]
    if len(lines) >= 150:
        raise ValueError('Summary exceeded 149 lines')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
