"""P4 development driver: router_train/router_val only; no model generation."""
import argparse
import itertools
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .data import (ROOT, MODELS, TfidfFeatures, available_layer_keys, load_f3,
                   load_labels, main_table, model_dir_name, probe_overhead, s2_table)
from .evaluate import Experiment, arm_metrics, learning_curve, pareto_frontier, point, write_json
from .models import CostPredictor, SuccessPredictor
from .predictor_report import predictor_report, policy_contrasts
from .policies import (LearnedConfidence, best_single_arm, cheapest_arm_meeting_target,
                       fixed_arm, lambda_grid)
from .workflow import (VALIDATION_NOTE, apply_recipe, apply_state, crossfit_calibration,
                       curve_auc, family_features, inputs_for, label_fingerprint, mean_auroc,
                       output_directory, pad_features, sha256, subset_features, summary)


def say(message):
    print(message, flush=True)


def fit_predictor(experiment, state, name, family, learner, mode, xt, xv, train, val, metrics):
    params = {'max_iter': 1500} if learner == 'mlp' else {}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always', ConvergenceWarning)
        model = SuccessPredictor(learner, mode, **params).fit(xt, train.success)
    raw = model.predict_proba(xv)
    calibration_diagnostics = {}
    crossfit, scores, folds = crossfit_calibration(raw, val.success, diagnostics=calibration_diagnostics)
    method = experiment.select('calibration/' + name, scores, lambda score, _: -score, selection_only=True)
    if method != 'none':
        model.calibrate(xv, val.success, method)
    state['predictors'][name] = model
    state['predictor_families'][name] = family
    metrics[name] = dict(raw=arm_metrics(val.success, raw), crossfit=arm_metrics(val.success, crossfit[method]),
                         selected_calibration=method, crossfit_brier=scores, validation_folds=folds,
                         calibration_diagnostics=calibration_diagnostics,
                         full_val_platt_nonpositive_count=model.platt_nonpositive_count,
                         warnings=sorted({str(w.message) for w in captured}), learner_params=params,
                         note=VALIDATION_NOTE)
    return crossfit[method]


def register(experiment, state, name, recipe, val, inputs, matrices, probabilities, costs, overheads):
    curve = apply_recipe(recipe, state, val, inputs, matrices, probabilities, costs)
    overhead = overheads.get(recipe.get('family'))
    recipe['chosen'] = experiment.select_operating_point(name, [point(d, val, overhead) for d in curve], curve)
    state['recipes'][name] = recipe
    return curve_auc(curve, val, overhead)


def state_base(domain, train, val, train_inputs):
    return dict(domain=domain, arms=val.arms, val=val, families={'F1': {}, 'F2': {}},
                tfidf=TfidfFeatures().fit(train_inputs[0]), predictors={}, predictor_families={},
                cost=CostPredictor('mean').fit(train_inputs[2], train.cost), confidences={}, recipes={})


def cascade_grids(table, ladder, confidence, quick):
    stages = len(ladder) - 1
    steps = 3 if quick else 11
    if confidence.startswith('learned'):
        return [np.full(stages, t).tolist() for t in np.r_[-np.inf, np.linspace(0, 1, steps), np.inf]]
    values = table.mean_logprob[:, [table.arms.index(a) for a in ladder[:-1]]]
    finite = values[np.isfinite(values)]
    log_grid = np.r_[-np.inf, np.quantile(finite, np.linspace(0, 1, steps)) if finite.size else [], np.inf]
    if confidence == 'logprob':
        return [np.full(stages, t).tolist() for t in log_grid]
    # The first executed arm has logprob confidence; later stages have binary
    # agreement. Explicit per-stage thresholds prevent mixing these scales.
    grid = []
    for first in log_grid:
        for later in [0., .5, 1.1]:
            grid.append([float(first)] + [later] * (stages - 1))
    if stages > 2 and not quick:
        for stage in range(1, stages):
            for later in [0., 1.1]:
                thresholds = [np.inf] + [.5] * (stages - 1)
                thresholds[stage] = later
                grid.append(thresholds)
    return grid


def fit_s2(raw, domain, items_dir, out, n_boot, quick):
    if not any(a.endswith('|S2') for a in raw['router_train'].arms):
        return None, None
    train, val = [s2_table(raw[s]) for s in ['router_train', 'router_val']]
    if not len(train.ids) or not len(val.ids):
        return dict(reason='No complete S2 train/validation subset'), None
    ti, vi = [inputs_for(t, items_dir, domain) for t in [train, val]]
    state = state_base(domain, train, val, ti)
    experiment = Experiment('s2', val, n_boot=n_boot, experiment_dir=out / 's2')
    metrics = {}
    p = fit_predictor(experiment, state, 'F1/logreg/per_arm', 'F1', 'logreg', 'per_arm',
                      ti[2], vi[2], train, val, metrics)
    matrices, overheads = family_features(state, val, vi, None)
    costs = state['cost'].predict(vi[2])
    probabilities = {'F1/logreg/per_arm': p}
    register(experiment, state, 's2/router', dict(kind='lambda', family='F1', predictor='F1/logreg/per_arm',
             grid=lambda_grid(costs, n=7 if quick else 25).tolist()), val, vi, matrices, probabilities, costs, overheads)
    for arm in val.arms:
        register(experiment, state, 'fixed/' + arm, dict(kind='fixed', arm=arm), val, vi,
                 matrices, probabilities, costs, overheads)
    state.update(selections=experiment.frozen, selection_only=list(experiment.selection_only))
    experiment.freeze()
    report = experiment.evaluate_on(val, apply_state(state, val, items_dir, val_probabilities=probabilities),
                                    fast_bootstrap=True, comparison_scope='best_single')
    report.update(note=VALIDATION_NOTE, prediction_metrics=metrics)
    write_json(out / 's2/report.json', report)
    return report, state


def run(labels, domain, items_dir, out, features_dir=None, n_boot=10000, lc_repeats=20,
        expected_arms=16, quick=False):
    started = time.perf_counter()
    labels, items_dir, out = Path(labels).resolve(), Path(items_dir).resolve(), output_directory(out)
    if (out / 'report.json').exists():
        previous = json.loads((out / 'report.json').read_text(encoding='utf-8'))
        if previous.get('p4_completed'):
            raise RuntimeError('Completed P4 report exists; use a new run name')
    n_boot, lc_repeats = (200, 2) if quick else (n_boot, lc_repeats)
    if n_boot < 1 or lc_repeats < 1:
        raise ValueError('Positive bootstrap and learning-curve repeat counts required')
    raw = load_labels(labels, splits=['router_train', 'router_val'])
    train, val = [main_table(raw[s], expected_n_arms=expected_arms) for s in ['router_train', 'router_val']]
    if train.arms != val.arms:
        raise ValueError('Train/validation arm order mismatch')
    ti, vi = [inputs_for(t, items_dir, domain) for t in [train, val]]
    experiment = Experiment('p4', val, n_boot=n_boot, experiment_dir=out)
    state = state_base(domain, train, val, ti)
    skipped, layer_scores, feature_paths = [], {}, set()
    features_dir = Path(features_dir).resolve() if features_dir is not None else None
    if features_dir is not None and features_dir.exists():
        for variant in ['own', 'small']:
            family = 'F3-' + variant
            models = sorted({a.split('|')[0] for a in train.arms}) if variant == 'own' else [MODELS[0]]
            try:
                indexes = [features_dir / model_dir_name(m) / 'index.json' for m in models]
                keys = available_layer_keys(features_dir, models[0])
                for model in models[1:]:
                    keys = [k for k in keys if k in available_layer_keys(features_dir, model)]
                if not keys:
                    raise ValueError('No shared layer/pooling keys')
                scores = {}
                for key in keys:
                    xt = load_f3(features_dir, train.ids, key, variant, train.arms)
                    xv = load_f3(features_dir, val.ids, key, variant, val.arms)
                    predictor = SuccessPredictor('logreg').fit(xt, train.success)
                    scores[key] = mean_auroc(val.success, predictor.predict_proba(xv))
                chosen = experiment.select('layer/' + family, scores, lambda score, _: score, selection_only=True)
                state['families'][family] = dict(layer=chosen, variant=variant)
                feature_paths.update(indexes)
                layer_scores[family] = dict(scores=scores, selected=chosen,
                                           rule='Mean per-arm validation AUROC; train-only logreg, first key wins ties')
                say(f'{family}: selected {chosen} on validation')
            except (FileNotFoundError, ValueError, KeyError) as exc:
                skipped.append(f'{family} skipped: {exc}')
    else:
        skipped.append('F3-own and F3-small skipped: no existing features directory supplied')
    train_matrices, _ = family_features(state, train, ti, features_dir)
    val_matrices, overheads = family_features(state, val, vi, features_dir)
    predictions, metrics = {}, {}
    for family in state['families']:
        for learner, mode in itertools.product(['logreg', 'hgb', 'mlp'], ['per_arm', 'shared']):
            name = f'{family}/{learner}/{mode}'
            predictions[name] = fit_predictor(experiment, state, name, family, learner, mode,
                                              train_matrices[family], val_matrices[family], train, val, metrics)
        say(f'{family}: six learners fitted and five-fold calibration selected')
    costs = state['cost'].predict(vi[2])
    rf = CostPredictor('rf').fit(ti[2], train.cost)
    cost_diagnostics = dict(mean_mae=float(np.mean(np.abs(costs - val.cost))),
                            rf_mae=float(np.mean(np.abs(rf.predict(vi[2]) - val.cost))),
                            routing_cost_model='mean', secondary='rf predictions only; not selected for routing')
    def add(name, recipe):
        return register(experiment, state, name, recipe, val, vi, val_matrices, predictions, costs, overheads)
    for arm in val.arms:
        add('fixed/' + arm, dict(kind='fixed', arm=arm))
    best = float(val.success[:, best_single_arm(val)].mean())
    targets = {}
    for offset in [0, .02, .05]:
        target = best - offset
        arm, met = cheapest_arm_meeting_target(val, target)
        name = f'target/best-{offset:.2f}'
        targets[name] = dict(target=target, met=met, arm=val.arms[arm])
        add(name, dict(kind='fixed', arm=val.arms[arm]))
    models = [m for m in MODELS if f'{m}|S0' in val.arms]
    heuristic_grid = [dict(small=small, large=large, k=k, length_threshold=length, strategy=strategy)
                      for i, small in enumerate(models) for large in models[i:]
                      for k in ([1, 3] if quick else [0, 1, 2, 3, 4])
                      for length in ([8, 12] if quick else [4, 8, 12, 16, 24])
                      for strategy in ['S3', 'S4']
                      if f'{small}|{strategy}' in val.arms and f'{large}|{strategy}' in val.arms]
    if heuristic_grid:
        add('heuristic', dict(kind='heuristic', grid=heuristic_grid))
    route_scores = {}
    for name in predictions:
        family = state['predictor_families'][name]
        route_scores['route/' + name] = add('route/' + name, dict(kind='lambda', family=family, predictor=name,
                                          grid=lambda_grid(costs, n=7 if quick else 25).tolist()))
    for name, grid in [('threshold/0.5', [.5]), ('threshold/sweep', [0, .25, .5, .6, .7, .8, .9, .95, 1.])]:
        add(name, dict(kind='threshold', family='F1', predictor='F1/logreg/per_arm', grid=grid))
    fixed = [dict(arm=a, **point(fixed_arm(len(val.ids), len(val.arms), j), val)) for j, a in enumerate(val.arms)]
    ladders = {'size_S0': [f'{m}|S0' for m in models],
               'fixed_frontier': [p['arm'] for p in pareto_frontier(fixed)]}
    ladders.update({m: [f'{m}|S0', f'{m}|S3'] for m in models if f'{m}|S3' in val.arms})
    cascade_scores = {}
    for ladder_name, ladder in ladders.items():
        if len(ladder) < 2:
            skipped.append(f'Cascade {ladder_name} skipped: fewer than two ladder arms')
            continue
        for confidence in ['logprob', 'agreement', 'learned_F4', 'learned_F4_F1']:
            name = f'cascade/{ladder_name}/{confidence}'
            if confidence != 'agreement' and not np.isfinite(val.mean_logprob[:, [val.arms.index(a) for a in ladder]]).any():
                skipped.append(f'{name} skipped: all ladder logprobs missing')
                continue
            recipe = dict(kind='cascade', ladder=ladder, confidence=confidence,
                          grid=cascade_grids(val, ladder, confidence, quick))
            if confidence.startswith('learned'):
                recipe['confidence_key'] = name
                state['confidences'][name] = LearnedConfidence('logreg').fit(
                    train, ladder, X=ti[2] if confidence == 'learned_F4_F1' else None)
            cascade_scores[name] = add(name, recipe)
    for family in state['families']:
        candidates = {k: s for k, s in route_scores.items() if k.startswith('route/' + family + '/')}
        chosen = experiment.select('selected/' + family, candidates, lambda score, _: score)
        state['recipes']['selected/' + family] = dict(kind='alias', source=chosen)
    chosen = experiment.select('selected/overall', route_scores, lambda score, _: score)
    state['recipes']['selected/overall'] = dict(kind='alias', source=chosen)
    if cascade_scores:
        cascade_choice = experiment.select('selected/cascade', cascade_scores, lambda score, _: score)
        state['recipes']['selected/cascade'] = dict(kind='alias', source=cascade_choice)
    else:
        skipped.append('H5 cascade - pre-generation contrast skipped: no eligible cascade')
    state['contrast_pairs'] = policy_contrasts(state)
    state.update(selections=experiment.frozen, selection_only=sorted(experiment.selection_only))
    experiment.freeze()
    say(f'Frozen {len(state["recipes"])} policies; evaluating validation with {n_boot} paired bootstrap replicates')
    policies = apply_state(state, val, items_dir, features_dir, val_probabilities=predictions)
    report = experiment.evaluate_on(val, policies, fast_bootstrap=True, comparison_scope='best_single',
                                    contrast_pairs=state['contrast_pairs'])
    report.update(predictor_report(state, val, val_matrices, n_boot))
    curves = {}
    for family in ['F1', 'F2', 'F3-own']:
        if family not in state['families']:
            continue
        def score(idx):
            xt, xv = subset_features(train_matrices[family], idx), val_matrices[family]
            if family == 'F2':
                texts = [ti[0][i] for i in idx]
                subset_tfidf = TfidfFeatures().fit(texts)
                xt, xv = subset_tfidf.transform(texts), subset_tfidf.transform(vi[0])
            model = SuccessPredictor('logreg').fit(xt, train.success[idx])
            p = model.predict_proba(xv)
            from .policies import predict_then_route
            cost = CostPredictor('mean').fit(ti[2][idx], train.cost[idx]).predict(vi[2])
            curve = [predict_then_route(p, cost, lam) for lam in lambda_grid(cost, n=7 if quick else 25)]
            return dict(auroc=mean_auroc(val.success, p), normalized_auc=curve_auc(curve, val, overheads[family]))
        sizes = sorted(set(min(n, len(train.ids)) for n in [50, 100, 200, 400]))
        sizes = [n for n in sizes if n != len(train.ids)] + [None]
        curves[family] = learning_curve(score, len(train.ids), sizes=sizes, repeats=lc_repeats)
        say(f'{family}: learning curves completed')
    s2_report, s2_state = fit_s2(raw, domain, items_dir, out, n_boot, quick)
    source_manifest = labels.with_suffix('.synthetic.json')
    synthetic = None
    if source_manifest.exists():
        synthetic = json.loads(source_manifest.read_text(encoding='utf-8'))
        if synthetic.get('generator') != 'v2.router.synth_labels' or synthetic['labels_sha256'] != sha256(labels):
            raise ValueError('Synthetic label provenance mismatch')
        for split in ['router_train', 'router_val']:
            if synthetic['item_sha256'][f'{split}.jsonl'] != sha256(items_dir / f'{split}.jsonl'):
                raise ValueError('Synthetic development item provenance mismatch')
    state['s2'] = s2_state
    frozen_dir = out / 'frozen'
    frozen_dir.mkdir(exist_ok=True)
    joblib.dump(state, frozen_dir / 'objects.joblib', compress=3)
    write_json(frozen_dir / 'selections.json', dict(selections=state['selections'],
               selection_only=state['selection_only'], recipes=state['recipes'], layer_selection=layer_scores,
               s2_selections=s2_state['selections'] if s2_state else None))
    input_hashes = {str(labels): sha256(labels),
                    **{str(items_dir / f'{s}.jsonl'): sha256(items_dir / f'{s}.jsonl') for s in ['router_train', 'router_val']},
                    **{str(p): sha256(p) for p in sorted(feature_paths)}}
    code_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'v2/router').glob('*.py'))}
    report.update(p4_completed=True, note=VALIDATION_NOTE, quick=quick, domain=domain,
                  expected_arms=expected_arms, synthetic_labels=synthetic, selected_router=chosen,
                  input_sha256=input_hashes, code_sha256=code_hashes,
                  development_labels_sha256=label_fingerprint(labels),
                  frozen_sha256={p.name: sha256(p) for p in sorted(frozen_dir.iterdir()) if p.is_file()},
                  labels_path=str(labels), items_dir=str(items_dir), features_dir=str(features_dir) if features_dir else None,
                  prediction_metrics=metrics, layer_selection=layer_scores, targets=targets,
                  cost_predictor=cost_diagnostics, learning_curves=curves, lc_repeats=lc_repeats,
                  selections=state['selections'], s2=s2_report, skipped=skipped,
                  probe_cost_source='S0 label prompt tokens; absent adapter S0 uses WS2 index n_tokens',
                  learning_curve_note='Train-subset-only uncalibrated logreg/TF-IDF; F3 layer fixed by full development selection; validation diagnostics; all-training fit once',
                  model_calls=0)
    from .plots import plot_cost_accuracy, plot_reliability, plot_learning_curves, plot_auroc_grid
    dest = out / 'figures'
    title = VALIDATION_NOTE
    plot_cost_accuracy(report['fixed_points'], report['fixed_frontier'],
                       {k: v['curve'] for k, v in report['policies'].items() if k.startswith('selected/')},
                       report['oracle'], dest / 'cost_accuracy.png', title=title)
    selected_predictor = state['recipes'][chosen]['predictor']
    plot_reliability(dict(zip(val.arms, metrics[selected_predictor]['crossfit']['reliability_bins'])),
                     dest / 'reliability.png', title=title)
    plot_learning_curves(curves, dest / 'learning_curve.png', title=title)
    grid = {f: {'/'.join(k.split('/')[1:]): v['crossfit']['auroc'] for k, v in metrics.items() if k.startswith(f + '/')}
            for f in state['families']}
    plot_auroc_grid(grid, dest / 'auroc.png', title=title, arm_labels=[str(i) for i in range(len(val.arms))])
    report['wall_seconds'] = time.perf_counter() - started
    write_json(out / 'report.json', report)
    summary(report, out / 'summary.md')
    say(f'P4 completed: {out}; {report["wall_seconds"]:.2f}s')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--domain', choices=['scan', 'chainops'], required=True)
    parser.add_argument('--items-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--features-dir', type=Path)
    parser.add_argument('--n-boot', type=int, default=10000)
    parser.add_argument('--lc-repeats', type=int, default=20)
    parser.add_argument('--expected-arms', choices=['16', 'none'], default='16')
    parser.add_argument('--quick', action='store_true')
    args = vars(parser.parse_args())
    args['expected_arms'] = None if args['expected_arms'] == 'none' else 16
    with threadpool_limits(limits=1):
        run(**args)


if __name__ == '__main__':
    main()
