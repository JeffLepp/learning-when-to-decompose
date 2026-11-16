"""Exploratory success classifiers and validation-selected routing. No test tuning."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from experiment import ROOT, SEED, save

STRATEGIES = ['direct', 'decomposed']


def features(command):
    words = command.split()
    return {'word_count': len(words), 'char_count': len(command), **{f'count_{w}': words.count(w) for w in ['walk', 'run', 'jump', 'look', 'turn', 'left', 'right', 'opposite', 'around', 'twice', 'thrice', 'and', 'after']}}


def load_run(name, partition):
    folder = ROOT / 'results' / name
    config = json.loads((folder / 'config.json').read_text())
    assert config['partition'] == partition, f'{name} is not {partition}'
    pairs = {}
    for line in (folder / 'records.jsonl').read_text().splitlines():
        row = json.loads(line)
        pair = pairs.setdefault(row['id'], {})
        assert row['strategy'] not in pair, 'Duplicate strategy record'
        pair[row['strategy']] = row
    assert len(pairs) == config['count'] and all(set(p) == set(STRATEGIES) for p in pairs.values()), 'Incomplete run'
    return config, pairs


def matrices(pairs):
    rows = list(pairs.values())
    x = [features(r['direct']['command']) for r in rows]
    y = np.array([[r[s]['correct'] for s in STRATEGIES] for r in rows], dtype=int)
    cost = np.array([[r[s]['prompt_tokens'] + r[s]['completion_tokens'] for s in STRATEGIES] for r in rows])
    seconds = np.array([[r[s]['wall_seconds'] for s in STRATEGIES] for r in rows])
    return x, y, cost, seconds


def probability(model, x):
    classes = list(model.classes_)
    return model.predict_proba(x)[:, classes.index(1)] if 1 in classes else np.zeros(len(x))


def route(probabilities, costs, threshold):
    eligible = probabilities >= threshold
    choice = np.argmin(np.where(eligible, costs, np.inf), axis=1)
    none = ~eligible.any(axis=1)
    choice[none] = probabilities[none].argmax(axis=1)
    return choice


def metrics(choice, y, costs, seconds):
    idx = np.arange(len(choice))
    successes = y[idx, choice]
    p, n, z = float(successes.mean()), len(choice), 1.96
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {'n': n, 'correct': int(successes.sum()), 'accuracy': p, 'accuracy_wilson_95': [center-half, center+half], 'mean_total_tokens': float(costs[idx, choice].mean()), 'mean_seconds_replayed': float(seconds[idx, choice].mean()), 'decomposition_fraction': float(choice.mean())}


def main(args):
    configurations, datasets = [], []
    for name, partition in [(args.train, 'controller_train'), (args.validation, 'validation'), (args.audit, 'audit')]:
        config, data = load_run(name, partition)
        configurations.append(config)
        datasets.append(data)
    for i in range(3):
        for j in range(i):
            assert not set(datasets[i]) & set(datasets[j]), 'Partition leakage'
    for key in ['model_digest', 'rules', 'plan_instruction', 'action_schema', 'plan_schema', 'temperature', 'think', 'num_ctx', 'num_predict', 'code_sha256', 'data_manifest_sha256']:
        assert all(c[key] == configurations[0][key] for c in configurations), f'Mismatched {key}'
    train, val, audit = [matrices(d) for d in datasets]
    vectorizer = DictVectorizer(sparse=False).fit(train[0])
    xt, xv, xa = [vectorizer.transform(d[0]) for d in [train, val, audit]]
    cost_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=SEED).fit(xt, train[2])
    cv, ca = cost_model.predict(xv), cost_model.predict(xa)
    results, models = {}, {}
    # Fixed small search; only validation determines classifier family and threshold.
    for family in ['logistic_regression', 'random_forest']:
        fitted = []
        for s in range(2):
            y = train[1][:, s]
            if len(set(y)) == 1:
                model = DummyClassifier(strategy='prior')
            elif family == 'logistic_regression':
                model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=SEED))
            else:
                model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=5, random_state=SEED)
            model.fit(xt, y)
            fitted.append(model)
        pv = np.column_stack([probability(m, xv) for m in fitted])
        candidates = []
        for threshold in [0, .25, .5, .6, .7, .8, .9, .95]:
            m = metrics(route(pv, cv, threshold), val[1], val[2], val[3])
            candidates.append({'threshold': threshold, **m})
        results[family] = {'validation_candidates': candidates, 'validation_brier_by_strategy': [brier_score_loss(val[1][:, s], pv[:, s]) for s in range(2)]}
        models[family] = fitted
    candidates = [(family, c) for family, r in results.items() for c in r['validation_candidates']]
    feasible = [(f, c) for f, c in candidates if c['accuracy'] >= args.target]
    if feasible:
        family, selection = min(feasible, key=lambda item: (item[1]['mean_total_tokens'], -item[1]['accuracy']))
    else:
        family, selection = min(candidates, key=lambda item: (-item[1]['accuracy'], item[1]['mean_total_tokens']))
    # Freeze selection before calculating the exploratory held-out audit results.
    start = time.perf_counter()
    pa = np.column_stack([probability(m, xa) for m in models[family]])
    choice = route(pa, ca, selection['threshold'])
    decision_seconds = time.perf_counter() - start
    audit_metrics = {s: metrics(np.full(len(xa), i), audit[1], audit[2], audit[3]) for i, s in enumerate(STRATEGIES)}
    audit_metrics['controller'] = metrics(choice, audit[1], audit[2], audit[3])
    report = {'runs': vars(args), 'selected_family': family, 'selected_threshold': selection['threshold'], 'validation_target_met': bool(feasible), 'selection_validation_metrics': selection, 'search': results, 'training_success_counts': train[1].sum(axis=0).tolist(), 'audit': audit_metrics, 'audit_oracle_accuracy_upper_bound': float(audit[1].max(axis=1).mean()), 'classifier_and_routing_batch_seconds': decision_seconds, 'caveats': ['Exploratory small sample; official test untouched.', 'Runtime is paired-run replay, not an end-to-end deployment benchmark.', 'Routing timing excludes feature extraction and cost prediction; measure full overhead in final evaluation.', 'Total input plus output tokens is a cost proxy, not FLOPs or energy.', 'Validation success threshold is empirical, not a per-example success guarantee.', 'Accuracy intervals are marginal Wilson intervals, not paired significance tests.']}
    dest = ROOT / 'results' / args.name
    save(dest / 'report.json', report)
    save(dest / 'audit_predictions.json', [{'id': id_, 'choice': STRATEGIES[int(choice[i])], 'success_probabilities': pa[i].tolist(), 'predicted_token_costs': ca[i].tolist()} for i, id_ in enumerate(datasets[2])])
    joblib.dump({'vectorizer': vectorizer, 'success_models': models[family], 'cost_model': cost_model, 'threshold': selection['threshold'], 'family': family}, dest / 'controller.joblib')
    print(json.dumps({k: v for k, v in report.items() if k != 'search'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True)
    p.add_argument('--validation', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--target', type=float, default=.8)
    p.add_argument('--name', default='controller-v1')
    main(p.parse_args())
