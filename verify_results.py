"""Read delivered files afresh and recompute scores, counts, and saved routing."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from controller import features, probability, route

ROOT = Path(__file__).resolve().parent


def main():
    checks = {}
    for name in ['pilot-v1', 'pilot-8b-v1', 'train-8b-v1', 'validation-8b-v1', 'audit-8b-v1']:
        folder = ROOT / 'results' / name
        config = json.loads((folder / 'config.json').read_text())
        assert hashlib.sha256((folder / 'experiment.py').read_bytes()).hexdigest() == config['code_sha256']
        partition = {r['id']: r for r in json.loads((ROOT / 'data' / (config['partition'] + '.json')).read_text())[:config['count']]}
        records = [json.loads(line) for line in (folder / 'records.jsonl').read_text().splitlines()]
        summary = json.loads((folder / 'summary.json').read_text())
        seen = set()
        for r in records:
            key = (r['id'], r['strategy'])
            assert key not in seen
            seen.add(key)
            assert partition[r['id']]['target'] == r['target']
            assert partition[r['id']]['command'] == r['command']
            assert r['correct'] == (r['error'] is None and r['prediction'] == r['target'])
            assert r['prompt_tokens'] == sum(c['prompt_eval_count'] for c in r['calls'])
            assert r['completion_tokens'] == sum(c['eval_count'] for c in r['calls'])
        assert len(records) == 2 * config['count']
        for strategy in ['direct', 'decomposed']:
            subset = [r for r in records if r['strategy'] == strategy]
            assert len(subset) == summary[strategy]['n']
            assert sum(r['correct'] for r in subset) == summary[strategy]['correct']
            assert abs(sum(r['wall_seconds'] for r in subset) / len(subset) - summary[strategy]['mean_seconds']) < 1e-9
        checks[name] = {'records': len(records), 'scoring_tokens_source_and_summary': 'passed'}
    folder = ROOT / 'results/controller-v1'
    model = joblib.load(folder / 'controller.joblib')
    predictions = json.loads((folder / 'audit_predictions.json').read_text())
    report = json.loads((folder / 'report.json').read_text())
    audit = {r['id']: r for r in json.loads((ROOT / 'data/audit.json').read_text())}
    x = model['vectorizer'].transform([features(audit[p['id']]['command']) for p in predictions])
    probabilities = np.column_stack([probability(m, x) for m in model['success_models']])
    costs = model['cost_model'].predict(x)
    choices = route(probabilities, costs, model['threshold'])
    strategy_names = ['direct', 'decomposed']
    assert [strategy_names[i] for i in choices] == [p['choice'] for p in predictions]
    records = [json.loads(line) for line in (ROOT / 'results/audit-8b-v1/records.jsonl').read_text().splitlines()]
    paired = {(r['id'], r['strategy']): r for r in records}
    selected = [paired[(p['id'], p['choice'])] for p in predictions]
    assert sum(r['correct'] for r in selected) == report['audit']['controller']['correct']
    assert abs(sum(r['prompt_tokens'] + r['completion_tokens'] for r in selected) / len(selected) - report['audit']['controller']['mean_total_tokens']) < 1e-9
    checks['saved_controller'] = {'predictions': len(predictions), 'reload_routing_and_metrics': 'passed'}
    (ROOT / 'results/verification.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
