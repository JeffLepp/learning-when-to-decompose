"""Rebuild the historical logistic v1 route using only the v2 library."""

import json
from pathlib import Path

from .data import ROOT, load_items, load_labels, structural_features, write_v1_adapter
from .models import CostPredictor, SuccessPredictor
from .policies import decision_cost, decision_success, threshold_route


def _metrics(decision, table):
    success = decision_success(decision, table)
    return dict(n=len(table.ids), correct=int(success.sum()),
                accuracy=float(success.mean()),
                mean_total_tokens=float(decision_cost(decision, table).mean()))


def reproduce(root=ROOT):
    adapter = write_v1_adapter(root)
    tables = load_labels(adapter / 'scan_v1_labels.csv')
    items = load_items(adapter / 'scan_v1_items.jsonl')
    train, val, audit = [tables[s] for s in ['router_train', 'router_val', 'v1_audit']]
    matrices = [structural_features([items[id_]['input'] for id_ in t.ids], 'scan')[0]
                for t in [train, val, audit]]
    xt, xv, xa = matrices
    success = SuccessPredictor('logreg').fit(xt, train.success)
    costs = CostPredictor('rf_joint').fit(xt, train.cost)
    pv, cv = success.predict_proba(xv), costs.predict(xv)
    candidates = []
    for threshold in [0, .25, .5, .6, .7, .8, .9, .95]:
        decision = threshold_route(pv, cv, threshold)
        candidates.append(dict(threshold=threshold, **_metrics(decision, val)))
    feasible = [c for c in candidates if c['accuracy'] >= .8]
    selection = min(feasible, key=lambda c: (c['mean_total_tokens'], -c['accuracy'])) \
        if feasible else min(candidates, key=lambda c: (-c['accuracy'], c['mean_total_tokens']))
    # Selection is fixed before historical audit predictions or expected choices are read.
    pa, ca = success.predict_proba(xa), costs.predict(xa)
    decision = threshold_route(pa, ca, selection['threshold'])
    audit_metrics = _metrics(decision, audit)
    choices = {id_: audit.arms[int(a)] for id_, a in zip(audit.ids, decision.final)}
    original = json.loads((Path(root) / 'results/controller-v1/audit_predictions.json')
                          .read_text(encoding='utf-8'))
    expected_choices = {
        r['id']: 'qwen3:8b|' + {'direct': 'S0', 'decomposed': 'S3'}[r['choice']]
        for r in original
    }
    measured = dict(
        selected_threshold=selection['threshold'], validation_n=selection['n'],
        validation_correct=selection['correct'],
        validation_mean_tokens=selection['mean_total_tokens'],
        audit_n=audit_metrics['n'], audit_correct=audit_metrics['correct'],
        audit_mean_tokens=audit_metrics['mean_total_tokens'], audit_choices=choices,
    )
    expected = dict(selected_threshold=.5, validation_n=32, validation_correct=10,
                    validation_mean_tokens=629.9375, audit_n=32, audit_correct=15,
                    audit_mean_tokens=625.09375, audit_choices=expected_choices)
    checks = {k: dict(measured=measured[k], expected=v, match=measured[k] == v)
              for k, v in expected.items()}
    report = dict(checks=checks, match=all(c['match'] for c in checks.values()),
                  validation_target=.8, validation_target_met=bool(feasible),
                  validation_candidates=candidates, selection=selection,
                  audit=audit_metrics,
                  note='Historical v1 regression only; no v2 final evaluation.')
    dest = Path(root) / 'results/v2/router/v1_repro'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    result = reproduce()
    print(json.dumps(dict(match=result['match'], selection=result['selection'],
                         audit=result['audit']), indent=2))
