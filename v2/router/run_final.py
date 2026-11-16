"""One-shot frozen P5 evaluation; synthetic Mode-B fixtures are the only tag bypass."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits

from .data import ROOT, load_labels, main_table, s2_table
from .evaluate import write_json
from .predictor_report import predictor_report
from .workflow import (apply_state, label_fingerprint, output_directory,
                       restore_experiment, sha256, summary, family_features, inputs_for)


def verify(p4_dir, labels, items_dir, synthetic_test):
    p4_dir = output_directory(p4_dir)
    report = json.loads((p4_dir / 'report.json').read_text(encoding='utf-8'))
    if not report.get('p4_completed'):
        raise RuntimeError('P4 must be completed before final evaluation')
    provenance = report.get('synthetic_labels') or {}
    if synthetic_test:
        if (provenance.get('generator') != 'v2.router.synth_labels' or provenance.get('mode') != 'B'
                or Path(items_dir).resolve().is_relative_to((ROOT / 'data').resolve())):
            raise RuntimeError('--synthetic-test requires Mode-B P4 and items outside data/')
    else:
        tag = subprocess.run(['git', 'rev-parse', '-q', '--verify', 'refs/tags/freeze-v2'],
                             cwd=ROOT, capture_output=True, text=True)
        if tag.returncode:
            raise RuntimeError('freeze-v2 git tag required; no final data were opened')
    current_code = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'v2/router').glob('*.py'))}
    if current_code != report['code_sha256']:
        raise RuntimeError('Router code sha256 mismatch; freeze invalid')
    frozen = p4_dir / 'frozen'
    current_artifacts = {p.name: sha256(p) for p in sorted(frozen.iterdir()) if p.is_file()}
    if current_artifacts != report['frozen_sha256']:
        raise RuntimeError('Frozen artifact sha256 mismatch')
    for path, digest in report['input_sha256'].items():
        if path == report['labels_path']:
            # Full-file hash is provenance. Final rows can be collected/appended
            # later; the frozen development-row hash is the integrity boundary.
            continue
        if sha256(path) != digest:
            raise RuntimeError(f'P4 input sha256 mismatch: {path}')
    for path in [report['labels_path'], labels]:
        if label_fingerprint(path) != report['development_labels_sha256']:
            raise RuntimeError('router_train/router_val labels sha256 mismatch')
    # Final evaluation may use a different directory, but its development items
    # must still be exactly those used in P4.
    for split in ['router_train', 'router_val']:
        original = str(Path(report['items_dir']) / f'{split}.jsonl')
        if sha256(Path(items_dir) / f'{split}.jsonl') != report['input_sha256'][original]:
            raise RuntimeError('Development item sha256 mismatch')
    return report


def run(p4_dir, labels, items_dir, splits, features_dir=None, synthetic_test=False):
    started = time.perf_counter()
    p4_dir, labels, items_dir = output_directory(p4_dir), Path(labels).resolve(), Path(items_dir).resolve()
    splits = splits.split(',') if isinstance(splits, str) else list(splits)
    if not splits or len(set(splits)) != len(splits) or not set(splits) <= {'holdout', 'ood_length', 'ood_depth'}:
        raise ValueError('Request distinct holdout/ood_length/ood_depth splits')
    report = verify(p4_dir, labels, items_dir, synthetic_test)
    allowed_ood = 'ood_length' if report['domain'] == 'scan' else 'ood_depth'
    if not set(splits) <= {'holdout', allowed_ood}:
        raise ValueError('OOD split does not match the frozen domain')
    # Check the persisted one-shot guard before opening any final item/label row.
    for split in splits:
        log = p4_dir / f'final_{split}/evaluate_calls.jsonl'
        if log.exists() and any(json.loads(line)['status'] == 'completed' for line in log.read_text().splitlines()):
            raise RuntimeError(f'Final evaluation already completed for {split}')
    # Only trusted, hash-verified locally generated artifacts are unpickled.
    state = joblib.load(p4_dir / 'frozen/objects.joblib')
    features_dir = Path(features_dir) if features_dir is not None else report['features_dir']
    results = {}
    for split in splits:
        split_started = time.perf_counter()
        if synthetic_test:
            digest = report['synthetic_labels']['item_sha256'].get(f'{split}.jsonl')
            if digest is None or sha256(items_dir / f'{split}.jsonl') != digest:
                raise RuntimeError('Synthetic final item provenance mismatch')
        raw = load_labels(labels, splits=[split])[split]
        table = main_table(raw, expected_n_arms=report['expected_arms'])
        if synthetic_test and not all(i.startswith('synthetic-') for i in table.ids):
            raise RuntimeError('Non-synthetic final IDs refused')
        dest = p4_dir / f'final_{split}'
        experiment = restore_experiment(state, dest, report['uncertainty']['n_boot'])
        policies = apply_state(state, table, items_dir, features_dir)
        final = experiment.evaluate_on(table, policies, final_eval=True,
                                       fast_bootstrap=True, comparison_scope='best_single',
                                       contrast_pairs=state['contrast_pairs'])
        matrices, _ = family_features(state, table, inputs_for(table, items_dir, state['domain']), features_dir)
        final.update(predictor_report(state, table, matrices, report['uncertainty']['n_boot']))
        s2_report = None
        if state['s2'] is not None and any(a.endswith('|S2') for a in raw.arms):
            subset = s2_table(raw)
            if subset.ids:
                s2_exp = restore_experiment(state['s2'], dest / 's2', report['uncertainty']['n_boot'])
                s2_report = s2_exp.evaluate_on(subset, apply_state(state['s2'], subset, items_dir),
                                              final_eval=True, fast_bootstrap=True, comparison_scope='best_single')
        final.update(note=f'{split}: frozen evaluation; no refitting or selection',
                     synthetic_labels=report['synthetic_labels'], s2=s2_report,
                     selected_router=report['selected_router'], p4_report_sha256=sha256(p4_dir / 'report.json'),
                     selections=state['selections'], skipped=report['skipped'],
                     labels_sha256=sha256(labels), items_sha256=sha256(items_dir / f'{split}.jsonl'),
                     wall_seconds=time.perf_counter() - split_started, model_calls=0)
        write_json(dest / 'report.json', final)
        summary(final, dest / 'summary.md', final['note'])
        results[split] = final
        print(f'Frozen {split} complete: {final["wall_seconds"]:.2f}s', flush=True)
    print(f'Final driver total: {time.perf_counter() - started:.2f}s', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p4-dir', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--items-dir', type=Path, required=True)
    parser.add_argument('--splits', required=True)
    parser.add_argument('--features-dir', type=Path)
    parser.add_argument('--synthetic-test', action='store_true')
    with threadpool_limits(limits=1):
        run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
