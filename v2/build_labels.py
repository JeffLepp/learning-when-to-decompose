"""Rebuild deterministic label CSVs from immutable run records."""
import argparse
import csv
import hashlib
from pathlib import Path

from experiment import ROOT
from v2.build_data import canonical_json, read_jsonl

COLUMNS = ['id', 'split', 'model', 'strategy', 'correct', 'total_tokens', 'prompt_tokens',
           'completion_tokens', 'wall_seconds', 'n_calls', 'error', 'mean_logprob',
           'min_logprob', 'answer_sha1', 'thinking_tokens', 'run_name']


def build(runs_dir=ROOT / 'results/v2/runs', out_dir=ROOT / 'results/v2/labels', allow_runs=None):
    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    selected = set(allow_runs) if allow_runs is not None else None
    paths = [p for p in sorted(runs_dir.glob('*/records.jsonl')) if selected is None or p.parent.name in selected]
    if selected is not None and selected != {p.parent.name for p in paths}:
        raise ValueError('An explicitly selected run is missing records.jsonl')
    groups, seen = {}, set()
    for path in paths:
        for record in read_jsonl(path):
            key = (record['domain'], record['id'], record['model'], record['strategy'])
            if key in seen:
                raise ValueError(f'Duplicate label {key}; use --allow-runs to select nonoverlapping runs explicitly')
            seen.add(key)
            row = {k: record[k] for k in COLUMNS if k in record}
            row.update(correct=int(record['correct']), error=record['error'] or '',
                       mean_logprob=record['logprob_stats']['mean'], min_logprob=record['logprob_stats']['min'],
                       answer_sha1='' if record['prediction'] is None else hashlib.sha1(canonical_json(record['prediction']).encode()).hexdigest(),
                       run_name=path.parent.name)
            groups.setdefault(record['domain'], []).append(row)
    # Validate all inputs before writing any output.
    out_dir.mkdir(parents=True, exist_ok=True)
    for domain, rows in sorted(groups.items()):
        rows.sort(key=lambda r: (r['id'], r['model'], r['strategy'], r['split'], r['run_name']))
        with (out_dir / (domain + '.csv')).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    return {domain: len(rows) for domain, rows in groups.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir', type=Path, default=ROOT / 'results/v2/runs')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'results/v2/labels')
    parser.add_argument('--allow-runs', help='Comma-separated run names; selection must still be unambiguous')
    args = parser.parse_args()
    print(build(args.runs_dir, args.out_dir, args.allow_runs.split(',') if args.allow_runs else None))
