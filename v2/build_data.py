"""Deterministically materialize v2 partitions without modifying v1 data."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from experiment import ROOT, SEED, parse_scan, save, verify_data
from v2.domains import chainops, scan

COUNTS = {'dev': 128, 'router_train': 600, 'router_val': 200, 'holdout': 400}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((''.join(canonical_json(row) + '\n' for row in rows)).encode('utf-8'))


def make_scan():
    sources = {'dev': ('development', slice(None)), 'router_train': ('controller_train', slice(600)),
               'router_val': ('validation', slice(200)), 'holdout': ('audit', slice(32, 432))}
    raw = {split: json.loads((ROOT / f'data/{name}.json').read_text())[selection]
           for split, (name, selection) in sources.items()}
    raw['ood_length'] = random.Random(SEED).sample(parse_scan(
        (ROOT / 'data/SCAN/length_split/tasks_test_length.txt').read_text()), 300)
    return {split: [{'id': r['id'], 'domain': 'scan', 'split': split, 'input': r['command'],
                     'target': r['target'], 'gold_plan': scan.parse(r['command']),
                     'meta': scan.meta(r['command'], r['target'])} for r in rows] for split, rows in raw.items()}


def make_chainops():
    rng, seen_text, seen_plan = random.Random(SEED), set(), set()
    splits = {}
    for split, count in {**COUNTS, 'ood_depth': 300}.items():
        ood = split == 'ood_depth'
        depths = (6, 7, 8) if ood else (1, 2, 3, 4)
        rows = []
        while len(rows) < count:
            depth = depths[len(rows) % len(depths)]
            start, ops, target = chainops.generate(rng, depth, ood)
            plan = {'start': start, 'ops': ops}
            text = chainops.render(start, ops, rng)
            key = canonical_json(plan)
            if text in seen_text or key in seen_plan:
                continue
            seen_text.add(text)
            seen_plan.add(key)
            rows.append({'id': hashlib.sha256(text.encode()).hexdigest()[:16], 'domain': 'chainops',
                         'split': split, 'input': text, 'target': target, 'gold_plan': plan,
                         'meta': chainops.meta(start, ops, target)})
        splits[split] = rows
    return splits


def verify(splits):
    train_commands = {r['command'] for r in parse_scan(
        (ROOT / 'data/SCAN/length_split/tasks_train_length.txt').read_text())}
    for domain, partitions in splits.items():
        ids, texts, plans = set(), set(), set()
        for split, rows in partitions.items():
            expected = COUNTS.get(split, 300)
            if len(rows) != expected:
                raise ValueError(f'{domain}/{split}: expected {expected} items')
            for row in rows:
                if row['id'] in ids or row['input'] in texts:
                    raise ValueError(f'Duplicate ID/text in {domain}: {row["id"]}')
                ids.add(row['id'])
                texts.add(row['input'])
                plan = row['gold_plan']
                if domain == 'scan':
                    target = [token for step in plan['steps'] for leaf in scan.expand(step) for token in scan.leaf_gold(leaf)]
                    if split == 'ood_length' and row['input'] in train_commands:
                        raise ValueError('OOD command overlaps train pool')
                else:
                    key = canonical_json(plan)
                    if key in plans:
                        raise ValueError('Duplicate ChainOps plan')
                    plans.add(key)
                    target = chainops.execute(plan['start'], plan['ops'])
                if target != row['target']:
                    raise ValueError(f'Gold executor mismatch: {row["id"]}')


def build(out_dir=ROOT / 'data/v2'):
    verify_data()
    out_dir = Path(out_dir)
    splits = {'scan': make_scan(), 'chainops': make_chainops()}
    verify(splits)
    source = json.loads((ROOT / 'data/manifest.json').read_text())
    code_paths = [Path(__file__), Path(scan.__file__), Path(chainops.__file__)]
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in code_paths}
    manifest = {'seed': SEED, 'generator_code_sha256': hashes,
                'source_partition_hashes': source['partition_sha256'],
                'source_manifest_sha256': sha256(ROOT / 'data/manifest.json'), 'files': {}}
    for domain, partitions in splits.items():
        for split, rows in partitions.items():
            name = f'{domain}/{split}.jsonl'
            path = out_dir / name
            write_jsonl(path, rows)
            manifest['files'][name] = {'count': len(rows), 'sha256': sha256(path),
                                       'locked': split in ('ood_length', 'ood_depth')}
    save(out_dir / 'manifest.json', manifest)
    # Read back the delivered bytes, not just the objects used to create them.
    verify({d: {s: read_jsonl(out_dir / d / f'{s}.jsonl') for s in partitions}
            for d, partitions in splits.items()})
    verify_data()
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'data/v2')
    args = parser.parse_args()
    result = build(args.out_dir)
    print(f'Wrote and verified {sum(f["count"] for f in result["files"].values())} items in {len(result["files"])} files.')
