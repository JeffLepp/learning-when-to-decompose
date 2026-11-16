"""Dev-only prompt scoring/selection; never performs inference.

CLI: init; log DOMAIN ROUND --rationales FILE; select.
Canonical block hashes use sorted keys, compact separators, ensure_ascii=False.
All output is confined to P1 prompt files and results/v2/prompt_dev.
"""
import argparse
import copy
import hashlib
import json
import random
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v2/prompt_dev'
PROMPTS = ROOT / 'v2/domains/prompts'
MODELS = ('qwen3:1.7b', 'qwen3:8b')
STRATEGIES = ('S0', 'S1', 'S3', 'S4')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def block_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def create(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def features(item):
    if item['domain'] == 'chainops':
        return set(item['meta']['ops']) | {'depth:' + str(item['meta']['depth'])}
    result = {'conj:' + item['meta']['conj']}
    for step in item['gold_plan']['steps']:
        result |= {f'{key}:{step[key]}' for key in ('action', 'direction', 'modifier', 'repeat')}
        result.add('clause:' + step['command'])
    return result


def init():
    for domain in ('scan', 'chainops'):
        items = rows(ROOT / f'data/v2/{domain}/dev.jsonl')
        assert len(items) == 128 and all(x['split'] == 'dev' for x in items)
        base = read(PROMPTS / f'{domain}_v0.json')
        pool = list(base['exemplar_ids'])
        by_id = {x['id']: x for x in items}
        covered = set().union(*(features(by_id[i]) for i in pool))
        candidates = [x for x in items if x['id'] not in pool]
        random.Random(437).shuffle(candidates)
        while len(pool) < 16:
            best = max(candidates, key=lambda x: len(features(x) - covered))
            pool.append(best['id'])
            covered |= features(best)
            candidates.remove(best)
        remaining = [x['id'] for x in items if x['id'] not in pool]
        score = random.Random(437).sample(remaining, 60)
        spec = {'domain': domain, 'seed': 437, 'exemplar_pool': pool,
                'score_ids': score, 'final_ids': remaining,
                'pool_method': 'v0 IDs then greedy uncovered grammar/operation features; seed-shuffled tie order',
                'covered_features': sorted(covered)}
        create(OUT / f'{domain}_ids.json', spec)
        create(OUT / f'{domain}_score_ids.json', score)
        create(OUT / f'{domain}_final_ids.json', remaining)
        base['version'] = domain + '_v0p'
        base['exemplar_ids'] = pool
        create(PROMPTS / f'{domain}_v0p.json', base)


def score_records(records, ids):
    """Require a complete paired round; failure labels remain in denominators."""
    ids = set(ids)
    expected = {(i, m, s) for i in ids for m in MODELS for s in STRATEGIES}
    keys = [(r['id'], r['model'], r['strategy']) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Round must contain exactly one record per score ID/model/strategy')
    if any(r.get('split') != 'dev' for r in records):
        raise ValueError('Only dev records may be scored')
    result = {}
    for strategy in STRATEGIES:
        groups = [[r for r in records if r['strategy'] == strategy and r['model'] == model] for model in MODELS]
        accuracies = [statistics.mean(r['correct'] for r in group) for group in groups]
        result[strategy] = {'n_items': len(ids), 'acc_1.7b': accuracies[0], 'acc_8b': accuracies[1],
                            'mean_acc': statistics.mean(accuracies),
                            'mean_tokens_1.7b': statistics.mean(r['total_tokens'] for r in groups[0]),
                            'mean_tokens_8b': statistics.mean(r['total_tokens'] for r in groups[1]),
                            'error_counts': {m: dict(Counter(r['error'] or 'semantic_mismatch' for r in group if not r['correct']))
                                             for m, group in zip(MODELS, groups)}}
    return result


def select_best(log):
    """Highest two-model mean, ties to earliest numeric round."""
    best = {}
    for row in sorted(log, key=lambda r: r['round']):
        key = (row['domain'], row['strategy'])
        if key not in best or row['mean_acc'] > best[key]['mean_acc']:
            best[key] = row
    return best


def log_round(domain, number, rationale_path):
    if number not in range(5):
        raise ValueError('At most five rounds including zero')
    version = '0p' if number == 0 else str(number)
    prompt_path = PROMPTS / f'{domain}_v{version}.json'
    prompts = read(prompt_path)
    ids = read(OUT / f'{domain}_ids.json')
    assert prompts['exemplar_ids'] == ids['exemplar_pool'] and prompts['S2'] == prompts['S0']
    for strategy in STRATEGIES:
        for block in prompts[strategy].values():
            assert all(e['id'] in ids['exemplar_pool'] for e in block['examples'])
    run_name = f'pd-{domain}-v{version}'
    run_path = ROOT / 'results/v2/runs' / run_name
    config = read(run_path / 'config.json')
    assert config['split'] == 'dev' and config['prompts_sha256'] == file_hash(prompt_path)
    assert set(config['item_ids']) == set(ids['score_ids'])
    records = rows(run_path / 'records.jsonl')
    assert all(r['domain'] == domain for r in records)
    scores = score_records(records, ids['score_ids'])
    rationales = read(rationale_path)
    path = OUT / 'log.jsonl'
    previous = rows(path) if path.exists() else []
    if any(r['domain'] == domain and r['round'] == number for r in previous):
        raise ValueError('Round already logged; earlier lines are immutable')
    if set(r['round'] for r in previous if r['domain'] == domain) != set(range(number)):
        raise ValueError('Rounds must be logged in order')
    additions = [{**scores[s], 'domain': domain, 'round': number, 'strategy': s,
                  'prompts_file': prompt_path.relative_to(ROOT).as_posix(),
                  'prompts_sha256': file_hash(prompt_path), 'block_sha256': block_hash(prompts[s]),
                  'run_name': run_name, 'rationale': rationales[s],
                  'time': datetime.now(timezone.utc).isoformat()} for s in STRATEGIES]
    with path.open('a', encoding='utf-8', newline='\n') as stream:
        for row in additions:
            stream.write(canonical(row) + '\n')
    print(json.dumps(additions, indent=2))


def select():
    best = select_best(rows(OUT / 'log.jsonl'))
    selection = {}
    for domain in ('scan', 'chainops'):
        final = {'domain': domain, 'version': domain + '_final',
                 'exemplar_ids': read(OUT / f'{domain}_ids.json')['exemplar_pool'], 'provenance': {}}
        chosen = {}
        for strategy in STRATEGIES:
            row = best[(domain, strategy)]
            source = ROOT / row['prompts_file']
            assert file_hash(source) == row['prompts_sha256']
            block = read(source)[strategy]
            assert block_hash(block) == row['block_sha256']
            final[strategy] = block
            final['provenance'][strategy] = {'round': row['round'], 'block_sha256': row['block_sha256']}
            chosen[strategy] = {key: row[key] for key in ('round', 'acc_1.7b', 'acc_8b', 'mean_acc', 'block_sha256')}
        final['S2'] = copy.deepcopy(final['S0'])
        final['provenance']['S2'] = {**final['provenance']['S0'], 'copied_from': 'S0'}
        path = PROMPTS / f'{domain}_final.json'
        create(path, final)
        selection[domain] = {'strategies': chosen, 'final_file': path.relative_to(ROOT).as_posix(),
                             'final_sha256': file_hash(path)}
    create(OUT / 'selection.json', selection)
    print(json.dumps(selection, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('init')
    sub.add_parser('select')
    log_parser = sub.add_parser('log')
    log_parser.add_argument('domain', choices=['scan', 'chainops'])
    log_parser.add_argument('round', type=int)
    log_parser.add_argument('--rationales', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'init': init()
    elif args.action == 'select': select()
    else: log_round(args.domain, args.round, args.rationales)
