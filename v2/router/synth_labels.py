"""Deterministic CPU fixtures, never evidence about real model capability."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from v2.build_labels import COLUMNS
from .data import ROOT, MODELS, DEPLOYABLE_MAIN, load_items, model_dir_name, structural_features
from .evaluate import write_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fake_items(domain, rng):
    """Fake, deliberately non-benchmark inputs; no real final files are read."""
    ood = 'ood_length' if domain == 'scan' else 'ood_depth'
    result = []
    for split, n in [('router_train', 600), ('router_val', 200), ('holdout', 400), (ood, 300)]:
        for i in range(n):
            depth = int(rng.integers(6, 9) if split == ood else rng.integers(1, 5))
            if domain == 'scan':
                clauses = [' '.join([str(rng.choice(['walk', 'run', 'jump', 'look'])),
                                     str(rng.choice(['', 'left', 'around right', 'opposite left'])),
                                     str(rng.choice(['', 'twice', 'thrice']))]).strip()
                           for _ in range(depth)]
                text = ' and '.join(clauses)
                meta = dict(depth=depth, n_clauses=depth, target_len=depth * 4)
            else:
                length = int(rng.integers(6, 9) if split == ood else rng.integers(3, 7))
                ops = rng.choice(['reverse', 'sort_asc', 'add', 'multiply', 'keep_even', 'rotate_left'], depth).tolist()
                wording = dict(reverse='Reverse the list.', sort_asc='Sort ascending.', add='Add 2 to each number.',
                               multiply='Multiply each by 2.', keep_even='Keep only even numbers.', rotate_left='Rotate left by 1.')
                text = f'Start with {rng.integers(0, 20, length).tolist()}. ' + ' Then '.join(wording[o] for o in ops)
                meta = dict(depth=depth, n_ops=depth, ops=ops, list_len=length, target_len=length)
            result.append(dict(id=f'synthetic-{domain}-{split}-{i:04}', domain=domain, split=split,
                               input=text, target=[], gold_plan={}, meta=meta))
    return result


def stratified_subset(items, difficulty, n, rng):
    """Proportional systematic sampling across difficulty-sorted, randomized ties."""
    order = np.lexsort((rng.random(len(items)), difficulty))
    groups = np.array_split(order, min(n, len(order)))
    return {items[int(rng.choice(g))]['id'] for g in groups}


def generate(domain, out, items_dir=None, synthetic_items_out=None, features_out=None, seed=437):
    rng = np.random.default_rng(seed)
    out = Path(out)
    if synthetic_items_out is not None:
        directory = Path(synthetic_items_out).resolve()
        if directory.is_relative_to((ROOT / 'data').resolve()):
            raise ValueError('Synthetic items must not be under data/')
        items = fake_items(domain, rng)
        directory.mkdir(parents=True, exist_ok=True)
        for split in dict.fromkeys(i['split'] for i in items):
            path = directory / f'{split}.jsonl'
            with path.open('w', encoding='utf-8') as stream:
                for item in items:
                    if item['split'] == split:
                        stream.write(json.dumps(item, sort_keys=True) + '\n')
        mode = 'B'
    else:
        directory = Path(items_dir or ROOT / 'data/v2' / domain).resolve()
        items = [i for s in ['router_train', 'router_val']
                 for i in load_items(directory / f'{s}.jsonl').values()]
        mode = 'A'
    x, names = structural_features([i['input'] for i in items], domain, [i['meta'] for i in items])
    word = x[:, names.index('word_count')]
    complexity = (sum(x[:, names.index('count_' + w)] for w in
                      ['and', 'after', 'around', 'opposite', 'twice', 'thrice']) if domain == 'scan'
                  else x[:, names.index('n_ops')])
    # Fixed scales: OOD remains harder instead of being normalized within split.
    structural = .12 * word + .36 * complexity
    latent = structural + rng.normal(0, .65, len(items))
    model_difficulty = np.column_stack([latent + rng.normal(0, .35, len(items)) for _ in MODELS])
    s2_ids, gold_ids = set(), set()
    counts = dict(router_train=150, router_val=100, holdout=150, ood_length=100, ood_depth=100)
    for split in dict.fromkeys(i['split'] for i in items):
        indices = [j for j, i in enumerate(items) if i['split'] == split]
        subset = [items[j] for j in indices]
        s2_ids |= stratified_subset(subset, structural[indices], counts[split], rng)
        if split == 'router_train':
            gold_ids |= stratified_subset(subset, structural[indices], 200, rng)
        elif split == 'holdout':
            gold_ids |= {i['id'] for i in subset}
    rows = []
    for j, item in enumerate(items):
        for m, model in enumerate(MODELS):
            strategies = list(DEPLOYABLE_MAIN) + (['S2'] if item['id'] in s2_ids else [])
            if m == 3 and item['id'] in gold_ids:
                strategies += ['G1', 'G2']
            for strategy in strategies:
                d = model_difficulty[j, m]
                effect = {'S0': 0., 'S1': .35 - .07 * complexity[j],
                          'S2': .25 + .13 * complexity[j],
                          'S3': -.8 + .38 * complexity[j],
                          'S4': -1.2 + .53 * complexity[j], 'G1': 1.4, 'G2': 1.2}[strategy]
                p = expit(2.8 + .50 * m - .62 * d + effect)
                error = bool(rng.random() < .008)
                correct = int(rng.random() < p and not error)
                calls = 1 if strategy in ['S0', 'S1', 'S2'] else 2 + int(complexity[j] // 2)
                prompt = int(65 + 2 * word[j] + 16 * m + 55 * (calls - 1) + (85 if strategy == 'S1' else 0))
                thinking = int(rng.integers(100, 350) + 12 * complexity[j]) if strategy == 'S2' else 0
                completion = int(16 + 3 * word[j] + 22 * (calls - 1) + rng.integers(0, 20)) + thinking
                logprob = float(-.2 - 1.2 * (1 - correct) + rng.normal(0, .4))
                answer = item['id'] + ('-correct' if correct else f'-wrong-{rng.integers(0, 20)}')
                rows.append(dict(id=item['id'], split=item['split'], model=model, strategy=strategy,
                                 correct=correct, total_tokens=prompt + completion, prompt_tokens=prompt,
                                 completion_tokens=completion, thinking_tokens=thinking,
                                 wall_seconds=round((prompt + completion) * (m + 1) / 1500, 6),
                                 n_calls=calls, error='synthetic-format-error' if error else '',
                                 mean_logprob=logprob, min_logprob=logprob - abs(float(rng.normal(1, .3))),
                                 answer_sha1=hashlib.sha1(answer.encode()).hexdigest(),
                                 run_name=f'synthetic-{model}-{strategy}'))
    rows.sort(key=lambda r: (r['id'], r['model'], r['strategy'], r['split'], r['run_name']))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    if features_out is not None:
        for m, model in enumerate(MODELS):
            dest = Path(features_out) / model_dir_name(model)
            dest.mkdir(parents=True, exist_ok=True)
            files = {}
            for layer in [4, 8]:
                for pooling in ['last', 'mean']:
                    key = f'L{layer:02}_{pooling}'
                    features = rng.normal(size=(len(items), 32))
                    if layer == 4:
                        features[:, 0] = model_difficulty[:, m] + rng.normal(0, .15, len(items))
                        features[:, 1] = complexity + rng.normal(0, .15, len(items))
                    np.save(dest / f'{key}.npy', features.astype(np.float16))
                    files[key] = f'{key}.npy'
            write_json(dest / 'index.json', dict(ids=[i['id'] for i in items],
                       splits=[i['split'] for i in items], n_tokens=(65 + 2 * word + 16 * m).astype(int),
                       layers=[4, 8], poolings=['last', 'mean'], files=files,
                       ollama_model=model, hidden_size=32, synthetic=True))
    manifest = dict(generator='v2.router.synth_labels', mode=mode, domain=domain, seed=seed,
                    labels_sha256=sha256(out), items_dir=str(directory),
                    item_sha256={f'{s}.jsonl': sha256(directory / f'{s}.jsonl')
                                 for s in dict.fromkeys(i['split'] for i in items)},
                    n_items=len(items), n_rows=len(rows), note='Synthetic labels; no model calls.')
    write_json(out.with_suffix('.synthetic.json'), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=['scan', 'chainops'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--items-dir', type=Path)
    parser.add_argument('--synthetic-items-out', type=Path)
    parser.add_argument('--features-out', type=Path)
    parser.add_argument('--seed', type=int, default=437)
    args = parser.parse_args()
    print(json.dumps(generate(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
