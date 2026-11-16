"""Label contracts and features; no generation or reference-answer access."""

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = (
    'id split model strategy correct total_tokens prompt_tokens completion_tokens '
    'wall_seconds n_calls error mean_logprob min_logprob answer_sha1'
).split()
NUMERIC = (
    'success cost prompt_tokens completion_tokens seconds n_calls error '
    'mean_logprob min_logprob'
).split()


@dataclass
class ArmTable:
    ids: list[str]
    arms: list[str]
    split: str
    success: np.ndarray
    cost: np.ndarray
    prompt_tokens: np.ndarray
    completion_tokens: np.ndarray
    seconds: np.ndarray
    n_calls: np.ndarray
    error: np.ndarray
    mean_logprob: np.ndarray
    min_logprob: np.ndarray
    answer_sha1: np.ndarray

    def __post_init__(self):
        shape = (len(self.ids), len(self.arms))
        for name in NUMERIC + ['answer_sha1']:
            if getattr(self, name).shape != shape:
                raise ValueError(f'{name} must have shape {shape}')
        if len(set(self.ids)) != len(self.ids):
            raise ValueError('Duplicate item ids')
        if len(set(self.arms)) != len(self.arms):
            raise ValueError('Duplicate arms')

    def select(self, arms=None, item_mask=None):
        arms = self.arms if arms is None else list(arms)
        cols = [self.arms.index(a) for a in arms]
        rows = np.arange(len(self.ids))
        if item_mask is not None:
            rows = rows[np.asarray(item_mask)]
        arrays = {
            name: getattr(self, name)[np.ix_(rows, cols)].copy()
            for name in NUMERIC + ['answer_sha1']
        }
        return ArmTable([self.ids[i] for i in rows], list(arms), self.split, **arrays)

    def complete(self, arms=None):
        """Retain all columns, restricting rows to observed requested arms."""
        arms = self.arms if arms is None else arms
        cols = [self.arms.index(a) for a in arms]
        return self.select(item_mask=np.isfinite(self.success[:, cols]).all(axis=1))


def write_labels(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            # CSV booleans must round-trip through the numeric label contract.
            row['correct'] = int(row['correct']) if isinstance(
                row['correct'], (bool, np.bool_)
            ) else row['correct']
            writer.writerow(row)


def load_labels(path):
    grouped, seen, splits = {}, set(), {}
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        missing = set(COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing label columns: {sorted(missing)}')
        for row in reader:
            key = (row['id'], row['model'], row['strategy'])
            if key in seen:
                raise ValueError(f'Duplicate label row: {key}')
            seen.add(key)
            if row['id'] in splits and splits[row['id']] != row['split']:
                raise ValueError(f"Item {row['id']} appears in two splits")
            splits[row['id']] = row['split']
            try:
                correct = float(row['correct'])
            except (TypeError, ValueError) as exc:
                raise ValueError('correct must be binary (0 or 1)') from exc
            if correct not in (0., 1.):
                raise ValueError('correct must be binary (0 or 1)')
            row['correct'] = correct
            grouped.setdefault(row['split'], []).append(row)
    tables = {}
    mapping = dict(success='correct', cost='total_tokens', seconds='wall_seconds')
    for split, rows in grouped.items():
        ids = list(dict.fromkeys(row['id'] for row in rows))
        arms = sorted({f"{r['model']}|{r['strategy']}" for r in rows})
        shape = (len(ids), len(arms))
        arrays = {name: np.full(shape, np.nan) for name in NUMERIC}
        arrays['answer_sha1'] = np.full(shape, None, dtype=object)
        id_index = {id_: i for i, id_ in enumerate(ids)}
        arm_index = {arm: i for i, arm in enumerate(arms)}
        for row in rows:
            i = id_index[row['id']]
            a = arm_index[f"{row['model']}|{row['strategy']}"]
            for name in NUMERIC:
                value = row[mapping.get(name, name)]
                arrays[name][i, a] = (
                    float(bool(value)) if name == 'error'
                    else float(value) if value not in ('', None) else np.nan
                )
            arrays['answer_sha1'][i, a] = row['answer_sha1'] or None
        tables[split] = ArmTable(ids, arms, split, **arrays)
    return tables


def v1_label_rows(root=ROOT):
    rows, inputs = [], {}
    for run, split in [('train', 'router_train'), ('validation', 'router_val'),
                       ('audit', 'v1_audit')]:
        path = Path(root) / 'results' / f'{run}-8b-v1' / 'records.jsonl'
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                r = json.loads(line)
                inputs[r['id']] = r['command']
                answer = json.dumps(r['prediction'], separators=(',', ':'))
                rows.append(dict(
                    id=r['id'], split=split, model='qwen3:8b',
                    strategy={'direct': 'S0', 'decomposed': 'S3'}[r['strategy']],
                    correct=int(r['correct']),
                    total_tokens=r['prompt_tokens'] + r['completion_tokens'],
                    prompt_tokens=r['prompt_tokens'],
                    completion_tokens=r['completion_tokens'],
                    wall_seconds=r['wall_seconds'], n_calls=len(r['calls']),
                    error=r['error'] or '', mean_logprob='', min_logprob='',
                    answer_sha1=hashlib.sha1(answer.encode()).hexdigest(),
                ))
    return rows, inputs


def write_v1_adapter(root=ROOT):
    rows, inputs = v1_label_rows(root)
    dest = Path(root) / 'results/v2/router/v1_adapter'
    write_labels(dest / 'scan_v1_labels.csv', rows)
    splits = {r['id']: r['split'] for r in rows}
    with (dest / 'scan_v1_items.jsonl').open('w', encoding='utf-8') as stream:
        for id_, text in inputs.items():
            stream.write(json.dumps(dict(
                id=id_, domain='scan', split=splits[id_], input=text
            )) + '\n')
    return dest


def load_items(path):
    items = {}
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            if 'id' not in item or 'input' not in item:
                raise ValueError('Items require id and input')
            if item['id'] in items:
                raise ValueError(f"Duplicate item id: {item['id']}")
            items[item['id']] = item
    return items


def scan_features(text, meta=None):
    words = text.split()
    vocabulary = ('walk run jump look turn left right opposite around twice '
                  'thrice and after').split()
    return {'word_count': len(words), 'char_count': len(text),
            **{f'count_{w}': words.count(w) for w in vocabulary}}


def chainops_features(text, meta=None):
    meta = meta or {}
    words = re.findall(r'\b\w+\b', text.lower())
    vocabulary = ('reverse sort remove keep add multiply rotate swap append '
                  'duplicates').split()
    counts = {f'count_{w}': words.count(w) for w in vocabulary}
    # Generator operation metadata describes the input, never the target.
    operation_names = ('reverse sort_asc sort_desc remove_first keep_first remove_all '
                       'add multiply keep_even keep_odd rotate_left swap_ends append unique').split()
    ops = meta.get('ops', [])
    if not isinstance(ops, (list, tuple)):
        ops = []
    op_counts = {f'count_{op}': ops.count(op) for op in operation_names}
    initial = re.search(r'\[([^]]*)\]', text)
    length = len(re.findall(r'-?\d+', initial[1])) if initial else 0
    operations = meta.get('n_ops', meta.get('ops', sum(
        counts[f'count_{w}'] for w in vocabulary if w != 'duplicates'
    )))
    if isinstance(operations, (list, tuple)):
        operations = len(operations)
    # Keep lexical counts for shared names when metadata is absent.
    counts.update({k: v for k, v in op_counts.items() if ops or k not in counts})
    return dict(word_count=len(words), char_count=len(text), **counts,
                n_ops=operations or 0,
                list_length=meta.get('list_len', meta.get('list_length', length)) or 0,
                depth=meta.get('depth', 0) or 0)


STRUCTURAL = {'scan': scan_features, 'chainops': chainops_features}


def structural_features(inputs, domain, metas=None):
    inputs = list(inputs)
    metas = [None] * len(inputs) if metas is None else list(metas)
    if len(inputs) != len(metas):
        raise ValueError('inputs and metas must have equal length')
    fn = STRUCTURAL[domain]
    names = sorted(fn('', None))  # DictVectorizer's default sorted column order.
    rows = [fn(text, meta) for text, meta in zip(inputs, metas)]
    if any(set(row) != set(names) for row in rows):
        raise ValueError('Structural feature registry must return a fixed schema')
    return np.array([[row[k] for k in names] for row in rows], dtype=float).reshape(
        len(rows), len(names)
    ), names


class TfidfFeatures:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3))

    def fit(self, train_inputs):
        self.vectorizer.fit(train_inputs)
        return self

    def transform(self, inputs):
        return self.vectorizer.transform(inputs)


def load_activations(npy_path, index_path, ids):
    matrix = np.load(npy_path, allow_pickle=False)
    index = json.loads(Path(index_path).read_text(encoding='utf-8'))
    order = index.get('ids', index.get('item_ids'))
    if not isinstance(order, list):
        raise ValueError('Activation index requires ids (or item_ids) list')
    if matrix.ndim != 2 or len(order) != matrix.shape[0]:
        raise ValueError('Activation index length must equal matrix rows (2-D)')
    if len(set(order)) != len(order):
        raise ValueError('Duplicate ids in activation index')
    lookup = {id_: i for i, id_ in enumerate(order)}
    missing = set(ids) - lookup.keys()
    if missing:
        raise ValueError(f'Missing activation ids: {sorted(missing)}')
    return np.asarray(matrix[[lookup[id_] for id_ in ids]], dtype=float)


def posthoc_features(table, executed, target=None):
    """Use only executed signals. target is reserved, never a label feature."""
    columns, names = [], []
    for arm in executed:
        a = table.arms.index(arm)
        mean, minimum = table.mean_logprob[:, a], table.min_logprob[:, a]
        missing = np.isnan(mean) | np.isnan(minimum)
        signals = dict(
            mean_logprob=np.nan_to_num(mean, nan=0.),
            min_logprob=np.nan_to_num(minimum, nan=0.),
            error=table.error[:, a],
            completion_tokens=table.completion_tokens[:, a],
            n_calls=table.n_calls[:, a], missing_logprob=missing.astype(float),
        )
        for name, values in signals.items():
            names.append(f'{arm}:{name}')
            columns.append(values)
    for left, right in combinations(executed, 2):
        x = table.answer_sha1[:, table.arms.index(left)]
        y = table.answer_sha1[:, table.arms.index(right)]
        columns.append(((x != None) & (y != None) & (x == y)).astype(float))
        names.append(f'{left}={right}:agreement')
    matrix = np.column_stack(columns) if columns else np.empty((len(table.ids), 0))
    return matrix, names


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['v1-adapter'])
    parser.parse_args()
    print(write_v1_adapter())
