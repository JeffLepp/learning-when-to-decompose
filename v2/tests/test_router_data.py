import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from v2.router.data import (
    ROOT, COLUMNS, NUMERIC, STRUCTURAL, TfidfFeatures, load_activations,
    load_items, load_labels, posthoc_features, structural_features, write_labels,
)


def label_rows():
    rows = []
    for i in range(6):
        for a in range(3):
            rows.append(dict(
                id=f'i{i}', split='router_train', model='m', strategy=f'S{a}',
                correct=int((i + a) % 3 != 0), total_tokens=10 + 10 * a + i,
                prompt_tokens=5, completion_tokens=5 + 10 * a + i,
                wall_seconds=1 + a, n_calls=1 + a, error='bad' if i == 5 else '',
                mean_logprob='' if i == 4 else -.2 * (i + 1),
                min_logprob='' if i == 4 else -1 - i,
                answer_sha1=f'answer{i}' if a < 2 else f'other{i}',
            ))
    return rows


class TempCase(unittest.TestCase):
    def setUp(self):
        base = ROOT / 'results/v2/router'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='test_', dir=base)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def table(self, rows=None):
        write_labels(self.path / 'labels.csv', label_rows() if rows is None else rows)
        return load_labels(self.path / 'labels.csv')['router_train']


class DataTests(TempCase):
    def test_chainops_generator_metadata(self):
        f = STRUCTURAL['chainops']('Retain the first three entries.',
                                  {'list_len': 5, 'n_ops': 1, 'ops': ['keep_first']})
        self.assertEqual(f['list_length'], 5)
        self.assertEqual(f['count_keep_first'], 1)

    def _dev_schema(self, domain, expected):
        items = list(load_items(ROOT / f'data/v2/{domain}/dev.jsonl').values())
        x, columns = structural_features([i['input'] for i in items], domain,
                                         [i.get('meta') for i in items])
        self.assertEqual(set(columns), set(expected))
        self.assertEqual(x.shape, (len(items), len(expected)))
        self.assertTrue(np.isfinite(x).all())

    @unittest.skipUnless((ROOT / 'data/v2/chainops/dev.jsonl').exists(), 'ChainOps dev absent')
    def test_chainops_dev_schema(self):
        words = ('reverse sort remove keep add multiply rotate swap append duplicates '
                 'sort_asc sort_desc remove_first keep_first remove_all keep_even keep_odd '
                 'rotate_left swap_ends unique').split()
        self._dev_schema('chainops', ['word_count', 'char_count', 'n_ops', 'list_length', 'depth'] +
                         [f'count_{w}' for w in words])

    @unittest.skipUnless((ROOT / 'data/v2/scan/dev.jsonl').exists(), 'SCAN dev absent')
    def test_scan_dev_schema(self):
        words = 'walk run jump look turn left right opposite around twice thrice and after'.split()
        self._dev_schema('scan', ['word_count', 'char_count'] + [f'count_{w}' for w in words])

    def test_round_trip_missing_selection_and_complete(self):
        rows = label_rows()[1:]
        table = self.table(rows)
        self.assertEqual(table.ids, [f'i{i}' for i in range(6)])
        self.assertEqual(table.arms, ['m|S0', 'm|S1', 'm|S2'])
        for name in NUMERIC:
            self.assertTrue(np.isnan(getattr(table, name)[0, 0]))
        self.assertIsNone(table.answer_sha1[0, 0])
        self.assertEqual(table.complete().ids, table.ids[1:])
        self.assertEqual(table.complete(['m|S1']).ids, table.ids)
        selected = table.select(['m|S2', 'm|S1'], [False, True, False, True, False, False])
        self.assertEqual(selected.arms, ['m|S2', 'm|S1'])
        self.assertEqual(selected.ids, ['i1', 'i3'])
        np.testing.assert_array_equal(selected.cost, table.cost[[1, 3]][:, [2, 1]])
        self.assertEqual(table.error[5, 0], 1.)
        with (self.path / 'labels.csv').open() as stream:
            self.assertEqual(next(csv.reader(stream)), COLUMNS)
        rows[0]['correct'] = True
        self.assertEqual(self.table(rows).success[0, 1], 1.)

    def test_label_failures(self):
        path = self.path / 'labels.csv'
        path.write_text('id,correct\ni,1\n')
        with self.assertRaisesRegex(ValueError, 'Missing label columns'):
            load_labels(path)
        rows = label_rows()
        for bad, message in [
            (rows + [rows[0]], 'Duplicate'),
            (rows + [dict(rows[0], split='router_val', strategy='S9')], 'two splits'),
            ([dict(rows[0], correct=.5)], 'binary'),
            ([dict(rows[0], correct='NaN')], 'binary'),
            ([dict(rows[0], correct='yes')], 'binary'),
        ]:
            with self.subTest(message=message):
                write_labels(path, bad)
                with self.assertRaisesRegex(ValueError, message):
                    load_labels(path)

    def test_structural(self):
        from controller import features
        commands = ['walk', 'walk twice after run', 'jump around left thrice and look',
                    'turn opposite right after run around left twice', '']
        x, names = structural_features(commands, 'scan')
        for command, row in zip(commands, x):
            self.assertEqual(STRUCTURAL['scan'](command, None), features(command))
            self.assertEqual(dict(zip(names, row)), features(command))
        from sklearn.feature_extraction import DictVectorizer
        expected = DictVectorizer(sparse=False).fit_transform([features(c) for c in commands])
        np.testing.assert_array_equal(x, expected)
        a = STRUCTURAL['chainops']('', None)
        b = STRUCTURAL['chainops']('Start [1, 2, 3]. Reverse. Then add 2.', {'depth': 4})
        self.assertEqual(set(a), set(b))
        self.assertEqual(b['list_length'], 3)
        self.assertEqual(b['n_ops'], 2)
        self.assertEqual(b['depth'], 4)

    def test_tfidf_train_only(self):
        feature = TfidfFeatures().fit(['walk twice', 'run and jump'])
        before = dict(feature.vectorizer.vocabulary_)
        val = feature.transform(['uniquetovalidation'])
        self.assertEqual(before, feature.vectorizer.vocabulary_)
        self.assertEqual(val.shape[1], len(before))
        self.assertEqual(val.nnz, 0)
        self.assertIn('run and jump', before)

    def test_activations(self):
        npy, index = self.path / 'layer.npy', self.path / 'index.json'
        np.save(npy, np.array([[1, 2], [3, 4], [5, 6]]))
        for key in ['ids', 'item_ids']:
            index.write_text(json.dumps({key: ['a', 'b', 'c']}))
            np.testing.assert_array_equal(load_activations(npy, index, ['c', 'a']),
                                          [[5, 6], [1, 2]])
        for order, ids, message in [
            (['a'], ['a'], 'length'), (['a', 'a', 'c'], ['a'], 'Duplicate'),
            (['a', 'b', 'c'], ['missing'], 'Missing'),
        ]:
            index.write_text(json.dumps({'ids': order}))
            with self.assertRaisesRegex(ValueError, message):
                load_activations(npy, index, ids)

    def test_items(self):
        path = self.path / 'items.jsonl'
        path.write_text('{"id":"a","input":"walk"}\n')
        self.assertEqual(load_items(path)['a']['input'], 'walk')
        path.write_text('{"id":"a"}\n')
        with self.assertRaisesRegex(ValueError, 'id and input'):
            load_items(path)

    def test_posthoc_invariance(self):
        table = self.table()
        before, names = posthoc_features(table, table.arms[:2])
        changed = table.select()
        rng = np.random.default_rng(437)
        changed.success[:] = rng.random(changed.success.shape)
        for name in NUMERIC:
            getattr(changed, name)[:, 2] = rng.normal(size=6)
        changed.answer_sha1[:, 2] = 'arbitrary'
        after, after_names = posthoc_features(changed, changed.arms[:2], target='ignored')
        np.testing.assert_array_equal(before, after)
        self.assertEqual(names, after_names)
        self.assertEqual(before[4, names.index('m|S0:mean_logprob')], 0)
        self.assertEqual(before[4, names.index('m|S0:missing_logprob')], 1)
        np.testing.assert_array_equal(before[:, -1], np.ones(6))


if __name__ == '__main__':
    unittest.main()
