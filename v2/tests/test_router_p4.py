import builtins
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from v2.build_labels import COLUMNS
from v2.router.bootstrap import curve_metrics, item_counts
from v2.router.predictor_report import bootstrap_auroc, predictor_report
from v2.router.models import SuccessPredictor, ConstantCalibration
from sklearn.metrics import roc_auc_score
from v2.router.data import ROOT, load_labels, main_table, s2_table
from v2.router.evaluate import (normalized_auc, curve_accuracy_at_cost, curve_cost_at_accuracy,
                                decision_cost, point)
from v2.router.policies import Decision, decision_success
from v2.router.run_p4 import run as run_p4
from v2.router.run_final import run as run_final
from v2.router.synth_labels import generate
from v2.router.workflow import (crossfit_calibration, apply_state, sha256, fit_calibrators,
                                apply_calibrators, family_features, inputs_for)


class P4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='test_p4_', dir=ROOT / 'results/v2/router')
        cls.base = Path(cls.temp.name)
        cls.labels, cls.items, cls.features = cls.base / 'labels.csv', cls.base / 'items', cls.base / 'features'
        cls.limits = threadpool_limits(limits=1)
        cls.limits.__enter__()
        generate('scan', cls.labels, synthetic_items_out=cls.items, features_out=cls.features)

    @classmethod
    def tearDownClass(cls):
        cls.limits.__exit__(None, None, None)
        cls.temp.cleanup()

    def test_01_synthetic_contract_and_split_filter(self):
        with self.labels.open(newline='') as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, COLUMNS)
            rows = list(reader)
        self.assertEqual(rows, sorted(rows, key=lambda r: (r['id'], r['model'], r['strategy'], r['split'], r['run_name'])))
        tables = load_labels(self.labels)
        for split, n in [('router_train', 600), ('router_val', 200), ('holdout', 400), ('ood_length', 300)]:
            table = main_table(tables[split])
            self.assertEqual(len(table.ids), n)
            self.assertEqual(len(table.arms), 16)
            self.assertFalse(any(a.endswith(('G1', 'G2')) for a in table.arms))
        self.assertEqual([len(s2_table(tables[s]).ids) for s in tables],
                         [dict(holdout=150, ood_length=100, router_train=150, router_val=100)[s] for s in tables])
        for r in rows:
            self.assertEqual(int(r['total_tokens']), int(r['prompt_tokens']) + int(r['completion_tokens']))
            self.assertLessEqual(int(r['thinking_tokens']), int(r['completion_tokens']))
            if r['error']:
                self.assertEqual(r['correct'], '0')
        # Invalid final rows must be skipped before parsing/duplicate checking.
        bad = self.base / 'filtered.csv'
        bad.write_text(self.labels.read_text() + ',holdout,m,S0,invalid\n')
        self.assertEqual(set(load_labels(bad, splits=['router_train', 'router_val'])), {'router_train', 'router_val'})
        for model_dir in self.features.iterdir():
            index = json.loads((model_dir / 'index.json').read_text())
            self.assertTrue(set(['ids', 'splits', 'n_tokens', 'layers', 'poolings', 'files', 'ollama_model', 'hidden_size']) <= index.keys())
            self.assertEqual(len(index['files']), 4)
            self.assertEqual(np.load(model_dir / 'L04_last.npy').dtype, np.float16)

    def test_02_crossfit_holds_out_labels(self):
        rng = np.random.default_rng(437)
        p, y = rng.random((50, 2)), rng.integers(0, 2, (50, 2)).astype(float)
        preds, _, folds = crossfit_calibration(p, y)
        for heldout in folds:
            changed = y.copy()
            changed[heldout] = 1 - changed[heldout]
            other, _, _ = crossfit_calibration(p, changed)
            for method in preds:
                np.testing.assert_array_equal(preds[method][heldout], other[method][heldout])

    def test_03_vector_bootstrap_matches_reference(self):
        rng = np.random.default_rng(437)
        y, c = rng.random((100, 20)), rng.integers(1, 200, (100, 20)).astype(float)
        lo, hi = c.min(axis=1), c.max(axis=1)
        best_y, best_c = rng.random(100), rng.uniform(0, 220, 100)
        fast = curve_metrics(y, c, lo, hi, best_y, best_c)
        for i in range(100):
            curve = [dict(accuracy=a, mean_cost=b) for a, b in zip(y[i], c[i])]
            for name, expected in [('normalized_auc', normalized_auc(curve, lo[i], hi[i])),
                                   ('accuracy_at_best_single_cost', curve_accuracy_at_cost(curve, best_c[i])),
                                   ('cost_at_best_single_accuracy', curve_cost_at_accuracy(curve, best_y[i]))]:
                np.testing.assert_allclose(fast[name][i], expected, atol=1e-12)

    def test_platt_anticorrelated_fold_cannot_invert_ranking(self):
        raw = np.linspace(.01, .99, 50)[:, None]
        y = (raw < .5).astype(float)
        diagnostics = {}
        preds, _, folds = crossfit_calibration(raw, y, diagnostics=diagnostics)
        self.assertEqual(diagnostics['platt_nonpositive_count'], 5)
        for heldout in folds:
            train = np.setdiff1d(np.arange(len(y)), heldout)
            np.testing.assert_allclose(preds['platt'][heldout], y[train].mean())
        calibrators = fit_calibrators(raw, y, 'platt')
        self.assertIsInstance(calibrators[0], ConstantCalibration)
        self.assertTrue((np.diff(apply_calibrators(raw, calibrators, 'platt')[:, 0]) >= 0).all())
        model = SuccessPredictor('logreg')
        model.n_arms = 1
        with patch.object(model, '_raw', return_value=raw):
            model.calibrate(raw, y, 'platt')
            self.assertEqual(model.platt_nonpositive_count, 1)
            np.testing.assert_allclose(model.predict_proba(raw), .5)
            model.calibrate(raw, y, 'isotonic')
            self.assertTrue((np.diff(model.predict_proba(raw)[:, 0]) >= 0).all())

    def test_predictor_bootstrap_matches_explicit_paired_resampling(self):
        y = np.array([[0, 0], [1, 0], [0, 0], [1, 1]], float)
        scores = np.array([[.1, .4], [.5, .2], [.5, .2], [.9, .8]])
        counts = item_counts(4, 50, 437)
        means, skipped = bootstrap_auroc(y, scores, counts)
        expected, missing = [], [0, 0]
        for b, weights in enumerate(counts):
            idx = np.repeat(np.arange(4), np.rint(weights * 4).astype(int))
            values = []
            for a in range(2):
                if len(np.unique(y[idx, a])) < 2:
                    if b:
                        missing[a] += 1
                else:
                    values.append(roc_auc_score(y[idx, a], scores[idx, a]))
            expected.append(np.mean(values) if values else np.nan)
        np.testing.assert_allclose(means, expected, atol=1e-12)
        self.assertEqual(skipped['single_class_by_arm'], missing)
        self.assertGreater(sum(missing), 0)

    def test_03b_endpoint_roundoff_is_not_missing_support(self):
        y = np.array([[.7, .9], [.7, .9]])
        c = np.array([[100. + 1e-13, 200.], [100., 200.]])
        result = curve_metrics(y, c, np.array([100., 100.]), np.array([200., 200.]),
                               np.array([.9 + 1e-16, .95]), np.array([100., 200. + 1e-13]))
        np.testing.assert_allclose(result['normalized_auc'], [.8, .8], atol=1e-12)
        np.testing.assert_allclose(result['accuracy_at_best_single_cost'], [.7, .9], atol=1e-12)
        self.assertAlmostEqual(result['cost_at_best_single_accuracy'][0], 200.)
        self.assertTrue(np.isnan(result['cost_at_best_single_accuracy'][1]))

    def test_04_p4_and_frozen_final_end_to_end(self):
        original_open, original_path_open = builtins.open, Path.open
        opened = []
        def check(file):
            if isinstance(file, (str, Path)):
                path = Path(file)
                opened.append(str(path))
                if path.name in {'holdout.jsonl', 'ood_length.jsonl', 'ood_depth.jsonl'}:
                    self.fail(f'P4 opened forbidden final items: {path}')
        def guarded(file, *args, **kwargs):
            check(file)
            return original_open(file, *args, **kwargs)
        def path_guard(file, *args, **kwargs):
            check(file)
            return original_path_open(file, *args, **kwargs)
        p4 = self.base / 'p4'
        with patch('builtins.open', guarded), patch.object(Path, 'open', path_guard):
            report = run_p4(self.labels, 'scan', self.items, p4, self.features, quick=True)
        self.assertEqual(report['uncertainty']['n_boot'], 200)
        self.assertEqual(len(report['prediction_metrics']), 24)
        self.assertEqual(len(report['predictor_metrics']), 24)
        self.assertEqual(len(report['predictor_contrasts']), 30)
        self.assertIn('selected/cascade', report['selections'])
        self.assertIn('selected/cascade - selected/overall', report['contrasts'])
        self.assertEqual(report['predictor_contrasts_skipped'], [])
        self.assertTrue(report['p4_completed'])
        self.assertTrue(report['synthetic_labels']['mode'] == 'B')
        self.assertLess(len((p4 / 'summary.md').read_text().splitlines()), 150)
        report_path = p4 / 'report.json'
        original_report = report_path.read_bytes()
        altered = json.loads(original_report)
        altered['synthetic_labels']['mode'] = 'A'
        report_path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(RuntimeError, 'Mode-B'):
            run_final(p4, self.labels, self.items, 'holdout', synthetic_test=True)
        report_path.write_bytes(original_report)
        with self.assertRaisesRegex(RuntimeError, 'outside data'):
            run_final(p4, self.labels, ROOT / 'data/v2/scan', 'holdout', synthetic_test=True)
        with patch('v2.router.run_final.sha256', return_value='tampered'):
            with self.assertRaisesRegex(RuntimeError, 'code sha256'):
                run_final(p4, self.labels, self.items, 'holdout', synthetic_test=True)
        changed_labels = self.base / 'changed_labels.csv'
        changed_labels.write_text(self.labels.read_text().replace('synthetic-scan-router_train-0000',
                                                                  'changed-train-id'))
        with self.assertRaisesRegex(RuntimeError, 'labels sha256'):
            run_final(p4, changed_labels, self.items, 'holdout', synthetic_test=True)
        with self.assertRaisesRegex(RuntimeError, 'Completed P4'):
            run_p4(self.labels, 'scan', self.items, p4, quick=True)
        with patch('v2.router.run_final.subprocess.run') as git:
            git.return_value.returncode = 1
            with self.assertRaisesRegex(RuntimeError, 'tag required'):
                run_final(p4, self.labels, self.items, 'holdout')
        artifact = p4 / 'frozen/objects.joblib'
        original = artifact.read_bytes()
        artifact.write_bytes(original + b'tamper')
        with self.assertRaisesRegex(RuntimeError, 'artifact sha256'):
            run_final(p4, self.labels, self.items, 'holdout', synthetic_test=True)
        artifact.write_bytes(original)
        with patch('v2.router.models.SuccessPredictor.fit', side_effect=AssertionError('refit')), \
             patch('v2.router.models.SuccessPredictor.calibrate', side_effect=AssertionError('recalibrate')), \
             patch('v2.router.models.CostPredictor.fit', side_effect=AssertionError('cost refit')), \
             patch('v2.router.data.TfidfFeatures.fit', side_effect=AssertionError('TF-IDF refit')), \
             patch('v2.router.policies.LearnedConfidence.fit', side_effect=AssertionError('confidence refit')):
            results = run_final(p4, self.labels, self.items, 'holdout,ood_length', self.features, synthetic_test=True)
        self.assertEqual(set(results), {'holdout', 'ood_length'})
        state = joblib.load(artifact)
        for split, final in results.items():
            table = main_table(load_labels(self.labels, splits=[split])[split])
            self.assertLess(len((p4 / f'final_{split}/summary.md').read_text().splitlines()), 150)
            self.assertEqual(len(final['predictor_metrics']), 24)
            self.assertEqual(len(final['predictor_contrasts']), 30)
            matrices, _ = family_features(state, table, inputs_for(table, self.items, 'scan'), self.features)
            for name, metric in final['predictor_metrics'].items():
                raw = state['predictors'][name]._raw(matrices[state['predictor_families'][name]])
                expected = [roc_auc_score(table.success[:, a], raw[:, a]) for a in range(16)]
                np.testing.assert_allclose(metric['auroc'], expected, atol=1e-12)
                self.assertAlmostEqual(metric['mean_auroc'], np.mean(expected))
            specs = apply_state(state, table, self.items, self.features)
            for name, spec in specs.items():
                expected = point(spec['decision'], table, spec['overhead'])
                for metric in ['accuracy', 'mean_cost']:
                    self.assertAlmostEqual(expected[metric], final['policies'][name][metric])
            h5 = final['contrasts']['selected/cascade - selected/overall']
            self.assertEqual(h5['normalized_auc']['n_boot'], 200)
            for name, contrast in final['contrasts'].items():
                left, right = name.split(' - ')
                left_auc = final['policies'][left]['normalized_auc']['value']
                if right in final['policies']:
                    self.assertAlmostEqual(contrast['normalized_auc']['value'],
                                           left_auc - final['policies'][right]['normalized_auc']['value'])
                elif right == 'fixed_frontier':
                    self.assertAlmostEqual(contrast['normalized_auc']['value'], left_auc - final['fixed_frontier_auc'])
            # Compare H5 CIs to a slow, independent explicit-index reference.
            rng = np.random.default_rng(437)
            deltas = []
            for _ in range(200):
                idx = rng.integers(len(table.ids), size=len(table.ids))
                arm_costs = table.cost[idx].mean(axis=0)
                values = []
                for name in ['selected/cascade', 'selected/overall']:
                    spec = specs[name]
                    points = []
                    for d in [spec['anchor']] + spec['curve']:
                        overhead = None if d is spec['anchor'] else spec['overhead']
                        y = decision_success(d, table)
                        c = decision_cost(d, table, overhead)
                        points.append(dict(accuracy=y[idx].mean(), mean_cost=c[idx].mean()))
                    values.append(normalized_auc(points, arm_costs.min(), arm_costs.max()))
                deltas.append(values[0] - values[1])
            np.testing.assert_allclose(h5['normalized_auc']['ci95'], np.quantile(deltas, [.025, .975]), atol=1e-12)
            with (p4 / f'final_{split}/decisions.csv').open() as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), len(table.ids) * (len(specs) + 2))
            for r in records:
                i = table.ids.index(r['id'])
                arms = [table.arms.index(a) for a in json.loads(r['executed_arms'])]
                overhead = specs[r['policy']]['overhead'][i] if r['policy'] in specs else 0
                self.assertAlmostEqual(float(r['cost']), table.cost[i, arms].sum() + overhead)
                self.assertEqual(float(r['correct']), table.success[i, table.arms.index(r['final_arm'])])
            with self.assertRaisesRegex(RuntimeError, 'already completed'):
                run_final(p4, self.labels, self.items, split, self.features, synthetic_test=True)
        from v2.router.readback_p4 import verify
        self.assertTrue(verify(p4)['all_passed'])

    def test_05_v1_adapter_with_and_without_features(self):
        labels = ROOT / 'results/v2/router/v1_adapter/scan_v1_labels.csv'
        items = ROOT / 'data/v2/scan'
        for name, features in [('plain', None), ('features', ROOT / 'results/v2/features/scan_v1')]:
            report = run_p4(labels, 'scan', items, self.base / name, features,
                            expected_arms=None, quick=True)
            self.assertEqual(len(report['fixed_points']), 2)
            self.assertEqual(report['best_single']['n'], 32)
            if features:
                self.assertEqual(set(report['layer_selection']), {'F3-own', 'F3-small'})
                self.assertGreater(report['policies']['selected/F3-small']['mean_overhead'], 0)
            else:
                self.assertEqual(len(report['predictor_contrasts']), 6)
                self.assertEqual(len(report['predictor_contrasts_skipped']), 24)

    def test_06_mode_a_both_domains_and_synthetic_bypass_refused(self):
        for domain in ['scan', 'chainops']:
            labels = self.base / f'{domain}_mode_a.csv'
            manifest = generate(domain, labels)
            self.assertEqual(manifest['mode'], 'A')
            self.assertEqual(set(load_labels(labels)), {'router_train', 'router_val'})
        # Refuse an out-of-tree output before any training or writes.
        with self.assertRaisesRegex(ValueError, 'under results'):
            run_p4(self.labels, 'scan', self.items, ROOT / 'outside_scope')


if __name__ == '__main__':
    unittest.main()
