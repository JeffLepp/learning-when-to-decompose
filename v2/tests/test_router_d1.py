import inspect
import json
import unittest

import numpy as np

from v2.router.data import (
    ArmTable, DEPLOYABLE_MAIN, DIAGNOSTIC, MODELS, NUMERIC, available_layer_keys,
    load_f3, main_arms, main_table, model_dir_name, probe_overhead, s2_table,
)
from v2.router.evaluate import (
    Experiment, bootstrap_curve_metric, operating_point, paired_bootstrap, point,
)
from v2.router.policies import fixed_arm
from v2.tests.test_router_data import TempCase


def p(accuracy, cost):
    return dict(accuracy=accuracy, mean_cost=cost)


def coverage_table():
    arms = [f'{m}|{s}' for m in reversed(MODELS)
            for s in ('G2', 'S4', 'S2', 'S0', 'G1', 'S3', 'S1')]
    shape = (3, len(arms))
    arrays = {name: np.ones(shape) for name in NUMERIC}
    arrays['answer_sha1'] = np.full(shape, 'answer', dtype=object)
    for i, model in enumerate(MODELS):
        arrays['prompt_tokens'][:, arms.index(f'{model}|S0')] = np.arange(3) + 10 * (i + 1)
    return ArmTable(['a', 'b', 'c'], arms, 'router_val', **arrays)


class OperatingPointTests(unittest.TestCase):
    def test_target_met_cheapest_and_tolerance(self):
        self.assertEqual(operating_point([p(.9, 30), p(.8, 10), p(.7, 1)], .8), 1)
        self.assertEqual(operating_point([p(.8 - 5e-13, 10), p(.9, 20)], .8), 0)
        self.assertEqual(operating_point([p(.8 - 2e-12, 10), p(.9, 20)], .8), 1)

    def test_unmet_and_all_ties(self):
        self.assertEqual(operating_point([p(.5, 1), p(.7, 20), p(.6, 10)], .8), 1)
        points = [p(.6, 5), p(.7, 5), p(.7, 5), p(.7, 10)]
        self.assertEqual(operating_point(points, .5), 1)  # Cost, accuracy, index.
        self.assertEqual(operating_point(points, .8), 1)  # Accuracy, cost, index.
        self.assertEqual(operating_point([p(.8, 3), p(.9, 4)], .8), 0)
        for points, target in [([], .5), ([p(np.nan, 3)], .5),
                               ([p(.8, np.inf)], .5), ([p(.8, 3)], np.nan)]:
            with self.assertRaises(ValueError):
                operating_point(points, target)

    def test_bootstrap_defaults(self):
        for fn in [paired_bootstrap, bootstrap_curve_metric, Experiment]:
            self.assertEqual(inspect.signature(fn).parameters['n_boot'].default, 10000)


class D1DataTests(TempCase):
    def test_frozen_record_and_selection_guards(self):
        table = coverage_table()
        experiment = Experiment('test', table, n_boot=2, experiment_dir=self.path)
        decision = fixed_arm(3, len(table.arms), 0)
        points = [p(.5, 1), point(decision, table)]
        self.assertEqual(experiment.select_operating_point('route', points,
                                                          [decision, decision], rule='D1.7'), 1)
        expected = dict(choice=1, rule='D1.7', validation_point=points[1],
                        best_single_val_accuracy=1.)
        self.assertEqual(experiment.frozen['route'], expected)
        points[1]['accuracy'] = 0
        snapshot = experiment.frozen
        snapshot['route']['validation_point']['accuracy'] = 0
        self.assertEqual(experiment.frozen['route']['validation_point']['accuracy'], 1.)
        self.assertEqual(experiment.select_operating_point('key', [p(1, 2)], ['lambda_0']), 'lambda_0')
        self.assertEqual(experiment.select_operating_point('mapping', [p(1, 2)], {'tau': decision}), 'tau')
        for name in ['best_single', 'oracle', 'bad@0', 'route']:
            with self.assertRaises(ValueError):
                experiment.select_operating_point(name, [p(1, 1)], [decision])
        with self.assertRaises(ValueError):
            experiment.select_operating_point('invalid', points, [decision])
        with self.assertRaises(ValueError):
            experiment.select_operating_point('invalid', [p(1, 1)], [decision], rule='unknown')
        experiment.freeze()
        with self.assertRaises(RuntimeError):
            experiment.select_operating_point('late', [p(1, 1)], [decision])
        # Serialized evaluation must retain the choice provenance as well.
        table.cost[:, 1] = 2
        experiment.evaluate_on(table, {name: decision for name in experiment.frozen})
        saved = json.loads((self.path / 'report.json').read_text())
        self.assertEqual(saved['evaluation_call']['frozen_choices'], experiment.frozen)

    def test_main_coverage_and_diagnostics(self):
        table = coverage_table()
        expected = [a for a in table.arms if a.split('|')[1] in DEPLOYABLE_MAIN]
        self.assertEqual(main_arms(table), expected)
        main = main_table(table)
        self.assertEqual(main.arms, expected)
        self.assertEqual(main.ids, table.ids)
        self.assertFalse(any(a.split('|')[1] in DIAGNOSTIC for a in main.arms))
        small = table.select(arms=['qwen3:8b|S0', 'qwen3:8b|S3'])
        with self.assertRaises(ValueError):
            main_table(small)
        self.assertEqual(main_table(small, expected_n_arms=None).arms, small.arms)
        for field in ['success', 'cost']:
            changed = table.select()
            getattr(changed, field)[0, changed.arms.index(expected[0])] = np.nan
            with self.assertRaisesRegex(ValueError, 'Missing main arm'):
                main_table(changed)

    def test_s2_subset(self):
        table = coverage_table()
        table.success[0, table.arms.index('qwen3:0.6b|S2')] = np.nan
        table.success[2, table.arms.index('qwen3:8b|S2')] = np.nan
        subset = s2_table(table)
        self.assertEqual(subset.ids, ['b'])
        self.assertEqual(len(subset.arms), 20)
        self.assertEqual(subset.arms, [a for a in table.arms if a.split('|')[1] not in DIAGNOSTIC])
        missing = table.select(arms=[a for a in table.arms if a != 'qwen3:4b|S2'])
        self.assertEqual(s2_table(missing).ids, [])

    def test_f3_variants_alignment_and_missing_inputs(self):
        arms = ['qwen3:8b|S3', 'qwen3:0.6b|S0', 'qwen3:8b|S0']
        matrices = {}
        for model, width in [('qwen3:8b', 4), ('qwen3:0.6b', 2)]:
            directory = self.path / model_dir_name(model)
            directory.mkdir()
            matrix = np.arange(3 * width).reshape(3, width)
            matrices[model] = matrix
            np.save(directory / 'L04_last.npy', matrix)
            (directory / 'index.json').write_text(json.dumps(dict(
                ids=['c', 'a', 'b'], splits=['v1_train'] * 3, n_tokens=[1, 2, 3],
                layers=[4], poolings=['last'], files={'L04_last': 'L04_last.npy'},
                ollama_model=model, hidden_size=width)))
        self.assertEqual(model_dir_name('qwen3:0.6b'), 'qwen3-0.6b')
        self.assertEqual(available_layer_keys(self.path, 'qwen3:8b'), ['L04_last'])
        for variant in ['own', 'small']:
            result = load_f3(self.path, ['b', 'c'], 'L04_last', variant, arms)
            self.assertEqual(len(result), 3)
            for arm, x in zip(arms, result):
                model = arm.split('|')[0] if variant == 'own' else 'qwen3:0.6b'
                np.testing.assert_array_equal(x, matrices[model][[2, 0]])
        with self.assertRaisesRegex(ValueError, 'Missing activation ids'):
            load_f3(self.path, ['absent'], 'L04_last', 'own', arms)
        with self.assertRaisesRegex(ValueError, 'Missing activation layer'):
            load_f3(self.path, ['a'], 'L99_last', 'own', arms)
        with self.assertRaises(FileNotFoundError):
            load_f3(self.path, ['a'], 'L04_last', 'own', ['qwen3:4b|S0'])
        with self.assertRaises(ValueError):
            load_f3(self.path, ['a'], 'L04_last', 'invalid', arms)
        (self.path / 'qwen3-8b/L04_last.npy').unlink()
        with self.assertRaises(FileNotFoundError):
            load_f3(self.path, ['a'], 'L04_last', 'own', arms)

    def test_probe_overhead_charges_each_model_once(self):
        table = coverage_table()
        arms = main_arms(table)
        np.testing.assert_array_equal(probe_overhead(table, 'own', arms), [100, 104, 108])
        np.testing.assert_array_equal(probe_overhead(table, 'small', arms), [10, 11, 12])
        np.testing.assert_array_equal(probe_overhead(table, 'own', ['qwen3:8b|S3']), [40, 41, 42])
        with self.assertRaises(ValueError):
            probe_overhead(table, 'invalid', arms)
        absent = table.select(arms=[a for a in table.arms if a != 'qwen3:0.6b|S0'])
        for variant in ['own', 'small']:
            with self.assertRaisesRegex(ValueError, 'Missing probe S0 arm'):
                probe_overhead(absent, variant, arms)
        for bad in [np.nan, np.inf, -1.]:
            table.prompt_tokens[0, table.arms.index('qwen3:0.6b|S0')] = bad
            with self.assertRaisesRegex(ValueError, 'prompt tokens'):
                probe_overhead(table, 'small', arms)


if __name__ == '__main__':
    unittest.main()
