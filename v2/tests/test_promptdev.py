import unittest

from v2.harness_promptdev import MODELS, STRATEGIES, score_records, select_best


class PromptDevTest(unittest.TestCase):
    def test_paired_scoring_and_earliest_selection(self):
        records = [{'id': item, 'model': model, 'strategy': strategy, 'split': 'dev',
                    'correct': item == 'a' or model == MODELS[1], 'total_tokens': 10 if item == 'a' else 30,
                    'error': 'invalid_plan' if item == 'b' and model == MODELS[0] else None}
                   for item in ('a', 'b') for model in MODELS for strategy in STRATEGIES]
        scores = score_records(records, ['a', 'b'])
        self.assertEqual(scores['S0']['acc_1.7b'], .5)
        self.assertEqual(scores['S0']['acc_8b'], 1)
        self.assertEqual(scores['S0']['mean_acc'], .75)
        self.assertEqual(scores['S0']['mean_tokens_1.7b'], 20)
        self.assertEqual(scores['S0']['error_counts'][MODELS[0]], {'invalid_plan': 1})
        log = [{'domain': 'scan', 'strategy': 'S0', 'round': n, 'mean_acc': acc}
               for n, acc in [(2, .75), (0, .5), (1, .75)]]
        self.assertEqual(select_best(log)[('scan', 'S0')]['round'], 1)
        for invalid in (records[:-1], records + [records[0]]):
            with self.assertRaises(ValueError): score_records(invalid, ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
