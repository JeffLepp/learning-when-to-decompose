import unittest
from experiment import parse_scan, validate_actions, validate_plan, execute
from unittest.mock import patch


class PipelineTests(unittest.TestCase):
    def test_parser_keeps_official_action_tokens(self):
        self.assertEqual(parse_scan('IN: walk twice OUT: I_WALK I_WALK\n')[0]['target'], ['I_WALK', 'I_WALK'])

    def test_rejects_bad_intermediates(self):
        for obj in [{'steps': []}, {'steps': [{'command': 'walk after run', 'repeat': 1}]}, {'steps': [{'command': 'jump', 'repeat': True}]}, {'steps': [{'command': 'jump', 'repeat': 100}]}]:
            with self.assertRaises(ValueError):
                validate_plan(obj)
        with self.assertRaises(ValueError):
            validate_actions({'actions': ['WALK']})

    @patch('experiment.generate')
    def test_combiner_obeys_model_plan_without_gold(self, generate):
        generate.side_effect = [{'steps': [{'command': 'run', 'repeat': 1}, {'command': 'walk', 'repeat': 2}]}, {'actions': ['I_RUN']}, {'actions': ['I_WALK']}]
        result = execute('walk twice after run', 'decomposed', 'test')
        self.assertEqual(result['prediction'], ['I_RUN', 'I_WALK', 'I_WALK'])
        self.assertIsNone(result['error'])

    @patch('experiment.generate', return_value={'actions': ['BAD']})
    def test_invalid_output_is_recorded(self, generate):
        self.assertIsNotNone(execute('walk', 'direct', 'test')['error'])


if __name__ == '__main__':
    unittest.main()
