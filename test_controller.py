import unittest
import numpy as np
from controller import features, route


class ControllerTests(unittest.TestCase):
    def test_cost_only_breaks_ties_between_eligible_strategies(self):
        probabilities = np.array([[.9, .95], [.3, .9], [.3, .4]])
        costs = np.array([[10, 50], [10, 50], [10, 50]])
        np.testing.assert_array_equal(route(probabilities, costs, .8), [0, 1, 1])

    def test_features_are_command_only(self):
        f = features('walk twice after run')
        self.assertEqual(f['count_after'], 1)
        self.assertEqual(f['word_count'], 4)
        self.assertFalse(any(k in f for k in ['target', 'correct', 'prediction']))


if __name__ == '__main__':
    unittest.main()
