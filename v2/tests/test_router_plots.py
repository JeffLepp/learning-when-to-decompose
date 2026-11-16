import unittest

from v2.router.evaluate import arm_metrics
from v2.router import plots
from v2.tests.test_router_data import TempCase


class PlotTests(TempCase):
    def test_all_pngs(self):
        points = [dict(mean_cost=10, accuracy=.5, arm='small'),
                  dict(mean_cost=30, accuracy=.9, arm='large')]
        paths = [plots.plot_cost_accuracy(points, points, {'router': points},
                                          dict(mean_cost=20, accuracy=1), self.path / 'cost.png'),
                 plots.plot_reliability({'arm 0': arm_metrics([0, 1], [.1, .9])['reliability_bins'][0]},
                                        self.path / 'reliability.png'),
                 plots.plot_learning_curves({'router': [dict(n_train=50, mean={'accuracy': .7}, std={'accuracy': .1}),
                                                        dict(n_train=100, mean={'accuracy': .8}, std={'accuracy': .05})]},
                                            self.path / 'learning.png'),
                 plots.plot_auroc_grid({'structural': {'logreg': [.7, .8], 'hgb': [.6, .9]}},
                                       self.path / 'auroc.png')]
        for path in paths:
            self.assertGreater(path.stat().st_size, 100)
            self.assertEqual(path.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')


if __name__ == '__main__':
    unittest.main()
