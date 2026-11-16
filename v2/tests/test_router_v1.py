import json
import unittest

from v2.router.data import ROOT, load_items, load_labels
from v2.router.v1_repro import reproduce


class V1Tests(unittest.TestCase):
    def test_v1_regression_and_artifact_readback(self):
        report = reproduce()
        expected = dict(selected_threshold=.5, validation_n=32, validation_correct=10,
                        validation_mean_tokens=629.9375, audit_n=32, audit_correct=15,
                        audit_mean_tokens=625.09375)
        for key, value in expected.items():
            self.assertEqual(report['checks'][key]['measured'], value)
        self.assertTrue(report['match'])
        self.assertTrue(all(check['match'] for check in report['checks'].values()))
        self.assertFalse(report['validation_target_met'])
        path = ROOT / 'results/v2/router/v1_repro/report.json'
        self.assertEqual(json.loads(path.read_text(encoding='utf-8')), report)
        adapter = ROOT / 'results/v2/router/v1_adapter'
        tables = load_labels(adapter / 'scan_v1_labels.csv')
        self.assertEqual(set(tables), {'router_train', 'router_val', 'v1_audit'})
        self.assertEqual(len(tables['router_train'].ids), 96)
        items = load_items(adapter / 'scan_v1_items.jsonl')
        self.assertEqual(len(items), 160)
        self.assertTrue(all(i['domain'] == 'scan' for i in items.values()))
        self.assertEqual(len(report['checks']['audit_choices']['measured']), 32)


if __name__ == '__main__':
    unittest.main()
