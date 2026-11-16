import copy
import csv
import hashlib
import json
import random
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from experiment import ROOT, parse_scan, verify_data
from v2 import build_data, build_labels, harness
from v2.domains import chainops, scan


def tempdir():
    return tempfile.TemporaryDirectory(dir=ROOT / 'workspace_logs')


def chain_text(text):
    start, instructions = text.removeprefix('Start with ').split('. ', 1)
    return json.loads(start), [chainops.parse_template(s) for s in instructions.split(' Then ')]


def response(value, think=False, reason='stop'):
    text = value if isinstance(value, str) else json.dumps(value)
    logs = [{'token': c, 'logprob': -0.1, 'bytes': list(c.encode())} for c in text]
    thoughts = [{'token': '<think>', 'logprob': -0.2}, {'token': 'check', 'logprob': -0.3},
                {'token': '</think>', 'logprob': -0.4}] if think else []
    return {'response': text, 'thinking': 'check' if think else '', 'logprobs': thoughts + logs,
            'done': True, 'done_reason': reason, 'eval_count': len(logs) + len(thoughts) + 1,
            'prompt_eval_count': 10, 'total_duration': 100, 'model': 'mock'}


def perfect(payload):
    prompt, schema = payload['prompt'], payload['format']
    fields = schema['properties']
    if 'actions' in fields or 'steps' in fields:
        command = prompt.rsplit('Command: ', 1)[1]
        if 'actions' in fields:
            value = {'actions': scan.interpret(command)}
        else:
            value = scan.parse(command)
            if 'action' not in fields['steps']['items']['properties']:
                value = {'steps': [{k: s[k] for k in ('command', 'repeat')} for s in value['steps']]}
    elif 'Current list: ' in prompt:
        state, command = prompt.rsplit('Current list: ', 1)[1].split('\nInstructions: ')
        ops = [chainops.parse_template(sentence) for sentence in command.split(' Then ')]
        value = {'answer': chainops.execute(json.loads(state), ops)}
    else:
        text = prompt.rsplit('Task: ', 1)[1]
        start, ops = chain_text(text)
        if 'answer' in fields:
            value = {'answer': chainops.execute(start, ops)}
        elif 'ops' in fields:
            value = {'ops': ops}
        else:
            value = {'chunks': [' Then '.join(chainops.canonical(op) for op in ops[i:i+2]) for i in range(0, len(ops), 2)]}
    return response(value, payload['think'])


class DomainTests(unittest.TestCase):
    def test_scan_all_official_lines_both_executors(self):
        count = 0
        for name in ('train', 'test'):
            for row in parse_scan((ROOT / f'data/SCAN/length_split/tasks_{name}_length.txt').read_text()):
                self.assertEqual(scan.interpret(row['command']), row['target'], row['command'])
                plan = scan.parse(row['command'])
                self.assertEqual([t for s in plan['steps'] for leaf in scan.expand(s) for t in scan.leaf_gold(leaf)],
                                 row['target'], row['command'])
                harness.validate_plan(plan, 'scan', True)
                count += 1
        self.assertEqual(count, 20910)

    def test_scan_order_and_metadata(self):
        plan = scan.parse('jump around right twice after turn opposite left thrice')
        self.assertEqual(plan['steps'][0]['command'], 'turn opposite left')
        self.assertEqual(plan['steps'][0]['repeat'], 3)
        self.assertEqual(scan.expand(plan['steps'][0]), ['turn left'] * 6)
        command = 'jump around right twice after walk'
        self.assertEqual(scan.meta(command, scan.interpret(command))['depth'], 4)

    def test_chain_templates_all_roundtrip_and_purity(self):
        self.assertEqual(len(chainops.FUNCTIONS), 14)
        start = [3, 8, 1, 8, 4]
        for name, templates in chainops.TEMPLATES.items():
            self.assertGreaterEqual(len(templates), 2)
            for arg in ([0, 1, 2, 9] if name in chainops.ARG_OPS else [None]):
                op = {'op': name, 'arg': arg}
                for template in templates:
                    parsed = chainops.parse_template(template.format(arg=arg))
                    self.assertEqual(parsed, op)
                    self.assertEqual(chainops.apply(start, parsed), chainops.apply(start, op))
                    self.assertEqual(start, [3, 8, 1, 8, 4])
                    self.assertIsNot(chainops.apply(start, parsed), start)
                self.assertEqual(chainops.parse_template(chainops.canonical(op)), op)
        with self.assertRaises(ValueError):
            chainops.parse_template('Reverse, maybe sort.')

    def test_chain_known_results(self):
        expected = {'reverse': [4, 2, 2, 3], 'sort_asc': [2, 2, 3, 4], 'sort_desc': [4, 3, 2, 2],
                    'remove_first': [2, 4], 'keep_first': [3, 2], 'remove_all': [3, 4],
                    'add': [5, 4, 4, 6], 'multiply': [6, 4, 4, 8], 'keep_even': [2, 2, 4],
                    'keep_odd': [3], 'rotate_left': [2, 4, 3, 2], 'swap_ends': [4, 2, 2, 3],
                    'append': [3, 2, 2, 4, 2], 'unique': [3, 2, 4]}
        for name, result in expected.items():
            self.assertEqual(chainops.apply([3, 2, 2, 4], {'op': name, 'arg': 2 if name in chainops.ARG_OPS else None}), result)

    def test_chain_all_generated_bounds_and_determinism(self):
        splits = build_data.make_chainops()
        self.assertEqual(splits, build_data.make_chainops())
        for split, rows in splits.items():
            depths = []
            for row in rows:
                start, ops = row['gold_plan']['start'], row['gold_plan']['ops']
                self.assertEqual(chain_text(row['input']), (start, ops))
                self.assertEqual(chainops.execute(start, ops), row['target'])
                self.assertEqual(row['meta']['depth'], len(ops))
                self.assertTrue(all(0 <= x <= 20 for x in start))
                self.assertIn(len(start), range(6, 9) if split == 'ood_depth' else range(3, 7))
                self.assertIn(len(ops), range(6, 9) if split == 'ood_depth' else range(1, 5))
                state = start
                for op in ops:
                    if op['op'] in ('remove_first', 'keep_first', 'rotate_left'):
                        self.assertTrue(0 < op['arg'] < len(state))
                    result = chainops.apply(state, op)
                    self.assertNotEqual(result, state)
                    self.assertTrue(1 <= len(result) <= 10)
                    self.assertTrue(all(0 <= x <= 99 for x in result))
                    state = result
                depths.append(len(ops))
            self.assertLessEqual(max(depths.count(d) for d in set(depths)) - min(depths.count(d) for d in set(depths)), 1)


class DataTests(unittest.TestCase):
    def test_build_readback_idempotent_manifest_and_disjoint(self):
        before = {p.name: build_data.sha256(p) for p in (ROOT / 'data').glob('*.json')}
        with tempdir() as directory:
            out = Path(directory)
            manifest = build_data.build(out)
            first = {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob('*') if p.is_file()}
            self.assertEqual(build_data.build(out), manifest)
            self.assertEqual(first, {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob('*') if p.is_file()})
            self.assertEqual(len(manifest['files']), 10)
            for name, info in manifest['files'].items():
                self.assertEqual(build_data.sha256(out / name), info['sha256'])
                self.assertEqual(len(build_data.read_jsonl(out / name)), info['count'])
                self.assertEqual(info['locked'], 'ood_' in name)
            for domain in ('scan', 'chainops'):
                all_rows = [r for p in (out / domain).glob('*.jsonl') for r in build_data.read_jsonl(p)]
                self.assertEqual(len(all_rows), 1628)
                self.assertEqual(len({r['id'] for r in all_rows}), len(all_rows))
                self.assertEqual(len({r['input'] for r in all_rows}), len(all_rows))
            holdout = build_data.read_jsonl(out / 'scan/holdout.jsonl')
            self.assertEqual([r['id'] for r in holdout], [r['id'] for r in json.loads((ROOT / 'data/audit.json').read_text())[32:432]])
        self.assertEqual(before, {p.name: build_data.sha256(p) for p in (ROOT / 'data').glob('*.json')})
        verify_data()


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompts = {d: json.loads((ROOT / f'v2/domains/prompts/{d}_v0.json').read_text()) for d in ('scan', 'chainops')}
        cls.items = {'scan': build_data.make_scan()['dev'], 'chainops': build_data.make_chainops()['dev']}

    def assert_record(self, r):
        types = {'id': str, 'domain': str, 'split': str, 'model': str, 'strategy': str,
                 'prediction': (list, type(None)), 'correct': bool, 'error': (str, type(None)),
                 'prompt_tokens': int, 'completion_tokens': int, 'thinking_tokens': int, 'total_tokens': int,
                 'wall_seconds': float, 'n_calls': int, 'plan': (dict, type(None)), 'logprob_stats': dict,
                 'final_call_logprob_stats': dict, 'calls': list, 'code_responsibility': str,
                 'deployable': bool, 'truncated': bool}
        for key, kind in types.items():
            self.assertIsInstance(r[key], kind, key)
        self.assertEqual(r['total_tokens'], r['prompt_tokens'] + r['completion_tokens'])
        self.assertEqual(r['n_calls'], len(r['calls']))
        self.assertEqual(r['completion_tokens'], sum(c['eval_count'] for c in r['calls']))
        for stat in (r['logprob_stats'], r['final_call_logprob_stats']):
            self.assertEqual(set(stat), {'mean', 'min', 'sum', 'n'})
            self.assertIsInstance(stat['n'], int)
            self.assertIsInstance(stat['sum'], (int, float))
            self.assertIsInstance(stat['mean'], (int, float, type(None)))
            self.assertIsInstance(stat['min'], (int, float, type(None)))
        for c in r['calls']:
            self.assertEqual(len(c['prompt_sha256']), 64)
            self.assertIn(c['role'], ('plan', 'answer'))
            self.assertIsInstance(c['logprobs'], list)
            self.assertTrue(c['request']['logprobs'])
            self.assertEqual(c['request']['options']['temperature'], 0)
            self.assertEqual(c['request']['options']['seed'], 437)
            self.assertEqual(c['request']['options']['num_ctx'], 4096)
            self.assertEqual(c['request']['options']['num_predict'], 2048 if c['request']['think'] else 512)

    @patch.object(harness, 'ollama_call', side_effect=perfect)
    def test_perfect_all_arms_both_domains_schema(self, mock):
        for domain in self.items:
            cache = {}
            # All 128 dev items, not only convenient tiny examples.
            for item in self.items[domain]:
                for strategy in harness.STRATEGIES:
                    record = harness.execute(item, 'mock', strategy, self.prompts[domain], cache)
                    self.assert_record(record)
                    self.assertTrue(record['correct'], (domain, strategy, item['id'], record['error']))
                    self.assertEqual(record['deployable'], strategy.startswith('S'))
                    if strategy == 'S2':
                        self.assertEqual(record['thinking_tokens'], 3)
                        self.assertEqual(record['logprob_stats']['n'], record['completion_tokens'] - 4)
                    if strategy == 'G2':
                        self.assertEqual(record['reused_plan_from'], 'S4')

    @patch.object(harness, 'ollama_call', side_effect=lambda p: response('garbage', p['think']))
    def test_garbage_all_arms(self, mock):
        for domain in self.items:
            for strategy in harness.STRATEGIES:
                r = harness.execute(self.items[domain][0], 'mock', strategy, self.prompts[domain])
                self.assert_record(r)
                self.assertFalse(r['correct'])
                self.assertEqual(r['error'], 'invalid_plan' if strategy in ('S3', 'S4', 'G2') else 'format')
                self.assertEqual(r['n_calls'], 1)

    def test_structural_validation_and_no_gold_repair(self):
        invalid = [{'steps': []}, {'steps': [{'command': 'walk', 'repeat': True}]},
                   {'steps': [{'command': 'walk twice', 'repeat': 2}]}]
        for value in invalid:
            with self.assertRaises(ValueError):
                harness.validate_plan(value, 'scan', False)
        for value in [{'ops': [{'op': 'reverse', 'arg': 1}]}, {'ops': [{'op': 'add', 'arg': None}]}]:
            with self.assertRaises(ValueError):
                harness.validate_plan(value, 'chainops', True)
        with self.assertRaises(ValueError):
            harness.validate_plan({'chunks': ['Reverse the list. Then Reverse the list. Then Reverse the list.']}, 'chainops', False)
        item = copy.deepcopy(self.items['scan'][0])
        item['target'] = ['I_RUN']
        with patch.object(harness, 'ollama_call', return_value=response(scan.parse('walk'))):
            r = harness.execute(item, 'mock', 'G2', self.prompts['scan'])
        self.assertIsNone(r['error'])
        self.assertEqual(r['prediction'], ['I_WALK'])
        self.assertFalse(r['correct'])

    def test_truncation_and_missing_telemetry(self):
        for strategy in harness.STRATEGIES:
            with patch.object(harness, 'ollama_call', side_effect=lambda p: response('{}', p['think'], 'length')):
                r = harness.execute(self.items['scan'][0], 'mock', strategy, self.prompts['scan'])
                self.assertEqual(r['error'], 'truncated')
                self.assertTrue(r['truncated'])
                self.assertFalse(r['correct'])
                self.assertEqual(r['n_calls'], 1)
        truncated = {'response': '', 'thinking': 'ongoing', 'done_reason': 'length', 'eval_count': 2,
                     'logprobs': [{'token': '<think>', 'logprob': -1}, {'token': 'ongoing', 'logprob': -2}]}
        self.assertEqual(harness.token_parts(truncated, True), ([], 2))
        with self.assertRaises(RuntimeError):
            harness.token_parts({'response': '{}', 'eval_count': 2}, False)

    @patch.object(harness, 'ollama_call', side_effect=perfect)
    def test_distinct_scan_leaves_cache_and_fresh_g2(self, mock):
        item = copy.deepcopy(self.items['scan'][0])
        item.update(input='jump around right twice and jump around right',
                    gold_plan=scan.parse('jump around right twice and jump around right'))
        item['target'] = scan.interpret(item['input'])
        for strategy in ('S3', 'S4', 'G1', 'G2'):
            r = harness.execute(item, 'mock', strategy, self.prompts['scan'])
            self.assertTrue(r['correct'])
            self.assertEqual(r['n_calls'], 2 if strategy in ('S3', 'S4') else 1)
            self.assertNotIn('reused_plan_from', r)

    def args(self, domain='scan', strategies=None):
        return harness.parser().parse_args(['run', '--run-name', 'test', '--domain', domain, '--split', 'dev',
            '--models', 'mock-a,mock-b', '--strategies', strategies or ','.join(harness.STRATEGIES),
            '--prompts', str(ROOT / f'v2/domains/prompts/{domain}_v0.json'), '--limit', '3'])

    def test_guards_before_model_calls(self):
        with patch.object(harness, 'ollama_call') as call, patch.object(harness, 'freeze_tag_exists', return_value=False):
            for split, final, collect in [('ood_length', False, False), ('ood_depth', False, False),
                                          ('ood_length', True, False), ('ood_depth', True, False), ('holdout', False, False)]:
                args = self.args()
                args.split, args.final_eval, args.collect_holdout = split, final, collect
                with self.assertRaises(ValueError):
                    harness.run(args)
            call.assert_not_called()
        with patch.object(harness, 'freeze_tag_exists', return_value=True):
            harness.guard_split('ood_length', True)
            harness.guard_split('ood_depth', True)
        harness.guard_split('holdout', collect_holdout=True)

    def test_run_resume_config_order_exemplars_labels(self):
        with tempdir() as directory:
            base = Path(directory)
            for domain in self.items:
                build_data.write_jsonl(base / 'data' / domain / 'dev.jsonl', self.items[domain])
            with patch.object(harness, 'gpu_lock', side_effect=lambda *a: nullcontext()) as lock, \
                 patch.object(harness, 'server_metadata', return_value={'mock-a': 'digest-a', 'mock-b': 'digest-b'}), \
                 patch.object(harness, 'environment', return_value={'test': True}), \
                 patch.object(harness, 'ollama_call', side_effect=perfect) as call:
                args = self.args()
                summary = harness.run(args, base / 'runs', base / 'data')
                self.assertTrue(all(s['accuracy'] == 1 for s in summary.values()))
                record_path = base / 'runs/test/records.jsonl'
                before = record_path.read_bytes()
                rows = build_data.read_jsonl(record_path)
                self.assertEqual(len(rows), 42)
                self.assertTrue(all(r['model'] == 'mock-a' for r in rows[:21]))
                self.assertTrue(all(r['model'] == 'mock-b' for r in rows[21:]))
                self.assertFalse({r['id'] for r in rows} & set(self.prompts['scan']['exemplar_ids']))
                self.assertEqual([r['strategy'] for r in rows[:7]], list(harness.STRATEGIES))
                self.assertEqual([r['strategy'] for r in rows[7:14]], ['G1', 'S4', 'G2', 'S3', 'S2', 'S1', 'S0'])
                self.assertEqual(len(build_data.read_jsonl(base / 'runs/test/warmup.jsonl')), 2)
                lock.assert_called_with('ws1 test')
                call.reset_mock()
                harness.run(args, base / 'runs', base / 'data')
                call.assert_not_called()
                self.assertEqual(record_path.read_bytes(), before)
                args.limit = 2
                with self.assertRaisesRegex(ValueError, 'new run name'):
                    harness.run(args, base / 'runs', base / 'data')
                call.assert_not_called()
                args.limit = 3
                with patch.object(harness, 'server_metadata', return_value={'mock-a': 'changed', 'mock-b': 'digest-b'}):
                    with self.assertRaisesRegex(ValueError, 'new run name'):
                        harness.run(args, base / 'runs', base / 'data')
                build_labels.build(base / 'runs', base / 'labels')
                labels = base / 'labels/scan.csv'
                saved = labels.read_bytes()
                with labels.open(newline='') as stream:
                    reader = csv.DictReader(stream)
                    self.assertEqual(reader.fieldnames, build_labels.COLUMNS)
                    label_rows = list(reader)
                self.assertEqual(len(label_rows), 42)
                first = label_rows[0]
                original = next(r for r in rows if all(r[k] == first[k] for k in ('id', 'model', 'strategy')))
                self.assertEqual(first['answer_sha1'], hashlib.sha1(build_data.canonical_json(original['prediction']).encode()).hexdigest())
                self.assertEqual(first['correct'], '1')
                self.assertEqual(first['total_tokens'], str(original['total_tokens']))
                self.assertEqual(first['mean_logprob'], str(original['logprob_stats']['mean']))
                build_labels.build(base / 'runs', base / 'labels')
                self.assertEqual(labels.read_bytes(), saved)
                duplicate = base / 'runs/duplicate/records.jsonl'
                build_data.write_jsonl(duplicate, rows)
                with self.assertRaisesRegex(ValueError, 'Duplicate label'):
                    build_labels.build(base / 'runs', base / 'labels')
                build_labels.build(base / 'runs', base / 'labels', ['test'])
                self.assertEqual(labels.read_bytes(), saved)

    def test_transport_error_halts_without_label_journals_calls(self):
        with tempdir() as directory:
            base = Path(directory)
            build_data.write_jsonl(base / 'data/scan/dev.jsonl', self.items['scan'])
            count = 0
            def transport(payload):
                nonlocal count
                count += 1
                if count == 3:  # warmup, successful plan, then transport failure
                    raise OSError('connection closed')
                return perfect(payload)
            with patch.object(harness, 'gpu_lock', side_effect=lambda *a: nullcontext()), \
                 patch.object(harness, 'server_metadata', return_value={'mock-a': 'a', 'mock-b': 'b'}), \
                 patch.object(harness, 'environment', return_value={}), \
                 patch.object(harness, 'ollama_call', side_effect=transport):
                with self.assertRaisesRegex(OSError, 'connection closed'):
                    harness.run(self.args(strategies='S3'), base / 'runs', base / 'data')
            self.assertFalse((base / 'runs/test/records.jsonl').exists())
            journal = build_data.read_jsonl(base / 'runs/test/calls.jsonl')
            self.assertEqual(len(journal), 3)
            self.assertIn('transport_error', journal[-1])
            self.assertEqual(journal[-2]['call']['purpose'], 'S3/plan')


if __name__ == '__main__':
    unittest.main()
