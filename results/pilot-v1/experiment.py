"""Reproducible SCAN exploration. Official test data is never sent to the model here."""
import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API = 'http://localhost:11434/api/'
SEED = 437
ACTIONS = ['I_WALK', 'I_RUN', 'I_JUMP', 'I_LOOK', 'I_TURN_LEFT', 'I_TURN_RIGHT']
RULES = """Translate SCAN commands into action tokens.
walk = I_WALK; run = I_RUN; jump = I_JUMP; look = I_LOOK.
turn left = I_TURN_LEFT; turn right = I_TURN_RIGHT.
For an action left/right, turn once that direction then perform the action.
For an action opposite left/right, turn twice then perform the action once.
For an action around left/right, repeat (one turn then the action) four times.
turn opposite left/right means two turns. turn around left/right means four turns.
twice repeats the entire preceding clause twice; thrice repeats it three times.
A and B executes A then B. A after B executes B then A.
Return only the requested JSON. Do not explain."""
ACTION_SCHEMA = {'type': 'object', 'properties': {'actions': {'type': 'array', 'items': {'type': 'string', 'enum': ACTIONS}, 'minItems': 1, 'maxItems': 128}}, 'required': ['actions'], 'additionalProperties': False}
PLAN_SCHEMA = {'type': 'object', 'properties': {'steps': {'type': 'array', 'minItems': 1, 'maxItems': 2, 'items': {'type': 'object', 'properties': {'command': {'type': 'string'}, 'repeat': {'type': 'integer', 'minimum': 1, 'maximum': 3}}, 'required': ['command', 'repeat'], 'additionalProperties': False}}}, 'required': ['steps'], 'additionalProperties': False}
PLAN_INSTRUCTION = """Make a plan, not action tokens. Return {"steps":[{"command":"...","repeat":1}]}.
Each step command is one original clause with only its final twice/thrice removed.
Put that repetition in repeat (1, 2, or 3). Preserve around/opposite and direction.
Split and/after into two steps. For after, reverse the clause order.
Do not expand around/opposite into additional steps. Do not solve the steps."""
LEAF = re.compile(r'(?:(?:walk|run|jump|look)(?: (?:(?:around|opposite) )?(?:left|right))?|turn (?:(?:around|opposite) )?(?:left|right))\Z')


def request(url, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json', 'User-Agent': 'SCAN-course-project'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def parse_scan(text):
    rows = []
    for line in text.splitlines():
        command, target = line.removeprefix('IN: ').split(' OUT: ')
        command = command.strip()
        rows.append({'id': hashlib.sha256(command.encode()).hexdigest()[:16], 'command': command, 'target': target.split()})
    return rows


def prepare():
    manifest_path = ROOT / 'data/manifest.json'
    if manifest_path.exists():
        verify_data()
        print('Existing data and partitions verified; retained.')
        return
    commit = json.loads(request('https://api.github.com/repos/brendenlake/SCAN/commits/master'))['sha']
    manifest = {'source': 'https://github.com/brendenlake/SCAN', 'commit': commit, 'seed': SEED, 'files': {}}
    for name in ['README.md', 'LICENSE', 'length_split/tasks_train_length.txt', 'length_split/tasks_test_length.txt']:
        blob = request(f'https://raw.githubusercontent.com/brendenlake/SCAN/{commit}/{name}')
        dest = ROOT / 'data/SCAN' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        manifest['files'][name] = hashlib.sha256(blob).hexdigest()
    train = parse_scan((ROOT / 'data/SCAN/length_split/tasks_train_length.txt').read_text())
    test = parse_scan((ROOT / 'data/SCAN/length_split/tasks_test_length.txt').read_text())
    assert not {r['id'] for r in train} & {r['id'] for r in test}
    random.Random(SEED).shuffle(train)
    # Partition the entire training pool before running or inspecting any model results.
    slices = {'development': train[:128], 'controller_train': train[128:10128], 'validation': train[10128:13128], 'audit': train[13128:]}
    for name, rows in slices.items():
        save(ROOT / f'data/{name}.json', rows)
    manifest['partition_counts'] = {k: len(v) for k, v in slices.items()}
    manifest['official_test_count'] = len(test)
    manifest['partition_sha256'] = {k: hashlib.sha256((ROOT / f'data/{k}.json').read_bytes()).hexdigest() for k in slices}
    save(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


def verify_data():
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((ROOT / 'data/SCAN' / name).read_bytes()).hexdigest() == digest, name
    seen = set()
    for name, digest in manifest['partition_sha256'].items():
        path = ROOT / f'data/{name}.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
        rows = json.loads(path.read_text())
        ids = {r['id'] for r in rows}
        assert len(ids) == len(rows) and not seen & ids
        seen.update(ids)
    test = parse_scan((ROOT / 'data/SCAN/length_split/tasks_test_length.txt').read_text())
    assert not seen & {r['id'] for r in test}


def validate_actions(obj):
    if not isinstance(obj, dict) or set(obj) != {'actions'}:
        raise ValueError('Expected only actions key')
    actions = obj['actions']
    if not isinstance(actions, list) or not 1 <= len(actions) <= 128 or any(x not in ACTIONS for x in actions):
        raise ValueError('Invalid action sequence')
    return actions


def validate_plan(obj):
    if not isinstance(obj, dict) or set(obj) != {'steps'} or not isinstance(obj['steps'], list) or not 1 <= len(obj['steps']) <= 2:
        raise ValueError('Invalid plan structure')
    for step in obj['steps']:
        if not isinstance(step, dict) or set(step) != {'command', 'repeat'} or type(step['repeat']) is not int or step['repeat'] not in (1, 2, 3) or not isinstance(step['command'], str) or not LEAF.fullmatch(step['command']):
            raise ValueError('Invalid plan step')
    return obj['steps']


def generate(model, prompt, schema, calls):
    payload = {'model': model, 'system': RULES, 'prompt': prompt, 'format': schema, 'stream': False, 'think': False, 'keep_alive': '15m', 'options': {'temperature': 0, 'seed': SEED, 'num_ctx': 4096, 'num_predict': 512}}
    start = time.perf_counter()
    response = json.loads(request(API + 'generate', payload))
    calls.append({'prompt': prompt, 'wall_seconds': time.perf_counter() - start, **response})
    if not response.get('done') or response.get('done_reason') == 'length':
        raise ValueError('Incomplete or truncated model response')
    return json.loads(response['response'])


def execute(command, strategy, model):
    calls, prediction, error = [], [], None
    start = time.perf_counter()
    try:
        if strategy == 'direct':
            prediction = validate_actions(generate(model, f'Command: {command}\nReturn {{"actions":[...]}}.', ACTION_SCHEMA, calls))
        else:
            plan = validate_plan(generate(model, PLAN_INSTRUCTION + '\nCommand: ' + command, PLAN_SCHEMA, calls))
            for step in plan:
                actions = validate_actions(generate(model, f'Command: {step["command"]}\nReturn {{"actions":[...]}}.', ACTION_SCHEMA, calls))
                prediction.extend(actions * step['repeat'])
    except (ValueError, KeyError, TypeError) as exc:
        error = str(exc)
    # Transport/server failures deliberately stop the run, rather than becoming incorrect labels.
    return {'strategy': strategy, 'prediction': prediction, 'error': error, 'wall_seconds': time.perf_counter() - start, 'prompt_tokens': sum(c.get('prompt_eval_count', 0) for c in calls), 'completion_tokens': sum(c.get('eval_count', 0) for c in calls), 'load_seconds': sum(c.get('load_duration', 0) for c in calls) / 1e9, 'calls': calls}


def run(args):
    verify_data()
    out = ROOT / 'results' / args.name
    out.mkdir(parents=True, exist_ok=True)
    tags = json.loads(request(API + 'tags'))
    model_info = next(m for m in tags['models'] if m['name'] == args.model)
    config = {'model': args.model, 'model_digest': model_info['digest'], 'seed': SEED, 'partition': args.partition, 'count': args.count, 'rules': RULES, 'plan_instruction': PLAN_INSTRUCTION, 'action_schema': ACTION_SCHEMA, 'plan_schema': PLAN_SCHEMA, 'temperature': 0, 'think': False, 'num_ctx': 4096, 'num_predict': 512, 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'data_manifest_sha256': hashlib.sha256((ROOT / 'data/manifest.json').read_bytes()).hexdigest()}
    if (out / 'config.json').exists():
        if json.loads((out / 'config.json').read_text()) != config:
            raise ValueError('Run configuration changed: use a new --name')
    else:
        save(out / 'config.json', config)
        save(out / 'environment.json', {'date': datetime.now(timezone.utc).isoformat(), 'python': sys.version, 'ollama': json.loads(request(API + 'version')), 'model': model_info})
    log = out / 'records.jsonl'
    existing = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    done = {(r['id'], r['strategy']) for r in existing}
    rows = json.loads((ROOT / f'data/{args.partition}.json').read_text())[:args.count]
    if len(rows) != args.count:
        raise ValueError('Requested count exceeds partition')
    warmup = []
    generate(args.model, 'Command: walk\nReturn {"actions":[...]}.', ACTION_SCHEMA, warmup)
    save(out / 'warmup.json', warmup)
    with log.open('a', encoding='utf-8') as stream:
        for i, row in enumerate(rows):
            order = ['direct', 'decomposed'] if i % 2 == 0 else ['decomposed', 'direct']
            for strategy in order:
                if (row['id'], strategy) in done:
                    continue
                result = execute(row['command'], strategy, args.model)
                result.update(row)
                result['correct'] = result['error'] is None and result['prediction'] == row['target']
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                print(f'{i + 1}/{len(rows)} {strategy}: {result["correct"]} {result["wall_seconds"]:.2f}s {row["command"]}', flush=True)
    summarize(out)


def summarize(out):
    records = [json.loads(line) for line in (out / 'records.jsonl').read_text().splitlines()]
    groups = {s: [r for r in records if r['strategy'] == s] for s in ['direct', 'decomposed']}
    summary = {}
    for strategy, rows in groups.items():
        summary[strategy] = {'n': len(rows), 'correct': sum(r['correct'] for r in rows), 'accuracy': sum(r['correct'] for r in rows) / len(rows), 'mean_seconds': sum(r['wall_seconds'] for r in rows) / len(rows), 'mean_prompt_tokens': sum(r['prompt_tokens'] for r in rows) / len(rows), 'mean_completion_tokens': sum(r['completion_tokens'] for r in rows) / len(rows), 'invalid': sum(r['error'] is not None for r in rows)}
    paired = {}
    for row in records:
        paired.setdefault(row['id'], {})[row['strategy']] = row['correct']
    cells = Counter(('both_correct' if p['direct'] and p['decomposed'] else 'direct_only_correct' if p['direct'] else 'decomposed_only_correct' if p['decomposed'] else 'both_wrong') for p in paired.values() if len(p) == 2)
    summary['paired_outcomes'] = dict(cells)
    save(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('prepare')
    sub.add_parser('verify')
    p = sub.add_parser('run')
    p.add_argument('--partition', choices=['development', 'controller_train', 'validation', 'audit'], default='development')
    p.add_argument('--count', type=int, default=24)
    p.add_argument('--name', default='pilot-v1')
    p.add_argument('--model', default='qwen3:1.7b')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'verify':
        verify_data()
        print('Dataset hashes, partition hashes, and disjointness verified.')
    else:
        run(args)
