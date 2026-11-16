"""Auditable local execution of S0--S4 and diagnostic G1/G2.

Endpoint check (Ollama 0.40.13, qwen3:0.6b, 2026-11-15) is preserved in
workspace_logs/ws1-endpoint-check.json. Both /api/chat and /api/generate accept
format + logprobs:true + think together. We consistently use /api/generate:
its token stream preserves <think> and </think>, unlike chat's filtered stream.
The initial generate check had eval_count=326, 325 logprob entries: 286 entries
through </think> (the thinking phase, INCLUDING its delimiters), 39 content
entries, and one unreported end token. Nonthinking had 23 generated / 22 logged.

thinking_tokens counts generated logprob entries in the thinking phase,
including delimiters; completion_tokens always uses the full eval_count. EOS
or other unreported special tokens remain in completion_tokens, not thinking.
On truncation before </think>, all entries from <think> onward are thinking.
Content statistics use ONLY entries outside that phase, excluding special
tokens, and check that their concatenated bytes equal response content. Missing
or incompatible telemetry halts instead of manufacturing confidence labels.

All prompts (including warmup) come only from the versioned JSON plus an item.
Runtime validation checks structure, enum and bounds, never gold semantics.
ChainOps S3 chunks must contain one or two period-terminated sentences joined
by ' Then '; structural checks cannot prove a sentence contains one semantic
operation. n_calls counts charged logical calls, including a reused S4 parse
in G2. Reused parse tokens AND measured call time are charged to G2, while the
call's reused flag distinguishes physical requests. Full-run resume does no
warmup or inference. Transport errors escape without an item-arm record.
"""
import argparse
import copy
import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from experiment import ACTION_SCHEMA, API, LEAF, PLAN_SCHEMA, ROOT, SEED, request, save
from v2.build_data import canonical_json, read_jsonl, sha256
from v2.domains import chainops, scan
from v2.gpulock import gpu_lock

STRATEGIES = ('S0', 'S1', 'S2', 'S3', 'S4', 'G1', 'G2')
OPTIONS = {'temperature': 0, 'seed': SEED, 'num_ctx': 4096, 'num_predict': 512}
TOKEN_METHOD = {'endpoint': API + 'generate', 'method': 'logprob <think>..</think> inclusive; remaining content only',
                'unreported_special_tokens': 'included in eval_count, excluded from thinking and content stats',
                'verified_ollama': '0.40.13', 'check_file': 'workspace_logs/ws1-endpoint-check.json',
                'generate_thinking_check': {'eval_count': 326, 'logprob_entries': 325,
                                            'thinking_phase': 286, 'content': 39, 'unreported': 1},
                'generate_nonthinking_check': {'eval_count': 23, 'logprob_entries': 22}}


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def array(items, low, high):
    return {'type': 'array', 'items': items, 'minItems': low, 'maxItems': high}


def enum(values):
    return {'type': 'string', 'enum': list(values)}


OP_SCHEMA = obj({'op': enum(chainops.FUNCTIONS), 'arg': {'type': ['integer', 'null'], 'minimum': 0, 'maximum': 99}})
SCHEMAS = {
    'scan': {'answer': ACTION_SCHEMA, 'plan': PLAN_SCHEMA,
             'parse': obj({'steps': array(obj({'command': {'type': 'string', 'minLength': 1, 'maxLength': 160},
                                              'action': enum(['walk', 'run', 'jump', 'look', 'turn']),
                                              'direction': enum(['left', 'right', 'none']),
                                              'modifier': enum(['none', 'opposite', 'around']),
                                              'repeat': {'type': 'integer', 'minimum': 1, 'maximum': 3}}), 1, 2)})},
    'chainops': {'answer': obj({'answer': array({'type': 'integer'}, 0, 100)}),
                 'plan': obj({'chunks': array({'type': 'string', 'minLength': 1, 'maxLength': 1000}, 1, 8)}),
                 'parse': obj({'ops': array(OP_SCHEMA, 1, 8)})},
}
RESPONSIBILITY = {
    'S0': 'JSON validation and exact-match scoring only',
    'S1': 'Fixed dev few-shot assembly; JSON validation and exact-match scoring',
    'S2': 'JSON validation and exact-match scoring; model thinking enabled',
    'S3': 'Model coarse plan; code caches distinct SCAN clauses, repeats/concatenates or passes ChainOps state',
    'S4': 'Model structured parse; code expands SCAN structure, caches atomic leaves or passes ChainOps state',
    'G1': 'Diagnostic supplied gold structure; code expands/caches leaves or passes state; model executes leaves',
    'G2': 'Diagnostic model structured parse; code performs all expansion and leaf execution',
}


def validate(value, schema):
    """Small strict validator for the JSON-schema subset used in this module."""
    kinds = schema.get('type', [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    matches = {'object': isinstance(value, dict), 'array': isinstance(value, list),
               'string': isinstance(value, str), 'integer': type(value) is int,
               'number': type(value) in (int, float) and math.isfinite(value),
               'boolean': type(value) is bool, 'null': value is None}
    if kinds and not any(matches[k] for k in kinds):
        raise ValueError('Wrong JSON type')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError('Invalid enum')
    if isinstance(value, dict):
        if not set(schema.get('required', ())) <= value.keys():
            raise ValueError('Missing field')
        properties = schema.get('properties', {})
        if schema.get('additionalProperties') is False and not value.keys() <= properties.keys():
            raise ValueError('Unexpected field')
        for key, entry in value.items():
            if key in properties:
                validate(entry, properties[key])
    elif isinstance(value, list):
        if not schema.get('minItems', 0) <= len(value) <= schema.get('maxItems', math.inf):
            raise ValueError('Array bounds')
        for entry in value:
            validate(entry, schema.get('items', {}))
    elif isinstance(value, str):
        if not schema.get('minLength', 0) <= len(value) <= schema.get('maxLength', math.inf):
            raise ValueError('String bounds')
    elif type(value) in (int, float):
        if not schema.get('minimum', -math.inf) <= value <= schema.get('maximum', math.inf):
            raise ValueError('Numeric bounds')
    return value


def validate_plan(plan, domain, fine):
    validate(plan, SCHEMAS[domain]['parse' if fine else 'plan'])
    if domain == 'scan':
        for step in plan['steps']:
            if fine:
                if step['direction'] == 'none' and (step['action'] == 'turn' or step['modifier'] != 'none'):
                    raise ValueError('Turn/modifier needs direction')
            elif not LEAF.fullmatch(step['command']):
                raise ValueError('Clause syntax')
    elif fine:
        for op in plan['ops']:
            if (op['op'] in chainops.ARG_OPS) != (type(op['arg']) is int):
                raise ValueError('Operation argument type')
    else:
        for chunk in plan['chunks']:
            if not re.fullmatch(r'[^.!?]+\.(?: Then [^.!?]+\.)?', chunk):
                raise ValueError('Chunk needs one or two operation sentences')
    return plan


def ollama_call(payload):
    """The only inference transport; injectable by patching this function."""
    response = json.loads(request(API + 'generate', payload, timeout=600))
    if 'error' in response or not response.get('done'):
        raise RuntimeError(f'Ollama server error or incomplete transport: {response}')
    return response


def token_parts(response, thinking):
    logs = response.get('logprobs')
    if not isinstance(logs, list) or (response.get('eval_count', 0) and not logs):
        raise RuntimeError('Missing Ollama logprobs; halt collection')
    content, n_thinking, in_thought, saw_open, saw_close = [], 0, False, False, False
    # qwen3:4b's template prefills <think>, so its stream starts inside the thought with no opening token.
    if thinking and response.get('thinking') and not any(e['token'] == '<think>' for e in logs):
        in_thought = saw_open = True
    for entry in logs:
        token = entry['token']
        if token == '<think>':
            in_thought, saw_open = True, True
        if in_thought:
            n_thinking += 1
        elif token not in ('<|im_end|>', '<|endoftext|>'):
            content.append(entry)
        if token == '</think>':
            in_thought, saw_close = False, True
    if response.get('thinking') and not saw_open:
        raise RuntimeError('Thinking text without explicit token boundary')
    if not thinking and n_thinking:
        raise RuntimeError('Unexpected thinking despite think:false')
    joined = b''.join(bytes(e['bytes']) if 'bytes' in e else e['token'].encode() for e in content).decode('utf-8')
    if joined != response.get('response', ''):
        raise RuntimeError('Logprob/content alignment changed; halt collection')
    if len(logs) > response['eval_count'] or n_thinking > response['eval_count']:
        raise RuntimeError('Inconsistent generated token counts')
    if saw_open and not saw_close and response.get('done_reason') != 'length':
        raise RuntimeError('Unclosed thinking boundary without truncation')
    return content, n_thinking


def stats(entries):
    values = [e['logprob'] for e in entries]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise RuntimeError('Invalid token logprob')
    return {'mean': statistics.mean(values) if values else None,
            'min': min(values) if values else None, 'sum': sum(values), 'n': len(values)}


def prompt_parts(prompts, arm, stage, **fields):
    block = prompts[arm][stage]
    lines = [block['instruction']]
    for example in block['examples']:
        lines.extend([example['input'], canonical_json(example['output'])])
    lines.append(block['template'].format(**fields))
    return block['system'], '\n\n'.join(lines)


class ArmError(Exception):
    pass


def execute(item, model, strategy, prompts, plan_cache=None, journal=None):
    calls, plan, prediction, error = [], None, None, None
    domain = item['domain']
    answer_key = 'actions' if domain == 'scan' else 'answer'
    started = time.perf_counter()
    charged_reuse_seconds = 0.0

    def call(arm, stage, schema_key, **fields):
        system, prompt = prompt_parts(prompts, arm, stage, input=item['input'], **fields)
        thinking = arm == 'S2'
        payload = {'model': model, 'system': system, 'prompt': prompt,
                   'format': SCHEMAS[domain][schema_key], 'stream': False, 'think': thinking,
                   'logprobs': True, 'keep_alive': '15m',
                   'options': {**OPTIONS, 'num_predict': 2048 if thinking else 512}}
        t0 = time.perf_counter()
        try:
            response = ollama_call(payload)
        except Exception as exc:
            if journal is not None:
                journal({'id': item['id'], 'model': model, 'strategy': strategy,
                         'request': payload, 'purpose': arm + '/' + stage,
                         'transport_error': type(exc).__name__ + ': ' + str(exc),
                         'wall_seconds': time.perf_counter() - t0})
            raise
        entry = {**response, 'request': payload, 'role': 'plan' if schema_key in ('plan', 'parse') else 'answer',
                 'purpose': arm + '/' + stage, 'prompt_sha256': hashlib.sha256(canonical_json(
                     {'system': system, 'prompt': prompt}).encode()).hexdigest(),
                 'wall_seconds': time.perf_counter() - t0, 'reused': False}
        calls.append(entry)
        if journal is not None:
            journal({'id': item['id'], 'model': model, 'strategy': strategy, 'call': entry})
        token_parts(entry, thinking)
        if arm == 'S4' and stage == 'parse' and plan_cache is not None:
            plan_cache[(item['id'], model)] = copy.deepcopy(entry)
        return decode(entry, schema_key)

    def decode(entry, schema_key):
        if entry.get('done_reason') == 'length':
            raise ArmError('truncated')
        is_plan = schema_key in ('plan', 'parse')
        try:
            value = json.loads(entry['response'])
            if is_plan:
                return validate_plan(value, domain, schema_key == 'parse')
            return validate(value, SCHEMAS[domain]['answer'])[answer_key]
        except (ValueError, TypeError, KeyError):
            raise ArmError('invalid_plan' if is_plan else 'format') from None

    try:
        if strategy in ('S0', 'S1', 'S2'):
            prediction = call(strategy, 'direct', 'answer')
        else:
            if strategy == 'G1':
                plan = copy.deepcopy(item['gold_plan'])
            elif strategy == 'S3':
                plan = call('S3', 'plan', 'plan')
            else:
                cached = plan_cache.get((item['id'], model)) if plan_cache is not None and strategy == 'G2' else None
                if cached:
                    cached = copy.deepcopy(cached)
                    cached['reused'] = True
                    calls.append(cached)
                    charged_reuse_seconds = cached['wall_seconds']
                    plan = decode(cached, 'parse')
                else:
                    plan = call('S4', 'parse', 'parse')
            if domain == 'scan':
                prediction, solved = [], {}
                if strategy == 'S3':
                    for step in plan['steps']:
                        command = step['command']
                        if command not in solved:
                            solved[command] = call('S3', 'execute', 'answer', command=command)
                        prediction.extend(solved[command] * step['repeat'])
                else:
                    for step in plan['steps']:
                        for leaf in scan.expand(step):
                            if leaf not in solved:
                                solved[leaf] = scan.leaf_gold(leaf) if strategy == 'G2' else call('S4', 'leaf', 'answer', command=leaf)
                            prediction.extend(solved[leaf])
            else:
                # Start list is visible input, never an oracle reference for deployable arms.
                match = re.match(r'Start with (\[[^\]]*\])\.', item['input'])
                if not match:
                    raise RuntimeError('Invalid source item start list')
                prediction = json.loads(match[1])
                if strategy == 'S3':
                    for chunk in plan['chunks']:
                        prediction = call('S3', 'execute', 'answer', state=json.dumps(prediction), command=chunk)
                else:
                    for op in plan['ops']:
                        prediction = chainops.apply(prediction, op) if strategy == 'G2' else call(
                            'S4', 'leaf', 'answer', state=json.dumps(prediction), command=chainops.canonical(op))
    except ArmError as exc:
        error = str(exc)
        prediction = None
    content_parts = [token_parts(c, c['request']['think']) for c in calls]
    prompt_tokens = sum(c['prompt_eval_count'] for c in calls)
    completion_tokens = sum(c['eval_count'] for c in calls)
    record = {'id': item['id'], 'domain': domain, 'split': item['split'], 'model': model,
              'strategy': strategy, 'prediction': prediction,
              'correct': error is None and prediction == item['target'], 'error': error,
              'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
              'thinking_tokens': sum(n for _, n in content_parts), 'total_tokens': prompt_tokens + completion_tokens,
              'wall_seconds': time.perf_counter() - started + charged_reuse_seconds,
              'n_calls': len(calls), 'plan': plan,
              'logprob_stats': stats([entry for entries, _ in content_parts for entry in entries]),
              'final_call_logprob_stats': stats(content_parts[-1][0] if content_parts else []),
              'calls': calls, 'code_responsibility': RESPONSIBILITY[strategy],
              'deployable': strategy not in ('G1', 'G2'), 'truncated': error == 'truncated'}
    if any(c['reused'] for c in calls):
        record['reused_plan_from'] = 'S4'
    return record


def freeze_tag_exists():
    result = subprocess.run(['git', 'tag', '-l', 'freeze-v2'], cwd=ROOT, capture_output=True, text=True, check=True)
    return 'freeze-v2' in result.stdout.splitlines()


def guard_split(split, final_eval=False, collect_holdout=False):
    if split in ('ood_length', 'ood_depth') and (not final_eval or not freeze_tag_exists()):
        raise ValueError('OOD locked: requires --final-eval AND existing freeze-v2 tag')
    if split == 'holdout' and not collect_holdout:
        raise ValueError('holdout requires --collect-holdout')


def server_metadata(models):
    available = {m['name']: m for m in json.loads(request(API + 'tags'))['models']}
    missing = set(models) - available.keys()
    if missing:
        raise ValueError(f'Ollama models unavailable: {sorted(missing)}')
    return {m: available[m]['digest'] for m in models}


def environment():
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        gpu = 'unavailable: ' + str(exc)
    return {'time': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
            'ollama': json.loads(request(API + 'version')), 'GPU': gpu, 'platform': platform.platform()}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['model'], row['strategy'])].append(row)
    return {model + '/' + strategy: {'n': len(group), 'accuracy': statistics.mean(r['correct'] for r in group),
            'mean_seconds': statistics.mean(r['wall_seconds'] for r in group),
            'median_seconds': statistics.median(r['wall_seconds'] for r in group),
            'mean_tokens': statistics.mean(r['total_tokens'] for r in group),
            'truncation_rate': statistics.mean(r['truncated'] for r in group),
            'format_failure_rate': statistics.mean(r['error'] == 'format' for r in group),
            'invalid_plan_rate': statistics.mean(r['error'] == 'invalid_plan' for r in group)}
            for (model, strategy), group in sorted(groups.items())}


def append_record(path, row):
    with path.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(canonical_json(row) + '\n')
        stream.flush()


def run(args, runs_dir=ROOT / 'results/v2/runs', data_dir=ROOT / 'data/v2'):
    guard_split(args.split, args.final_eval, args.collect_holdout)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name):
        raise ValueError('Run name must be a simple directory name')
    models, strategies = args.models.split(','), args.strategies.split(',')
    if not models or len(set(models)) != len(models) or not strategies or len(set(strategies)) != len(strategies) or set(strategies) - set(STRATEGIES):
        raise ValueError('Invalid or duplicate models/strategies')
    item_path = Path(data_dir) / args.domain / (args.split + '.jsonl')
    all_items = read_jsonl(item_path)
    if any(r['domain'] != args.domain or r['split'] != args.split for r in all_items):
        raise ValueError('Item domain/split mismatch')
    prompts = json.loads(Path(args.prompts).read_text(encoding='utf-8'))
    if prompts['domain'] != args.domain or prompts['S2'] != prompts['S0']:
        raise ValueError('Prompt domain mismatch or S2 differs from S0')
    rows = [r for r in all_items if args.split != 'dev' or r['id'] not in prompts['exemplar_ids']]
    if args.ids_file:
        raw = Path(args.ids_file).read_text(encoding='utf-8')
        ids = json.loads(raw) if raw.lstrip().startswith('[') else raw.split()
        if len(set(ids)) != len(ids) or set(ids) - {r['id'] for r in rows}:
            raise ValueError('Unknown, duplicate, or excluded exemplar IDs')
        rows = [r for r in rows if r['id'] in set(ids)]
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(rows):
            raise ValueError('Invalid limit')
        rows = rows[:args.limit]
    if not rows:
        raise ValueError('No selected items')
    with gpu_lock(f'ws1 {args.run_name}'):
        digests = server_metadata(models)
        paths = [Path(__file__), Path(scan.__file__), Path(chainops.__file__), ROOT / 'experiment.py', ROOT / 'v2/build_data.py', ROOT / 'v2/gpulock.py']
        config = {'domain': args.domain, 'split': args.split, 'models': digests, 'model_order': models,
                  'strategies': strategies, 'prompts': prompts, 'prompts_sha256': sha256(args.prompts),
                  'schemas': SCHEMAS[args.domain], 'decoding': {'options': OPTIONS, 'thinking_num_predict': 2048,
                  'thinking_strategies': ['S2'], 'logprobs': True, 'stream': False, 'keep_alive': '15m'},
                  'code_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths},
                  'items_sha256': sha256(item_path), 'item_ids': [r['id'] for r in rows],
                  'token_accounting': TOKEN_METHOD, 'endpoint': API + 'generate',
                  'final_eval': args.final_eval, 'collect_holdout': args.collect_holdout}
        out = Path(runs_dir) / args.run_name
        out.mkdir(parents=True, exist_ok=True)
        config_path = out / 'config.json'
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise ValueError('Run configuration changed; pick a new run name')
        else:
            save(config_path, config)
            save(out / 'environment.json', environment())
        log = out / 'records.jsonl'
        records = read_jsonl(log) if log.exists() else []
        done = {(r['id'], r['model'], r['strategy']) for r in records}
        if len(done) != len(records):
            raise ValueError('Duplicate existing record keys')
        cache = {(r['id'], r['model']): r['calls'][0] for r in records if r['strategy'] == 'S4' and r['calls']}
        warmup_path = out / 'warmup.jsonl'
        warmed = {r['model'] for r in read_jsonl(warmup_path)} if warmup_path.exists() else set()
        for model in models:
            if not any((r['id'], model, s) not in done for r in rows for s in strategies):
                continue
            if model not in warmed:
                warm = execute(rows[0], model, 'S0', prompts,
                               journal=lambda entry: append_record(out / 'calls.jsonl', {**entry, 'warmup': True}))
                warm['excluded_warmup'] = True
                append_record(warmup_path, warm)
            for index, row in enumerate(rows):
                order = list(strategies if index % 2 == 0 else reversed(strategies))
                if 'S4' in order and 'G2' in order and order.index('G2') < order.index('S4'):
                    order.remove('G2')
                    order.insert(order.index('S4') + 1, 'G2')
                for strategy in order:
                    key = (row['id'], model, strategy)
                    if key in done:
                        continue
                    record = execute(row, model, strategy, prompts, cache,
                                     journal=lambda entry: append_record(out / 'calls.jsonl', entry))
                    append_record(log, record)
                    records.append(record)
                    done.add(key)
                    print(f'{model} {index + 1}/{len(rows)} {strategy}: {record["correct"]} '
                          f'{record["error"]} {record["wall_seconds"]:.3f}s', flush=True)
        summary = summarize(records)
        save(out / 'summary.json', summary)
        return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='action', required=True)
    p = commands.add_parser('run')
    p.add_argument('--run-name', required=True)
    p.add_argument('--domain', choices=['scan', 'chainops'], required=True)
    p.add_argument('--split', choices=['dev', 'router_train', 'router_val', 'holdout', 'ood_length', 'ood_depth'], required=True)
    p.add_argument('--models', required=True)
    p.add_argument('--strategies', default=','.join(STRATEGIES))
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--ids-file', type=Path)
    p.add_argument('--final-eval', action='store_true')
    p.add_argument('--collect-holdout', action='store_true')
    return result


if __name__ == '__main__':
    run(parser().parse_args())
