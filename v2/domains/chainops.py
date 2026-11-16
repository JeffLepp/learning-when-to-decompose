"""Bounded integer-list chains. All 14 explicitly requested operations are kept.

Every operation is pure and returns a fresh list. Python slicing defines remove/
keep/rotate semantics for parsed plans; generator arguments are stricter and are
chosen against the current state. parse_template is a test helper, never a model
plan verifier. Generated chains reject every no-op and invalid intermediate.
"""
import json
import re


def reverse(xs, arg=None): return list(reversed(xs))
def sort_asc(xs, arg=None): return sorted(xs)
def sort_desc(xs, arg=None): return sorted(xs, reverse=True)
def remove_first(xs, arg): return list(xs[arg:])
def keep_first(xs, arg): return list(xs[:arg])
def remove_all(xs, arg): return [x for x in xs if x != arg]
def add(xs, arg): return [x + arg for x in xs]
def multiply(xs, arg): return [x * arg for x in xs]
def keep_even(xs, arg=None): return [x for x in xs if x % 2 == 0]
def keep_odd(xs, arg=None): return [x for x in xs if x % 2 != 0]
def rotate_left(xs, arg): return list(xs[arg:] + xs[:arg])
def swap_ends(xs, arg=None): return [xs[-1], *xs[1:-1], xs[0]] if len(xs) > 1 else list(xs)
def append(xs, arg): return [*xs, arg]
def unique(xs, arg=None): return list(dict.fromkeys(xs))


FUNCTIONS = {f.__name__: f for f in (reverse, sort_asc, sort_desc, remove_first, keep_first, remove_all,
                                   add, multiply, keep_even, keep_odd, rotate_left, swap_ends, append, unique)}
TEMPLATES = {
    'reverse': ['Reverse the list.', 'Put the list in reverse order.'],
    'sort_asc': ['Sort the list in ascending order.', 'Sort the numbers from smallest to largest.'],
    'sort_desc': ['Sort the list in descending order.', 'Sort the numbers from largest to smallest.'],
    'remove_first': ['Remove the first {arg} elements.', 'Delete the first {arg} entries of the list.'],
    'keep_first': ['Keep only the first {arg} elements.', 'Retain just the first {arg} entries of the list.'],
    'remove_all': ['Remove all occurrences of {arg}.', 'Delete every number equal to {arg}.'],
    'add': ['Add {arg} to each number.', 'Increase every number by {arg}.'],
    'multiply': ['Multiply each number by {arg}.', 'Scale every number by a factor of {arg}.'],
    'keep_even': ['Keep only even numbers.', 'Remove every odd number.'],
    'keep_odd': ['Keep only odd numbers.', 'Remove every even number.'],
    'rotate_left': ['Rotate left by {arg} positions (move the first {arg} elements to the end).',
                    'Move the first {arg} elements to the end, preserving their order.'],
    'swap_ends': ['Swap the first and last elements.', 'Exchange the first entry with the last entry.'],
    'append': ['Append {arg} to the end of the list.', 'Place {arg} after the last element.'],
    'unique': ['Remove duplicates, keeping the first occurrence of each number.',
               'Keep only the first occurrence of each number, in original order.'],
}
ARG_OPS = {name for name, templates in TEMPLATES.items() if '{arg}' in templates[0]}


def apply(lst: list[int], op: dict) -> list[int]:
    name, arg = op['op'], op['arg']
    if name not in FUNCTIONS or (type(arg) is not int if name in ARG_OPS else arg is not None):
        raise ValueError('Invalid operation or argument')
    if name in ARG_OPS and not 0 <= arg <= 99:
        raise ValueError('Argument outside 0..99')
    return FUNCTIONS[name](lst, arg)


def execute(start: list[int], ops: list[dict]) -> list[int]:
    state = list(start)
    for op in ops:
        state = apply(state, op)
    return state


def canonical(op: dict) -> str:
    return TEMPLATES[op['op']][0].format(arg=op['arg'])


def render(start: list[int], ops: list[dict], rng) -> str:
    sentences = [rng.choice(TEMPLATES[op['op']]).format(arg=op['arg']) for op in ops]
    return 'Start with ' + json.dumps(start) + '. ' + ' Then '.join(sentences)


def parse_template(sentence: str) -> dict:
    matches = []
    for name, templates in TEMPLATES.items():
        for template in templates:
            parts = template.split('{arg}')
            pattern = re.escape(parts[0])
            for index, part in enumerate(parts[1:]):
                pattern += (r'(?P<arg>\d+)' if index == 0 else r'(?P=arg)') + re.escape(part)
            match = re.fullmatch(pattern, sentence)
            if match:
                matches.append({'op': name, 'arg': int(match['arg']) if name in ARG_OPS else None})
    if len(matches) != 1:
        raise ValueError(f'Template must match exactly once: {sentence}')
    return matches[0]


def meta(start: list[int], ops: list[dict], target: list[int]) -> dict:
    return {'depth': len(ops), 'n_ops': len(ops), 'list_len': len(start),
            'target_len': len(target), 'ops': [op['op'] for op in ops]}


def generate(rng, depth: int, ood: bool = False) -> tuple:
    """Rejection sample bounded, nonempty chains with no unchanged step."""
    names = list(FUNCTIONS)
    for _ in range(10000):
        start = [rng.randint(0, 20) for _ in range(rng.randint(6, 8) if ood else rng.randint(3, 6))]
        state, ops = start, []
        for _ in range(depth):
            for attempt in range(1000):
                name = rng.choice(names)
                if name in ('remove_first', 'keep_first', 'rotate_left'):
                    if len(state) < 2:
                        continue
                    arg = rng.randint(1, len(state) - 1)
                elif name == 'remove_all': arg = rng.choice(state)
                elif name == 'add': arg = rng.randint(1, 5)
                elif name == 'multiply': arg = rng.randint(2, 4)
                elif name == 'append': arg = rng.randint(0, 20)
                else: arg = None
                op = {'op': name, 'arg': arg}
                result = apply(state, op)
                if result != state and 1 <= len(result) <= 10 and all(0 <= x <= 99 for x in result):
                    ops.append(op)
                    state = result
                    break
            else:
                break
        if len(ops) == depth:
            return start, ops, state
    raise RuntimeError('Could not generate a valid chain')
