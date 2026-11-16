"""Official SCAN grammar, gold structure, and deterministic atomic execution."""
import re

from experiment import ACTIONS, LEAF


def parse(command: str) -> dict:
    parts = re.split(r' (and|after) ', command)
    if len(parts) not in (1, 3):
        raise ValueError('SCAN has one or two clauses')
    clauses = parts[::2]
    if len(parts) == 3 and parts[1] == 'after':
        clauses.reverse()
    steps = []
    for clause in clauses:
        words = clause.split()
        repeat = {'twice': 2, 'thrice': 3}.get(words[-1], 1) if words else 1
        if repeat != 1:
            words.pop()
        bare = ' '.join(words)
        if not LEAF.fullmatch(bare):
            raise ValueError(f'Invalid SCAN clause: {clause}')
        steps.append({'command': bare, 'action': words[0],
                      'direction': words[-1] if words[-1] in ('left', 'right') else 'none',
                      'modifier': next((w for w in words if w in ('around', 'opposite')), 'none'),
                      'repeat': repeat})
    return {'steps': steps}


def leaf_gold(leaf: str) -> list[str]:
    words = leaf.split()
    if len(words) == 1 and words[0] in ('walk', 'run', 'jump', 'look'):
        return ['I_' + words[0].upper()]
    if len(words) == 2 and words[0] in ('walk', 'run', 'jump', 'look', 'turn') and words[1] in ('left', 'right'):
        return ['I_TURN_' + words[1].upper()] + ([] if words[0] == 'turn' else ['I_' + words[0].upper()])
    raise ValueError(f'Invalid atomic leaf: {leaf}')


def expand(step: dict) -> list[str]:
    action, direction, modifier = (step[k] for k in ('action', 'direction', 'modifier'))
    leaf = action + ('' if direction == 'none' else ' ' + direction)
    if modifier == 'opposite':
        leaves = ['turn ' + direction, leaf]
    elif modifier == 'around':
        leaves = [leaf] * 4
    elif modifier == 'none':
        leaves = [leaf]
    else:
        raise ValueError('Unknown modifier')
    return leaves * step['repeat']


def interpret(command: str) -> list[str]:
    # Direct clause semantics are deliberately independent of expand/leaf_gold:
    # the exhaustive test checks both implementations against official targets.
    result = []
    for step in parse(command)['steps']:
        action, direction, modifier = (step[k] for k in ('action', 'direction', 'modifier'))
        primitive = [] if action == 'turn' else ['I_' + action.upper()]
        turn = [] if direction == 'none' else ['I_TURN_' + direction.upper()]
        if modifier == 'around':
            clause = (turn + primitive) * 4
        else:
            clause = turn * (2 if modifier == 'opposite' else 1) + primitive
        result.extend(clause * step['repeat'])
    return result


def meta(command: str, target: list[str]) -> dict:
    """Depth is 1 (atomic clause) plus each nested modifier and repetition,
    plus one conjunction node, taking the maximum path across clauses.
    Direction alone does not add depth. Counts count clauses, not repetitions.
    """
    steps = parse(command)['steps']
    return {'depth': max(1 + (s['modifier'] != 'none') + (s['repeat'] > 1) for s in steps) + (len(steps) > 1),
            'n_clauses': len(steps), 'target_len': len(target), 'command_len': len(command.split()),
            'n_around': sum(s['modifier'] == 'around' for s in steps),
            'n_opposite': sum(s['modifier'] == 'opposite' for s in steps),
            'n_repeat': sum(s['repeat'] > 1 for s in steps),
            'conj': 'after' if ' after ' in command else 'and' if ' and ' in command else 'none'}
