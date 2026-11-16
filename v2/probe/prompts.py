"""Use the harness's versioned S0 builder without duplicating prompt text."""
import json
from pathlib import Path


def s0_messages(item):
    try:
        from v2.harness import prompt_parts
    except ImportError as exc:
        # TODO(WS1): expose the S0 prompt builder if the harness is unavailable.
        raise NotImplementedError('TODO(WS1): S0 prompt builder unavailable') from exc
    path = Path(__file__).resolve().parents[1] / 'domains/prompts' / (item['domain'] + '_v0.json')
    if not path.exists():
        raise NotImplementedError('TODO(WS1): versioned domain prompts unavailable')
    prompts = json.loads(path.read_text(encoding='utf-8'))
    return prompt_parts(prompts, 'S0', 'direct', input=item['input'])
