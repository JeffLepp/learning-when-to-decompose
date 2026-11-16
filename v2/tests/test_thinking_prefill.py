import unittest

from v2.harness import token_parts


def entry(token):
    return {'token': token, 'logprob': -0.1, 'bytes': list(token.encode())}


class PrefilledThinkTest(unittest.TestCase):
    """qwen3:4b emits no opening <think> token (seen in pilot-s2-* calls.jsonl)."""

    def test_no_open_token_counts_thinking_until_close(self):
        response = {'logprobs': [entry(t) for t in ('We', ' think', '</think>', '{"a":1}')], 'eval_count': 5,
                    'thinking': 'We think', 'response': '{"a":1}', 'done_reason': 'stop'}
        content, n_thinking = token_parts(response, True)
        self.assertEqual((n_thinking, [e['token'] for e in content]), (3, ['{"a":1}']))

    def test_truncated_inside_prefilled_thought(self):
        response = {'logprobs': [entry('We'), entry(' think')], 'eval_count': 2,
                    'thinking': 'We think', 'response': '', 'done_reason': 'length'}
        self.assertEqual(token_parts(response, True), ([], 2))


if __name__ == '__main__':
    unittest.main()
