"""Exact counting against the pinned tokenizer, when it is available offline."""
import os
import unittest
from pathlib import Path

CHECKPOINT = os.environ.get('SAINA_TEST_CHECKPOINT')


@unittest.skipUnless(CHECKPOINT and Path(CHECKPOINT).is_dir(), 'set SAINA_TEST_CHECKPOINT to a staged checkpoint')
class TokenCounterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from saina.tokenize import TokenCounter
        cls.counter = TokenCounter.from_checkpoint(CHECKPOINT)

    def test_counter_matches_server_usage_and_needs_no_torch(self):
        import sys
        from saina.contract import AskRequest, ask
        request = AskRequest.model_validate({'model': 'helm-0.8b', 'state': {'ticket': 'Charged twice, please help'},
            'questions': {'team': {'type': 'single_choice', 'question': 'Which team?',
                                   'options': {'billing': 'Money', 'tech': None}},
                          'tags': {'type': 'multi_choice', 'question': 'Tags?', 'options': {'a': None, 'b': None}},
                          'sev': {'type': 'rating', 'question': 'Severity?', 'levels': ['low', 'mid', 'high']}}})
        counts = self.counter.count_ask(request)
        self.assertEqual(len(counts), 3)
        self.assertTrue(all(n > 50 for n in counts))

        class Scorer:
            supported_modes = self.counter.supported_modes
            encode = self.counter.encode
            def forward(_, ids, count, mode='single_label'):
                return [1 / count] * count
            count_text = self.counter.count_text
        billed = self.counter.bill_ask(request)
        self.assertEqual(ask(request, Scorer())['usage']['input_tokens'], billed.total)
        self.assertLess(billed.total, sum(counts))  # prompt formatting and repeated context are not billed
        self.assertNotIn('torch', sys.modules)

    def test_context_is_billed_once_but_read_per_question(self):
        from saina.contract import AskRequest
        one = {'type': 'yes_no', 'question': 'Urgent?'}
        single = AskRequest.model_validate({'model': 'helm-0.8b', 'state': 'x' * 400, 'questions': {'a': one}})
        double = AskRequest.model_validate({'model': 'helm-0.8b', 'state': 'x' * 400, 'questions': {'a': one, 'b': one}})
        # The current model reads the context once per question; billing counts it once.
        self.assertEqual(sum(self.counter.count_ask(double)), 2 * sum(self.counter.count_ask(single)))
        s, d = self.counter.bill_ask(single), self.counter.bill_ask(double)
        self.assertEqual(d.context_tokens, s.context_tokens)
        self.assertEqual(d.total, s.total + s.question_tokens[0])

    def test_shared_pass_counts_and_bills_the_one_model_input(self):
        from saina.contract import AskRequest
        from saina.prompt import SHARED_PROMPT
        from saina.tokenize import TokenCounter
        shared = TokenCounter(self.counter.tokenizer, self.counter.codes, SHARED_PROMPT,
                              ('single_label', 'multi_label'), max_length=262144)
        request = AskRequest.model_validate({'model': 'saina-helm', 'state': 'Parcel arrived crushed.',
            'questions': {f'q{i}': {'type': 'yes_no', 'question': f'Question {i}?'} for i in range(5)}})
        counts = shared.count_ask(request)
        self.assertEqual(len(counts), 1)
        self.assertEqual(shared.bill_ask(request).total, counts[0])
        self.assertLess(counts[0], sum(self.counter.count_ask(request)))  # the state is read once
        self.assertTrue(shared.manifest()['shared_trunk_pass'])

    def test_counter_names_its_release(self):
        # Checkpoints from before releases named themselves count for the original Helm.
        self.assertEqual(self.counter.model_id, 'saina-helm-0.8b')
        self.assertEqual(self.counter.manifest()['model'], 'saina-helm-0.8b')

    def test_manifest_pins_tokenizer(self):
        m = self.counter.manifest()
        self.assertRegex(m['tokenizer_sha256'], '^[0-9a-f]{64}$')
        self.assertEqual(m['prompt_format'], 'AgenticScorer.encode_prompt')


if __name__ == '__main__': unittest.main()
