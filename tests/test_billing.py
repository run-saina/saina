"""Billable input: the text a customer sends, counted once, independent of prompt formatting."""
import unittest
from saina.contract import AskRequest
from saina.jev import SystemOneRequest
from saina.prepare import as_sent, bill, billable_ask, billable_systemone


class Words:
    """One token per whitespace-separated word, so expected counts can be read off the request."""
    def count_text(self, text):
        return len(text.split())


def ask(state, questions):
    return AskRequest.model_validate({'model': 'helm-0.8b', 'state': state, 'questions': questions})


class BillableTextTests(unittest.TestCase):
    def test_strings_as_sent_and_structures_as_compact_json_in_order(self):
        self.assertEqual(as_sent('Charged  twice.'), 'Charged  twice.')
        self.assertEqual(as_sent({'b': 1, 'a': [1, 2], 'é': 'ü'}), '{"b":1,"a":[1,2],"é":"ü"}')
        self.assertEqual(billable_ask(ask({'z': 1, 'a': 2}, {'q': {'type': 'yes_no', 'question': 'Ok?'}})).context,
                         '{"z":1,"a":2}')

    def test_native_questions_bill_keys_descriptions_and_levels(self):
        b = billable_ask(ask('ctx', {
            'team': {'type': 'single_choice', 'question': 'Which team?',
                     'options': {'billing': 'Payments and refunds', 'other': None}},
            'tags': {'type': 'multi_choice', 'question': 'Tags?', 'options': {'a': None, 'b': {'hint': 'x'}}},
            'sev': {'type': 'rating', 'question': 'How urgent?', 'levels': ['low', 'high']},
            'yn': {'type': 'yes_no', 'question': 'Refund?', 'descriptions': {'yes': 'Refund it', 'no': 'Keep it'}},
            'bare': {'type': 'yes_no', 'question': 'Spam?'}}))
        self.assertEqual(b.questions, (
            ('Which team?', 'billing', 'Payments and refunds', 'other'),
            ('Tags?', 'a', 'b', '{"hint":"x"}'),
            ('How urgent?', 'low', 'high'),
            ('Refund?', 'Refund it', 'Keep it'),
            ('Spam?',)))  # defaults we add ("The answer is yes") are formatting, not billed

    def test_system_one_criteria(self):
        r = SystemOneRequest.model_validate({'model': 'm', 'state': 's', 'questions': {
            'u': {'type': 'noul', 'instructions': 'Urgent?'},
            't': {'type': 'choice', 'instructions': 'Team?', 'criteria': {'billing': 'Money', 'tech': None}},
            's': {'type': 'score', 'instructions': 'Severity?', 'criteria': ['low', 'high']}}})
        self.assertEqual(billable_systemone(r).questions,
                         (('Urgent?',), ('Team?', 'billing', 'Money', 'tech'), ('Severity?', 'low', 'high')))


class BillTests(unittest.TestCase):
    def test_context_is_billed_once_however_many_questions(self):
        q = {'type': 'yes_no', 'question': 'Is it urgent?'}
        one = bill(billable_ask(ask('a b c d e', {'a': q})), Words())
        eight = bill(billable_ask(ask('a b c d e', {k: q for k in 'abcdefgh'})), Words())
        self.assertEqual((one.context_tokens, one.total), (5, 8))
        self.assertEqual((eight.context_tokens, eight.question_tokens, eight.total), (5, (3,) * 8, 29))

    def test_each_part_counted_on_its_own(self):
        b = bill(billable_ask(ask('x', {'t': {'type': 'single_choice', 'question': 'Pick one',
                                              'options': {'left side': 'go left', 'right': None}}})), Words())
        self.assertEqual(b.question_tokens, (2 + 2 + 2 + 1,))


if __name__ == '__main__': unittest.main()


class SharedPassExecutionTests(unittest.TestCase):
    """Shared-pass models read every question from one model input, and are billed for that input."""

    class Shared(Words):
        shared_trunk_pass = True
        supported_modes = ('single_label', 'multi_label')

        def __init__(self):
            self.passes = []

        def encode(self, *args):
            raise AssertionError('shared-pass models are never encoded per question')

        def encode_shared(self, context, questions):
            return list(range(10 * len(questions) + len(context)))

        def forward_shared(self, ids, counts, modes):
            self.passes.append(len(ids))
            return [[1 / n] * n for n in counts]

    def test_one_pass_one_count_and_billed_for_the_model_input(self):
        from saina.prepare import MeteringMismatch, execute, prepare_ask
        request = ask('a long shared state', {
            'team': {'type': 'single_choice', 'question': 'Which team?', 'options': {'a': None, 'b': None}},
            'tags': {'type': 'multi_choice', 'question': 'Tags?', 'options': {'x': None, 'y': None, 'z': None}}})
        scorer = self.Shared()
        execution = execute(prepare_ask(request), scorer, billable_ask(request))
        self.assertEqual(scorer.passes, [39])
        self.assertEqual(execution.input_tokens, [39])
        self.assertEqual(execution.billable_tokens, 39)  # not the text-as-sent count of the request
        self.assertNotEqual(bill(billable_ask(request), scorer).total, 39)
        self.assertEqual(len(execution.probabilities), 2)
        with self.assertRaises(MeteringMismatch):
            execute(prepare_ask(request), scorer, billable_ask(request), expected_input_tokens=40)
