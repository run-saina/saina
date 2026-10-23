"""POST /v1/decisions: dialect detection, the OpenAI Decisions translation, and golden System One fixtures.

Every answer here comes from a deterministic fake model whose probabilities are a hash of the exact prompt it
reads, so equal probabilities across dialects mean equal model inputs. These are contract tests with mocked
inference, not a measurement of model quality. Regenerate the golden files with SAINA_REGEN_FIXTURES=1.
"""
import hashlib
import json
import os
import unittest
from pathlib import Path

from saina.decisions import DialectError, detect_dialect, native_error_path, respond, translate

try:
    from fastapi.testclient import TestClient
    from saina.server import create_app
    HAVE_FASTAPI = True
except ImportError:  # fastapi or httpx not installed
    HAVE_FASTAPI = False

FIXTURES = Path(__file__).parent / 'fixtures' / 'decisions'
AUTH = {'Authorization': 'Bearer k'}


def hashed(context, question, choices):
    """A probability vector that depends on every character of the prompt and on nothing else."""
    weights = [1 + hashlib.sha256(f'{context}\0{question}\0{c}'.encode()).digest()[0] for c in choices]
    return [w / sum(weights) for w in weights]


class HashScorer:
    """A shared-pass model like Helm 2: one input for all questions, prompt-determined probabilities."""
    shared_trunk_pass = True
    max_length = 262144
    supported_modes = ['single_label', 'multi_label']
    model_id = 'saina-helm-2-0.8b'
    inputs = ('text',)

    def __init__(self):
        self.calls = []

    def count_text(self, text):
        return len(text.split())

    def media_tokens(self, media):
        return 10 * len(media.images) if media else 0

    def encode_shared(self, context, questions, media=None):
        self.pending = (context, questions, media)
        return list(range(40)) + [0] * self.media_tokens(media)

    def forward_shared(self, ids, counts, modes, media=None):
        context, questions, media = self.pending
        self.calls.append({'context': context, 'questions': [(q, list(c), m) for q, c, m in questions],
                           'media': media})
        out = []
        for (question, choices, mode), n in zip(questions, counts):
            p = hashed(context, question, choices)
            out.append([min(1., 2 * v) for v in p] if mode == 'multi_label' else p)
        return out


class OmniHashScorer(HashScorer):
    model_id = 'saina-helm-2-omni-0.8b'
    inputs = ('text', 'image', 'audio')


def png(color=(200, 10, 10), size=(40, 30)):
    import base64
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.new('RGB', size, color).save(buffer, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


try:
    import numpy, PIL, scipy, soundfile  # noqa: E401,F401
    HAVE_MEDIA = True
except ImportError:
    HAVE_MEDIA = False


# -- the same three questions in each dialect ----------------------------------------------------------------

STATE = 'I was charged twice for my order and need the extra payment back today.'
NATIVE = {'model': 'saina-helm', 'state': STATE, 'questions': {
    'billing': {'type': 'yes_no', 'question': 'Is this about billing?'},
    'team': {'type': 'single_choice', 'question': 'Which team should handle it?',
             'options': {'billing': 'Payments, invoices and refunds', 'technical': 'Problems using the product',
                         'other': None}},
    'urgency': {'type': 'rating', 'question': 'How urgent is it?',
                'levels': ['Can wait', 'This week', 'Today']}}}
SYSTEMONE = {'model': 'saina-helm', 'state': STATE, 'questions': {
    'billing': {'type': 'noul', 'instructions': 'Is this about billing?'},
    'team': {'type': 'choice', 'instructions': 'Which team should handle it?',
             'criteria': {'billing': 'Payments, invoices and refunds', 'technical': 'Problems using the product',
                          'other': None}},
    'urgency': {'type': 'score', 'instructions': 'How urgent is it?', 'criteria': ['Can wait', 'This week', 'Today']}}}
OPENAI = {'model': 'saina-helm', 'input': STATE, 'questions': [
    {'type': 'predicate', 'name': 'billing', 'instructions': 'Is this about billing?'},
    {'type': 'choice', 'name': 'team', 'instructions': 'Which team should handle it?', 'choices': [
        {'value': 'billing', 'description': 'Payments, invoices and refunds'},
        {'value': 'technical', 'description': 'Problems using the product'}, {'value': 'other'}]},
    {'type': 'score', 'name': 'urgency', 'instructions': 'How urgent is it?',
     'levels': [{'label': 'Can wait'}, {'label': 'This week'}, {'label': 'Today'}]}]}


def openai_question(q, **changes):
    return {**OPENAI, 'questions': [{**q, **changes}]}


class DetectDialectTests(unittest.TestCase):
    def test_body_shape_decides(self):
        self.assertEqual(detect_dialect(OPENAI), 'openai')
        self.assertEqual(detect_dialect({'questions': []}), 'openai')  # an array is OpenAI even without input
        self.assertEqual(detect_dialect(SYSTEMONE), 'systemone')
        self.assertEqual(detect_dialect(OPENAI, 'OpenAI'), 'openai')
        self.assertEqual(detect_dialect(SYSTEMONE, 'systemone'), 'systemone')

    def test_ambiguity_and_absence_are_errors(self):
        cases = [({**OPENAI, 'state': 's'}, 'not both'),
                 ({'state': 's', 'questions': OPENAI['questions']}, '`state` is not an OpenAI Decisions field'),
                 ({**OPENAI, 'assets': {}}, '`assets` is not'),
                 ({**OPENAI, 'images': []}, '`images` is not'),
                 (openai_question(OPENAI['questions'][1], criteria={'a': None}), r'questions\[0\]\.criteria'),
                 ({'state': 's', 'questions': 'x'}, 'questions` as an object'),
                 ({'model': 'm'}, 'Send `input`'),
                 ([], 'JSON object')]
        for body, message in cases:
            with self.assertRaisesRegex(DialectError, message):
                detect_dialect(body)

    def test_header_never_contradicts_the_body(self):
        with self.assertRaisesRegex(DialectError, 'says systemone but the body is a openai'):
            detect_dialect(OPENAI, 'systemone')
        with self.assertRaisesRegex(DialectError, 'openai or systemone'):
            detect_dialect(OPENAI, 'jev')


class TranslateTests(unittest.TestCase):
    def test_three_types_translate_to_the_native_contract(self):
        t = translate(OPENAI)
        self.assertEqual(t.native, {'model': 'saina-helm', 'state': STATE, 'questions': {
            '0': NATIVE['questions']['billing'], '1': NATIVE['questions']['team'], '2': NATIVE['questions']['urgency']}})
        self.assertEqual((t.names, t.kinds), (['billing', 'team', 'urgency'], ['predicate', 'choice', 'score']))

    def test_messages_and_tool_items_become_an_ordered_state(self):
        body = {**OPENAI, 'input': [
            {'role': 'developer', 'content': 'Route support tickets.'},
            {'type': 'message', 'role': 'user', 'content': [
                {'type': 'input_text', 'text': 'Evaluate the ticket.'},
                {'type': 'input_json', 'data': {'order': {'charges': [49, 49]}}}]},
            {'role': 'assistant', 'content': [{'type': 'refusal', 'refusal': 'No.'}], 'phase': 'final_answer'},
            {'type': 'function_call', 'call_id': 'c1', 'name': 'lookup', 'arguments': '{"id":1}', 'status': 'completed'},
            {'type': 'function_call_output', 'call_id': 'c1', 'output': [{'type': 'input_text', 'text': 'paid twice'}]}]}
        self.assertEqual(translate(body).native['state'], [
            {'role': 'developer', 'content': 'Route support tickets.'},
            {'role': 'user', 'content': ['Evaluate the ticket.', {'order': {'charges': [49, 49]}}]},
            {'role': 'assistant', 'content': [{'refusal': 'No.'}]},
            {'type': 'function_call', 'call_id': 'c1', 'name': 'lookup', 'arguments': '{"id":1}'},
            {'type': 'function_call_output', 'call_id': 'c1', 'output': 'paid twice'}])

    def test_unnamed_questions_and_boolean_choices(self):
        body = {**OPENAI, 'questions': [{'type': 'choice', 'instructions': 'Refund?', 'choices': [
            {'value': True, 'description': 'Refund it'}, {'value': False}]}]}
        t = translate(body)
        self.assertEqual(t.names, [None])
        self.assertEqual(t.native['questions']['0']['options'], {'true': 'Refund it', 'false': None})
        self.assertEqual(t.values, [[True, False]])

    def test_extensions_and_ignored_fields(self):
        body = {**OPENAI, 'mode': 'decision', 'threshold': 0.9, 'specialist': 'mechanical',
                'prompt_cache_key': 'k', 'prompt_cache_options': {'ttl': '30m'}, 'safety_identifier': 'u1',
                'metadata': {'a': 'b'}}
        native = translate(body).native
        self.assertEqual({k: native[k] for k in ('mode', 'threshold', 'specialist')},
                         {'mode': 'decision', 'threshold': 0.9, 'specialist': 'mechanical'})
        self.assertFalse({'prompt_cache_key', 'prompt_cache_options', 'safety_identifier', 'metadata'} & set(native))

    def test_refusals_name_the_field(self):
        choice, score, predicate = OPENAI['questions'][1], OPENAI['questions'][2], OPENAI['questions'][0]
        cases = [
            ({**OPENAI, 'temperature': 0}, 'temperature is not supported'),
            ({**OPENAI, 'seed': 1}, 'seed is not supported'),
            ({**OPENAI, 'tools': []}, r'Request body\.tools is not supported'),
            ({k: v for k, v in OPENAI.items() if k != 'model'}, 'model is required'),
            ({**OPENAI, 'questions': []}, 'non-empty array'),
            ({**OPENAI, 'questions': [predicate, predicate]}, r'questions\[1\]\.name repeats questions\[0\]\.name'),
            (openai_question(predicate, type='box'), r'questions\[0\]\.type must be one of: predicate, choice, score'),
            (openai_question(predicate, type='multi_choice'), 'predicate, choice, score'),
            (openai_question(predicate, instructions=''), r'questions\[0\]\.instructions must not be empty'),
            (openai_question(predicate, instructions=[]), 'non-empty array'),
            (openai_question(predicate, instructions=[{'type': 'input_image', 'image_url': png() if HAVE_MEDIA else 'x'}]),
             r'questions\[0\]\.instructions\[0\] is an image'),
            (openai_question(choice, choices=[{'value': 'a'}, {'value': 'a'}]), r'choices\[1\]\.value repeats'),
            (openai_question(choice, choices=[{'value': 'true'}, {'value': True}]), 'a string and a boolean'),
            (openai_question(choice, choices=[{'value': 'a', 'weight': 1}, {'value': 'b'}]), r'choices\[0\]\.weight'),
            (openai_question(score, levels=['low', 'high']), r'levels\[0\] must be an object'),
            (openai_question(score, min_margin=0.1), r'questions\[0\]\.min_margin is not supported'),
            ({**OPENAI, 'input': [{'role': 'tool', 'content': 'x'}]}, r'input\[0\]\.role'),
            ({**OPENAI, 'input': [{'role': 'system', 'content': [{'type': 'input_json', 'data': {}}]}]},
             r'input\[0\]\.content\[0\]\.type must be one of: input_text'),
            ({**OPENAI, 'input': [{'type': 'function_call_output', 'call_id': 'x', 'output': 'y'}]},
             'does not match an earlier function_call'),
            ({**OPENAI, 'input': [{'type': 'item_reference', 'id': 'x'}]}, r'input\[0\]\.type'),
            ({**OPENAI, 'input': [{'role': 'user', 'content': [
                {'type': 'input_image', 'image_url': 'https://example.com/a.png'}]}]}, 'image URLs are not fetched'),
            ({**OPENAI, 'input': [{'role': 'user', 'content': [
                {'type': 'input_image', 'image_url': 'data:image/png;base64,AAAA', 'detail': 'high'}]}]},
             r'detail must be auto'),
            ({**OPENAI, 'input': [{'role': 'user', 'content': [{'type': 'input_file', 'file_id': 'f'}]}]},
             r'content\[0\]\.type'),
        ]
        for body, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(DialectError, message):
                translate(body)

    def test_native_error_paths_use_the_callers_words(self):
        self.assertEqual(native_error_path(('questions', '3', 'single_choice', 'options')), 'questions[3].choices')
        self.assertEqual(native_error_path(('questions', '0', 'yes_no', 'threshold')), 'questions[0].threshold')
        self.assertEqual(native_error_path(('questions',)), 'questions')
        self.assertEqual(native_error_path(()), 'request')

    def test_respond_keeps_request_order_and_typed_values(self):
        t = translate({**OPENAI, 'questions': [{'type': 'choice', 'instructions': 'Refund?', 'choices': [
            {'value': True}, {'value': False}]}]})
        native = {'model': 'saina-helm-2-0.8b', 'usage': {'input_tokens': 7, 'output_tokens': 0}, 'answers': {
            '0': {'type': 'single_choice', 'selection': 'false', 'probabilities': {'true': 0.25, 'false': 0.75},
                  'confidence': 0.5}}}
        self.assertEqual(respond(t, native), {'model': 'saina-helm-2-0.8b', 'answers': [
            {'type': 'choice', 'name': None, 'choice': False, 'confidence': 0.5,
             'probabilities': [{'value': True, 'probability': 0.25}, {'value': False, 'probability': 0.75}]}],
            'usage': {'input_tokens': 7, 'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0},
                      'output_tokens': 0, 'output_tokens_details': {'reasoning_tokens': 0}, 'total_tokens': 7}})


@unittest.skipUnless(HAVE_FASTAPI, 'fastapi and httpx required')
class RouteTests(unittest.TestCase):
    def setUp(self):
        self.scorer = HashScorer()
        self.client = TestClient(create_app(scorer=self.scorer, api_key='k'))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def post(self, body, route='/v1/decisions', **headers):
        return self.client.post(route, json=body, headers={**AUTH, **headers})

    def test_same_request_three_ways_gets_the_same_answers_and_bill(self):
        native = self.post(NATIVE, '/v1/ask').json()
        jev = self.post(SYSTEMONE, '/v1/systemone').json()
        jev_here = self.post(SYSTEMONE).json()
        oai = self.post(OPENAI)
        self.assertEqual(oai.status_code, 200, oai.text)
        oai = oai.json()
        # One model input, identical across dialects.
        self.assertEqual(len({json.dumps(c, default=str) for c in self.scorer.calls}), 1)
        self.assertEqual(jev, jev_here)
        bill = native['usage']['input_tokens']
        self.assertEqual((jev['usage']['input_tokens'], oai['usage']['input_tokens']), (bill, bill))
        self.assertEqual(oai['usage']['output_tokens'], 0)
        billing, team, urgency = oai['answers']
        self.assertEqual(billing, {'type': 'predicate', 'name': 'billing', 'probability': native['answers']['billing']['yes']})
        self.assertEqual(jev['answers']['billing']['noul'], billing['probability'])
        n = native['answers']['team']
        self.assertEqual((team['choice'], team['confidence']), (n['selection'], n['confidence']))
        self.assertEqual({p['value']: p['probability'] for p in team['probabilities']}, n['probabilities'])
        self.assertEqual(jev['answers']['team']['probabilities'], n['probabilities'])
        n = native['answers']['urgency']
        self.assertEqual((urgency['score'], urgency['confidence']), (n['expected_level'], n['confidence']))
        self.assertEqual([(p['value'], p['label'], p['probability']) for p in urgency['probabilities']],
                         [(int(k), lvl, v) for (k, v), lvl in zip(n['probabilities'].items(), n['levels'])])

    def test_alias_route_and_dialect_header(self):
        a, b = self.post(OPENAI), self.post(OPENAI, '/api/v1/decisions')
        self.assertEqual((a.status_code, b.status_code), (200, 200))
        self.assertEqual(a.json(), b.json())
        self.assertEqual(a.headers['x-saina-dialect'], 'openai')
        self.assertEqual(self.post(SYSTEMONE, '/api/v1/decisions').headers['x-saina-dialect'], 'systemone')
        r = self.post(OPENAI, **{'X-Saina-Dialect': 'systemone'})
        self.assertEqual((r.status_code, r.json()['error']['code']), (400, 'invalid_request'))
        self.assertEqual(self.post(OPENAI, **{'X-Saina-Dialect': 'openai'}).status_code, 200)

    def test_models_are_exact_on_this_route_only(self):
        for body in ({**OPENAI, 'model': 'gpt-6-luna'}, {**SYSTEMONE, 'model': 'jev-1.14'}, {**OPENAI, 'model': 'saina-helm-0.8b'}):
            r = self.post(body)
            self.assertEqual((r.status_code, r.json()['error']['code']), (404, 'not_found'), body['model'])
            self.assertIn('saina-helm-2-0.8b', r.json()['error']['message'])
        self.assertEqual(self.scorer.calls, [])
        # Unchanged: /v1/systemone still sends unknown System One labels to the default model.
        self.assertEqual(self.post({**SYSTEMONE, 'model': 'jev-1.14'}, '/v1/systemone').json()['model'], 'saina-helm-2-0.8b')
        self.assertEqual(self.post({**OPENAI, 'model': 'saina-helm-2-0.8b'}).json()['model'], 'saina-helm-2-0.8b')

    def test_invalid_requests_never_run(self):
        q = OPENAI['questions']
        bodies = [{**OPENAI, 'state': 's'}, {**OPENAI, 'questions': [q[0], q[0]]},
                  openai_question(q[0], type='box'), {**OPENAI, 'top_p': 1},
                  {**OPENAI, 'questions': q + [{'type': 'choice', 'instructions': 'x', 'choices': [{'value': 'only'}]}]},
                  {**OPENAI, 'threshold': 0.9},  # a policy needs decision mode
                  {**OPENAI, 'questions': [q[0]] * 0 + [dict(q[0], name=str(i)) for i in range(257)]}]
        for body in bodies:
            r = self.post(body)
            self.assertEqual((r.status_code, r.json()['error']['code']), (400, 'invalid_request'), r.text)
        r = self.post({**OPENAI, 'questions': q + [{'type': 'choice', 'instructions': 'x', 'choices': [{'value': 'only'}]}]})
        self.assertRegex(r.json()['error']['message'], r'^questions\[3\]\.choices: ')
        r = self.post({**SYSTEMONE, 'questions': {'b': {'type': 'box', 'instructions': 'Find it'}}})
        self.assertEqual((r.status_code, r.json()['error']['code']), (422, 'invalid_request'))
        self.assertEqual(self.client.post('/v1/decisions', content=b'{"input": NaN}', headers=AUTH).status_code, 400)
        self.assertEqual(self.scorer.calls, [])

    def test_native_limits_not_gateway_limits(self):
        many = [{'value': f'option {i}'} for i in range(120)]
        r = self.post(openai_question(OPENAI['questions'][1], choices=many))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(r.json()['answers'][0]['probabilities']), 120)
        self.assertAlmostEqual(sum(p['probability'] for p in r.json()['answers'][0]['probabilities']), 1)
        r = self.post(openai_question(OPENAI['questions'][1], choices=[{'value': f'{i}'} for i in range(256)]))
        self.assertEqual(r.status_code, 400)
        ten = [{'label': f'level {i}'} for i in range(10)]
        r = self.post(openai_question(OPENAI['questions'][2], levels=ten))
        self.assertEqual(r.status_code, 200)
        score = r.json()['answers'][0]
        self.assertEqual([p['value'] for p in score['probabilities']], list(range(10)))
        self.assertNotEqual(score['score'], round(score['score']))  # lands between two levels
        self.assertEqual(self.post(openai_question(OPENAI['questions'][2], levels=ten + [{'label': 'x'}])).status_code, 400)

    def test_policy_refusals_are_explicit_and_answers_otherwise(self):
        r = self.post(OPENAI).json()
        self.assertNotIn('refusal', [a['type'] for a in r['answers']])  # no policy, no refusal, however low
        r = self.post({**OPENAI, 'mode': 'decision', 'threshold': 0.99})
        self.assertEqual(r.status_code, 200)
        billing, team, urgency = r.json()['answers']
        self.assertEqual((billing['type'], billing['name'], billing['reason']), ('refusal', 'billing', 'below_threshold'))
        self.assertIn('probability', billing)
        self.assertEqual(len(team['probabilities']), 3)
        self.assertEqual(urgency['reason'], 'below_threshold')
        self.assertGreater(r.json()['usage']['input_tokens'], 0)  # a refusal is a completed, billed inference
        r = self.post({**OPENAI, 'mode': 'decision', 'threshold': 0.01, 'questions': [
            dict(OPENAI['questions'][0], threshold=0.99), OPENAI['questions'][1]]}).json()
        self.assertEqual([a['type'] for a in r['answers']], ['refusal', 'choice'])

    def test_text_model_refuses_images_before_running(self):
        body = {**OPENAI, 'input': [{'role': 'user', 'content': [
            {'type': 'input_text', 'text': 'Inspect the photo.'},
            {'type': 'input_image', 'image_url': 'data:image/png;base64,iVBORw0KGgo='}]}]}
        r = self.post(body)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()['error']['message'], 'input[0].content[1] is an image, and saina-helm-2-0.8b reads '
                         'text only; send images to a model that reads them (GET /v1/models)')
        self.assertEqual(self.scorer.calls, [])


@unittest.skipUnless(HAVE_FASTAPI and HAVE_MEDIA, 'fastapi, httpx and saina[media] required')
class OmniImageTests(unittest.TestCase):
    def test_image_parts_reach_the_model_as_native_images_in_order(self):
        scorer = OmniHashScorer()
        red, blue = png((220, 0, 0)), png((0, 0, 220), (30, 40))
        oai = {'model': 'saina-helm-2-omni-0.8b', 'input': [
            {'role': 'user', 'content': [{'type': 'input_text', 'text': 'Inspect the product.'},
                                         {'type': 'input_image', 'image_url': red},
                                         {'type': 'input_image', 'image_url': blue, 'detail': 'auto'}]}],
            'questions': [{'type': 'predicate', 'name': 'red', 'instructions': 'Is the first image red?'}]}
        native = {'model': 'saina-helm-2-omni-0.8b', 'images': [red, blue],
                  'state': [{'role': 'user', 'content': 'Inspect the product.'}],
                  'questions': {'red': {'type': 'yes_no', 'question': 'Is the first image red?'}}}
        with TestClient(create_app(scorer=scorer, api_key='k')) as c:
            a = c.post('/v1/decisions', json=oai, headers=AUTH)
            b = c.post('/v1/ask', json=native, headers=AUTH)
            self.assertEqual((a.status_code, b.status_code), (200, 200), a.text)
            first, second = scorer.calls
            self.assertEqual([im.size for im in first['media'].images], [(40, 30), (30, 40)])
            self.assertEqual(first['media'].images[0].getpixel((0, 0)), (220, 0, 0))
            self.assertEqual((first['context'], first['questions']), (second['context'], second['questions']))
            self.assertEqual(a.json()['answers'][0]['probability'], b.json()['answers']['red']['yes'])
            self.assertEqual(a.json()['usage']['input_tokens'], b.json()['usage']['input_tokens'])
            self.assertEqual(a.headers['x-saina-media-tokens'], '20')
            bad = {**oai, 'input': [{'role': 'user', 'content': [
                {'type': 'input_text', 'text': 'x'}, {'type': 'input_image', 'image_url': 'data:image/bmp;base64,Qk0='}]}]}
            r = c.post('/v1/decisions', json=bad, headers=AUTH)
            self.assertEqual(r.status_code, 422)
            self.assertRegex(r.json()['error']['message'], r'^input\[0\]\.content\[1\] ')
            r = c.post('/v1/decisions', json={**oai, 'input': [{'role': 'user', 'content': [
                {'type': 'input_image', 'image_url': red}] * 9}]}, headers=AUTH)
            self.assertEqual((r.status_code, r.json()['error']['message']), (422, 'At most 8 images per request'))
            self.assertEqual(len(scorer.calls), 2)


# -- Phase 0: golden System One fixtures ------------------------------------------------------------------------

GOLDEN = {
    'noul': {'model': 'saina-helm', 'state': STATE, 'questions': {
        'billing': {'type': 'noul', 'instructions': 'Is this about billing?'}}},
    'noul_criteria': {'model': 'saina-helm', 'state': {'subject': 'Refund', 'body': STATE}, 'questions': {
        'spam': {'type': 'noul', 'instructions': 'Is this spam?',
                 'criteria': {'true': 'Unsolicited advertising', 'false': 'A real customer'}}}},
    'choice': {'model': 'saina-helm', 'state': STATE, 'questions': {'team': SYSTEMONE['questions']['team']}},
    'choice_two_options': {'model': 'saina-helm', 'state': STATE, 'questions': {
        'refund': {'type': 'choice', 'instructions': 'Refund?', 'criteria': {'yes': None, 'no': None}}}},
    'choice_120_options': {'model': 'saina-helm', 'state': STATE, 'questions': {
        'code': {'type': 'choice', 'instructions': 'Which code?', 'criteria': {f'c{i}': None for i in range(120)}}}},
    'score_two_levels': {'model': 'saina-helm', 'state': STATE, 'questions': {
        'urgent': {'type': 'score', 'instructions': 'Urgency?', 'criteria': ['low', 'high']}}},
    'score_ten_levels': {'model': 'saina-helm', 'state': STATE, 'questions': {
        'urgent': {'type': 'score', 'instructions': 'Urgency?', 'criteria': [f'level {i}' for i in range(10)]}}},
    'mixed': SYSTEMONE,
}


@unittest.skipUnless(HAVE_FASTAPI, 'fastapi and httpx required')
class SystemOneGoldenTests(unittest.TestCase):
    """Pins the shipped /v1/systemone adapter: the same responses on /v1/systemone and on /v1/decisions."""

    def test_golden_responses(self):
        regen = os.environ.get('SAINA_REGEN_FIXTURES') == '1'
        with TestClient(create_app(scorer=HashScorer(), api_key='k')) as c:
            for name, body in GOLDEN.items():
                with self.subTest(name):
                    r = c.post('/v1/systemone', json=body, headers=AUTH)
                    self.assertEqual(r.status_code, 200, r.text)
                    path = FIXTURES / f'systemone_{name}.json'
                    record = {'request': body, 'response': r.json(),
                              'headers': {k: r.headers[k] for k in ('x-saina-billable-tokens', 'x-saina-input-tokens')}}
                    if regen:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + '\n')
                    self.assertEqual(record, json.loads(path.read_text()))
                    same = c.post('/v1/decisions', json=body, headers=AUTH)
                    self.assertEqual(same.json(), record['response'])
                    self.assertEqual(same.headers['x-saina-billable-tokens'], record['headers']['x-saina-billable-tokens'])

    def test_golden_openai_responses(self):
        regen = os.environ.get('SAINA_REGEN_FIXTURES') == '1'
        bodies = {'mixed': OPENAI, 'policy_refusal': {**OPENAI, 'mode': 'decision', 'threshold': 0.99},
                  'unnamed_boolean_choice': {**OPENAI, 'questions': [{'type': 'choice', 'instructions': 'Refund?',
                                                                      'choices': [{'value': True}, {'value': False}]}]}}
        with TestClient(create_app(scorer=HashScorer(), api_key='k')) as c:
            for name, body in bodies.items():
                with self.subTest(name):
                    r = c.post('/v1/decisions', json=body, headers=AUTH)
                    self.assertEqual(r.status_code, 200, r.text)
                    path = FIXTURES / f'openai_{name}.json'
                    record = {'request': body, 'response': r.json()}
                    if regen:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + '\n')
                    self.assertEqual(record, json.loads(path.read_text()))


if __name__ == '__main__':
    unittest.main()


def serve_in_thread(app):
    """A real HTTP server on a free local port, for clients that speak HTTP (the Saina client, official SDKs)."""
    import socket
    import threading
    import time
    import uvicorn
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level='warning', lifespan='on'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(.01)
    return server, thread, f'http://127.0.0.1:{sock.getsockname()[1]}'


@unittest.skipUnless(HAVE_FASTAPI, 'fastapi and httpx required')
class ClientEndToEndTests(unittest.TestCase):
    def test_python_client_decisions_helper(self):
        from saina import Saina, SainaError
        server, thread, url = serve_in_thread(create_app(scorer=HashScorer(), api_key='k'))
        try:
            client = Saina(url, api_key='k')
            result = client.decisions(model='saina-helm', input=STATE, questions=OPENAI['questions'])
            self.assertEqual([a['name'] for a in result['answers']], ['billing', 'team', 'urgency'])
            native = client.ask(model='saina-helm', state=STATE, questions=NATIVE['questions'])
            self.assertEqual(result['answers'][0]['probability'], native['answers']['billing']['yes'])
            refused = client.decisions(model='saina-helm', input=STATE, questions=OPENAI['questions'][:1],
                                       mode='decision', threshold=0.99)
            self.assertEqual(refused['answers'][0]['type'], 'refusal')
            with self.assertRaises(SainaError) as e:
                client.decisions(model='gpt-6-luna', input=STATE, questions=OPENAI['questions'])
            self.assertEqual((e.exception.status, e.exception.code), (404, 'not_found'))
        finally:
            server.should_exit = True
            thread.join(5)


@unittest.skipUnless(HAVE_FASTAPI, 'fastapi and httpx required')
class CompressedBodyTests(unittest.TestCase):
    """The TypeSafe SDKs zstd-compress bodies of 1 KiB or more by default."""

    def post(self, raw, encoding, limit=None):
        import os as _os
        from unittest.mock import patch
        env = {'SAINA_MAX_DECODED_BODY_BYTES': str(limit)} if limit else {}
        with patch.dict(_os.environ, env):
            app = create_app(scorer=HashScorer(), api_key='k')
        with TestClient(app) as c:
            return c.post('/v1/systemone', content=raw,
                          headers={**AUTH, 'Content-Type': 'application/json', 'Content-Encoding': encoding})

    def test_gzip_and_zstd_bodies_answer_like_plain_ones(self):
        import gzip
        from saina.encoding import _zstd
        body = {**SYSTEMONE, 'state': STATE * 30}
        raw = json.dumps(body).encode()
        with TestClient(create_app(scorer=HashScorer(), api_key='k')) as c:
            plain = c.post('/v1/systemone', json=body, headers=AUTH).json()
        self.assertEqual(self.post(gzip.compress(raw), 'gzip').json(), plain)
        self.assertEqual(self.post(gzip.compress(raw[:500]) + gzip.compress(raw[500:]), 'x-gzip').json(), plain)
        if _zstd is not None:
            self.assertEqual(self.post(_zstd.compress(raw), 'zstd').json(), plain)
        self.assertEqual(self.post(raw, 'identity').json(), plain)

    def test_bad_encodings_are_refused(self):
        import gzip
        raw = json.dumps(SYSTEMONE).encode()
        for data, encoding, status in ((raw, 'br', 415), (gzip.compress(gzip.compress(raw)), 'gzip, gzip', 415),
                                       (b'not gzip', 'gzip', 400), (gzip.compress(raw)[:-9], 'gzip', 400)):
            r = self.post(data, encoding)
            self.assertEqual((r.status_code, r.json()['error']['code']), (status, 'invalid_request'), encoding)
        r = self.post(gzip.compress(b' ' * 5000 + raw), 'gzip', limit=4096)  # a small body that expands past the limit
        self.assertEqual(r.status_code, 413)
