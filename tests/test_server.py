import os
import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    from saina.server import create_app
    HAVE_FASTAPI = True
except ImportError:  # fastapi or httpx not installed
    HAVE_FASTAPI = False

PREFLIGHT = {'Origin': 'https://saina.run', 'Access-Control-Request-Method': 'POST',
             'Access-Control-Request-Headers': 'authorization,content-type'}


@unittest.skipUnless(HAVE_FASTAPI, 'fastapi and httpx required')
class CorsTests(unittest.TestCase):
    def app(self, **env):
        with patch.dict(os.environ, env, clear=False):
            if 'SAINA_CORS_ORIGINS' not in env: os.environ.pop('SAINA_CORS_ORIGINS', None)
            return create_app(scorer=object(), api_key='k')

    def test_no_cors_origin_by_default(self):
        with TestClient(self.app()) as c:
            r = c.options('/v1/ask', headers=PREFLIGHT)
            self.assertNotIn('access-control-allow-origin', r.headers)

    def test_listed_origin_is_allowed(self):
        with TestClient(self.app(SAINA_CORS_ORIGINS='https://saina.run')) as c:
            r = c.options('/v1/ask', headers=PREFLIGHT)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.headers['access-control-allow-origin'], 'https://saina.run')
            self.assertIn('POST', r.headers['access-control-allow-methods'])
            self.assertIn('authorization', r.headers['access-control-allow-headers'].lower())
            self.assertEqual(r.headers['access-control-max-age'], '600')
            self.assertNotIn('access-control-allow-credentials', r.headers)
            h = c.get('/healthz', headers={'Origin': 'https://saina.run'})
            self.assertEqual(h.headers['access-control-allow-origin'], 'https://saina.run')

    def test_other_origin_gets_nothing(self):
        with TestClient(self.app(SAINA_CORS_ORIGINS='https://saina.run')) as c:
            r = c.options('/v1/ask', headers={**PREFLIGHT, 'Origin': 'https://evil.example'})
            self.assertNotIn('access-control-allow-origin', r.headers)

    def test_custom_list_and_disable(self):
        with TestClient(self.app(SAINA_CORS_ORIGINS='http://127.0.0.1:3200, https://saina.run')) as c:
            r = c.options('/v1/ask', headers={**PREFLIGHT, 'Origin': 'http://127.0.0.1:3200'})
            self.assertEqual(r.headers['access-control-allow-origin'], 'http://127.0.0.1:3200')
        with TestClient(self.app(SAINA_CORS_ORIGINS='')) as c:
            r = c.options('/v1/ask', headers=PREFLIGHT)
            self.assertNotIn('access-control-allow-origin', r.headers)


class FakeScorer:
    supported_modes = ['single_label', 'multi_label']

    def __init__(self):
        self.forwards = 0
        self.encodes = 0

    def encode(self, context, question, choices, mode='single_label'):
        self.encodes += 1
        if question == 'too long':
            from saina.errors import InputTooLong
            raise InputTooLong('too long')
        return list(range(7))

    def count_text(self, text):
        return len(text.split())  # one token per word: billable counts are easy to read in assertions

    def forward(self, ids, count, mode='single_label'):
        self.forwards += 1
        return [1 / count] * count if mode == 'multi_label' else [0.7] + [0.3 / (count - 1)] * (count - 1)


class SharedFakeScorer(FakeScorer):
    shared_trunk_pass = True
    max_length = 262144

    def __init__(self):
        super().__init__()
        self.calls = []

    def encode(self, *args, **kwargs):
        raise AssertionError('shared scorers must not be encoded per question')

    def forward(self, *args, **kwargs):
        raise AssertionError('shared scorers must not be called per question')

    def encode_shared(self, context, questions):
        self.encoded = [{'question': q, 'choices': list(c), 'mode': m} for q, c, m in questions]
        return list(range(40))  # one model input for every question

    def forward_shared(self, ids, counts, modes):
        self.calls.append(self.encoded)
        return [FakeScorer.forward(self, ids, n, m) for n, m in zip(counts, modes)]


@unittest.skipUnless(HAVE_FASTAPI, 'fastapi and httpx required')
class SharedPassRouteTests(unittest.TestCase):
    def test_one_call_for_all_questions_on_both_routes(self):
        scorer = SharedFakeScorer()
        questions = {f'q{i}': {'type': 'single_choice', 'question': f'Q{i}?', 'options': {'a': None, 'b': 'B'}}
                     for i in range(9)}
        questions['tags'] = {'type': 'multi_choice', 'question': 'Tags?', 'options': {'x': None, 'y': None}}
        questions['ok'] = {'type': 'yes_no', 'question': 'OK?'}
        questions['level'] = {'type': 'rating', 'question': 'How much?', 'levels': ['low', 'high']}
        with TestClient(create_app(scorer=scorer, api_key='k')) as c:
            r = c.post('/v1/ask', json={'model': 'saina-helm-0.8b', 'state': {'a': 1}, 'questions': questions},
                       headers={'Authorization': 'Bearer k'})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(len(scorer.calls), 1)
            self.assertEqual([q['mode'] for q in scorer.calls[0]], ['single_label'] * 9 + ['multi_label', 'single_label', 'single_label'])
            self.assertEqual(scorer.calls[0][0]['choices'], ['a', 'b: B'])
            self.assertEqual(list(r.json()['answers']), list(questions))
            # Billed for the text sent, like every model; the header reports the one model input.
            # One token per word: the state {"a":1}, then 9 x (Q?, a, b, B), Tags? x y, OK?, How much? low high.
            self.assertEqual(r.json()['usage']['input_tokens'], 45)
            self.assertEqual(r.headers['x-saina-billable-tokens'], '45')
            self.assertEqual(r.headers['x-saina-input-tokens'], '40')
            r = c.post('/v1/systemone', headers={'Authorization': 'Bearer k'}, json={'model': 'saina-helm-0.8b', 'state': 's',
                'questions': {'u': {'type': 'noul', 'instructions': 'Urgent?'},
                              't': {'type': 'choice', 'instructions': 'Team?', 'criteria': {'a': None, 'b': None}}}})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(len(scorer.calls), 2)
            self.assertEqual(r.json()['usage']['input_tokens'], 5)  # s, Urgent?, Team?, a, b
            self.assertEqual(r.headers['x-saina-input-tokens'], '40')
            ctx = c.get('/v1/models').json()['data'][0]['input_modalities'][0]['supported_inputs']['max_context_length']
            self.assertEqual(ctx['value'], 262144)


@unittest.skipUnless(HAVE_FASTAPI, 'fastapi and httpx required')
class ModelNameTests(unittest.TestCase):
    def post(self, scorer, model):
        with TestClient(create_app(scorer=scorer, api_key='k')) as c:
            return c.post('/v1/ask', headers={'Authorization': 'Bearer k'}, json={'model': model, 'state': 's',
                'questions': {'ok': {'type': 'yes_no', 'question': 'OK?'}}})

    def test_unversioned_alias_reaches_the_served_model_and_versions_are_exact(self):
        helm2 = SharedFakeScorer(); helm2.model_id = 'saina-helm-2-0.8b'
        for model in ('saina-helm', 'saina-helm-2-0.8b'):
            r = self.post(helm2, model)
            self.assertEqual(r.status_code, 200, model)
            self.assertEqual(r.json()['model'], 'saina-helm-2-0.8b')
        for model in ('helm-2-0.8b', 'helm-0.8b', 'saina-helm-0.8b', 'other'):
            r = self.post(helm2, model)
            self.assertEqual(r.status_code, 404, model)
            self.assertEqual(r.json()['error']['code'], 'not_found')
            self.assertIn('saina-helm-2-0.8b', r.json()['detail'])
        legacy = FakeScorer()
        for model in ('saina-helm', 'saina-helm-0.8b', 'helm-0.8b'):  # helm-0.8b: the 0.1 request name
            self.assertEqual(self.post(legacy, model).json()['model'], 'saina-helm-0.8b', model)
        self.assertEqual(self.post(legacy, 'saina-helm-2-0.8b').status_code, 404)

    def test_health_and_listing_name_the_loaded_model(self):
        helm2 = SharedFakeScorer(); helm2.model_id = 'saina-helm-2-0.8b'
        with TestClient(create_app(scorer=helm2, api_key='k')) as c:
            self.assertEqual(c.get('/healthz').json()['model'], 'saina-helm-2-0.8b')
            m = c.get('/v1/models').json()['data'][0]
            self.assertEqual((m['id'], m['hugging_face_id']), ('saina-helm-2-0.8b', 'run-saina/saina-helm-2-0.8b'))
            r = c.post('/v1/systemone', headers={'Authorization': 'Bearer k'}, json={'model': 'anything', 'state': 's',
                'questions': {'u': {'type': 'noul', 'instructions': 'Urgent?'}}})
            self.assertEqual(r.json()['model'], 'saina-helm-2-0.8b')
            self.assertTrue(m['description'].startswith('Saina Helm is an open 0.8B decision model'))

    def test_listing_names_helm_3(self):
        helm3 = SharedFakeScorer(); helm3.model_id = 'saina-helm-3-0.8b'
        with TestClient(create_app(scorer=helm3, api_key='k')) as c:
            m = c.get('/v1/models').json()['data'][0]
        self.assertEqual((m['id'], m['name'], m['hugging_face_id']),
                         ('saina-helm-3-0.8b', 'Saina: Helm 3 0.8B', 'run-saina/saina-helm-3-0.8b'))
        self.assertIn('native multimodal decision model', m['description'])


@unittest.skipUnless(HAVE_FASTAPI, 'fastapi and httpx required')
class TwoModelTests(unittest.TestCase):
    def setUp(self):
        self.helm2 = SharedFakeScorer(); self.helm2.model_id, self.helm2.revision = 'saina-helm-2-0.8b', 'b' * 40
        self.legacy = FakeScorer(); self.legacy.revision = 'a' * 40; self.legacy.max_length = 8192
        self.app = create_app(scorers=[self.helm2, self.legacy], api_key='k')

    def ask(self, c, model, route='/v1/ask'):
        body = ({'model': model, 'state': 's', 'questions': {'ok': {'type': 'yes_no', 'question': 'OK?'}}}
                if route == '/v1/ask' else
                {'model': model, 'state': 's', 'questions': {'u': {'type': 'noul', 'instructions': 'Urgent?'}}})
        return c.post(route, headers={'Authorization': 'Bearer k'}, json=body)

    def test_names_route_to_the_right_model(self):
        with TestClient(self.app) as c:
            for model, answered in (('saina-helm', 'saina-helm-2-0.8b'), ('saina-helm-2-0.8b', 'saina-helm-2-0.8b'),
                                    ('helm-0.8b', 'saina-helm-0.8b'), ('saina-helm-0.8b', 'saina-helm-0.8b')):
                r = self.ask(c, model)
                self.assertEqual((r.status_code, r.json()['model']), (200, answered), model)
            self.assertEqual(self.legacy.forwards, 2)
            self.assertEqual(len(self.helm2.calls), 2)
            for unknown in ('helm-2-0.8b', 'saina-helm-3-0.8b'):
                self.assertEqual(self.ask(c, unknown).status_code, 404, unknown)
            # System One: known names route, other labels reach the default.
            self.assertEqual(self.ask(c, 'saina-helm-0.8b', '/v1/systemone').json()['model'], 'saina-helm-0.8b')
            self.assertEqual(self.ask(c, 'openrouter/saina', '/v1/systemone').json()['model'], 'saina-helm-2-0.8b')

    def test_health_and_listing_cover_both(self):
        with TestClient(self.app) as c:
            h = c.get('/healthz').json()
            self.assertEqual((h['model'], h['revision']), ('saina-helm-2-0.8b', 'b' * 40))
            self.assertEqual(h['inputs'], ['text'])
            self.assertEqual(h['models'], [{'model': 'saina-helm-2-0.8b', 'revision': 'b' * 40, 'inputs': ['text']},
                                           {'model': 'saina-helm-0.8b', 'revision': 'a' * 40, 'inputs': ['text']}])
            data = c.get('/v1/models').json()['data']
            self.assertEqual([m['id'] for m in data], ['saina-helm-2-0.8b', 'saina-helm-0.8b'])
            self.assertEqual([m['input_modalities'][0]['supported_inputs']['max_context_length']['value'] for m in data],
                             [262144, 8192])

    def test_models_share_the_bounded_queue(self):
        # One GPU: every served model waits for the same execution slots.
        with TestClient(self.app) as c:
            executor = c.app.state.executor
            for _ in range(executor.slots + executor.depth):
                executor.reserve()
            try:
                for model in ('saina-helm', 'saina-helm-0.8b'):
                    r = self.ask(c, model)
                    self.assertEqual((r.status_code, r.json()['error']['code']), (429, 'overloaded'), model)
            finally:
                for _ in range(executor.slots + executor.depth):
                    executor.release()
            self.assertEqual(self.ask(c, 'saina-helm-0.8b').status_code, 200)

    def test_duplicate_releases_are_refused(self):
        other = SharedFakeScorer(); other.model_id = 'saina-helm-2-0.8b'
        with self.assertRaisesRegex(RuntimeError, 'different release'):
            with TestClient(create_app(scorers=[self.helm2, other], api_key='k')):
                pass

    def test_models_spec(self):
        from saina.server import parse_models
        self.assertEqual(parse_models('run-saina/saina-helm-2-0.8b@' + 'b' * 40 + ':32768, /models/old@' + 'a' * 40),
                         [('run-saina/saina-helm-2-0.8b', 'b' * 40, 32768), ('/models/old', 'a' * 40, None)])
        for bad in ('', 'x@r:big', '@r'):
            with self.assertRaises(ValueError):
                parse_models(bad)


@unittest.skipUnless(HAVE_FASTAPI, 'fastapi and httpx required')
class RouteTests(unittest.TestCase):
    def app(self, api_key='k1, k2', **kwargs):
        self.scorer = kwargs.pop('scorer', None) or FakeScorer()
        return create_app(scorer=self.scorer, api_key=api_key, **kwargs)

    def test_any_listed_key_is_accepted(self):
        body = {'model': 'saina-helm-0.8b', 'state': 'Charged twice.', 'questions': {
            'team': {'type': 'single_choice', 'question': 'Which team?', 'options': {'billing': None, 'tech': None}}}}
        with TestClient(self.app()) as c:
            for key, status in (('k1', 200), ('k2', 200), ('k3', 401), ('', 401)):
                r = c.post('/v1/ask', json=body, headers={'Authorization': f'Bearer {key}'})
                self.assertEqual(r.status_code, status, key)
            self.assertEqual(c.post('/v1/ask', json=body).status_code, 401)

    def test_system_one_route(self):
        body = {'model': 'saina-helm-0.8b', 'state': 'Help! Payouts failing for 3 days.', 'questions': {
            'urgent': {'type': 'noul', 'instructions': 'Is this urgent?'},
            'team': {'type': 'choice', 'instructions': 'Which team?', 'criteria': {'billing': 'Money', 'tech': 'Bugs'}},
            'severity': {'type': 'score', 'instructions': 'How severe?', 'criteria': ['low', 'mid', 'high']}}}
        with TestClient(self.app()) as c:
            r = c.post('/v1/systemone', json=body, headers={'Authorization': 'Bearer k2'})
            self.assertEqual(r.status_code, 200, r.text)
            a = r.json()['answers']
            self.assertAlmostEqual(a['urgent']['noul'], 0.7)
            self.assertEqual(a['team']['choice'], 'billing')
            self.assertIn('score', a['severity'])
            # Billed on the text sent: state 6 + urgent 3 + team 2+4 + severity 2+3 words; the model read 3 × 7.
            self.assertEqual(r.json()['usage'], {'input_tokens': 20, 'output_tokens': 0})
            self.assertEqual(r.headers['X-Saina-Input-Tokens'], '21')
            bad = c.post('/v1/systemone', json={**body, 'questions': {'x': {'type': 'choice', 'instructions': 'q', 'criteria': {'only': None}}}},
                         headers={'Authorization': 'Bearer k1'})
            self.assertEqual(bad.status_code, 422)
            self.assertNotIn('Payouts', bad.text)

    def test_healthz_reports_pinned_revision(self):
        with patch.dict(os.environ, {'SAINA_REVISION': 'e82055b'}):
            with TestClient(self.app()) as c:
                h = c.get('/healthz').json()
        self.assertEqual((h['status'], h['model'], h['revision']), ('ready', 'saina-helm-0.8b', 'e82055b'))
        self.assertEqual(h['queue'], {'slots': 1, 'depth': 4, 'running': 0, 'queued': 0})

    def test_plain_http_behind_proxy_is_refused(self):
        with TestClient(self.app()) as c:
            for h in ({'X-Forwarded-Proto': 'http'}, {'CF-Visitor': '{"scheme":"http"}'}):
                r = c.get('/healthz', headers=h)
                self.assertEqual(r.status_code, 426, h)
                self.assertNotIn('Strict-Transport-Security', r.headers)
            r = c.get('/healthz', headers={'X-Forwarded-Proto': 'https'})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.headers['Strict-Transport-Security'], 'max-age=15552000')
            self.assertEqual(c.get('/healthz').status_code, 200)  # no proxy headers: served as-is

    def test_models_listing_is_public_and_priced_from_env(self):
        with patch.dict(os.environ, {'SAINA_PROMPT_PRICE_USD': '0.00000002'}):
            with TestClient(self.app()) as c:
                m = c.get('/v1/models').json()['data'][0]
        self.assertEqual(m['id'], 'saina-helm-0.8b')
        self.assertEqual(m['input_modalities'][0]['pricing'], [{'type': 'prompt', 'unit': 'token', 'cost_usd': '0.00000002'}])
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('SAINA_PROMPT_PRICE_USD', None)
            with TestClient(self.app()) as c:
                self.assertNotIn('pricing', c.get('/v1/models').json()['data'][0]['input_modalities'][0])



ASK = {'model': 'saina-helm-0.8b', 'state': 'Charged twice.', 'questions': {
    'team': {'type': 'single_choice', 'question': 'Which team?', 'options': {'billing': None, 'tech': None}},
    'urgent': {'type': 'yes_no', 'question': 'Urgent?'}}}


@unittest.skipUnless(HAVE_FASTAPI, 'fastapi and httpx required')
class MeteringAndQueueTests(unittest.TestCase):
    def app(self, **kwargs):
        self.scorer = kwargs.pop('scorer', None) or FakeScorer()
        return create_app(scorer=self.scorer, api_key='k', service_token='svc', **kwargs)

    def test_usage_is_billable_input_and_header_is_model_input(self):
        with TestClient(self.app()) as c:
            r = c.post('/v1/ask', json=ASK, headers={'Authorization': 'Bearer k'})
        # Billed: 'Charged twice.' once (2) + 'Which team?' billing tech (4) + 'Urgent?' (1).
        self.assertEqual(r.json()['usage'], {'input_tokens': 7, 'output_tokens': 0})
        self.assertEqual(r.headers['X-Saina-Billable-Tokens'], '7')
        # The model read two 7-token prompts; the gateway's metering check compares this one.
        self.assertEqual(r.headers['X-Saina-Input-Tokens'], '14')
        self.assertNotIn('X-Saina-Queue-Ms', r.headers)

    def test_invalid_later_question_runs_no_inference(self):
        body = {**ASK, 'questions': {**ASK['questions'], 'z': {'type': 'yes_no', 'question': 'too long'}}}
        with TestClient(self.app()) as c:
            r = c.post('/v1/ask', json=body, headers={'Authorization': 'Bearer k'})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()['error']['code'], 'invalid_request')
        self.assertEqual(self.scorer.forwards, 0)
        self.assertEqual(c.app.state.executor.admitted, 0)  # the queue place taken for encoding was given back

    def test_full_queue_rejects_before_tokenizing_and_is_retryable(self):
        from saina.client import _error, _retryable
        app = self.app(queue_depth=0)
        with TestClient(app) as c:
            app.state.executor.reserve()  # the only place is taken
            r = c.post('/v1/ask', json=ASK, headers={'Authorization': 'Bearer k'})
            self.assertEqual(r.status_code, 429)
            self.assertEqual(r.headers['Retry-After'], '1')
            self.assertEqual(r.json()['error'], {'code': 'overloaded', 'message': 'Inference queue is full',
                'admitted': False, 'state': 'not_admitted', 'retry': 'same_operation'})
            self.assertEqual(self.scorer.encodes, 0)
            err, typed = _error(r.status_code, r.content, r.headers)
            self.assertTrue(typed and _retryable(err))
            app.state.executor.release()
            self.assertEqual(c.post('/v1/ask', json=ASK, headers={'Authorization': 'Bearer k'}).status_code, 200)

    def test_cancelled_queued_job_is_503_not_admitted(self):
        import threading, time
        app = self.app()
        with TestClient(app) as c:
            ex = app.state.executor
            gate, started = threading.Event(), threading.Event()
            busy, _ = ex.submit(lambda: (started.set(), gate.wait()))
            self.assertTrue(started.wait(5))
            result = {}
            t = threading.Thread(target=lambda: result.update(r=c.post('/v1/ask', json=ASK, headers={
                'Authorization': 'Bearer svc', 'X-Saina-Request-Id': 'q1'})))
            t.start()
            for _ in range(500):  # until the request is queued behind the busy slot
                if ex.cancel('q1') == 'cancelled':
                    break
                time.sleep(0.01)
            else:
                self.fail('request never queued')
            t.join(5)
            gate.set()
            busy.result(timeout=5)
            r = result['r']
            self.assertEqual(r.status_code, 503, r.text)
            self.assertEqual(r.json()['error']['code'], 'overloaded')
            self.assertIs(r.json()['error']['admitted'], False)
            self.assertEqual(self.scorer.forwards, 0)

    def test_scorer_without_explicit_usage_is_rejected(self):
        class Legacy:
            last_input_tokens = 7
            def predict(self, *a, **k): return [0.5, 0.5]
        with TestClient(self.app(scorer=Legacy())) as c:
            r = c.post('/v1/ask', json=ASK, headers={'Authorization': 'Bearer k'})
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.json()['error']['code'], 'inference_failed')

    def test_service_token_metering_headers(self):
        with TestClient(self.app()) as c:
            ok = c.post('/v1/ask', json=ASK, headers={'Authorization': 'Bearer svc', 'X-Saina-Expected-Input-Tokens': '14',
                                                     'X-Saina-Request-Id': 'r1'})
            self.assertEqual(ok.status_code, 200, ok.text)
            self.assertIn('X-Saina-Queue-Ms', ok.headers)
            self.assertIn('X-Saina-Exec-Ms', ok.headers)
            bad = c.post('/v1/ask', json=ASK, headers={'Authorization': 'Bearer svc', 'X-Saina-Expected-Input-Tokens': '13'})
            self.assertEqual((bad.status_code, bad.json()['error']['code']), (409, 'metering_mismatch'))
            self.assertEqual(self.scorer.forwards, 2)  # the mismatched request ran nothing
            # Customer keys cannot send metering headers or reach internal routes.
            ignored = c.post('/v1/ask', json=ASK, headers={'Authorization': 'Bearer k', 'X-Saina-Expected-Input-Tokens': '1'})
            self.assertEqual(ignored.status_code, 200)
            self.assertEqual(c.delete('/v1/internal/requests/r1', headers={'Authorization': 'Bearer k'}).status_code, 403)
            self.assertEqual(c.delete('/v1/internal/requests/r1', headers={'Authorization': 'Bearer svc'}).json()['state'], 'unknown')
            self.assertEqual(c.post('/v1/ask', json=ASK, headers={'Authorization': 'Bearer nope'}).json()['error']['code'],
                             'invalid_api_key')

    def test_service_token_alone_is_enough(self):
        app = create_app(scorer=FakeScorer(), api_key='', service_token='svc')
        with TestClient(app) as c:
            self.assertEqual(c.post('/v1/ask', json=ASK, headers={'Authorization': 'Bearer svc'}).status_code, 200)
        with self.assertRaises(RuntimeError):
            create_app(scorer=FakeScorer(), api_key='', service_token='')


@unittest.skipUnless(HAVE_FASTAPI, 'fastapi and httpx required')
class WithoutLifespanTests(unittest.TestCase):
    """Embedders that never run the app's lifespan (a bare TestClient, another framework) still work."""
    AUTH = {'Authorization': 'Bearer k'}

    def check(self, app, model='saina-helm-0.8b', answered='saina-helm-0.8b'):
        c = TestClient(app)  # not a context manager: the lifespan never runs
        r = c.post('/v1/ask', headers=self.AUTH, json={**ASK, 'model': model})
        self.assertEqual((r.status_code, r.json().get('model')), (200, answered), r.text)
        self.assertEqual(c.get('/healthz').json()['model'], answered)
        self.assertEqual(c.get('/v1/models').json()['data'][0]['id'], answered)

    def test_scorer_set_on_app_state(self):
        app = create_app(api_key='k')
        app.state.scorer = FakeScorer()  # the 0.2 way of injecting a model
        self.check(app)

    def test_scorer_passed_to_create_app(self):
        self.check(create_app(scorer=FakeScorer(), api_key='k'))

    def test_several_scorers_passed_to_create_app(self):
        helm2 = SharedFakeScorer(); helm2.model_id = 'saina-helm-2-0.8b'
        app = create_app(scorers=[helm2, FakeScorer()], api_key='k')
        self.check(app, 'saina-helm', 'saina-helm-2-0.8b')  # the first model is the default
        r = TestClient(app).post('/v1/ask', headers=self.AUTH, json=ASK)
        self.assertEqual((r.status_code, r.json()['model']), (200, 'saina-helm-0.8b'))

    def test_no_model_is_unavailable_not_a_crash(self):
        r = TestClient(create_app(api_key='k')).post('/v1/ask', headers=self.AUTH, json=ASK)
        self.assertEqual((r.status_code, r.json()['error']['code']), (503, 'inference_unavailable'))
        self.assertFalse(r.json()['error']['admitted'])


class BoundedExecutorTests(unittest.TestCase):
    def test_full_queue_rejects_promptly_and_slot_held_until_work_finishes(self):
        import threading
        from saina.server import BoundedExecutor, Overloaded, QueueExpired
        ex = BoundedExecutor(slots=1, depth=1, queue_timeout=5)
        gate, started = threading.Event(), threading.Event()
        first, _ = ex.submit(lambda: (started.set(), gate.wait())[1], request_id='a')
        # A job the pool has not picked up yet is still cancellable; wait until 'a' really runs.
        self.assertTrue(started.wait(5))
        second, _ = ex.submit(lambda: 2, request_id='b')
        with self.assertRaises(Overloaded):
            ex.submit(lambda: 3)
        self.assertEqual(ex.cancel('b'), 'cancelled')  # queued work can be dropped
        self.assertEqual(ex.cancel('a'), 'running')    # running work cannot
        self.assertEqual(ex.stats()['running'], 1)
        third, _ = ex.submit(lambda: 3)               # the cancelled slot was freed
        gate.set()
        self.assertTrue(first.result(timeout=5))
        self.assertEqual(third.result(timeout=5), 3)
        ex.shutdown()
        self.assertEqual(ex.stats(), {'slots': 1, 'depth': 1, 'running': 0, 'queued': 0})

    def test_queued_work_past_its_deadline_never_runs(self):
        import threading, time
        from saina.server import BoundedExecutor, QueueExpired
        ex = BoundedExecutor(slots=1, depth=2, queue_timeout=5)
        gate, started = threading.Event(), threading.Event()
        ran = []
        # The blocker must be running (not itself expiring in the queue) before the late job is queued.
        ex.submit(lambda: (started.set(), gate.wait()))
        self.assertTrue(started.wait(5))
        ex.queue_timeout = 0.05  # applies to jobs submitted from now on
        late, _ = ex.submit(lambda: ran.append(1))
        time.sleep(0.1)  # the late job cannot start before gate.set(), so it is past its 0.05 s deadline
        gate.set()
        with self.assertRaises(QueueExpired):
            late.result(timeout=5)
        self.assertEqual(ran, [])
        ex.shutdown()


if __name__ == '__main__': unittest.main()


class MediaFakeScorer(SharedFakeScorer):
    """A shared-pass model that reads images and audio; every image is 10 tokens."""
    inputs = ('text', 'image', 'audio')
    model_id = 'saina-helm-2-mm-0.8b'

    def media_tokens(self, media):
        return 10 * len(media.images) if media else 0

    def encode_shared(self, context, questions, media=None):
        self.media = media
        return super().encode_shared(context, questions) + [0] * self.media_tokens(media)

    def forward_shared(self, ids, counts, modes, media=None):
        self.forwarded_media = media
        return super().forward_shared(ids, counts, modes)


def _png():
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.new('RGB', (40, 30), (10, 200, 10)).save(buffer, 'PNG')
    return 'data:image/png;base64,' + __import__('base64').b64encode(buffer.getvalue()).decode()


def _wav(seconds=1.0):
    import numpy as np
    from saina.media import _wav
    x = 0.1 * np.sin(np.arange(int(16000 * seconds)) / 5)
    return 'data:audio/wav;base64,' + __import__('base64').b64encode(_wav(x, 16000)).decode()


try:
    import numpy, PIL, scipy, soundfile  # noqa: E401,F401
    HAVE_MEDIA = True
except ImportError:
    HAVE_MEDIA = False


@unittest.skipUnless(HAVE_FASTAPI, 'fastapi and httpx required')
class MediaRouteTests(unittest.TestCase):
    QUESTIONS = {'ok': {'type': 'yes_no', 'question': 'OK?'}}

    def post(self, scorer, route='/v1/ask', **fields):
        body = ({'model': 'saina-helm', 'state': 's', 'questions': self.QUESTIONS} if route == '/v1/ask' else
                {'model': 'saina-helm', 'state': 's', 'questions': {'u': {'type': 'noul', 'instructions': 'Urgent?'}}})
        with TestClient(create_app(scorer=scorer, api_key='k')) as c:
            return c.post(route, headers={'Authorization': 'Bearer k'}, json={**body, **fields})

    def test_text_only_model_refuses_media_before_running(self):
        helm2 = SharedFakeScorer(); helm2.model_id = 'saina-helm-2-0.8b'
        for route in ('/v1/ask', '/v1/systemone'):
            for fields in ({'images': ['data:image/png;base64,iVBORw0KGgo=']},
                           {'audios': ['data:audio/wav;base64,UklGRg==']}):
                r = self.post(helm2, route, **fields)
                self.assertEqual(r.status_code, 422, r.text)
                self.assertEqual(r.json()['error'], {'code': 'invalid_request',
                                                     'message': 'saina-helm-2-0.8b accepts text only'})
        self.assertEqual(helm2.calls, [])
        self.assertEqual(self.post(helm2, images=[], audios=None).status_code, 200)  # empty lists are no media

    def test_videos_and_invalid_media_are_invalid_requests(self):
        scorer = MediaFakeScorer()
        for fields, message in (({'videos': ['data:video/mp4;base64,AAAA']}, 'videos are not supported yet'),
                                ({'images': ['data:image/png;base64,iVBORw0KGgo='] * 9}, 'At most 8 images per request'),
                                ({'audios': ['data:audio/wav;base64,UklGRg=='] * 5}, 'At most 4 audios per request'),
                                ({'images': ['https://example.com/cat.png']}, r'images\[0\] must be a base64 data URL'),
                                ({'images': ['data:image/bmp;base64,Qk0=']}, 'accepted types are image/png')):
            r = self.post(scorer, **fields)
            self.assertEqual((r.status_code, r.json()['error']['code']), (422, 'invalid_request'), fields)
            self.assertRegex(r.json()['error']['message'], message)
        self.assertEqual(scorer.calls, [])
        r = self.post(scorer, images=[1])  # not a string: schema validation
        self.assertEqual((r.status_code, r.json()['error']['code']), (422, 'invalid_request'))

    @unittest.skipUnless(HAVE_MEDIA, 'requires saina[media]')
    def test_media_is_read_in_order_and_billed(self):
        scorer = MediaFakeScorer()
        r = self.post(scorer, images=[_png(), _png()], audios=[_wav()])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(scorer.forwarded_media.sources, ['image', 'image', 'audio'])
        self.assertEqual([im.size for im in scorer.forwarded_media.images], [(40, 30), (40, 30), (100, 128)])
        # Billed: the text sent (s, OK?) plus the input tokens of the two images and the spectrogram.
        self.assertEqual(r.json()['usage']['input_tokens'], 2 + 30)
        self.assertEqual(r.headers['x-saina-media-tokens'], '30')
        self.assertEqual(r.headers['x-saina-input-tokens'], str(40 + 30))
        r = self.post(scorer, '/v1/systemone', images=[_png()])
        self.assertEqual((r.status_code, r.json()['usage']['input_tokens']), (200, 2 + 10))
        r = self.post(scorer, audios=[_wav(11)])
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()['error']['message'],
                         'audios[0] is 11.0 s long; at most 10.0 s are accepted (shorten it; nothing is cut off)')

    def test_models_declare_their_inputs(self):
        with TestClient(create_app(scorers=[MediaFakeScorer(), FakeScorer()], api_key='k')) as c:
            h = c.get('/healthz').json()
            self.assertEqual(h['inputs'], ['text', 'image', 'audio'])
            self.assertEqual([m['inputs'] for m in h['models']], [['text', 'image', 'audio'], ['text']])
            mm, text = c.get('/v1/models').json()['data']
            self.assertEqual((mm['inputs'], text['inputs']), (['text', 'image', 'audio'], ['text']))
            self.assertEqual([m['type'] for m in mm['input_modalities']], ['text', 'image', 'audio'])
            self.assertEqual(mm['input_modalities'][1]['supported_inputs']['max_items'], 8)
            self.assertEqual(mm['input_modalities'][2]['supported_inputs']['max_seconds'], 10.0)
            self.assertEqual([m['type'] for m in text['input_modalities']], ['text'])


@unittest.skipUnless(HAVE_FASTAPI, 'fastapi and httpx required')
class SpecialistTests(unittest.TestCase):
    def setUp(self):
        self.omni = SharedFakeScorer(); self.omni.model_id, self.omni.revision = 'saina-helm-2-omni-0.8b', 'c' * 40
        self.omni.inputs = ('text', 'image', 'audio')
        self.mech = SharedFakeScorer(); self.mech.model_id = 'saina-helm-2-omni-0.8b'
        self.mech.inputs, self.mech.specialist = ('text', 'audio'), 'mechanical'
        self.mech.specialist_revision, self.mech.specialist_description = 'd' * 40, 'Machine faults from sound'
        self.omni.specialists = {'mechanical': self.mech}
        self.text = SharedFakeScorer(); self.text.model_id = 'saina-helm-2-0.8b'
        self.app = create_app(scorers=[self.text, self.omni], api_key='k')

    def ask(self, c, specialist=None, model='saina-helm-2-omni-0.8b', route='/v1/ask'):
        body = ({'model': model, 'state': 's', 'questions': {'ok': {'type': 'yes_no', 'question': 'OK?'}}}
                if route == '/v1/ask' else
                {'model': model, 'state': 's', 'questions': {'u': {'type': 'noul', 'instructions': 'Urgent?'}}})
        if specialist is not None:
            body['specialist'] = specialist
        return c.post(route, headers={'Authorization': 'Bearer k'}, json=body)

    def test_specialist_answers_only_when_selected(self):
        with TestClient(self.app) as c:
            r = self.ask(c)
            self.assertEqual((r.status_code, r.json()['model']), (200, 'saina-helm-2-omni-0.8b'))
            self.assertNotIn('specialist', r.json())
            r = self.ask(c, 'mechanical')
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual((r.json()['model'], r.json()['specialist']), ('saina-helm-2-omni-0.8b', 'mechanical'))
            r = self.ask(c, 'mechanical', route='/v1/systemone')
            self.assertEqual((r.status_code, r.json()['specialist']), (200, 'mechanical'))
            self.assertEqual((len(self.omni.calls), len(self.mech.calls)), (1, 2))

    def test_unknown_or_foreign_specialist_is_not_found(self):
        with TestClient(self.app) as c:
            r = self.ask(c, 'legal')
            self.assertEqual(r.status_code, 404)
            self.assertIn('available: mechanical', r.json()['error']['message'])
            # Specialists belong to one model: the text model has none.
            self.assertEqual(self.ask(c, 'mechanical', model='saina-helm-2-0.8b').status_code, 404)
            self.assertEqual(self.ask(c, 'Bad Name').status_code, 422)
            self.assertEqual(self.mech.calls, [])

    def test_listing(self):
        with TestClient(self.app) as c:
            self.assertEqual(c.get('/v1/specialists').json()['data'], [{
                'id': 'mechanical', 'model': 'saina-helm-2-omni-0.8b', 'revision': 'd' * 40,
                'inputs': ['text', 'audio'], 'description': 'Machine faults from sound'}])
            text, omni = c.get('/v1/models').json()['data']
            self.assertEqual((text['specialists'], omni['specialists']), ([], ['mechanical']))
            self.assertEqual(c.get('/healthz').json()['specialists'][0]['id'], 'mechanical')

    def test_attach_finds_the_matching_base(self):
        from saina.server import attach_specialists
        text, omni = SharedFakeScorer(), SharedFakeScorer()

        def loader(base, checkpoint, revision):
            if base is not omni:
                raise ValueError('trained for different base weights')
            special = SharedFakeScorer(); special.specialist = 'mechanical'
            return special
        attach_specialists([text, omni], 'run-saina/x@' + 'e' * 40, loader)
        self.assertEqual(list(omni.specialists), ['mechanical'])
        self.assertFalse(hasattr(text, 'specialists'))
        with self.assertRaises(RuntimeError):
            attach_specialists([text], 'run-saina/x@' + 'e' * 40, loader)


@unittest.skipUnless(HAVE_FASTAPI, 'fastapi and httpx required')
class StandaloneSpecialistTests(unittest.TestCase):
    """A checkpoint with a specialist merged in serves that specialist without its base model."""
    def merged(self, name='mechanical'):
        s = SharedFakeScorer(); s.model_id, s.revision = 'saina-helm-2-omni-0.8b', 'f' * 40
        s.inputs, s.specialist, s.specialist_revision, s.specialist_description = ('text', 'audio'), name, 'f' * 40, 'Machines'
        s.specialists, s.specialist_only = {name: s}, True
        return s

    def ask(self, c, specialist=None):
        body = {'model': 'saina-helm-2-omni-0.8b', 'state': 's', 'questions': {'ok': {'type': 'yes_no', 'question': 'OK?'}}}
        if specialist is not None:
            body['specialist'] = specialist
        return c.post('/v1/ask', headers={'Authorization': 'Bearer k'}, json=body)

    def test_answers_only_as_its_specialist(self):
        mech = self.merged()
        with TestClient(create_app(scorers=[mech], api_key='k')) as c:
            r = self.ask(c, 'mechanical')
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual((r.json()['model'], r.json()['specialist']), ('saina-helm-2-omni-0.8b', 'mechanical'))
            r = self.ask(c)
            self.assertEqual(r.status_code, 404)
            self.assertIn('only as its specialist mechanical', r.json()['error']['message'])
            self.assertEqual(self.ask(c, 'legal').status_code, 404)
            self.assertEqual(len(mech.calls), 1)
            self.assertEqual(c.get('/v1/specialists').json()['data'], [{
                'id': 'mechanical', 'model': 'saina-helm-2-omni-0.8b', 'revision': 'f' * 40,
                'inputs': ['text', 'audio'], 'description': 'Machines'}])

    def test_gather_joins_a_base_or_another_standalone_of_the_same_release(self):
        from saina.server import gather_specialists
        text = SharedFakeScorer(); text.model_id = 'saina-helm-2-0.8b'
        base = SharedFakeScorer(); base.model_id = 'saina-helm-2-omni-0.8b'
        mech, legal = self.merged(), self.merged('legal')
        self.assertEqual(gather_specialists([text, mech, legal]), [text, mech])
        self.assertEqual(sorted(mech.specialists), ['legal', 'mechanical'])
        mech, legal = self.merged(), self.merged('legal')
        self.assertEqual(gather_specialists([mech, base, legal]), [base])
        self.assertEqual(sorted(base.specialists), ['legal', 'mechanical'])
        with self.assertRaises(RuntimeError):
            gather_specialists([self.merged(), self.merged()])
