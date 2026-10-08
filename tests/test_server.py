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

    def encode(self, context, question, choices, mode='single_label'):
        if question == 'too long':
            from saina.errors import InputTooLong
            raise InputTooLong('too long')
        return list(range(7))

    def forward(self, ids, count, mode='single_label'):
        self.forwards += 1
        return [1 / count] * count if mode == 'multi_label' else [0.7] + [0.3 / (count - 1)] * (count - 1)


@unittest.skipUnless(HAVE_FASTAPI, 'fastapi and httpx required')
class RouteTests(unittest.TestCase):
    def app(self, api_key='k1, k2', **kwargs):
        self.scorer = kwargs.pop('scorer', None) or FakeScorer()
        return create_app(scorer=self.scorer, api_key=api_key, **kwargs)

    def test_any_listed_key_is_accepted(self):
        body = {'model': 'helm-0.8b', 'state': 'Charged twice.', 'questions': {
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
            self.assertEqual(r.json()['usage'], {'input_tokens': 21, 'output_tokens': 0})
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



ASK = {'model': 'helm-0.8b', 'state': 'Charged twice.', 'questions': {
    'team': {'type': 'single_choice', 'question': 'Which team?', 'options': {'billing': None, 'tech': None}},
    'urgent': {'type': 'yes_no', 'question': 'Urgent?'}}}


@unittest.skipUnless(HAVE_FASTAPI, 'fastapi and httpx required')
class MeteringAndQueueTests(unittest.TestCase):
    def app(self, **kwargs):
        self.scorer = kwargs.pop('scorer', None) or FakeScorer()
        return create_app(scorer=self.scorer, api_key='k', service_token='svc', **kwargs)

    def test_usage_is_the_encoded_length_of_every_question(self):
        with TestClient(self.app()) as c:
            r = c.post('/v1/ask', json=ASK, headers={'Authorization': 'Bearer k'})
        self.assertEqual(r.json()['usage'], {'input_tokens': 14, 'output_tokens': 0})
        self.assertEqual(r.headers['X-Saina-Input-Tokens'], '14')
        self.assertNotIn('X-Saina-Queue-Ms', r.headers)

    def test_invalid_later_question_runs_no_inference(self):
        body = {**ASK, 'questions': {**ASK['questions'], 'z': {'type': 'yes_no', 'question': 'too long'}}}
        with TestClient(self.app()) as c:
            r = c.post('/v1/ask', json=body, headers={'Authorization': 'Bearer k'})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()['error']['code'], 'invalid_request')
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


class BoundedExecutorTests(unittest.TestCase):
    def test_full_queue_rejects_promptly_and_slot_held_until_work_finishes(self):
        import threading
        from saina.server import BoundedExecutor, Overloaded, QueueExpired
        ex = BoundedExecutor(slots=1, depth=1, queue_timeout=5)
        gate = threading.Event()
        first, _ = ex.submit(gate.wait, request_id='a')
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
        ex = BoundedExecutor(slots=1, depth=2, queue_timeout=0.05)
        gate = threading.Event()
        ran = []
        ex.submit(gate.wait)
        late, _ = ex.submit(lambda: ran.append(1))
        time.sleep(0.1)
        gate.set()
        with self.assertRaises(QueueExpired):
            late.result(timeout=5)
        self.assertEqual(ran, [])
        ex.shutdown()


if __name__ == '__main__': unittest.main()
