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

    def test_default_allows_saina_run(self):
        with TestClient(self.app()) as c:
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
        with TestClient(self.app()) as c:
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
    last_input_tokens = 7

    def predict(self, context, question, choices, temperature=1.0, mode='single_label'):
        n = len(choices)
        return [1 / n] * n if mode == 'multi_label' else [0.7] + [0.3 / (n - 1)] * (n - 1)


class SharedFakeScorer(FakeScorer):
    shared_trunk_pass = True
    max_length = 262144

    def __init__(self):
        self.calls = []

    def predict(self, *args, **kwargs):
        raise AssertionError('shared scorers must not be called per question')

    def predict_questions(self, context, questions, temperature=1.0):
        self.calls.append(questions)
        self.last_input_tokens = 40
        return [FakeScorer.predict(self, context, q['question'], q['choices'], mode=q['mode']) for q in questions]


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
            r = c.post('/v1/ask', json={'model': 'helm-0.8b', 'state': {'a': 1}, 'questions': questions},
                       headers={'Authorization': 'Bearer k'})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(len(scorer.calls), 1)
            self.assertEqual([q['mode'] for q in scorer.calls[0]], ['single_label'] * 9 + ['multi_label', 'single_label', 'single_label'])
            self.assertEqual(scorer.calls[0][0]['choices'], ['a', 'b: B'])
            self.assertEqual(list(r.json()['answers']), list(questions))
            self.assertEqual(r.json()['usage']['input_tokens'], 40)
            r = c.post('/v1/systemone', headers={'Authorization': 'Bearer k'}, json={'model': 'saina-helm-0.8b', 'state': 's',
                'questions': {'u': {'type': 'noul', 'instructions': 'Urgent?'},
                              't': {'type': 'choice', 'instructions': 'Team?', 'criteria': {'a': None, 'b': None}}}})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(len(scorer.calls), 2)
            self.assertEqual(r.json()['usage']['input_tokens'], 40)
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
        for model in ('saina-helm', 'helm-2-0.8b', 'saina-helm-2-0.8b'):
            r = self.post(helm2, model)
            self.assertEqual(r.status_code, 200, model)
            self.assertEqual(r.json()['model'], 'helm-2-0.8b')
        for model in ('helm-0.8b', 'saina-helm-0.8b', 'other'):
            r = self.post(helm2, model)
            self.assertEqual(r.status_code, 404, model)
            self.assertIn('saina-helm-2-0.8b', r.json()['detail'])
        legacy = FakeScorer()
        self.assertEqual(self.post(legacy, 'saina-helm').json()['model'], 'helm-0.8b')
        self.assertEqual(self.post(legacy, 'helm-2-0.8b').status_code, 404)

    def test_health_and_listing_name_the_loaded_model(self):
        helm2 = SharedFakeScorer(); helm2.model_id = 'saina-helm-2-0.8b'
        with TestClient(create_app(scorer=helm2, api_key='k')) as c:
            self.assertEqual(c.get('/healthz').json()['model'], 'saina-helm-2-0.8b')
            m = c.get('/v1/models').json()['data'][0]
            self.assertEqual((m['id'], m['hugging_face_id']), ('saina-helm-2-0.8b', 'run-saina/saina-helm-2-0.8b'))
            r = c.post('/v1/systemone', headers={'Authorization': 'Bearer k'}, json={'model': 'anything', 'state': 's',
                'questions': {'u': {'type': 'noul', 'instructions': 'Urgent?'}}})
            self.assertEqual(r.json()['model'], 'saina-helm-2-0.8b')


@unittest.skipUnless(HAVE_FASTAPI, 'fastapi and httpx required')
class RouteTests(unittest.TestCase):
    def app(self, api_key='k1, k2'):
        return create_app(scorer=FakeScorer(), api_key=api_key)

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
                self.assertEqual(c.get('/healthz').json(), {'status': 'ready', 'model': 'saina-helm-0.8b', 'revision': 'e82055b'})

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


if __name__ == '__main__': unittest.main()
