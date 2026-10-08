import email.message
import io
import json
import threading
import time
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock
from urllib.error import HTTPError, URLError
from saina.client import (SainaHelm, SainaHelmError, SainaError, SainaConnectionError, new_idempotency_key,
                          InsufficientCredits, InferenceFailed, IdempotencyConflict, AccountingUnavailable,
                          Overloaded, AccountSuspended, ERROR_CLASSES, DEFAULT_BASE_URL)

ASK = dict(model='saina-helm-0.8b', state='text',
           questions={'q': {'type': 'yes_no', 'question': 'Yes?'}})
RESULT = {'model': 'saina-helm-0.8b', 'answers': {'q': {'type': 'yes_no', 'yes': .7, 'no': .3, 'confidence': .4}},
          'usage': {'input_tokens': 3, 'output_tokens': 0}}
KEY = '01890a5d-ac96-774b-bcce-b302099a8057'


def headers(**values):
    message = email.message.Message()
    for name, value in values.items():
        message[name.replace('_', '-')] = value
    return message


class Reply(io.BytesIO):
    def __init__(self, body, **hdrs):
        super().__init__(json.dumps(body).encode() if not isinstance(body, bytes) else body)
        self.headers = headers(**hdrs)


def fail(status, error=None, raw=None, **hdrs):
    body = raw if raw is not None else json.dumps({'error': error, 'detail': 'x'}).encode()
    return HTTPError('https://example.com', status, 'error', headers(**hdrs), io.BytesIO(body))


def error(code, **fields):
    return dict(code=code, message=code, request_id='req_1', **fields)


class FakeOpener:
    """Replays scripted outcomes and records every request."""
    def __init__(self, *outcomes):
        self.outcomes, self.requests = list(outcomes), []

    def open(self, request, timeout=None):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def client(*outcomes, **options):
    c = SainaHelm('https://example.com', api_key='test', **options)
    c._opener = FakeOpener(*outcomes)
    c.sleeps = []
    c._sleep = c.sleeps.append
    c._random = lambda: 1.
    return c


class ClientTests(unittest.TestCase):
    def test_ask_memberships(self):
        c=SainaHelm('https://example.com',api_key='test')
        c._opener=Mock()
        request=dict(model='saina-helm-0.8b',state='text',questions={'tags':{'type':'multi_choice','question':'Tags?','options':{'a':None,'b':None}}})
        result={'model':'saina-helm-0.8b','answers':{'tags':{'type':'multi_choice','memberships':{'a':.9,'b':.8}}},'usage':{'input_tokens':1,'output_tokens':0}}
        def reply(req, **kwargs):
            self.assertEqual(req.full_url,'https://example.com/v1/ask')
            self.assertEqual(json.loads(req.data),dict(**request,mode='distribution'))
            return io.BytesIO(json.dumps(result).encode())
        c._opener.open.side_effect=reply
        self.assertEqual(c.ask(**request),result)
        for answers in [{}, {'tags':{'type':'yes_no','yes':.9,'no':.1}}, {'tags':{'type':'multi_choice','memberships':{'a':float('nan'),'b':.8}}}]:
            result['answers']=answers
            with self.assertRaises(SainaHelmError): c.ask(**request)

    def test_decision_request(self):
        c=SainaHelm('https://example.com',api_key='test')
        c._opener=Mock()
        result={'model':'saina-helm-0.8b','answers':{'q':{'type':'yes_no','yes':.1,'no':.9,'confidence':.8,'selected':False,'reason':'accepted'}}}
        c._opener.open.side_effect=lambda *a,**k:io.BytesIO(json.dumps(result).encode())
        c.ask(model='saina-helm-0.8b',state='text',mode='decision',threshold=.85,questions={'q':{'type':'yes_no','question':'Yes?'}})
        self.assertEqual(json.loads(c._opener.open.call_args.args[0].data)['threshold'],.85)

    def test_url(self):
        with self.assertRaises(ValueError): SainaHelm('http://example.com',api_key='test')



class IdempotencyKeyTests(unittest.TestCase):
    def test_uuid7_format_and_timestamp(self):
        before = time.time_ns() // 1_000_000
        keys = {new_idempotency_key() for _ in range(100)}
        after = time.time_ns() // 1_000_000
        self.assertEqual(len(keys), 100)
        for key in keys:
            u = uuid.UUID(key)
            self.assertEqual(str(u), key)
            self.assertEqual(u.version, 7)
            self.assertEqual(u.variant, uuid.RFC_4122)
            self.assertTrue(before <= int.from_bytes(u.bytes[:6], 'big') <= after)

    def test_rejects_non_uuid7_key(self):
        with self.assertRaises(ValueError): client().ask(**ASK, idempotency_key=str(uuid.uuid4()))


class RetryTests(unittest.TestCase):
    def test_no_key_no_retry(self):
        c = client(fail(503, error('accounting_unavailable', admitted=True, state='unknown', retry='same_operation')))
        with self.assertRaises(AccountingUnavailable): c.ask(**ASK)
        self.assertEqual(len(c._opener.requests), 1)
        self.assertNotIn('Idempotency-key', c._opener.requests[0].headers)

    def test_retry_same_key_and_body(self):
        c = client(URLError('reset'), fail(503, error('accounting_unavailable', admitted=True, state='unknown')),
                   Reply(RESULT, X_Request_Id='req_9', X_Saina_Credits_Charged='3',
                         X_Saina_Balance='9007199254740993', X_Saina_Price_Version='1', X_Saina_Replayed='true'))
        result, meta = c.ask_with_metadata(**ASK, idempotency_key=KEY)
        self.assertEqual(result, RESULT)
        sent = c._opener.requests
        self.assertEqual(len(sent), 3)
        self.assertEqual({r.get_header('Idempotency-key') for r in sent}, {KEY})
        self.assertEqual(len({r.data for r in sent}), 1)
        self.assertEqual((meta.request_id, meta.credits_charged, meta.balance, meta.price_version, meta.replayed,
                          meta.idempotency_key, meta.attempts), ('req_9', 3, 9007199254740993, '1', True, KEY, 3))
        self.assertIs(c.last_metadata, meta)
        self.assertEqual(c.sleeps, [0.5, 1.0])

    def test_auto_key_generated_once(self):
        c = client(fail(500, raw=b'<html>'), Reply(RESULT), auto_idempotency_key=True)
        self.assertEqual(c.ask(**ASK), RESULT)
        keys = [r.get_header('Idempotency-key') for r in c._opener.requests]
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(uuid.UUID(keys[0]).version, 7)
        self.assertEqual(c.last_metadata.idempotency_key, keys[0])

    def test_terminal_errors_not_retried(self):
        for status, err, cls in [(502, error('inference_failed', admitted=True, state='terminal', retry='new_operation'), InferenceFailed),
                                 (409, error('idempotency_conflict', retry='no'), IdempotencyConflict),
                                 (402, error('insufficient_credits', admitted=False, state='not_admitted', retry='no'), InsufficientCredits),
                                 (503, error('overloaded', admitted=True, retry='new_operation'), Overloaded),
                                 (429, error('overloaded'), Overloaded)]:
            c = client(fail(status, err), Reply(RESULT))
            with self.assertRaises(cls) as caught: c.ask(**ASK, idempotency_key=KEY)
            self.assertEqual(len(c._opener.requests), 1, err['code'])
            self.assertIsInstance(caught.exception, SainaError)
            self.assertEqual(caught.exception.status, status)

    def test_overloaded_retried_when_not_admitted(self):
        c = client(fail(429, error('overloaded', admitted=False, state='not_admitted', retry='same_operation'), Retry_After='2'),
                   fail(503, error('inference_unavailable', admitted=False)), Reply(RESULT))
        self.assertEqual(c.ask(**ASK, idempotency_key=KEY), RESULT)
        self.assertEqual(c.sleeps, [2, 1.0])

    def test_bounded_retries(self):
        c = client(*[fail(409, error('request_in_progress', state='in_progress'), Retry_After='1') for _ in range(4)])
        with self.assertRaises(ERROR_CLASSES['request_in_progress']): c.ask(**ASK, idempotency_key=KEY)
        self.assertEqual(len(c._opener.requests), 3)
        c = client(fail(429, error('rate_limited', admitted=False), Retry_After='120'), Reply(RESULT))
        with self.assertRaises(ERROR_CLASSES['rate_limited']) as caught: c.ask(**ASK, idempotency_key=KEY)
        self.assertEqual(caught.exception.retry_after, 120)
        self.assertEqual(c.sleeps, [])

    def test_error_fields(self):
        suspension = {'types': ['financial'], 'next_step': 'buy_credits'}
        c = client(fail(403, error('account_suspended', admitted=False, state='not_admitted', retry='no', suspension=suspension)))
        with self.assertRaises(AccountSuspended) as caught: c.ask(**ASK)
        e = caught.exception
        self.assertEqual((e.status, e.code, e.request_id, e.admitted, e.state, e.retry, e.suspension),
                         (403, 'account_suspended', 'req_1', False, 'not_admitted', 'no', suspension))
        self.assertIsInstance(e, SainaHelmError)

    def test_legacy_and_unparseable_errors(self):
        c = client(fail(503, raw=json.dumps({'detail': 'Model is busy'}).encode()))
        with self.assertRaises(SainaError) as caught: c.ask(**ASK)
        self.assertEqual((type(caught.exception), caught.exception.status, caught.exception.code), (SainaError, 503, None))
        self.assertIn('Model is busy', str(caught.exception))
        c = client(fail(401, raw=b'nope'))
        with self.assertRaises(SainaError) as caught: c.ask(**ASK)
        self.assertEqual(caught.exception.status, 401)
        c = client(TimeoutError(), TimeoutError(), TimeoutError(), auto_idempotency_key=True)
        with self.assertRaises(SainaConnectionError): c.ask(**ASK)
        self.assertEqual(len(c._opener.requests), 3)


class AccountTests(unittest.TestCase):
    def test_credit_strings_become_ints(self):
        big = '18446744073709551615'
        c = client(Reply({'account_id': 'acc', 'balance': big, 'reserved': '0', 'available': big,
                          'overdraft_allowance': '0', 'debt': '-5', 'suspensions': []}),
                   Reply({'rows': [{'day': '2026-10-01', 'credits': '250000000', 'input_tokens': 7}]}),
                   Reply({'requests': [{'request_id': 'req_1', 'credits_charged': '12', 'balance_after': big}], 'next_cursor': None}))
        self.assertEqual(c.balance(), {'account_id': 'acc', 'balance': 2**64 - 1, 'reserved': 0, 'available': 2**64 - 1,
                                       'overdraft_allowance': 0, 'debt': -5, 'suspensions': []})
        self.assertEqual(c.usage(from_='2026-10-01', to=None, channel='api')['rows'][0]['credits'], 250000000)
        row = c.requests(limit=10, status='settled')['requests'][0]
        self.assertEqual((row['credits_charged'], row['balance_after']), (12, 2**64 - 1))
        urls = [r.full_url for r in c._opener.requests]
        self.assertEqual(urls, ['https://example.com/v1/account/balance',
                                'https://example.com/v1/account/usage?from=2026-10-01&channel=api',
                                'https://example.com/v1/account/requests?limit=10&status=settled'])
        self.assertTrue(all(r.get_method() == 'GET' and r.get_header('Authorization') == 'Bearer test'
                            for r in c._opener.requests))

    def test_default_base_url(self):
        self.assertEqual(SainaHelm(api_key='k').base_url, DEFAULT_BASE_URL)
        self.assertEqual(SainaHelm('https://x.example/', api_key='k').base_url, 'https://x.example')


class LocalServerTests(unittest.TestCase):
    """End to end through urllib against a local HTTP server."""
    def test_retry_over_http(self):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                seen.append((self.path, self.headers['Idempotency-Key'], body))
                if len(seen) == 1:
                    data, status, extra = {'error': error('accounting_unavailable', admitted=True, state='unknown', retry='same_operation'), 'detail': 'x'}, 503, {'Retry-After': '0'}
                else:
                    data, status, extra = {'model': 'm', 'answers': {'a': {'type': 'noul', 'noul': .5}}}, 200, {'X-Saina-Credits-Charged': '5'}
                raw = json.dumps(data).encode()
                self.send_response(status)
                for name, value in {'Content-Type': 'application/json', 'Content-Length': str(len(raw)), **extra}.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(raw)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            c = SainaHelm(f'http://127.0.0.1:{server.server_port}', api_key='test', auto_idempotency_key=True)
            question = {'a': {'type': 'noul', 'instructions': 'Is it?'}}
            result, meta = c.system_one_with_metadata(model='m', state='s', questions=question)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(result['answers']['a']['noul'], .5)
        self.assertEqual(meta.credits_charged, 5)
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0], seen[1])
        self.assertEqual(seen[0][0], '/v1/systemone')

if __name__=='__main__': unittest.main()
