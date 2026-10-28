"""Regression coverage for the 1.0 client contract amendments."""
import copy
import json
import uuid

import pytest

import saina
from saina.client import _error
from test_client import ASK, RESULT, Reply, client, fail, error, KEY


@pytest.mark.parametrize('status,category', [
    (401, saina.AuthenticationError), (403, saina.PermissionDeniedError),
    (404, saina.NotFoundError), (422, saina.InvalidRequestError),
    (429, saina.RateLimitError), (503, saina.ModelBusyError),
    (500, saina.ServerError), (502, saina.ServerError), (409, saina.HTTPError),
])
@pytest.mark.parametrize('body', [b'not json', b'{"detail":"legacy"}', b'{"error":{"code":"future_code","custom":9}}'])
def test_status_fallback(status, category, body):
    caught, _ = _error(status, body, {})
    assert type(caught) is category


def test_code_precedes_status_and_preserves_fields():
    fields = error('rate_limited', admitted=False, future={'nested': [1, 2]})
    caught, typed = _error(500, json.dumps({'error': fields}), {})
    assert typed and isinstance(caught, saina.RateLimited)
    assert isinstance(caught, saina.RateLimitError)
    assert caught.fields == fields
    assert caught.raw == {'error': fields}
    assert saina.SainaHelmError is saina.SainaError
    assert saina.SainaConnectionError is saina.TransportError
    from saina.errors import RateLimited
    assert RateLimited is saina.RateLimited


@pytest.mark.parametrize('operation', ['ask', 'ask_with_metadata', 'system_one', 'system_one_with_metadata',
                                      'decisions', 'decisions_with_metadata'])
def test_default_key_and_identical_retry_bytes(operation):
    request = ASK
    response = RESULT
    if operation.startswith('system_one'):
        request = dict(model='m', state='s', questions={'q': {'type': 'noul', 'instructions': 'Q?'}})
        response = dict(model='m', answers={'q': {'type': 'noul', 'noul': .9}})
    elif operation.startswith('decisions'):
        request = dict(model='m', input='s', questions=[{'name': 'q', 'type': 'predicate', 'question': 'Q?'}])
        response = dict(model='m', answers=[{'name': 'q', 'type': 'predicate', 'probability': .9}])
    c = client(fail(503, error('accounting_unavailable')), Reply(response))
    getattr(c, operation)(**request)
    sent = c._opener.requests
    assert len(sent) == 2
    assert sent[0].data == sent[1].data
    assert sent[0].get_header('Idempotency-key') == sent[1].get_header('Idempotency-key')
    assert uuid.UUID(sent[0].get_header('Idempotency-key')).version == 7


def test_explicit_key_wins_even_when_generation_disabled():
    for enabled in (True, False):
        c = client(fail(503, error('accounting_unavailable')), Reply(RESULT), auto_idempotency_key=enabled)
        c.ask(**ASK, idempotency_key=KEY)
        assert {r.get_header('Idempotency-key') for r in c._opener.requests} == {KEY}


def test_extensions_flatten_without_mutating_input():
    request = dict(model='m', state='s', questions={'q': {
        'type': 'single_choice', 'question': 'Q?', 'options': ['a', 'b'], 'extra': {'future': True}}},
        extra={'new_request_feature': 2})
    original = copy.deepcopy(request)
    response = dict(request_id='req_1', model='m', answers={'q': dict(type='single_choice', selection='a',
                    probabilities={'a': .9, 'b': .1}, confidence=.8)})
    c = client(Reply(response))
    c.ask(**request)
    wire = json.loads(c._opener.requests[0].data)
    assert wire['questions']['q']['options'] == {'a': None, 'b': None}
    assert wire['questions']['q']['future'] is True
    assert wire['new_request_feature'] == 2
    assert 'extra' not in wire and 'extra' not in wire['questions']['q']
    assert request == original


@pytest.mark.parametrize('field', ['type', 'question', 'options', 'levels', 'threshold', 'min_margin',
                                    'descriptions', 'extra'])
def test_question_collisions_before_network(field):
    c = client()
    request = copy.deepcopy(ASK)
    request['questions']['q']['extra'] = {field: 'override'}
    with pytest.raises(ValueError):
        c.ask(**request)
    assert not c._opener.requests


@pytest.mark.parametrize('field', ['model', 'mode', 'questions', 'images', 'threshold', 'extra'])
def test_top_level_collisions_before_network(field):
    c = client()
    with pytest.raises(ValueError):
        c.ask(**ASK, extra={field: 'override'})
    assert not c._opener.requests


def test_metadata_serialization_and_wire_snapshot():
    raw = dict(RESULT, billing={'balance': '18446744073709551615'})
    c = client(Reply(raw, X_Saina_Balance='18446744073709551615'), Reply(raw, X_Saina_Balance='18446744073709551615'))
    answers = c.ask(**ASK)
    assert answers['q'].to_dict() == RESULT['answers']['q']
    assert answers.metadata.balance == 2**64 - 1
    assert answers.to_dict()['metadata']['balance'] == 2**64 - 1
    assert json.loads(answers.to_json())['metadata']['balance'] == 2**64 - 1
    assert answers.raw['billing']['balance'] == str(2**64 - 1)
    answers.to_dict()['metadata']['balance'] = 0
    compat, metadata = c.ask_with_metadata(**ASK)
    assert compat.metadata is metadata
    assert metadata.balance == answers.metadata.balance
