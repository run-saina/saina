import copy
import json
import threading
from dataclasses import FrozenInstanceError
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType

import pytest

from saina import (Saina, YesNoAnswer, SingleChoiceAnswer, RatingAnswer, MultiChoiceAnswer,
                   YesNoSelection, SingleChoiceSelection, RatingSelection, MultiChoiceSelection,
                   ProtocolError, TransportError)
from test_client import ASK, RESULT, Reply, client


@pytest.mark.parametrize('answer,threshold,margin,value,reason', [
    (YesNoAnswer(.5, .5, 0), .9, 0, None, 'below_threshold'),
    (YesNoAnswer(.5, .5, 0), .4, 0, None, 'tie'),
    (YesNoAnswer(.1, .9, .8), .9, .8, False, 'accepted'),
    (YesNoAnswer(.6, .4, .2), .5, .3, None, 'below_margin'),
    (SingleChoiceAnswer({'a': .9, 'b': .1}, .8), .9, .8, 'a', 'accepted'),
    (SingleChoiceAnswer({'a': .5000000000004, 'b': .4999999999996}, 0), .4, 0, None, 'tie'),
    (RatingAnswer({0: .9, 1: .1}, ['low', 'high'], .1, .8), .9, 1, 0, 'accepted'),
    (RatingAnswer({0: .5, 1: .5}, ['same', 'same'], .5, 0), .4, 1, None, 'tie'),
    (RatingAnswer({0: .5, 1: .5}, ['same', 'same'], .5, 0), .9, 1, None, 'below_threshold'),
    (MultiChoiceAnswer({'b': .9, 'a': .8, 'c': .1}), .8, 1, ('b', 'a'), 'accepted'),
    (MultiChoiceAnswer({'b': .2, 'a': .1}), .9, 0, (), 'no_options_selected'),
])
def test_selection_rules(answer, threshold, margin, value, reason):
    selected = answer.select(threshold=threshold, min_margin=margin)
    assert selected.value == value
    assert selected.reason == reason
    assert selected.accepted == (reason == 'accepted')
    assert selected.answer is answer
    assert selected.min_margin == (0 if isinstance(answer, (RatingAnswer, MultiChoiceAnswer)) else margin)
    expected_type = {YesNoAnswer: YesNoSelection, SingleChoiceAnswer: SingleChoiceSelection,
                     RatingAnswer: RatingSelection, MultiChoiceAnswer: MultiChoiceSelection}[type(answer)]
    assert isinstance(selected, expected_type)
    if isinstance(answer, RatingAnswer):
        assert selected.label == (None if value is None else answer.levels[value])
    serialized = selected.to_dict()
    assert serialized['type'] == answer.type
    assert json.loads(selected.to_json()) == serialized


@pytest.mark.parametrize('bad,error', [(True, TypeError), ('0.9', TypeError), (None, TypeError),
                                     (-.1, ValueError), (1.1, ValueError), (float('nan'), ValueError),
                                     (float('inf'), ValueError)])
def test_selection_validation(bad, error):
    a = YesNoAnswer(.9, .1, .8)
    with pytest.raises(error): a.select(threshold=bad)
    with pytest.raises(error): a.select(threshold=.9, min_margin=bad)
    with pytest.raises(TypeError): a.select()


def test_typed_mapping_immutability_raw_and_serialization():
    request = dict(model='m', state={}, questions={
        'model': {'type': 'single_choice', 'question': 'Q?', 'options': ['a', 'b']},
        'severity': {'type': 'rating', 'question': 'Q?', 'levels': [{'text': 'low'}, {'text': 'high'}]},
        'tags': {'type': 'multi_choice', 'question': 'Q?', 'options': ['b', 'a']}})
    raw = dict(model='m', request_id='r', specialist='s', usage={'input_tokens': 3, 'output_tokens': 0},
               future={'balance': '18446744073709551615'}, answers={
        'model': dict(type='single_choice', probabilities={'a': .9, 'b': .1}, confidence=.8),
        'severity': dict(type='rating', probabilities={'0': .9, '1': .1}, confidence=.8, expected_level=.1,
                         levels=[{'text': 'low'}, {'text': 'high'}]),
        'tags': dict(type='multi_choice', memberships={'b': .9, 'a': .8})})
    c = client(Reply(raw, X_Saina_Balance='18446744073709551615'))
    answers = c.ask(**request)
    assert len(answers) == 3 and list(answers) == ['model', 'severity', 'tags']
    assert answers.model == 'm' and 'model' in answers
    assert dict(answers)['model'] is answers['model']
    assert answers.usage.input_tokens == 3
    assert answers['severity'].probabilities[0] == .9
    assert answers['tags'].select(threshold=.8).value == ('b', 'a')
    assert answers.select(threshold=.9, min_margin=.3)['severity'].value == 0
    assert answers.metadata.balance == 2**64-1
    assert not hasattr(answers['tags'], 'confidence')
    with pytest.raises(TypeError): answers['x'] = None
    with pytest.raises(FrozenInstanceError): answers.model = 'other'
    with pytest.raises(TypeError): answers['model'].probabilities['a'] = 0
    with pytest.raises(TypeError): answers['severity'].levels[0]['text'] = 'changed'
    with pytest.raises(FrozenInstanceError): answers.metadata.balance = 0
    snapshot = answers.raw
    snapshot['future']['balance'] = '0'
    assert answers.raw == raw
    serialized = answers.to_dict()
    serialized['answers']['severity']['levels'][0]['text'] = 'changed'
    assert answers['severity'].levels[0]['text'] == 'low'
    assert json.loads(answers.to_json())['metadata']['balance'] == 2**64-1
    assert '0' in answers.to_dict()['answers']['severity']['probabilities']


@pytest.mark.parametrize('body_id,header_id,valid', [(None,None,False), ('body',None,True),
    (None,'header',True), ('same','same',True), ('body','header',False), ('',None,False), (3,None,False)])
def test_request_ids(body_id, header_id, valid):
    raw = copy.deepcopy(RESULT)
    raw.pop('request_id', None)
    if body_id is not None: raw['request_id'] = body_id
    c = client(Reply(raw, **({'X_Request_Id':header_id} if header_id is not None else {})))
    if valid:
        answers = c.ask(**ASK)
        assert answers.request_id == (body_id or header_id)
        assert c.last_metadata is answers.metadata
    else:
        with pytest.raises(ProtocolError): c.ask(**ASK)
    assert len(c._opener.requests) == 1


@pytest.mark.parametrize('change', [lambda r:r['answers']['q'].update(confidence=.5),
    lambda r:r['answers']['q'].update(yes=.8), lambda r:r['answers']['q'].update(no=float('nan')),
    lambda r:r.update(answers={}), lambda r:r.update(usage={'input_tokens':-1})])
def test_invalid_response_rejected_without_retry(change):
    raw = copy.deepcopy(RESULT)
    change(raw)
    c = client(Reply(raw), Reply(RESULT))
    with pytest.raises(ProtocolError): c.ask(**ASK)
    assert len(c._opener.requests) == 1


def test_rating_derived_fields_and_echo():
    request = dict(model='m', state='s', questions={'q':{'type':'rating','question':'Q?', 'levels':['a','b','c']}})
    raw = dict(model='m', request_id='r', answers={'q':dict(type='rating', probabilities={'0':.1,'1':.2,'2':.7},
               levels=['a','b','c'], expected_level=1.6, confidence=.4)})
    # mean distance from midpoint = 2/3; weighted distance from peak = .4 => confidence .4
    assert client(Reply(raw)).ask(**request)['q'].select(threshold=.7).value == 2
    for update in [dict(confidence=.5), dict(expected_level=1.5), dict(levels=['a','b','wrong'])]:
        bad = copy.deepcopy(raw)
        bad['answers']['q'].update(update)
        with pytest.raises(ProtocolError): client(Reply(bad)).ask(**request)


@pytest.mark.parametrize('update', [dict(questions={'q':{'type':'unknown','question':'Q?'}}),
    dict(questions={'q':{'type':'yes_no','question':'Q?', 'typo': True}}),
    dict(questions={'q':{'type':'single_choice','question':'Q?','options':['a','a']}}),
    dict(questions={'q':{'type':'single_choice','question':'Q?','options':['a',' ']}}),
    dict(questions={'q':{'type':'rating','question':'Q?','levels':['one']}}),
    dict(state={3:'bad'}), dict(state={'n':float('inf')}), dict(extra={'idempotency_key':'override'})])
def test_invalid_requests_fail_before_media_io(update, monkeypatch):
    c = client()
    def download(*a): raise AssertionError('Media I/O should not occur')
    monkeypatch.setattr('saina.media.image_data_url', download)
    request = dict(ASK, images=['https://example.com/image'])
    request.update(update)
    with pytest.raises((TypeError, ValueError)): c.ask(**request)
    assert not c._opener.requests


def test_cyclic_json_and_mapping_inputs():
    cycle = []; cycle.append(cycle)
    with pytest.raises(ValueError): client().ask(**dict(ASK,state=cycle))
    c = client(Reply(RESULT))
    c.ask(**dict(ASK, questions=MappingProxyType(ASK['questions']), state=MappingProxyType({'a':(1,2)})))
    assert json.loads(c._opener.requests[0].data)['state'] == {'a':[1,2]}


def test_environment_configuration_and_http_opt_in(monkeypatch):
    monkeypatch.setenv('SAINA_API_KEY','env-key')
    monkeypatch.setenv('SAINA_BASE_URL','https://env.example')
    monkeypatch.setenv('SAINA_TIMEOUT','12.5')
    c = Saina()
    assert (c.base_url, c.timeout) == ('https://env.example',12.5)
    assert Saina('https://explicit.example', api_key='key', timeout=3).timeout == 3
    with pytest.raises(ValueError): Saina('http://saina:8700')
    assert Saina('http://saina:8700', allow_http=True).base_url == 'http://saina:8700'
    assert Saina('http://127.0.0.2:8700').base_url.startswith('http:')
    monkeypatch.setenv('SAINA_BASE_URL','https://new.example')
    assert Saina().base_url == 'https://new.example'
    assert c.base_url == 'https://env.example'


def test_last_metadata_is_thread_local():
    c = Saina(api_key='key')
    barrier = threading.Barrier(2)
    def attempt(method, path, body, headers):
        name = json.loads(body)['state']
        return dict(RESULT, request_id=name), {}
    c._attempt = attempt
    def work(name):
        a = c.ask(**dict(ASK,state=name))
        barrier.wait(timeout=5)
        assert c.last_metadata is a.metadata
        return a.request_id
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(work, ['first','second'])) == ['first','second']
    assert c.last_metadata is None


def test_transport_timeout_flag_and_protocol_no_retry():
    c = client(TimeoutError(), max_retries=0)
    with pytest.raises(TransportError) as caught: c.ask(**ASK)
    assert caught.value.timed_out is True
    c = client(Reply(b'invalid json'), Reply(RESULT))
    with pytest.raises(ProtocolError): c.ask(**ASK)
    assert len(c._opener.requests) == 1


def test_invalid_key_precedes_media_io(monkeypatch):
    def download(*a): raise AssertionError('Media I/O should not occur')
    monkeypatch.setattr('saina.media.image_data_url', download)
    for key in ['not-a-uuid', '01890a5d-ac96-774b-bcce-b302099a8057\n']:
        with pytest.raises(ValueError):
            client().ask(**ASK, images=['https://example.com/image'], idempotency_key=key)


def test_invalid_account_credit_is_not_a_float():
    with pytest.raises(ProtocolError): client(Reply({'available':1.5})).balance()
