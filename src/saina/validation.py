"""Standard-library request validation, before media downloads or inference."""
from collections.abc import Mapping
import math

QUESTION_FIELDS = frozenset({'type', 'question', 'options', 'levels', 'descriptions', 'threshold',
                              'min_margin', 'extra', 'instructions', 'criteria', 'name', 'choices'})
REQUEST_FIELDS = frozenset({'model', 'state', 'questions', 'mode', 'threshold', 'min_margin',
                             'images', 'audios', 'videos', 'specialist', 'extra', 'idempotency_key',
                             'input', 'idempotencyKey'})


def json_value(value, active=None):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('JSON numbers must be finite')
        return
    if not isinstance(value, (Mapping, list, tuple)):
        raise TypeError('Unsupported JSON value')
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ValueError('Cyclic JSON value')
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError('JSON object keys must be strings')
                json_value(item, active)
        else:
            for item in value:
                json_value(item, active)
    finally:
        active.remove(identity)


def flatten_extra(value, known):
    if not isinstance(value, Mapping):
        raise TypeError('Expected a mapping')
    result = dict(value)
    extra = result.pop('extra', None)
    if extra is not None:
        if not isinstance(extra, Mapping):
            raise TypeError('extra must be a mapping')
        if set(extra) & (known | result.keys()):
            raise ValueError('extra cannot override known fields')
        json_value(extra)
        result.update(extra)
    return result


def label(value, name):
    if not isinstance(value, str):
        raise TypeError(f'{name} must be a string')
    if not value.strip():
        raise ValueError(f'{name} must not be blank')



def content(value, name):
    if not isinstance(value, (str, Mapping, list, tuple)):
        raise TypeError(f'{name} must be text or structured content')
    json_value(value)


def native_questions(questions):
    if not isinstance(questions, Mapping):
        raise TypeError('questions must be a mapping')
    if not 1 <= len(questions) <= 256:
        raise ValueError('Expected 1 to 256 questions')
    result = {}
    for key, question in questions.items():
        label(key, 'Question ID')
        if not isinstance(question, Mapping):
            raise TypeError('Each question must be a mapping')
        kind = question.get('type')
        if kind not in ('yes_no', 'single_choice', 'multi_choice', 'rating'):
            raise ValueError('Unsupported question type')
        allowed = {'type', 'question', 'extra'} | ({'descriptions'} if kind == 'yes_no' else
                  {'levels'} if kind == 'rating' else {'options'})
        if set(question) - allowed:
            raise ValueError('Unknown question fields; use extra for extensions and select() for thresholds')
        if 'question' not in question:
            raise ValueError('question is required')
        content(question['question'], 'question')
        q = dict(question)
        if kind in ('single_choice', 'multi_choice'):
            options = q.get('options')
            if not isinstance(options, (Mapping, list, tuple)):
                raise TypeError('options must be a mapping or list')
            if not 2 <= len(options) <= 255:
                raise ValueError('Expected 2 to 255 options')
            labels = list(options)
            for item in labels:
                label(item, 'Option label')
            if len(set(labels)) != len(labels):
                raise ValueError('Option labels must be unique')
            q['options'] = dict(options) if isinstance(options, Mapping) else dict.fromkeys(labels)
            for value in q['options'].values():
                if value is not None:
                    content(value, 'Option description')
        elif kind == 'rating':
            if not isinstance(q.get('levels'), (list, tuple)):
                raise TypeError('levels must be a list')
            if not 2 <= len(q['levels']) <= 10:
                raise ValueError('Expected 2 to 10 levels')
            q['levels'] = list(q['levels'])
            for value in q['levels']:
                content(value, 'Level description')
        elif q.get('descriptions') is not None:
            if not isinstance(q['descriptions'], Mapping) or set(q['descriptions']) != {'yes', 'no'}:
                raise ValueError('descriptions must contain yes and no')
            for value in q['descriptions'].values():
                content(value, 'Description')
        q = flatten_extra(q, QUESTION_FIELDS)
        json_value(q)
        result[key] = q
    return result


def native_request(model, state, questions, specialist, extra):
    label(model, 'model')
    content(state, 'state')
    if len(model) > 64:
        raise ValueError('model exceeds 64 characters')
    if specialist is not None:
        label(specialist, 'specialist')
    result = dict(model=model, state=state, questions=native_questions(questions), mode='distribution', extra=extra)
    if specialist is not None:
        result['specialist'] = specialist
    result = flatten_extra(result, REQUEST_FIELDS)
    json_value(result)
    return json_copy(result)


def json_copy(value):
    """Convert validated mappings/sequences to independent JSON containers."""
    if isinstance(value, Mapping):
        return {k: json_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_copy(v) for v in value]
    return value
