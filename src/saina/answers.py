"""Immutable typed answers and synchronous, local selection."""
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Literal

from .errors import ProtocolError

SelectionReason = Literal['accepted', 'below_threshold', 'tie', 'below_margin', 'no_options_selected']


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value


def plain(value):
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return deepcopy(value)


class Serializable:
    def to_dict(self):
        return plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), allow_nan=False)


def _require_probability(value, name):
    if type(value) not in (int, float):
        raise TypeError(f'{name} must be a number')
    if not 0 <= value <= 1 or not math.isfinite(value):
        raise ValueError(f'{name} must be finite and in [0, 1]')


class Selectable(Serializable):
    def select(self, *, threshold, min_margin=0.0):
        _require_probability(threshold, 'threshold')
        _require_probability(min_margin, 'min_margin')
        if isinstance(self, MultiChoiceAnswer):
            value = tuple(k for k, p in self.memberships.items() if p >= threshold)
            reason = 'accepted' if value else 'no_options_selected'
            return MultiChoiceSelection(value, bool(value), reason, self, threshold, 0.0)
        vector = ({True: self.yes, False: self.no} if isinstance(self, YesNoAnswer) else self.probabilities)
        ranked = sorted(vector, key=vector.get, reverse=True)
        first, second = ranked[:2]
        lead = vector[first] - vector[second]
        margin = 0.0 if isinstance(self, RatingAnswer) else min_margin
        reason = ('below_threshold' if vector[first] < threshold else 'tie' if lead <= 1e-12 else
                  'below_margin' if lead < margin else 'accepted')
        accepted = reason == 'accepted'
        value = first if accepted else None
        cls = YesNoSelection if isinstance(self, YesNoAnswer) else SingleChoiceSelection
        if isinstance(self, RatingAnswer):
            return RatingSelection(value, accepted, reason, self, threshold, margin,
                                   self.levels[value] if accepted else None)
        return cls(value, accepted, reason, self, threshold, margin)


@dataclass(frozen=True)
class YesNoAnswer(Selectable):
    yes: float
    no: float
    confidence: float
    type: str = 'yes_no'


@dataclass(frozen=True)
class SingleChoiceAnswer(Selectable):
    probabilities: Mapping
    confidence: float
    type: str = 'single_choice'

    def __post_init__(self):
        object.__setattr__(self, 'probabilities', freeze(self.probabilities))


@dataclass(frozen=True)
class RatingAnswer(Selectable):
    probabilities: Mapping
    levels: tuple
    expected_level: float
    confidence: float
    type: str = 'rating'

    def __post_init__(self):
        object.__setattr__(self, 'probabilities', freeze(self.probabilities))
        object.__setattr__(self, 'levels', freeze(self.levels))


@dataclass(frozen=True)
class MultiChoiceAnswer(Selectable):
    memberships: Mapping
    type: str = 'multi_choice'

    def __post_init__(self):
        object.__setattr__(self, 'memberships', freeze(self.memberships))


@dataclass(frozen=True)
class Selection(Serializable):
    value: object
    accepted: bool
    reason: SelectionReason
    answer: object
    threshold: float
    min_margin: float

    def to_dict(self):
        return dict(super().to_dict(), type=self.answer.type)


@dataclass(frozen=True)
class YesNoSelection(Selection):
    pass


@dataclass(frozen=True)
class SingleChoiceSelection(Selection):
    pass


@dataclass(frozen=True)
class RatingSelection(Selection):
    label: object


@dataclass(frozen=True)
class MultiChoiceSelection(Selection):
    pass


@dataclass(frozen=True)
class Usage(Serializable):
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, eq=False)
class Selections(Mapping, Serializable):
    _items: Mapping

    def __post_init__(self):
        object.__setattr__(self, '_items', freeze(self._items))

    def __getitem__(self, key):
        return self._items[key]

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def to_dict(self):
        return {k: v.to_dict() for k, v in self.items()}


@dataclass(frozen=True, eq=False, init=False)
class Answers(Selections):
    request_id: str
    model: str
    specialist: str | None
    usage: Usage
    metadata: object
    _raw: Mapping

    def __init__(self, raw, metadata, questions=None):
        answers = {}
        for key in (questions if questions is not None else raw['answers']):
            item = raw['answers'][key]
            kind = item['type']
            if kind == 'yes_no':
                answer = YesNoAnswer(item['yes'], item['no'], item['confidence'])
            elif kind == 'single_choice':
                vector = item['probabilities']
                labels = questions[key]['options'] if questions is not None else vector
                answer = SingleChoiceAnswer({k: vector[k] for k in labels}, item['confidence'])
            elif kind == 'rating':
                answer = RatingAnswer({int(k): v for k, v in item['probabilities'].items()},
                                      item['levels'], item['expected_level'], item['confidence'])
            else:
                vector = item['memberships']
                labels = questions[key]['options'] if questions is not None else vector
                answer = MultiChoiceAnswer({k: vector[k] for k in labels})
            answers[key] = answer
        usage = raw.get('usage', {})
        for name, value in dict(_items=freeze(answers), request_id=metadata.request_id, model=raw['model'],
                                specialist=raw.get('specialist'), metadata=metadata,
                                usage=Usage(usage.get('input_tokens'), usage.get('output_tokens')),
                                _raw=freeze(deepcopy(raw))).items():
            object.__setattr__(self, name, value)

    @property
    def raw(self):
        return plain(self._raw)

    def select(self, *, threshold, min_margin=0.0):
        _require_probability(threshold, 'threshold')
        _require_probability(min_margin, 'min_margin')
        return Selections({key: answer.select(threshold=threshold, min_margin=min_margin)
                           for key, answer in self.items()})

    def to_dict(self):
        return dict(request_id=self.request_id, model=self.model, specialist=self.specialist,
                    usage=self.usage.to_dict(), metadata=plain(self.metadata),
                    answers={key: answer.to_dict() for key, answer in self.items()})


def validate_answers(request, result):
    def fail():
        raise ProtocolError('Invalid typed answer response')

    def probability(value):
        return type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value)

    def near(actual, expected):
        return type(actual) in (int, float) and expected - 1e-9 <= actual <= expected + 1e-9

    if (not isinstance(result, dict) or not isinstance(result.get('model'), str) or not result['model'] or
            not isinstance(result.get('answers'), dict) or set(result['answers']) != set(request['questions']) or
            result.get('specialist') is not None and not isinstance(result['specialist'], str)):
        fail()
    usage = result.get('usage', {})
    if not isinstance(usage, dict) or any(type(v) is not int or v < 0 for k, v in usage.items()
                                        if k in ('input_tokens', 'output_tokens')):
        fail()
    for key, q in request['questions'].items():
        a = result['answers'][key]
        if not isinstance(a, dict) or a.get('type') != q['type']:
            fail()
        kind = q['type']
        labels = (['yes', 'no'] if kind == 'yes_no' else [str(i) for i in range(len(q['levels']))]
                  if kind == 'rating' else list(q['options']))
        vector = ({'yes': a.get('yes'), 'no': a.get('no')} if kind == 'yes_no' else
                  a.get('memberships' if kind == 'multi_choice' else 'probabilities'))
        if not isinstance(vector, dict) or set(vector) != set(labels) or not all(map(probability, vector.values())):
            fail()
        if kind == 'multi_choice':
            continue
        if abs(sum(vector.values()) - 1) > 1e-5:
            fail()
        p = list(vector.values())
        if kind == 'yes_no':
            confidence = 2 * max(p) - 1
        else:
            n = len(p)
            confidence = (max(p) - 1 / n) / (1 - 1 / n)
            if kind == 'rating':
                # Follow the server's first-maximum distance-weighted formula.
                p = [vector[str(i)] for i in range(n)]
                peak = p.index(max(p))
                mean_distance = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
                confidence = 1 - sum(v * abs(i - peak) for i, v in enumerate(p)) / mean_distance
                if a.get('levels') != q['levels'] or not near(a.get('expected_level'), sum(i*v for i, v in enumerate(p))):
                    fail()
        if not probability(a.get('confidence')) or not near(a['confidence'], max(0., min(1., confidence))):
            fail()
