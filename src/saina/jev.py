"""TypeSafe System One wire adapter; model quality/calibration remain Helm's."""
import json
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from .prepare import execute, prepare_systemone


def render(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)


class Question(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: Literal['choice', 'score', 'noul']
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    criteria: dict[str, JsonValue] | list[JsonValue] | None = None

    @model_validator(mode='after')
    def valid(self):
        c = self.criteria
        if self.type == 'choice' and (not isinstance(c, dict) or not 2 <= len(c) <= 255):
            raise ValueError('Choice requires 2–255 options')
        if self.type == 'score' and (not isinstance(c, list) or not 2 <= len(c) <= 10):
            raise ValueError('Score requires 2–10 levels')
        if self.type == 'noul' and c is not None and (not isinstance(c, dict) or set(c) - {'true', 'false'}):
            raise ValueError('Noul criteria must contain true/false descriptions')
        values = c.values() if isinstance(c, dict) else c or []
        if any(v is not None and not isinstance(v, (str, dict, list)) for v in values):
            raise ValueError('Descriptions must be strings, objects or arrays')
        if isinstance(c, dict) and any(not k.strip() for k in c):
            raise ValueError('Option keys must not be blank')
        return self


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str = Field(min_length=1)
    state: str | dict[str, JsonValue] | list[JsonValue]
    questions: Annotated[dict[str, Question], Field(min_length=1, max_length=256)]


def answer(q, labels, p):
    n, peak = len(p), max(range(len(p)), key=p.__getitem__)
    if q.type == 'noul':
        return {'type': 'noul', 'noul': p[0]}
    confidence = (max(p) - 1 / n) / (1 - 1 / n)
    result = {'type': q.type, 'probabilities': dict(zip(labels, p))}
    if q.type == 'choice':
        result['choice'] = labels[peak]
    else:
        confidence = 1 - sum(v * abs(i - peak) for i, v in enumerate(p)) / (sum(abs(i - (n - 1) / 2) for i in range(n)) / n)
        result.update(score=sum(i * v for i, v in enumerate(p)),
                      legend={str(i): render(v) for i, v in enumerate(q.criteria)})
    result['confidence'] = max(0., min(1., confidence))
    return result


def respond(request, prepared, execution):
    answers = {q.key: answer(request.questions[q.key], list(q.labels), p)
               for q, p in zip(prepared, execution.probabilities)}
    return {'model': 'saina-helm-0.8b', 'answers': answers,
            'usage': {'input_tokens': execution.total_input_tokens, 'output_tokens': 0}}


def evaluate(request, scorer, expected_input_tokens=None):
    prepared = prepare_systemone(request)
    return respond(request, prepared, execute(prepared, scorer, expected_input_tokens=expected_input_tokens))
