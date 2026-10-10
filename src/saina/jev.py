"""TypeSafe System One wire adapter; model quality/calibration remain Helm's."""
import json
import math
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from .contract import LEGACY_MODEL_ID, score_questions


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


def evaluate(request, scorer):
    entries = []
    for key, q in request.questions.items():
        if q.type == 'noul':
            criteria = q.criteria or {}
            labels = ['true', 'false']
            choices = [f'Yes: {render(criteria.get("true", "The answer is yes"))}',
                       f'No: {render(criteria.get("false", "The answer is no"))}']
        else:
            labels = list(q.criteria) if q.type == 'choice' else [str(i) for i in range(len(q.criteria))]
            values = list(q.criteria.values()) if q.type == 'choice' else q.criteria
            choices = [label if value is None else f'{label}: {render(value)}'
                       for label, value in zip(labels, values)]
        entries.append((key, q, labels, choices))
    vectors, input_tokens = score_questions(scorer, render(request.state), [
        (render(q.instructions), choices, 'single_label') for _, q, _, choices in entries])
    answers = {}
    for (key, q, labels, _), p in zip(entries, vectors):
        if len(p) != len(labels) or any(not math.isfinite(x) or not 0 <= x <= 1 for x in p) or not math.isclose(sum(p), 1, abs_tol=1e-5):
            raise ValueError('Invalid probability distribution')
        n, peak = len(p), max(range(len(p)), key=p.__getitem__)
        if q.type == 'noul':
            answer = {'type': 'noul', 'noul': p[0]}
        else:
            confidence = (max(p) - 1 / n) / (1 - 1 / n)
            answer = {'type': q.type, 'probabilities': dict(zip(labels, p))}
            if q.type == 'choice':
                answer['choice'] = labels[peak]
            else:
                confidence = 1 - sum(v * abs(i - peak) for i, v in enumerate(p)) / (sum(abs(i - (n - 1) / 2) for i in range(n)) / n)
                answer.update(score=sum(i * v for i, v in enumerate(p)),
                              legend={str(i): render(v) for i, v in enumerate(q.criteria)})
            answer['confidence'] = max(0., min(1., confidence))
        answers[key] = answer
    # Report the model that answered; System One callers send their own model labels.
    return {'model': getattr(scorer, 'model_id', LEGACY_MODEL_ID), 'answers': answers,
            'usage': {'input_tokens': input_tokens, 'output_tokens': 0}}
