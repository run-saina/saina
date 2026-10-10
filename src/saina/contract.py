"""The native typed-question contract shared by HTTP and local integrations."""
import json
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from .errors import ModelNotServed
from .prepare import billable_ask, execute, prepare_ask, validate_distribution  # noqa: F401 (re-exported)

Content = str | dict[str, JsonValue] | list[JsonValue]
Probability = Annotated[float, Field(ge=0, le=1, strict=True)]
LEGACY_MODEL_ID = 'saina-helm-0.8b'


def short_name(model_id):
    return model_id.removeprefix('saina-')


def model_names(model_id):
    """Names that select this model. Unversioned `saina-helm` means whichever Helm the
    endpoint serves; versioned names only ever reach that exact model."""
    return {model_id, short_name(model_id), 'saina-helm'}


def served_model(scorer, requested):
    model_id = getattr(scorer, 'model_id', LEGACY_MODEL_ID)
    if requested not in model_names(model_id):
        raise ModelNotServed(f'This endpoint serves {model_id}')
    return model_id


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Descriptions(StrictModel):
    yes: Content
    no: Content


class BaseQuestion(StrictModel):
    question: Content
    threshold: Probability = .8


class YesNo(BaseQuestion):
    type: Literal['yes_no']
    descriptions: Descriptions | None = None
    min_margin: Probability = 0.


class ChoiceQuestion(BaseQuestion):
    options: Annotated[dict[str, Content | None], Field(min_length=2, max_length=255)]

    @model_validator(mode='after')
    def valid_options(self):
        if any(not key.strip() for key in self.options):
            raise ValueError('Option keys must not be blank')
        return self


class SingleChoice(ChoiceQuestion):
    type: Literal['single_choice']
    min_margin: Probability = 0.


class MultiChoice(ChoiceQuestion):
    type: Literal['multi_choice']


class Rating(BaseQuestion):
    type: Literal['rating']
    levels: Annotated[list[Content], Field(min_length=2, max_length=10)]


TypedQuestion = Annotated[YesNo | SingleChoice | Rating | MultiChoice, Field(discriminator='type')]


class AskRequest(StrictModel):
    model: Annotated[str, Field(min_length=1, max_length=64)]
    state: Content
    mode: Literal['distribution', 'decision'] = 'distribution'
    threshold: Probability = .8
    min_margin: Probability = 0.
    questions: Annotated[dict[str, TypedQuestion], Field(min_length=1, max_length=256)]

    @model_validator(mode='after')
    def valid_policy(self):
        if self.mode == 'distribution' and any(
                {'threshold', 'min_margin'} & obj.model_fields_set
                for obj in [self, *self.questions.values()]):
            raise ValueError('Threshold and margin require decision mode')
        return self


class BaseAnswer(StrictModel):
    reason: Literal['accepted', 'below_threshold', 'below_margin', 'tie'] | None = None


class YesNoAnswer(BaseAnswer):
    type: Literal['yes_no']
    yes: Probability
    no: Probability
    confidence: Probability
    selected: bool | None = None


class SingleChoiceAnswer(BaseAnswer):
    type: Literal['single_choice']
    selection: str | None
    probabilities: dict[str, Probability]
    confidence: Probability


class RatingAnswer(BaseAnswer):
    type: Literal['rating']
    expected_level: Annotated[float, Field(ge=0, le=9)]
    levels: list[Content]
    probabilities: dict[str, Probability]
    confidence: Probability
    level: int | None = None


class MultiChoiceAnswer(BaseAnswer):
    type: Literal['multi_choice']
    memberships: dict[str, Probability]
    selections: list[str] | None = None


class Usage(StrictModel):
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Literal[0]


class AskResponse(StrictModel):
    model: str
    answers: dict[str, Annotated[YesNoAnswer | SingleChoiceAnswer | RatingAnswer | MultiChoiceAnswer,
                                Field(discriminator='type')]]
    usage: Usage


def render(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)


def question_options(q):
    if q.type == 'yes_no':
        yes = q.descriptions.yes if q.descriptions else 'The answer is yes'
        no = q.descriptions.no if q.descriptions else 'The answer is no'
        return ['yes', 'no'], [f'Yes: {render(yes)}', f'No: {render(no)}']
    if q.type == 'rating':
        labels, values = [str(i) for i in range(len(q.levels))], q.levels
    else:
        labels, values = list(q.options), list(q.options.values())
    return labels, [label if value is None else f'{label}: {render(value)}'
                    for label, value in zip(labels, values)]


def decision_reason(p, threshold, min_margin=0):
    top, second = sorted(p, reverse=True)[:2]
    margin = top-second
    return ('tie' if margin <= 1e-12 else 'below_threshold' if top < threshold
            else 'below_margin' if margin < min_margin else 'accepted')


def answer(request, q, labels, p):
    """Typed answer for one question from its validated probability vector."""
    independent = q.type == 'multi_choice'
    n, peak = len(p), max(range(len(p)), key=p.__getitem__)
    result = {'type': q.type}
    if independent:
        result['memberships'] = dict(zip(labels, p))
    elif q.type == 'yes_no':
        result.update(yes=p[0], no=1-p[0], confidence=2*max(p[0], 1-p[0])-1)
    else:
        result['probabilities'] = dict(zip(labels, p))
        confidence = (max(p)-1/n)/(1-1/n)
        if q.type == 'single_choice':
            result['selection'] = labels[peak]
        else:
            mean_distance = sum(abs(i-(n-1)/2) for i in range(n))/n
            confidence = 1-sum(v*abs(i-peak) for i, v in enumerate(p))/mean_distance
            result.update(expected_level=sum(i*v for i, v in enumerate(p)), levels=q.levels)
        result['confidence'] = max(0., min(1., confidence))
    if request.mode == 'decision':
        threshold = q.threshold if 'threshold' in q.model_fields_set else request.threshold
        if independent:
            selections = [label for label, v in zip(labels, p) if v >= threshold]
            result.update(selections=selections, reason='accepted' if selections else 'below_threshold')
        else:
            margin = (q.min_margin if 'min_margin' in q.model_fields_set else request.min_margin) if q.type != 'rating' else 0.
            reason = decision_reason(p, threshold, margin)
            selected = reason == 'accepted'
            result['reason'] = reason
            if q.type == 'yes_no':
                result['selected'] = (peak == 0) if selected else None
            elif q.type == 'single_choice':
                result['selection'] = labels[peak] if selected else None
            else:
                result['level'] = peak if selected else None
    return result


def respond(request, prepared, execution, model_id=LEGACY_MODEL_ID):
    """Response body from an all-or-nothing execution; usage is the billable input."""
    answers = {q.key: answer(request, request.questions[q.key], list(q.labels), p)
               for q, p in zip(prepared, execution.probabilities)}
    return {'model': short_name(model_id), 'answers': answers,
            'usage': {'input_tokens': execution.billable_tokens, 'output_tokens': 0}}


def ask(request: AskRequest, scorer, expected_input_tokens=None):
    model_id = served_model(scorer, request.model)
    # Encode and check capability for every question before any forward pass in a mixed request.
    prepared = prepare_ask(request)
    return respond(request, prepared, execute(prepared, scorer, billable_ask(request),
                                              expected_input_tokens=expected_input_tokens), model_id)
