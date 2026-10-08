"""Turn validated requests into exact model inputs, and run them with explicit metering.

Shared by the inference server and the hosted gateway, so both count the same tokens.
No torch import: preparation and counting need only the tokenizer.
"""
import json
import math
from dataclasses import dataclass


class MeteringMismatch(RuntimeError):
    """Encoded inputs do not match the token count the caller was charged for."""


@dataclass(frozen=True)
class PreparedQuestion:
    key: str
    kind: str
    labels: tuple
    context: str
    question: str
    choices: tuple
    mode: str = 'single_label'


def render(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)


def _labelled(labels, values):
    return [label if value is None else f'{label}: {render(value)}' for label, value in zip(labels, values)]


def prepare_ask(request):
    """Model inputs for a native /v1/ask request, one per question, in request order."""
    context, prepared = render(request.state), []
    for key, q in request.questions.items():
        if q.type == 'yes_no':
            yes = q.descriptions.yes if q.descriptions else 'The answer is yes'
            no = q.descriptions.no if q.descriptions else 'The answer is no'
            labels, choices = ['yes', 'no'], [f'Yes: {render(yes)}', f'No: {render(no)}']
        elif q.type == 'rating':
            labels = [str(i) for i in range(len(q.levels))]
            choices = _labelled(labels, q.levels)
        else:
            labels = list(q.options)
            choices = _labelled(labels, q.options.values())
        prepared.append(PreparedQuestion(key, q.type, tuple(labels), context, render(q.question), tuple(choices),
                                         'multi_label' if q.type == 'multi_choice' else 'single_label'))
    return prepared


def prepare_systemone(request):
    """Model inputs for a TypeSafe System One request, one per question, in request order."""
    context, prepared = render(request.state), []
    for key, q in request.questions.items():
        if q.type == 'noul':
            criteria = q.criteria or {}
            labels = ['true', 'false']
            choices = [f'Yes: {render(criteria.get("true", "The answer is yes"))}',
                       f'No: {render(criteria.get("false", "The answer is no"))}']
        elif q.type == 'choice':
            labels = list(q.criteria)
            choices = _labelled(labels, q.criteria.values())
        else:
            labels = [str(i) for i in range(len(q.criteria))]
            choices = _labelled(labels, q.criteria)
        prepared.append(PreparedQuestion(key, q.type, tuple(labels), context, render(q.instructions), tuple(choices)))
    return prepared


def check_capabilities(prepared, supported_modes):
    """Reject the whole request before any forward pass if a question needs an untrained mode."""
    if any(q.mode not in supported_modes for q in prepared):
        raise ValueError('Checkpoint does not declare trained multi-label support')


def encode_all(prepared, encoder):
    """Encode every question before any inference, so a late invalid question fails the request early."""
    return [list(encoder.encode(q.context, q.question, q.choices, q.mode)) for q in prepared]


def validate_distribution(p, size, *, independent=False):
    if (len(p) != size or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in p)
            or (not independent and not math.isclose(sum(p), 1, rel_tol=0, abs_tol=1e-5))):
        raise ValueError('Invalid model probability vector')


@dataclass(frozen=True)
class Execution:
    probabilities: list
    input_tokens: list

    @property
    def total_input_tokens(self):
        return sum(self.input_tokens)


def execute(prepared, scorer, encoded=None, expected_input_tokens=None):
    """Run every question and report usage from the encoded inputs actually fed to the model.

    Usage is never read from mutable scorer state: each question's count is the length of the
    token sequence passed to `scorer.forward`. A request is all-or-nothing.
    """
    check_capabilities(prepared, getattr(scorer, 'supported_modes', ('single_label',)))
    if encoded is None:
        encoded = encode_all(prepared, scorer)
    if len(encoded) != len(prepared):
        raise MeteringMismatch('Missing encoded input')
    counts = [len(ids) for ids in encoded]
    if expected_input_tokens is not None and sum(counts) != expected_input_tokens:
        raise MeteringMismatch('Encoded input does not match the expected token count')
    probabilities = []
    for q, ids in zip(prepared, encoded):
        p = list(scorer.forward(ids, len(q.choices), q.mode))
        validate_distribution(p, len(q.labels), independent=q.mode == 'multi_label')
        probabilities.append(p)
    return Execution(probabilities, counts)
