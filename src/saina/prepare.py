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


# -- billing ------------------------------------------------------------------------------------
# A request is billed for the text the customer sends, not for the prompts built from it: the
# context once, plus each question and its options or levels with their keys and descriptions.
# Prompt formatting and any re-reading of the context are not billed. Model-input lengths are
# still counted separately, for the context-window limit and the gateway/server metering check.

def as_sent(value):
    """Billing text of one field: a string as sent; an object or array as compact JSON in the order sent."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False,
                                                           separators=(',', ':'))


@dataclass(frozen=True)
class Billable:
    context: str
    questions: tuple  # one tuple of text parts per question, in request order


@dataclass(frozen=True)
class Bill:
    context_tokens: int
    question_tokens: tuple  # per question, in request order; the context is not included

    @property
    def total(self):
        return self.context_tokens + sum(self.question_tokens)


def _keyed(mapping):
    return [part for key, value in mapping.items() for part in (key, None if value is None else as_sent(value))
            if part is not None]


def billable_ask(request):
    questions = []
    for q in request.questions.values():
        parts = [as_sent(q.question)]
        if q.type == 'yes_no':
            if q.descriptions:
                parts += [as_sent(q.descriptions.yes), as_sent(q.descriptions.no)]
        elif q.type == 'rating':
            parts += [as_sent(level) for level in q.levels]
        else:
            parts += _keyed(q.options)
        questions.append(tuple(parts))
    return Billable(as_sent(request.state), tuple(questions))


def billable_systemone(request):
    questions = []
    for q in request.questions.values():
        parts, c = [as_sent(q.instructions)], q.criteria
        if isinstance(c, dict):
            parts += _keyed(c)
        elif c is not None:
            parts += [as_sent(v) for v in c if v is not None]
        questions.append(tuple(parts))
    return Billable(as_sent(request.state), tuple(questions))


def bill(billable, counter):
    """Billable input tokens. `counter.count_text` tokenizes one part with the pinned tokenizer;
    each part is counted on its own, so a client with the same tokenizer can reproduce the count."""
    return Bill(counter.count_text(billable.context),
                tuple(sum(counter.count_text(p) for p in parts) for parts in billable.questions))


def check_capabilities(prepared, supported_modes):
    """Reject the whole request before any forward pass if a question needs an untrained mode."""
    if any(q.mode not in supported_modes for q in prepared):
        raise ValueError('Checkpoint does not declare trained multi-label support')


def shared(encoder):
    """Whether this scorer or counter answers every question from one pass over one prompt."""
    return getattr(encoder, 'shared_trunk_pass', False) is True


def encode_all(prepared, encoder):
    """Every model input of the request, encoded before any inference, so a late invalid question fails
    the request early: one input per question, or a single input for all of them on a shared-pass model."""
    if shared(encoder):
        if len({q.context for q in prepared}) != 1:
            raise ValueError('A shared pass needs one context for every question')
        return [list(encoder.encode_shared(prepared[0].context, [(q.question, q.choices, q.mode) for q in prepared]))]
    return [list(encoder.encode(q.context, q.question, q.choices, q.mode)) for q in prepared]


def validate_distribution(p, size, *, independent=False):
    if (len(p) != size or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in p)
            or (not independent and not math.isclose(sum(p), 1, rel_tol=0, abs_tol=1e-5))):
        raise ValueError('Invalid model probability vector')


@dataclass(frozen=True)
class Execution:
    probabilities: list
    input_tokens: list  # model-input length per forward pass, for the metering check
    billable_tokens: int  # what the request is billed for, reported as usage.input_tokens

    @property
    def total_input_tokens(self):
        return sum(self.input_tokens)


def execute(prepared, scorer, billable, encoded=None, expected_input_tokens=None):
    """Run every question and report both counts from what was actually encoded and sent.

    Counts are never read from mutable scorer state: each model-input count is the length of the
    token sequence passed to the model (`scorer.forward` per question, or `scorer.forward_shared` once
    for a shared-pass model), and the billable count comes from the request text.
    `expected_input_tokens` checks model-input tokens. Every model is billed for the request text, however
    many passes it makes over it. A request is all-or-nothing.
    """
    check_capabilities(prepared, getattr(scorer, 'supported_modes', ('single_label',)))
    if encoded is None:
        encoded = encode_all(prepared, scorer)
    billable_tokens = bill(billable, scorer).total
    if len(encoded) != (1 if shared(scorer) else len(prepared)):
        raise MeteringMismatch('Missing encoded input')
    counts = [len(ids) for ids in encoded]
    if expected_input_tokens is not None and sum(counts) != expected_input_tokens:
        raise MeteringMismatch('Encoded input does not match the expected token count')
    if shared(scorer):
        vectors = [list(p) for p in scorer.forward_shared(
            encoded[0], [len(q.choices) for q in prepared], [q.mode for q in prepared])]
        if len(vectors) != len(prepared):
            raise ValueError('Model returned the wrong number of answers')
    else:
        vectors = [list(scorer.forward(ids, len(q.choices), q.mode)) for q, ids in zip(prepared, encoded)]
    for q, p in zip(prepared, vectors):
        validate_distribution(p, len(q.labels), independent=q.mode == 'multi_label')
    return Execution(vectors, counts, billable_tokens)
