"""Acceptance: the official OpenAI Python SDK, unmodified, against a Saina /v1/decisions endpoint.

    python scripts/acceptance/openai_python.py BASE_URL API_KEY [--omni MODEL] [--text MODEL]
BASE_URL is the API root including /v1 (for example http://127.0.0.1:8090/v1 or https://api.saina.run/v1).
Only base_url and api_key differ from the OpenAI guide's examples. Prints one line per check; exits 1 on failure.
"""
import base64
import io
import sys

import openai
from openai.types import decision as types

checks = []


def check(name, condition, detail=''):
    checks.append((name, bool(condition)))
    print(('PASS ' if condition else 'FAIL ') + name + (f' ({detail})' if detail and not condition else ''))


def png(rgb):
    from PIL import Image
    buffer = io.BytesIO()
    Image.new('RGB', (32, 32), rgb).save(buffer, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def main(base_url, api_key, text_model='saina-helm', omni_model=None):
    client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
    print(f'openai-python {openai.__version__} -> {base_url}')

    # The guide's choice example, model name changed.
    decision = client.decisions.create(
        model=text_model, input='I was charged twice for my order.',
        questions=[{'type': 'choice', 'name': 'department', 'instructions': 'Which department should handle this complaint?',
                    'choices': [{'value': 'billing', 'description': 'Payments, invoices, and refunds.'},
                                {'value': 'technical', 'description': 'Problems using the product.'},
                                {'value': 'shipping', 'description': 'Delivery and tracking.'},
                                {'value': 'other', 'description': 'Requests outside these categories.'}]}])
    answer = decision.answers[0]
    check('choice: parsed into the SDK model', isinstance(decision, types.Decision)
          and isinstance(answer, types.AnswerAnswerResourceChoice), type(answer).__name__)
    check('choice: value is one of the choices', answer.choice in ('billing', 'technical', 'shipping', 'other'))
    check('choice: probabilities list sums to 1',
          abs(sum(p.probability for p in answer.probabilities) - 1) < 1e-6 and len(answer.probabilities) == 4)
    check('choice: confidence in [0, 1]', 0 <= answer.confidence <= 1)
    check('usage: typed, output 0', decision.usage.output_tokens == 0 and decision.usage.total_tokens == decision.usage.input_tokens > 0)
    check('model: release name reported', decision.model.startswith('saina-helm'), decision.model)

    # Predicate and score in one call, one unnamed question, conversation input.
    decision = client.decisions.create(
        model=text_model,
        input=[{'role': 'developer', 'content': 'Triage support tickets.'},
               {'role': 'user', 'content': [{'type': 'input_text', 'text': 'Export fails in Safari but works in Chrome.'},
                                            {'type': 'input_json', 'data': {'browser': 'Safari', 'plan': 'pro'}}]}],
        questions=[{'type': 'predicate', 'name': 'bug', 'instructions': 'Is this a product bug?'},
                   {'type': 'score', 'instructions': 'How severe is this issue?',
                    'levels': [{'label': 'Cosmetic', 'description': 'Appearance only.'},
                               {'label': 'Workaround available'}, {'label': 'Fully blocked'}]}])
    bug, severity = decision.answers
    check('predicate: probability', isinstance(bug, types.AnswerAnswerResourcePredicate) and 0 <= bug.probability <= 1)
    check('score: levels indexed from 0 with labels',
          [(p.value, p.label) for p in severity.probabilities] == [(0, 'Cosmetic'), (1, 'Workaround available'), (2, 'Fully blocked')])
    check('score: weighted average', abs(severity.score - sum(p.value * p.probability for p in severity.probabilities)) < 1e-9)
    check('unnamed question: name is None', severity.name is None)

    # Saina decision policy through extra_body: refusals stay typed for the SDK.
    decision = client.decisions.create(
        model=text_model, input='Maybe billing, maybe not.',
        questions=[{'type': 'predicate', 'name': 'billing', 'instructions': 'Is this about billing?'}],
        extra_body={'mode': 'decision', 'threshold': 0.999})
    refusal = decision.answers[0]
    check('policy: refusal entry', isinstance(refusal, types.AnswerAnswerResourceRefusal) and refusal.name == 'billing',
          type(refusal).__name__)
    check('policy: Saina extension fields readable', getattr(refusal, 'reason', None) == 'below_threshold')

    # Errors map to the SDK's typed exceptions.
    for name, kwargs, error in (
            ('unknown model is NotFoundError', {'model': 'gpt-6-luna'}, openai.NotFoundError),
            ('duplicate names are BadRequestError', {'questions': [{'type': 'predicate', 'name': 'a', 'instructions': 'A?'}] * 2},
             openai.BadRequestError),
            ('box type is BadRequestError', {'questions': [{'type': 'box', 'instructions': 'Find it'}]}, openai.BadRequestError)):
        base = {'model': text_model, 'input': 'x', 'questions': [{'type': 'predicate', 'instructions': 'A?'}]}
        try:
            client.decisions.create(**{**base, **kwargs})
            check(name, False, 'no error')
        except error:
            check(name, True)
        except Exception as e:  # noqa: BLE001
            check(name, False, type(e).__name__)

    image_question = [{'type': 'predicate', 'name': 'red', 'instructions': 'Is the product red?'}]
    image_input = [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'Inspect the product in this photo.'},
                                                {'type': 'input_image', 'image_url': png((220, 0, 0))}]}]
    try:
        client.decisions.create(model=text_model, input=image_input, questions=image_question)
        check('image on a text model is refused', False, 'answered')
    except openai.BadRequestError as e:
        check('image on a text model is refused', 'reads text only' in str(e), str(e))
    if omni_model:
        decision = client.decisions.create(model=omni_model, input=image_input, questions=image_question)
        check('image on Omni: answered', decision.answers[0].type == 'predicate' and decision.model == omni_model)

    failed = [n for n, ok in checks if not ok]
    print(f'{len(checks) - len(failed)}/{len(checks)} checks passed')
    return 1 if failed else 0


if __name__ == '__main__':
    args = sys.argv[1:]
    opts = {}
    while len(args) > 2:
        flag, value = args.pop(2), args.pop(2)
        opts[{'--omni': 'omni_model', '--text': 'text_model'}[flag]] = value
    sys.exit(main(args[0], args[1], **opts))
