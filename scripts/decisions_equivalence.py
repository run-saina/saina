"""Cross-dialect equivalence on real, pinned weights: the same request sent natively (/v1/ask), as System One
(/v1/systemone and /v1/decisions) and in the OpenAI Decisions shape (/v1/decisions) must get the same
probabilities, selections and bill.

This guards the adapter against drifting from the native contract. It is not a quality evaluation: it says
nothing about whether the answers are right, only that every dialect asks the model the same thing.

    python scripts/decisions_equivalence.py run-saina/saina-helm-2-0.8b <revision> [--device cpu] [--out report.json]
"""
import argparse
import json
import platform
import sys
import time

TOLERANCE = 1e-6  # probabilities from one model on one device for the same input; differences mean different inputs

TICKET = 'I was charged twice for my order last week and support has not answered. I need the extra payment back today.'
FIXTURES = [
    # name, state, questions as (key, kind, instructions, options or levels or None)
    ('predicate', TICKET, [('billing', 'predicate', 'Is this about billing?', None)]),
    ('choice_described', TICKET, [('team', 'choice', 'Which team should handle it?',
                                   {'billing': 'Payments, invoices and refunds', 'technical': 'Problems using the product',
                                    'shipping': 'Delivery and tracking', 'other': 'Anything else'})]),
    ('choice_bare', TICKET, [('tone', 'choice', 'What is the tone?', {'calm': None, 'frustrated': None, 'angry': None})]),
    ('choice_many', TICKET, [('code', 'choice', 'Which reason code applies?',
                              {f'R{i:03d}': f'Reason {i}' for i in range(120)})]),
    ('score_two', TICKET, [('urgent', 'score', 'How urgent is it?', ['Can wait', 'Today'])]),
    ('score_ten', TICKET, [('severity', 'score', 'Rate the severity from 0 to 9.', [f'Severity {i}' for i in range(10)])]),
    ('mixed', TICKET, [('billing', 'predicate', 'Is this about billing?', None),
                       ('team', 'choice', 'Which team should handle it?', {'billing': None, 'technical': None, 'other': None}),
                       ('urgency', 'score', 'How urgent is it?', ['Can wait', 'This week', 'Today'])]),
    ('object_state', {'subject': 'Duplicate charge', 'body': TICKET, 'plan': 'pro'},
     [('refund', 'predicate', 'Does the customer ask for a refund?', None)]),
    ('nine_questions', TICKET, [(f'q{i}', 'predicate', f'Is statement {i} about money?', None) for i in range(9)]),
]


def native(state, questions):
    out = {}
    for key, kind, text, extra in questions:
        if kind == 'predicate':
            out[key] = {'type': 'yes_no', 'question': text}
        elif kind == 'choice':
            out[key] = {'type': 'single_choice', 'question': text, 'options': extra}
        else:
            out[key] = {'type': 'rating', 'question': text, 'levels': extra}
    return {'model': 'saina-helm', 'state': state, 'questions': out}


def systemone(state, questions):
    out = {}
    for key, kind, text, extra in questions:
        out[key] = {'type': 'noul' if kind == 'predicate' else kind, 'instructions': text}
        if extra is not None:
            out[key]['criteria'] = extra
    return {'model': 'saina-helm', 'state': state, 'questions': out}


def openai(state, questions):
    if not isinstance(state, str):
        return None  # an OpenAI input is a string or input items, not an arbitrary object
    out = []
    for key, kind, text, extra in questions:
        q = {'type': kind, 'name': key, 'instructions': text}
        if kind == 'choice':
            q['choices'] = [{'value': v} if d is None else {'value': v, 'description': d} for v, d in extra.items()]
        elif kind == 'score':
            q['levels'] = [{'label': level} for level in extra]
        out.append(q)
    return {'model': 'saina-helm', 'input': state, 'questions': out}


def png(rgb, size=(64, 64)):
    import base64
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.new('RGB', size, rgb).save(buffer, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def image_fixtures():
    """For models that read images: OpenAI image parts against the same request with native `images`."""
    red, blue = png((220, 20, 20)), png((20, 20, 220), (48, 80))
    out = []
    for name, images, questions in (
            ('image_one', [red], [('red', 'predicate', 'Is the square in the image red?', None),
                                  ('colour', 'choice', 'What colour is the square?', {'red': None, 'green': None, 'blue': None})]),
            ('image_two', [red, blue], [('second_blue', 'predicate', 'Is the second image blue?', None)])):
        text = 'Inspect the attached product photos.'
        parts = [{'type': 'input_text', 'text': text}] + [{'type': 'input_image', 'image_url': u} for u in images]
        oai = {**openai(text, questions), 'model': 'saina-helm-2-omni-0.8b', 'input': [{'role': 'user', 'content': parts}]}
        nat = {**native([{'role': 'user', 'content': text}], questions), 'model': 'saina-helm-2-omni-0.8b', 'images': images}
        out.append((name, {'native': ('/v1/ask', nat), 'openai': ('/v1/decisions', oai)}, len(questions)))
    return out


def distributions(kind, body):
    """{question: (probabilities in option order, selection)} from any dialect's response."""
    out = {}
    if kind == 'native':
        for key, a in body['answers'].items():
            if a['type'] == 'yes_no':
                out[key] = ([a['yes'], a['no']], a['yes'] >= a['no'])
            elif a['type'] == 'single_choice':
                out[key] = (list(a['probabilities'].values()), a['selection'])
            else:
                out[key] = (list(a['probabilities'].values()), max(range(len(a['probabilities'])),
                                                                     key=lambda i: a['probabilities'][str(i)]))
    elif kind == 'systemone':
        for key, a in body['answers'].items():
            if a['type'] == 'noul':
                out[key] = ([a['noul'], 1 - a['noul']], a['noul'] >= 1 - a['noul'])
            else:
                probs = list(a['probabilities'].values())
                out[key] = (probs, a['choice'] if a['type'] == 'choice' else max(range(len(probs)), key=probs.__getitem__))
    else:
        for a in body['answers']:
            if a['type'] == 'predicate':
                out[a['name']] = ([a['probability'], 1 - a['probability']], a['probability'] >= 1 - a['probability'])
            else:
                probs = [p['probability'] for p in a['probabilities']]
                out[a['name']] = (probs, a['choice'] if a['type'] == 'choice' else max(range(len(probs)), key=probs.__getitem__))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('revision')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out')
    args = parser.parse_args()
    import torch
    from fastapi.testclient import TestClient
    import saina
    from saina.loader import load_scorer
    from saina.server import create_app
    started = time.time()
    scorer = load_scorer(args.checkpoint, revision=args.revision, device=args.device)
    report = {'checkpoint': args.checkpoint, 'revision': args.revision, 'model': scorer.model_id, 'device': args.device,
              'saina': saina.__version__, 'torch': torch.__version__, 'python': platform.python_version(),
              'tolerance': TOLERANCE, 'note': 'equivalence of dialects on pinned weights; not a quality measurement',
              'fixtures': []}
    headers = {'Authorization': 'Bearer k'}
    failures = 0
    with TestClient(create_app(scorer=scorer, api_key='k')) as c:
        cases = [(name, {'native': ('/v1/ask', native(state, questions)),
                         'systemone': ('/v1/systemone', systemone(state, questions)),
                         'systemone_on_decisions': ('/v1/decisions', systemone(state, questions)),
                         'openai': ('/v1/decisions', openai(state, questions))}, len(questions))
                 for name, state, questions in FIXTURES]
        if 'image' in getattr(scorer, 'inputs', ()):
            cases += image_fixtures()
        for name, calls, count in cases:
            results = {}
            for label, (route, body) in calls.items():
                if body is None:
                    continue
                r = c.post(route, json=body, headers=headers)
                if r.status_code != 200:
                    raise SystemExit(f'{name} {label}: {r.status_code} {r.text[:300]}')
                results[label] = (r.json()['usage']['input_tokens'], distributions(
                    'native' if label == 'native' else 'openai' if label == 'openai' else 'systemone', r.json()))
            base_tokens, base = results['native']
            worst, same_selection, same_bill = 0., True, True
            for label, (tokens, dists) in results.items():
                same_bill &= tokens == base_tokens
                for key, (probs, selection) in dists.items():
                    ref_probs, ref_selection = base[key]
                    worst = max(worst, max(abs(a - b) for a, b in zip(probs, ref_probs)))
                    same_selection &= (selection == ref_selection) or (str(selection) == str(ref_selection))
            ok = worst <= TOLERANCE and same_selection and same_bill
            failures += not ok
            report['fixtures'].append({'name': name, 'dialects': list(results), 'questions': count,
                                       'billed_input_tokens': base_tokens, 'max_abs_probability_difference': worst,
                                       'identical_selections': same_selection, 'identical_bill': same_bill, 'pass': ok})
            print(f"{'PASS' if ok else 'FAIL'} {name}: {len(results)} dialects, max |dp| = {worst:.2e}, "
                  f'bill {base_tokens} tokens', flush=True)
    report['seconds'] = round(time.time() - started, 1)
    report['passed'] = len(cases) - failures
    print(f"{report['passed']}/{len(cases)} fixtures equivalent within {TOLERANCE}")
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(report, f, indent=1)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
