"""Acceptance: the official TypeSafe Python SDK, unmodified, against Saina's System One route.

    python scripts/acceptance/typesafe_python.py BASE_URL API_KEY     (BASE_URL is the API root, without /v1)
The SDK always posts to {BASE_URL}/v1/systemone, so it cannot be pointed at /v1/decisions without changes.
"""
import sys

import typesafe_sdk
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

checks = []


def check(name, condition, detail=''):
    checks.append(bool(condition))
    print(('PASS ' if condition else 'FAIL ') + name + (f' ({detail})' if detail and not condition else ''))


def ask(client, **kw):
    return client.system_one(state={'document': 'I was charged twice. Please fix this ASAP.'}, questions={
        'billing': Noul(instructions='Is this ticket about billing?'),
        'tone': Choice(instructions="What is the customer's tone?", criteria={'calm': None, 'frustrated': None, 'angry': None}),
        'urgency': Score(instructions='How urgent is this ticket?', criteria=['can wait', 'this week', 'today'])}, **kw)


def main(base_url, api_key):
    print(f'typesafe-sdk {typesafe_sdk.__version__} -> {base_url}')
    for label, options in (('default compression', {}), ('compression off', {'compression': False})):
        with TypeSafeClient(api_key=api_key, base_url=base_url, model='saina-helm', **options) as client:
            try:
                r = ask(client)
                check(f'{label}: typed answers', 0 <= r.nouls['billing'].noul <= 1 and r.choices['tone'].choice in
                      ('calm', 'frustrated', 'angry') and 0 <= r.scores['urgency'].score <= 2)
            except Exception as e:  # noqa: BLE001
                check(f'{label}: typed answers', False, f'{type(e).__name__}: {e}')
            try:
                big = client.system_one(state='x ' * 2000, questions={'b': Noul(instructions='Billing?')})
                check(f'{label}: body over 1 KiB', 0 <= big.nouls['b'].noul <= 1)
            except Exception as e:  # noqa: BLE001
                check(f'{label}: body over 1 KiB', False, f'{type(e).__name__}: {str(e)[:120]}')
    with TypeSafeClient(api_key=api_key, base_url=base_url, compression=False) as client:
        try:
            r = ask(client)  # SDK default model jev-latest: /v1/systemone answers with Saina's default model
            check('default model jev-latest on /v1/systemone', r.model.startswith('saina-helm'), r.model)
        except Exception as e:  # noqa: BLE001
            check('default model jev-latest on /v1/systemone', False, f'{type(e).__name__}: {e}')
    print(f'{sum(checks)}/{len(checks)} checks passed')
    return 0 if all(checks) else 1


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:3]))
