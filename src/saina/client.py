"""Dependency-free Saina HTTP client. Does not download model weights."""
import json
import math
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError, URLError

__version__ = '0.1.0'


class SainaHelmError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class SainaHelm:
    def __init__(self, base_url, *, api_key, timeout=60):
        parsed = urlparse(base_url)
        if (not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or
                not (parsed.scheme == 'https' or parsed.scheme == 'http' and
                     parsed.hostname in ('localhost', '127.0.0.1', '::1'))):
            raise ValueError('Use HTTPS, or HTTP on localhost; no credentials/query in URL')
        if not isinstance(api_key, str) or not api_key or any(c in api_key for c in '\r\n'):
            raise ValueError('A nonempty API key is required')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Timeout must be positive and finite')
        self.base_url, self._key, self.timeout = base_url.rstrip('/'), api_key, timeout
        self._opener = build_opener(_NoRedirect())

    def ask(self, *, model, state, questions, mode='distribution', **policy):
        """Ask typed questions; policy defaults/overrides require decision mode."""
        if not isinstance(questions, dict) or not questions:
            raise ValueError('At least one typed question is required')
        payload = dict(model=model, state=state, questions=questions, mode=mode, **policy)
        body = json.dumps(payload, allow_nan=False).encode()
        request = Request(self.base_url + '/v1/ask', data=body, headers={
            'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self._key})
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            raise SainaHelmError(f'Saina HTTP {exc.code}', exc.code) from None
        except (URLError, TimeoutError):
            raise SainaHelmError('Saina connection failed or timed out') from None
        except (ValueError, UnicodeError):
            raise SainaHelmError('Invalid JSON response') from None
        _validate_answers(payload, result)
        return result


def _validate_answers(request, result):
    def probability(value):
        return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1

    def fail():
        raise SainaHelmError('Invalid typed answer response')

    if (not isinstance(result, dict) or not isinstance(result.get('model'), str) or
            not isinstance(result.get('answers'), dict) or
            set(result['answers']) != set(request['questions'])):
        fail()
    for key, q in request['questions'].items():
        a = result['answers'][key]
        kind = q['type']
        if not isinstance(a, dict) or a.get('type') != kind:
            fail()
        labels = (['yes', 'no'] if kind == 'yes_no' else
                  [str(i) for i in range(len(q['levels']))] if kind == 'rating' else list(q['options']))
        vector = ({'yes': a.get('yes'), 'no': a.get('no')} if kind == 'yes_no' else
                  a.get('memberships' if kind == 'multi_choice' else 'probabilities'))
        if not isinstance(vector, dict) or set(vector) != set(labels) or not all(probability(v) for v in vector.values()):
            fail()
        if kind != 'multi_choice' and (not math.isclose(sum(vector.values()), 1, rel_tol=0, abs_tol=1e-5) or not probability(a.get('confidence'))):
            fail()
        if kind == 'single_choice' and not (a.get('selection') in labels or request['mode'] == 'decision' and 'selection' in a and a['selection'] is None):
            fail()
        if kind == 'rating':
            level = a.get('expected_level')
            if (type(level) not in (int, float) or not math.isfinite(level) or not 0 <= level <= len(labels)-1 or
                    not isinstance(a.get('levels'), list) or len(a['levels']) != len(labels)):
                fail()
        if request['mode'] == 'decision':
            if a.get('reason') not in ('accepted', 'below_threshold', 'below_margin', 'tie'):
                fail()
            if kind == 'yes_no' and ('selected' not in a or not (a['selected'] is None or type(a['selected']) is bool)):
                fail()
            if kind == 'rating' and ('level' not in a or not (a['level'] is None or type(a['level']) is int and 0 <= a['level'] < len(labels))):
                fail()
            if kind == 'multi_choice':
                selections = a.get('selections')
                if (not isinstance(selections, list) or not all(isinstance(k, str) and k in labels for k in selections) or
                        len(set(selections)) != len(selections)):
                    fail()

Saina = SainaHelm
SainaError = SainaHelmError
