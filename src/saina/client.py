"""Dependency-free Saina HTTP client. Does not download model weights."""
import email.utils
import http.client
import json
import math
import random
import re
import secrets
import time
import uuid
from dataclasses import dataclass
from urllib.parse import urlencode, urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError, URLError

__version__ = '0.2.0'

DEFAULT_BASE_URL = 'https://api.saina.run'

# Every credit amount in the hosted API is a decimal string holding a 64-bit integer.
CREDIT_FIELDS = frozenset({'balance', 'reserved', 'available', 'overdraft_allowance', 'debt', 'credits',
                           'credits_charged', 'balance_after'})

_UUID7 = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$')
_INTEGER = re.compile(r'^-?[0-9]+$')
_LOCAL_HOSTS = ('localhost', '127.0.0.1', '::1')


def new_idempotency_key():
    """Return a fresh RFC 9562 UUIDv7 string (48-bit Unix ms timestamp, then random bits)."""
    raw = bytearray((time.time_ns() // 1_000_000).to_bytes(6, 'big') + secrets.token_bytes(10))
    raw[6] = 0x70 | raw[6] & 0x0F  # version 7
    raw[8] = 0x80 | raw[8] & 0x3F  # variant 10
    return str(uuid.UUID(bytes=bytes(raw)))


class SainaHelmError(RuntimeError):
    """A failed Saina call. Fields come from the typed error body when the server sent one.

    ``code``, ``request_id``, ``admitted``, ``state``, ``retry`` and ``suspension`` are ``None`` when absent;
    ``retry_after`` is in seconds.
    """
    code = None

    def __init__(self, message, status=None, *, code=None, request_id=None, admitted=None, state=None,
                 retry=None, retry_after=None, suspension=None):
        super().__init__(message)
        self.status = status
        self.code = code or type(self).code
        self.request_id, self.admitted, self.state = request_id, admitted, state
        self.retry, self.retry_after, self.suspension = retry, retry_after, suspension


class SainaConnectionError(SainaHelmError):
    """The connection failed, timed out, or the response was cut off; the outcome may be unknown."""


def _error_class(code):
    name = ''.join(part.title() for part in code.split('_'))
    return type(name, (SainaHelmError,), {'code': code, '__module__': __name__,
                                          '__doc__': f'Saina error code ``{code}``.'})


ERROR_CLASSES = {code: _error_class(code) for code in (
    'invalid_api_key', 'key_revoked', 'account_suspended', 'insufficient_credits', 'invalid_request',
    'rate_limited', 'overloaded', 'inference_failed', 'inference_unavailable', 'accounting_unavailable',
    'idempotency_conflict', 'request_in_progress', 'idempotency_result_expired', 'idempotency_unverifiable',
    'fresh_auth_required', 'permission_denied', 'not_found')}
InvalidApiKey = ERROR_CLASSES['invalid_api_key']
KeyRevoked = ERROR_CLASSES['key_revoked']
AccountSuspended = ERROR_CLASSES['account_suspended']
InsufficientCredits = ERROR_CLASSES['insufficient_credits']
InvalidRequest = ERROR_CLASSES['invalid_request']
RateLimited = ERROR_CLASSES['rate_limited']
Overloaded = ERROR_CLASSES['overloaded']
InferenceFailed = ERROR_CLASSES['inference_failed']
InferenceUnavailable = ERROR_CLASSES['inference_unavailable']
AccountingUnavailable = ERROR_CLASSES['accounting_unavailable']
IdempotencyConflict = ERROR_CLASSES['idempotency_conflict']
RequestInProgress = ERROR_CLASSES['request_in_progress']
IdempotencyResultExpired = ERROR_CLASSES['idempotency_result_expired']
IdempotencyUnverifiable = ERROR_CLASSES['idempotency_unverifiable']
FreshAuthRequired = ERROR_CLASSES['fresh_auth_required']
PermissionDenied = ERROR_CLASSES['permission_denied']
NotFound = ERROR_CLASSES['not_found']

_RETRY_CODES = {'accounting_unavailable', 'request_in_progress', 'rate_limited'}
_RETRY_IF_NOT_ADMITTED = {'overloaded', 'inference_unavailable'}


@dataclass(frozen=True)
class ResponseMetadata:
    """Billing metadata from response headers; credit fields are ``None`` when the server omits them
    (self-hosted servers, provider-channel traffic)."""
    request_id: str | None
    credits_charged: int | None
    balance: int | None
    price_version: str | None
    replayed: bool
    idempotency_key: str | None
    attempts: int = 1


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class _Uncertain(Exception):
    """Internal: the attempt failed in a way a same-key retry may resolve."""

    def __init__(self, error, retry_after=None):
        self.error, self.retry_after = error, retry_after


def _int(value):
    return int(value) if isinstance(value, str) and _INTEGER.match(value) else None


def _parse_credits(value):
    """Recursively convert known credit fields from decimal strings to ``int``."""
    if isinstance(value, list):
        return [_parse_credits(v) for v in value]
    if isinstance(value, dict):
        return {k: (_int(v) if k in CREDIT_FIELDS and _int(v) is not None else _parse_credits(v))
                for k, v in value.items()}
    return value


def _retry_after(headers):
    value = headers.get('Retry-After') if headers else None
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return int(value)
    try:
        return max(0., email.utils.parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def _error(status, raw, headers):
    """Build an error from an HTTP error response; returns ``(error, typed)``. Old ``{"detail"}`` and
    non-JSON bodies still give a :class:`SainaHelmError` with ``status``."""
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeError):
        body = None
    retry_after = _retry_after(headers)
    request_id = headers.get('X-Request-Id') if headers else None
    error = body.get('error') if isinstance(body, dict) else None
    if not isinstance(error, dict) or not isinstance(error.get('code'), str):
        detail = body.get('detail') if isinstance(body, dict) else None
        message = f'Saina HTTP {status}' + (f': {detail[:500]}' if isinstance(detail, str) and detail else '')
        return SainaHelmError(message, status, request_id=request_id, retry_after=retry_after), False
    code = error['code']
    message = error.get('message')
    text = f'Saina HTTP {status} {code}' + (f': {message[:500]}' if isinstance(message, str) and message else '')
    admitted = error.get('admitted')
    return ERROR_CLASSES.get(code, SainaHelmError)(
        text, status, code=code, request_id=error.get('request_id') or request_id,
        admitted=admitted if isinstance(admitted, bool) else None, state=error.get('state'),
        retry=error.get('retry'), retry_after=retry_after, suspension=error.get('suspension')), True


def _retryable(error):
    if error.code in _RETRY_CODES:
        return True
    return error.code in _RETRY_IF_NOT_ADMITTED and error.admitted is False


class SainaHelm:
    """Saina HTTP client.

    Inference calls are retried automatically only when they carry an ``Idempotency-Key``: pass
    ``idempotency_key=`` per call, or set ``auto_idempotency_key=True`` so each logical call gets one new
    UUIDv7 key that every retry of that call reuses with identical body bytes. Read-only account calls
    are retried on the same conditions without a key.
    """

    def __init__(self, base_url=DEFAULT_BASE_URL, *, api_key, timeout=60, max_retries=2, max_retry_delay=30,
                 auto_idempotency_key=False):
        parsed = urlparse(base_url)
        if (not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or
                not (parsed.scheme == 'https' or parsed.scheme == 'http' and parsed.hostname in _LOCAL_HOSTS)):
            raise ValueError('Use HTTPS, or HTTP on localhost; no credentials/query in URL')
        if not isinstance(api_key, str) or not api_key or any(c in api_key for c in '\r\n'):
            raise ValueError('A nonempty API key is required')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Timeout must be positive and finite')
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError('max_retries must be a nonnegative integer')
        if not math.isfinite(max_retry_delay) or max_retry_delay < 0:
            raise ValueError('max_retry_delay must be nonnegative and finite')
        self.base_url, self._key, self.timeout = base_url.rstrip('/'), api_key, timeout
        self.max_retries, self.max_retry_delay = max_retries, max_retry_delay
        self.auto_idempotency_key = auto_idempotency_key
        self.last_metadata = None
        self._opener = build_opener(_NoRedirect())
        self._sleep, self._random = time.sleep, random.random

    # Inference -----------------------------------------------------------------------------------------

    def ask(self, *, model, state, questions, mode='distribution', idempotency_key=None, **policy):
        """Ask typed questions; policy defaults/overrides require decision mode."""
        return self.ask_with_metadata(model=model, state=state, questions=questions, mode=mode,
                                      idempotency_key=idempotency_key, **policy)[0]

    def ask_with_metadata(self, *, model, state, questions, mode='distribution', idempotency_key=None, **policy):
        """Like :meth:`ask`, returning ``(result, ResponseMetadata)``."""
        if not isinstance(questions, dict) or not questions:
            raise ValueError('At least one typed question is required')
        payload = dict(model=model, state=state, questions=questions, mode=mode, **policy)
        result, meta = self._post('/v1/ask', payload, idempotency_key)
        _validate_answers(payload, result)
        return result, meta

    def system_one(self, *, model, state, questions, idempotency_key=None):
        """Call ``POST /v1/systemone`` (System One wire format: ``noul``, ``choice``, ``score``)."""
        return self.system_one_with_metadata(model=model, state=state, questions=questions,
                                             idempotency_key=idempotency_key)[0]

    def system_one_with_metadata(self, *, model, state, questions, idempotency_key=None):
        if not isinstance(questions, dict) or not questions:
            raise ValueError('At least one question is required')
        payload = dict(model=model, state=state, questions=questions)
        result, meta = self._post('/v1/systemone', payload, idempotency_key)
        if (not isinstance(result, dict) or not isinstance(result.get('model'), str) or
                not isinstance(result.get('answers'), dict) or set(result['answers']) != set(questions)):
            raise SainaHelmError('Invalid System One response')
        return result, meta

    # Account (read-only, API key) ----------------------------------------------------------------------

    def balance(self):
        """``GET /v1/account/balance``; credit fields are ints."""
        return self._get('/v1/account/balance', {})

    def usage(self, **filters):
        """``GET /v1/account/usage`` daily aggregates. Filters: ``from_``, ``to``, ``channel``."""
        return self._get('/v1/account/usage', filters)

    def requests(self, **filters):
        """``GET /v1/account/requests``. Filters: ``cursor``, ``limit``, ``from_``, ``to``, ``key_id``,
        ``channel``, ``status``."""
        return self._get('/v1/account/requests', filters)

    # Transport -----------------------------------------------------------------------------------------

    def _post(self, path, payload, idempotency_key):
        if idempotency_key is None and self.auto_idempotency_key:
            idempotency_key = new_idempotency_key()  # one key per logical call, reused by every retry
        if idempotency_key is not None and (not isinstance(idempotency_key, str) or
                                            not _UUID7.match(idempotency_key.lower())):
            raise ValueError('idempotency_key must be a UUIDv7 string; use new_idempotency_key()')
        body = json.dumps(payload, allow_nan=False).encode()  # serialized once: retries send identical bytes
        headers = {'Content-Type': 'application/json'}
        if idempotency_key is not None:
            headers['Idempotency-Key'] = idempotency_key
        return self._call('POST', path, body, headers, idempotency_key, retries=idempotency_key is not None)

    def _get(self, path, filters):
        query = {}
        for name, value in filters.items():
            if value is None:
                continue
            name = name.rstrip('_')
            query[name] = value.isoformat() if hasattr(value, 'isoformat') else value
        result, _ = self._call('GET', path + ('?' + urlencode(query) if query else ''), None, {}, None,
                               retries=True, record=False)
        return _parse_credits(result)

    def _call(self, method, path, body, headers, idempotency_key, *, retries, record=True):
        headers = dict(headers, Accept='application/json', Authorization='Bearer ' + self._key)
        attempts, waited = 0, 0.
        while True:
            attempts += 1
            try:
                result, response_headers = self._attempt(method, path, body, headers)
                break
            except _Uncertain as failure:
                if not retries or attempts > self.max_retries:
                    raise failure.error from None
                cap = min(8., 0.5 * 2 ** (attempts - 1))
                delay = failure.retry_after if failure.retry_after is not None else self._random() * cap
                if waited + delay > self.max_retry_delay:
                    raise failure.error from None
                waited += delay
                self._sleep(delay)
        get = response_headers.get if response_headers is not None else (lambda name: None)
        meta = ResponseMetadata(
            request_id=get('X-Request-Id'), credits_charged=_int(get('X-Saina-Credits-Charged')),
            balance=_int(get('X-Saina-Balance')), price_version=get('X-Saina-Price-Version'),
            replayed=(get('X-Saina-Replayed') or '').lower() == 'true', idempotency_key=idempotency_key,
            attempts=attempts)
        if record:
            self.last_metadata = meta
        return result, meta

    def _attempt(self, method, path, body, headers):
        request = Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read()
                response_headers = getattr(response, 'headers', None)
        except HTTPError as exc:
            try:
                raw = exc.read()
            except (OSError, http.client.HTTPException):
                raw = b''
            error, typed = _error(exc.code, raw, exc.headers)
            if _retryable(error) or exc.code >= 500 and not typed:  # untyped 5xx: outcome uncertain
                raise _Uncertain(error, error.retry_after)
            raise error from None
        except (URLError, OSError, http.client.HTTPException):
            # Connection loss, timeout, or a response cut off: the outcome is unknown.
            raise _Uncertain(SainaConnectionError('Saina connection failed or timed out')) from None
        try:
            return json.loads(raw), response_headers
        except (ValueError, UnicodeError):
            raise _Uncertain(SainaHelmError('Invalid JSON response')) from None


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
