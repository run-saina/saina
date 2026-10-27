"""Dependency-free Saina HTTP client. Does not download model weights."""
import email.utils
import http.client
import json
import math
import os
import ipaddress
import threading
import random
import re
import secrets
import time
import uuid
from dataclasses import dataclass, replace
from urllib.parse import urlencode, urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError as UrllibHTTPError, URLError

from .errors import (
    SainaError, SainaHelmError, SainaConnectionError, HTTPError, AuthenticationError,
    PermissionDeniedError, NotFoundError, InvalidRequestError, RateLimitError, IdempotencyError,
    ServerError, ModelBusyError, TransportError, ProtocolError, SpoolFullError,
    TraceStorageError, InvalidApiKey, KeyRevoked, FreshAuthRequired, PermissionDenied,
    AccountSuspended, NotFound, InvalidRequest, RateLimited, InsufficientCredits,
    IdempotencyConflict, RequestInProgress, IdempotencyResultExpired, IdempotencyUnverifiable, InferenceFailed,
    AccountingUnavailable, Overloaded, InferenceUnavailable, ERROR_CLASSES
)

from .answers import Answers, validate_answers
from .validation import native_request, flatten_extra as _flatten_extra, json_value, json_copy, QUESTION_FIELDS as _QUESTION_FIELDS, REQUEST_FIELDS as _REQUEST_FIELDS

__version__ = '0.2.0'

DEFAULT_BASE_URL = 'https://api.saina.run'

# Every credit amount in the hosted API is a decimal string holding a 64-bit integer.
CREDIT_FIELDS = frozenset({'balance', 'reserved', 'available', 'overdraft_allowance', 'debt', 'credits',
                           'credits_charged', 'balance_after'})

_UUID7 = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$')
_INTEGER = re.compile(r'^-?[0-9]+$')


def new_idempotency_key():
    """Return a fresh RFC 9562 UUIDv7 string (48-bit Unix ms timestamp, then random bits)."""
    raw = bytearray((time.time_ns() // 1_000_000).to_bytes(6, 'big') + secrets.token_bytes(10))
    raw[6] = 0x70 | raw[6] & 0x0F  # version 7
    raw[8] = 0x80 | raw[8] & 0x3F  # variant 10
    return str(uuid.UUID(bytes=bytes(raw)))



def _finite(value):
    try:
        return math.isfinite(value)
    except OverflowError:
        return False

def _status_class(status):
    return {401: AuthenticationError, 403: PermissionDeniedError, 404: NotFoundError,
            422: InvalidRequestError, 429: RateLimitError, 503: ModelBusyError}.get(
                status, ServerError if 500 <= status < 600 else HTTPError)


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
    return int(value) if isinstance(value, str) and _INTEGER.fullmatch(value) else None



def _credit(value):
    if value is None or type(value) is int:
        return value
    parsed = _int(value)
    if parsed is None:
        raise ProtocolError('Invalid credit amount')
    return parsed


def _parse_credits(value):
    """Recursively convert known credit fields from decimal strings to ``int``."""
    if isinstance(value, list):
        return [_parse_credits(v) for v in value]
    if isinstance(value, dict):
        return {k: (_credit(v) if k in CREDIT_FIELDS else _parse_credits(v))
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
        return _status_class(status)(message, status, request_id=request_id, retry_after=retry_after,
                                     raw=body, fields=error if isinstance(error, dict) else None), False
    code = error['code']
    message = error.get('message')
    text = f'Saina HTTP {status} {code}' + (f': {message[:500]}' if isinstance(message, str) and message else '')
    admitted = error.get('admitted')
    return ERROR_CLASSES.get(code, _status_class(status))(
        text, status, code=code, request_id=error.get('request_id') or request_id,
        admitted=admitted if isinstance(admitted, bool) else None, state=error.get('state'),
        retry=error.get('retry'), retry_after=retry_after, suspension=error.get('suspension'),
        raw=body, fields=error), True


def _validate_idempotency_key(key):
    if key is not None and (not isinstance(key, str) or not _UUID7.fullmatch(key.lower())):
        raise ValueError('idempotency_key must be a UUIDv7 string; use new_idempotency_key()')


def _retryable(error):
    if error.code in _RETRY_CODES:
        return True
    return error.code in _RETRY_IF_NOT_ADMITTED and error.admitted is False


class SainaHelm:
    """Saina HTTP client.

    Inference calls are retried automatically only when they carry an ``Idempotency-Key``: pass
    ``idempotency_key=`` per call, or use the default ``auto_idempotency_key=True``. Each logical call gets one new
    UUIDv7 key that every retry of that call reuses with identical body bytes. Read-only account calls
    are retried on the same conditions without a key.
    """

    def __init__(self, base_url=None, *, api_key=None, timeout=None, max_retries=2, max_retry_delay=30,
                 auto_idempotency_key=True, allow_http=False):
        api_key = os.environ.get('SAINA_API_KEY') if api_key is None else api_key
        base_url = os.environ.get('SAINA_BASE_URL', DEFAULT_BASE_URL) if base_url is None else base_url
        if timeout is None:
            timeout = float(os.environ.get('SAINA_TIMEOUT', '60'))
        if not isinstance(base_url, str):
            raise TypeError('base_url must be a string')
        if type(allow_http) is not bool or type(auto_idempotency_key) is not bool:
            raise TypeError('allow_http and auto_idempotency_key must be bools')
        parsed = urlparse(base_url)
        try:
            loopback = ipaddress.ip_address(parsed.hostname or '').is_loopback
        except ValueError:
            loopback = parsed.hostname == 'localhost'
        try:
            parsed.port
        except ValueError:
            raise ValueError('Invalid URL port') from None
        if any(c.isspace() for c in base_url):
            raise ValueError('URL must not contain whitespace')
        if (not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or
                not (parsed.scheme == 'https' or parsed.scheme == 'http' and (loopback or allow_http))):
            raise ValueError('Use HTTPS, or HTTP on localhost; no credentials/query in URL')
        if not isinstance(api_key, str) or not api_key.strip() or any(c in api_key for c in '\r\n'):
            raise ValueError('A nonempty API key is required')
        if type(timeout) not in (int, float):
            raise TypeError('Timeout must be a number')
        if type(max_retry_delay) not in (int, float):
            raise TypeError('max_retry_delay must be a number')
        if not _finite(timeout) or timeout <= 0:
            raise ValueError('Timeout must be positive and finite')
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError('max_retries must be a nonnegative integer')
        if not _finite(max_retry_delay) or max_retry_delay < 0:
            raise ValueError('max_retry_delay must be nonnegative and finite')
        self.base_url, self._key, self.timeout = base_url.rstrip('/'), api_key, timeout
        self.max_retries, self.max_retry_delay = max_retries, max_retry_delay
        self.auto_idempotency_key = auto_idempotency_key
        self._metadata_local = threading.local()
        self._opener = build_opener(_NoRedirect())
        self._sleep, self._random = time.sleep, random.random

    # Inference -----------------------------------------------------------------------------------------

    @property
    def last_metadata(self):
        return getattr(self._metadata_local, 'value', None)

    @last_metadata.setter
    def last_metadata(self, value):
        self._metadata_local.value = value

    def ask(self, *, model, state, questions, images=None, audios=None, specialist=None,
            idempotency_key=None, extra=None):
        """Ask for distributions; apply acceptance requirements locally with select()."""
        return self.ask_with_metadata(model=model, state=state, questions=questions, images=images,
                                      audios=audios, specialist=specialist,
                                      idempotency_key=idempotency_key, extra=extra)[0]

    def ask_with_metadata(self, *, model, state, questions, images=None, audios=None, specialist=None,
                          idempotency_key=None, extra=None):
        _validate_idempotency_key(idempotency_key)
        payload = native_request(model, state, questions, specialist, extra)
        payload.update(_media(images, audios))
        result, meta = self._post('/v1/ask', payload, idempotency_key)
        validate_answers(payload, result)
        body_id = result.get('request_id')
        if body_id is not None and (not isinstance(body_id, str) or not body_id.strip()):
            raise ProtocolError('Invalid request ID')
        if body_id is not None and meta.request_id is not None and body_id != meta.request_id:
            raise ProtocolError('Body and header request IDs disagree')
        request_id = body_id if body_id is not None else meta.request_id
        if not isinstance(request_id, str) or not request_id.strip():
            raise ProtocolError('Typed answers require a request ID')
        meta = replace(meta, request_id=request_id)
        self.last_metadata = meta
        return Answers(result, meta, payload['questions']), meta

    def system_one(self, *, model, state, questions, images=None, audios=None, idempotency_key=None, extra=None, specialist=None):
        """Call ``POST /v1/systemone`` (System One wire format: ``noul``, ``choice``, ``score``); ``images`` and
        ``audios`` as in :meth:`ask`."""
        return self.system_one_with_metadata(model=model, state=state, questions=questions, images=images,
                                             audios=audios, idempotency_key=idempotency_key, extra=extra, specialist=specialist)[0]

    def system_one_with_metadata(self, *, model, state, questions, images=None, audios=None, idempotency_key=None, extra=None, specialist=None):
        _validate_idempotency_key(idempotency_key)
        if not isinstance(questions, dict) or not questions:
            raise ValueError('At least one question is required')
        payload = _flatten_extra(dict(model=model, state=state, questions=_questions(questions),
                                      extra=extra), _REQUEST_FIELDS)
        if specialist is not None:
            payload['specialist'] = specialist
        json_value(payload)
        payload.update(_media(images, audios))
        result, meta = self._post('/v1/systemone', payload, idempotency_key)
        if (not isinstance(result, dict) or not isinstance(result.get('model'), str) or
                not isinstance(result.get('answers'), dict) or set(result['answers']) != set(questions)):
            raise ProtocolError('Invalid System One response')
        return result, meta

    def decisions(self, *, model, input, questions, idempotency_key=None, **extensions):
        """Call ``POST /v1/decisions`` in the OpenAI Decisions request shape: ``input`` (a string or OpenAI input
        items) and a ``questions`` array of ``predicate``, ``choice`` and ``score`` questions. Answers come back in
        question order, in the same shape the official OpenAI SDK reads. Saina extensions are passed as keywords:
        ``mode='decision'`` with ``threshold``/``min_margin`` turns answers the policy does not accept into
        ``refusal`` entries (still billed), and ``specialist`` selects a domain specialist."""
        return self.decisions_with_metadata(model=model, input=input, questions=questions,
                                            idempotency_key=idempotency_key, **extensions)[0]

    def decisions_with_metadata(self, *, model, input, questions, idempotency_key=None, **extensions):
        """Like :meth:`decisions`, returning ``(result, ResponseMetadata)``."""
        _validate_idempotency_key(idempotency_key)
        if not isinstance(questions, (list, tuple)) or not questions:
            raise ValueError('At least one question is required')
        payload = dict(model=model, input=input, questions=[_flatten_extra(q, _QUESTION_FIELDS) for q in questions], **extensions)
        payload = _flatten_extra(payload, _REQUEST_FIELDS)
        result, meta = self._post('/v1/decisions', payload, idempotency_key)
        _validate_decisions(payload, result)
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
        _validate_idempotency_key(idempotency_key)
        json_value(payload)
        body = json.dumps(json_copy(payload), allow_nan=False).encode()  # serialized once: retries send identical bytes
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
        except UrllibHTTPError as exc:
            try:
                raw = exc.read()
            except (OSError, http.client.HTTPException):
                raw = b''
            error, typed = _error(exc.code, raw, exc.headers)
            if _retryable(error) or exc.code >= 500 and not typed:  # untyped 5xx: outcome uncertain
                raise _Uncertain(error, error.retry_after)
            raise error from None
        except (URLError, OSError, http.client.HTTPException) as exc:
            # Connection loss, timeout, or a response cut off: the outcome is unknown.
            raise _Uncertain(TransportError('Saina connection failed or timed out', timed_out=isinstance(exc, TimeoutError) or isinstance(getattr(exc, 'reason', None), TimeoutError))) from None
        try:
            return json.loads(raw), response_headers
        except (ValueError, UnicodeError):
            raise ProtocolError('Invalid JSON response') from None


def _media(images, audios):
    """Request fields for images and audio clips, each converted to a base64 data URL."""
    from .media import audio_data_url, image_data_url
    fields = {}
    for name, items, convert in (('images', images, image_data_url), ('audios', audios, audio_data_url)):
        if items is None:
            continue
        if isinstance(items, (str, bytes, bytearray)) or not hasattr(items, '__iter__'):
            raise ValueError(f'{name} must be a list')
        fields[name] = [convert(v) for v in items]
    return fields


Saina = SainaHelm


def _questions(questions):
    result = {}
    for key, question in questions.items():
        if not isinstance(question, dict):
            raise ValueError('Each question must be a dictionary')
        question = _flatten_extra(question, _QUESTION_FIELDS)
        if isinstance(question.get('options'), list):
            question['options'] = dict.fromkeys(question['options'])
        result[key] = question
    return result


def _validate_decisions(request, result):
    def probability(value):
        return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1

    answers = result.get('answers') if isinstance(result, dict) else None
    if (not isinstance(result.get('model') if isinstance(result, dict) else None, str) or not isinstance(answers, list)
            or len(answers) != len(request['questions'])):
        raise ProtocolError('Invalid Decisions response')
    for q, a in zip(request['questions'], answers):
        if (not isinstance(a, dict) or a.get('name') != q.get('name') or a.get('type') not in (q.get('type'), 'refusal')
                or a['type'] == 'predicate' and not probability(a.get('probability'))
                or a['type'] in ('choice', 'score') and not (isinstance(a.get('probabilities'), list)
                                                             and probability(a.get('confidence')))):
            raise ProtocolError('Invalid Decisions response')
