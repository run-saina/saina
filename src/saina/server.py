"""Authenticated inference API serving one or more Helm models through a bounded queue.
Run with uvicorn's factory option.

Concurrency: requests wait in a bounded FIFO queue for a fixed number of execution slots
(default one). A full queue rejects promptly with `429 overloaded`; a queued request that
waits longer than the queue timeout is dropped before it runs. A slot is released only when
its model work has actually finished, never when a client disconnects. Every served model shares
the same execution slots.
"""
import asyncio
from concurrent.futures import CancelledError as JobCancelled, ThreadPoolExecutor
from contextlib import asynccontextmanager
import hmac
import os
import threading
import time
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from . import __version__
from .errors import InputTooLong, ModelNotServed
from .loader import load_scorer
from .contract import (LEGACY_MODEL_ID, AskRequest, AskResponse, billable_ask, model_names, prepare_ask,
                       respond as respond_ask)
from .jev import SystemOneRequest, billable_systemone, prepare_systemone, respond as respond_systemone
from .prepare import MeteringMismatch, check_capabilities, encode_all, execute


class Overloaded(Exception):
    pass


class QueueExpired(Exception):
    pass


def error(status, code, message, headers=None, extra=None):
    # `detail` keeps older clients that read FastAPI's default shape working.
    return JSONResponse(status_code=status, headers=headers,
                        content={'error': {'code': code, 'message': message, **(extra or {})}, 'detail': message})


class ApiError(Exception):
    def __init__(self, status, code, message, headers=None, extra=None):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers
        self.extra = extra


# Work rejected or dropped by the queue never ran and nothing was recorded, so the same request can be
# sent again after Retry-After. Same fields as the hosted gateway's error contract, so clients retry it.
NOT_ADMITTED = {'admitted': False, 'state': 'not_admitted', 'retry': 'same_operation'}


class BoundedExecutor:
    """Hard limit of `slots` concurrent model executions plus `depth` waiting requests."""

    def __init__(self, slots=1, depth=4, queue_timeout=30.0):
        if slots < 1 or depth < 0 or queue_timeout <= 0:
            raise ValueError('Invalid queue configuration')
        self.slots, self.depth, self.queue_timeout = slots, depth, queue_timeout
        self._pool = ThreadPoolExecutor(max_workers=slots, thread_name_prefix='saina-exec')
        self._lock = threading.Lock()
        self.admitted = 0
        self.running = 0
        self._jobs = {}

    def stats(self):
        with self._lock:
            return {'slots': self.slots, 'depth': self.depth, 'running': self.running,
                    'queued': self.admitted - self.running}

    def reserve(self):
        """Take one place in the queue, or raise `Overloaded`. Pair with `submit(..., reserved=True)` or `release()`."""
        with self._lock:
            if self.admitted >= self.slots + self.depth:
                raise Overloaded()
            self.admitted += 1

    def release(self):
        """Give back a place taken by `reserve()` that will not be submitted."""
        with self._lock:
            self.admitted -= 1

    def submit(self, fn, request_id=None, deadline=None, reserved=False):
        enqueued = time.monotonic()
        expires = enqueued + self.queue_timeout
        if deadline is not None:
            expires = min(expires, enqueued + max(0., deadline - time.time()))
        timing = {}

        def run():
            started = time.monotonic()
            if started > expires:
                raise QueueExpired()
            timing['queue_ms'] = int((started - enqueued) * 1000)
            with self._lock:
                self.running += 1
            try:
                return fn()
            finally:
                timing['exec_ms'] = int((time.monotonic() - started) * 1000)
                with self._lock:
                    self.running -= 1

        with self._lock:
            if not reserved:
                if self.admitted >= self.slots + self.depth:
                    raise Overloaded()
                self.admitted += 1
            try:
                future = self._pool.submit(run)
            except BaseException:
                self.admitted -= 1
                raise
            if request_id:
                self._jobs[request_id] = future

        def done(_):
            # Runs when the work has really finished (or was cancelled before starting).
            with self._lock:
                self.admitted -= 1
                if request_id and self._jobs.get(request_id) is future:
                    del self._jobs[request_id]
        future.add_done_callback(done)
        return future, timing

    def cancel(self, request_id):
        with self._lock:
            future = self._jobs.get(request_id)
        if future is None:
            return 'unknown'
        return 'cancelled' if future.cancel() else ('finished' if future.done() else 'running')

    def shutdown(self):
        self._pool.shutdown(wait=True, cancel_futures=True)


def _keys(value):
    return [k.strip().encode() for k in (value or '').split(',') if k.strip()]


# Public listing details per released model (OpenRouter provider format).
CATALOG = {
    'saina-helm-0.8b': {'name': 'Saina: Helm 0.8B', 'created': 1791331200,
                        'hugging_face_id': 'run-saina/saina-helm-0.8b'},
    'saina-helm-2-0.8b': {'name': 'Saina: Helm 2 0.8B', 'created': 1791590400,
                          'hugging_face_id': 'run-saina/saina-helm-2-0.8b'},
}


def model_id(scorer):
    return getattr(scorer, 'model_id', LEGACY_MODEL_ID)


def parse_models(spec):
    """SAINA_MODELS: comma-separated `checkpoint@revision[:max_length]`; the first is the default.

    `checkpoint` is a Hub repo (revision required) or a local staged directory."""
    models = []
    for item in (part.strip() for part in spec.split(',')):
        if not item:
            continue
        checkpoint, _, rest = item.partition('@')
        revision, _, limit = rest.partition(':')
        if not checkpoint or (limit and not limit.isdigit()):
            raise ValueError(f'Invalid SAINA_MODELS entry: {item!r}')
        models.append((checkpoint, revision or None, int(limit) if limit else None))
    if not models:
        raise ValueError('SAINA_MODELS lists no models')
    return models


def load_models(loader=load_scorer):
    env = os.environ.get
    device = env('SAINA_DEVICE', 'cpu')
    if env('SAINA_MODELS'):
        return [loader(c, revision=r, device=device, max_length=m) for c, r, m in parse_models(env('SAINA_MODELS'))]
    # Without SAINA_MAX_LENGTH the checkpoint's own context limit applies.
    return [loader(env('SAINA_CHECKPOINT'), revision=env('SAINA_REVISION'), device=device,
                   max_length=int(env('SAINA_MAX_LENGTH')) if env('SAINA_MAX_LENGTH') else None)]


def create_app(scorer=None, api_key=None, cors_origins=None, service_token=None,
               execution_slots=None, queue_depth=None, queue_timeout=None, scorer_loader=None, scorers=None):
    env = os.environ.get
    # One key per self-hosted caller, comma-separated, so a key can be revoked without rotating the rest.
    keys = _keys(api_key if api_key is not None else env('SAINA_API_KEY', ''))
    # A gateway in front of this server authenticates with its own token and may send metering headers.
    service = _keys(service_token if service_token is not None else env('SAINA_SERVICE_TOKEN', ''))
    if not keys and not service:
        raise RuntimeError('SAINA_API_KEY or SAINA_SERVICE_TOKEN is required')
    executor = BoundedExecutor(
        slots=execution_slots or int(env('SAINA_EXECUTION_SLOTS', '1')),
        depth=queue_depth if queue_depth is not None else int(env('SAINA_QUEUE_DEPTH', '4')),
        queue_timeout=queue_timeout or float(env('SAINA_QUEUE_TIMEOUT', '30')))

    @asynccontextmanager
    async def lifespan(app):
        loaded = list(scorers) if scorers else [scorer] if scorer is not None else load_models(scorer_loader or load_scorer)
        ids = [model_id(s) for s in loaded]
        if len(set(ids)) != len(ids):
            raise RuntimeError('Each served model must be a different release')
        # The first model is the default: it answers the unversioned `saina-helm` name.
        app.state.scorers, app.state.scorer = loaded, loaded[0]
        yield
        executor.shutdown()

    def pick(name, fallback=False):
        for s in app.state.scorers:
            if name != 'saina-helm' and name in model_names(model_id(s)):
                return s
        if name == 'saina-helm' or fallback:
            return app.state.scorer
        raise ApiError(404, 'not_found', 'This endpoint serves ' + ', '.join(map(model_id, app.state.scorers)))

    app = FastAPI(title='Saina Helm', version=__version__, lifespan=lifespan)
    app.state.executor = executor
    if cors_origins is None:
        cors_origins = env('SAINA_CORS_ORIGINS', '')
    if isinstance(cors_origins, str):
        cors_origins = [o.strip() for o in cors_origins.split(',') if o.strip()]
    if cors_origins:
        # Opt-in for self-hosters with their own browser UI. Only the listed origins; no cookies.
        app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=['GET', 'POST'],
                           allow_headers=['Authorization', 'Content-Type'], allow_credentials=False, max_age=600)
    bearer = HTTPBearer(auto_error=False)

    @app.middleware('http')
    async def https_only(request, call_next):
        # Behind a TLS-terminating proxy, refuse plain-HTTP requests before reading any token.
        # Without proxy headers (local use or a private gateway link), requests are served as-is.
        proto = request.headers.get('x-forwarded-proto', '').split(',')[0].strip().lower()
        if not proto and '"scheme":"http"' in request.headers.get('cf-visitor', '').replace(' ', ''):
            proto = 'http'
        if proto == 'http':
            return JSONResponse(status_code=426, content={
                'error': {'code': 'https_required', 'message': 'Use https'}, 'detail': 'Use https'},
                headers={'Upgrade': 'TLS/1.2'})
        response = await call_next(request)
        if proto == 'https':
            response.headers['Strict-Transport-Security'] = 'max-age=15552000'
        return response

    @app.exception_handler(ApiError)
    async def api_error(request, exc):
        return error(exc.status, exc.code, exc.message, exc.headers, exc.extra)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Validation responses must not echo customer context or input values.
        detail = [{'loc': e['loc'], 'type': e['type'], 'msg': e['msg']} for e in exc.errors()]
        return JSONResponse(status_code=422, content={
            'error': {'code': 'invalid_request', 'message': 'Request validation failed'}, 'detail': detail})

    def caller(auth: HTTPAuthorizationCredentials | None = Depends(bearer)):
        token = auth.credentials.encode() if auth else b''
        # Compare against every key so timing does not reveal which list matched.
        is_service = any([hmac.compare_digest(token, k) for k in service])
        is_key = any([hmac.compare_digest(token, k) for k in keys])
        if not token or not (is_service or is_key):
            raise ApiError(401, 'invalid_api_key', 'Invalid bearer token', {'WWW-Authenticate': 'Bearer'})
        return 'service' if is_service else 'key'

    def service_only(kind: str = Depends(caller)):
        if kind != 'service':
            raise ApiError(403, 'permission_denied', 'Service token required')

    def int_header(request, name):
        value = request.headers.get(name)
        if value is None:
            return None
        if not value.isdigit() or len(value) > 15:
            raise ApiError(400, 'invalid_request', f'Invalid {name} header')
        return int(value)

    async def result_of(future):
        # Waits for a job without tying its cancellation to the handler's: a job cancelled in the queue
        # surfaces as concurrent.futures.CancelledError from `result()`, while cancelling this handler
        # (asyncio.CancelledError) drops the job if it is still queued and propagates unchanged.
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()

        def wake(_):
            try:
                loop.call_soon_threadsafe(lambda: waiter.done() or waiter.set_result(None))
            except RuntimeError:  # loop already closed
                pass
        future.add_done_callback(wake)
        try:
            await waiter
        except asyncio.CancelledError:
            future.cancel()  # no effect once running: the slot is held until the work finishes
            raise
        return future.result()

    async def serve(request, kind, prepared, billable, respond, model):
        # Metering headers are honored only from the gateway's service token.
        trusted = kind == 'service'
        request_id = request.headers.get('x-saina-request-id') if trusted else None
        expected = int_header(request, 'x-saina-expected-input-tokens') if trusted else None
        deadline_ms = int_header(request, 'x-saina-deadline-ms') if trusted else None
        try:
            check_capabilities(prepared, getattr(model, 'supported_modes', ('single_label',)))
            # Take a queue place before tokenizing, so a full queue rejects without encoding anything.
            # Encoding happens outside the execution slot; the place is given back if it fails.
            executor.reserve()
            submitted = False
            try:
                encoded = await asyncio.to_thread(encode_all, prepared, model)
                if expected is not None and sum(map(len, encoded)) != expected:
                    raise MeteringMismatch()
                future, timing = executor.submit(lambda: execute(prepared, model, billable, encoded), reserved=True,
                    request_id=request_id, deadline=deadline_ms / 1000 if deadline_ms else None)
                submitted = True
            finally:
                if not submitted:
                    executor.release()
            execution = await result_of(future)
        except Overloaded:
            raise ApiError(429, 'overloaded', 'Inference queue is full', {'Retry-After': '1'}, NOT_ADMITTED) from None
        except (QueueExpired, JobCancelled):
            # Dropped before running (expired, or cancelled by the gateway): nothing ran.
            raise ApiError(503, 'overloaded', 'Request expired in the inference queue', {'Retry-After': '1'},
                           NOT_ADMITTED) from None
        except ModelNotServed as e:
            raise ApiError(404, 'not_found', str(e)) from None
        except MeteringMismatch:
            raise ApiError(409, 'metering_mismatch', 'Encoded input does not match the expected token count') from None
        except InputTooLong:
            raise ApiError(422, 'invalid_request', 'Input exceeds model context window') from None
        except ValueError:
            raise ApiError(422, 'invalid_request', 'Unsupported input, checkpoint capability, or invalid model output') from None
        except Exception:
            # Errors never carry request content.
            raise ApiError(500, 'inference_failed', 'Inference failed') from None
        body = respond(prepared, execution, model_id(model))
        # X-Saina-Input-Tokens is the model-input length the gateway's metering check compares;
        # X-Saina-Billable-Tokens (and usage.input_tokens) is what the request is billed for.
        headers = {'X-Saina-Input-Tokens': str(execution.total_input_tokens),
                   'X-Saina-Billable-Tokens': str(execution.billable_tokens)}
        if trusted:
            headers.update({'X-Saina-Queue-Ms': str(timing.get('queue_ms', 0)),
                            'X-Saina-Exec-Ms': str(timing.get('exec_ms', 0))})
        return body, headers

    @app.get('/livez')
    def live():
        return {'status': 'ok'}

    @app.get('/healthz')
    def health():
        # The pinned revision is public, so callers can verify which weights answered them.
        default = app.state.scorer
        health = {'status': 'ready', 'model': model_id(default), 'version': __version__,
                  'revision': getattr(default, 'revision', None) or env('SAINA_REVISION') or None,
                  'queue': executor.stats()}
        if len(app.state.scorers) > 1:
            health['models'] = [{'model': model_id(s), 'revision': getattr(s, 'revision', None)}
                                for s in app.state.scorers]
        return health

    @app.get('/v1/models')
    def models():
        # Model listing in OpenRouter's provider format (schema 2.4). Public: no secrets in it.
        # SAINA_PROMPT_PRICE_USD is the per-token input price as a string; unset means unpriced.
        price = env('SAINA_PROMPT_PRICE_USD')

        def listing(scorer):
            # The limit the model enforces: the checkpoint's unless configured lower.
            limit = getattr(scorer, 'max_length', None) or int(env('SAINA_MAX_LENGTH', '8192'))
            text = {'type': 'text', 'supported_inputs': {
                'max_context_length': {'value': limit, 'unit': 'token'}}}
            if price:
                text['pricing'] = [{'type': 'prompt', 'unit': 'token', 'cost_usd': price}]
            known = CATALOG.get(model_id(scorer), {'name': 'Saina: ' + model_id(scorer), 'created': 0,
                                                   'hugging_face_id': 'run-saina/' + model_id(scorer)})
            return {
                'schema_version': '2.4',
                'id': model_id(scorer),
                **known,
                'quantization': 'bf16',
                'description': ('Saina Helm is an open 0.8B decision model for workflow automation. It reads a state and '
                                'answers typed questions (noul, choice, score) with a probability for every option, '
                                'instead of generating text.'),
                'input_modalities': [text],
                'output_modalities': [{'type': 'decisions', 'supported_parameters': {}}],
                'capacity': [{'type': 'request', 'unit': 'request', 'per': 'minute', 'value': 300}],
                'is_ready': True,
                'datacenters': [{'country_code': env('SAINA_COUNTRY', 'CA')}],
            }
        return {'data': [listing(s) for s in app.state.scorers]}

    @app.post('/v1/ask', response_model=AskResponse, response_model_exclude_unset=True)
    async def native_questions(body: AskRequest, request: Request, kind: str = Depends(caller)):
        result, headers = await serve(request, kind, prepare_ask(body), billable_ask(body),
                                      lambda prepared, execution, mid: respond_ask(body, prepared, execution, mid),
                                      pick(body.model))
        return JSONResponse(AskResponse.model_validate(result).model_dump(exclude_unset=True), headers=headers)

    @app.post('/v1/systemone')
    async def system_one(body: SystemOneRequest, request: Request, kind: str = Depends(caller)):
        # TypeSafe System One wire format, as used by Jev clients and OpenRouter's Decisions API.
        # System One callers send their own model labels; unknown labels reach the default model.
        result, headers = await serve(request, kind, prepare_systemone(body), billable_systemone(body),
                                      lambda prepared, execution, mid: respond_systemone(body, prepared, execution, mid),
                                      pick(body.model, fallback=True))
        return JSONResponse(result, headers=headers)

    @app.delete('/v1/internal/requests/{request_id}', dependencies=[Depends(service_only)])
    def cancel(request_id: str):
        # Drops queued work the gateway abandoned. Running work keeps its slot until it finishes.
        return {'request_id': request_id, 'state': executor.cancel(request_id)}

    return app
