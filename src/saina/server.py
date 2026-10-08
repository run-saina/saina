"""Authenticated inference API with a bounded queue. Run with uvicorn's factory option.

Concurrency: requests wait in a bounded FIFO queue for a fixed number of execution slots
(default one). A full queue rejects promptly with `429 overloaded`; a queued request that
waits longer than the queue timeout is dropped before it runs. A slot is released only when
its model work has actually finished, never when a client disconnects.
"""
import asyncio
from concurrent.futures import CancelledError, ThreadPoolExecutor
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
from .errors import InputTooLong
from .loader import load_scorer
from .contract import AskRequest, AskResponse, prepare_ask, respond as respond_ask
from .jev import SystemOneRequest, prepare_systemone, respond as respond_systemone
from .prepare import MeteringMismatch, check_capabilities, encode_all, execute


class Overloaded(Exception):
    pass


class QueueExpired(Exception):
    pass


def error(status, code, message, headers=None):
    # `detail` keeps older clients that read FastAPI's default shape working.
    return JSONResponse(status_code=status, headers=headers,
                        content={'error': {'code': code, 'message': message}, 'detail': message})


class ApiError(Exception):
    def __init__(self, status, code, message, headers=None):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers


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

    def submit(self, fn, request_id=None, deadline=None):
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
            if self.admitted >= self.slots + self.depth:
                raise Overloaded()
            self.admitted += 1
            future = self._pool.submit(run)
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


def create_app(scorer=None, api_key=None, cors_origins=None, service_token=None,
               execution_slots=None, queue_depth=None, queue_timeout=None, scorer_loader=None):
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
        app.state.scorer = scorer or (scorer_loader or load_scorer)(env('SAINA_CHECKPOINT'),
            revision=env('SAINA_REVISION'), device=env('SAINA_DEVICE', 'cpu'),
            max_length=int(env('SAINA_MAX_LENGTH', '8192')))
        yield
        executor.shutdown()

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
        return error(exc.status, exc.code, exc.message, exc.headers)

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

    async def serve(request, kind, prepared, respond):
        # Metering headers are honored only from the gateway's service token.
        trusted = kind == 'service'
        request_id = request.headers.get('x-saina-request-id') if trusted else None
        expected = int_header(request, 'x-saina-expected-input-tokens') if trusted else None
        deadline_ms = int_header(request, 'x-saina-deadline-ms') if trusted else None
        model = app.state.scorer
        try:
            # Validate and encode every question before taking an execution slot.
            check_capabilities(prepared, getattr(model, 'supported_modes', ('single_label',)))
            encoded = await asyncio.to_thread(encode_all, prepared, model)
            if expected is not None and sum(map(len, encoded)) != expected:
                raise MeteringMismatch()
            future, timing = executor.submit(lambda: execute(prepared, model, encoded),
                request_id=request_id, deadline=deadline_ms / 1000 if deadline_ms else None)
            execution = await asyncio.wrap_future(future)
        except Overloaded:
            raise ApiError(429, 'overloaded', 'Inference queue is full', {'Retry-After': '1'}) from None
        except (QueueExpired, CancelledError):
            raise ApiError(503, 'overloaded', 'Request expired in the inference queue', {'Retry-After': '1'}) from None
        except MeteringMismatch:
            raise ApiError(409, 'metering_mismatch', 'Encoded input does not match the expected token count') from None
        except InputTooLong:
            raise ApiError(422, 'invalid_request', 'Input exceeds model context window') from None
        except ValueError:
            raise ApiError(422, 'invalid_request', 'Unsupported input, checkpoint capability, or invalid model output') from None
        except Exception:
            # Errors never carry request content.
            raise ApiError(500, 'inference_failed', 'Inference failed') from None
        body = respond(prepared, execution)
        headers = {'X-Saina-Input-Tokens': str(execution.total_input_tokens)}
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
        return {'status': 'ready', 'model': 'saina-helm-0.8b', 'version': __version__,
                'revision': env('SAINA_REVISION') or None, 'queue': executor.stats()}

    @app.get('/v1/models')
    def models():
        # Model listing in OpenRouter's provider format (schema 2.4). Public: no secrets in it.
        # SAINA_PROMPT_PRICE_USD is the per-token input price as a string; unset means unpriced.
        price = env('SAINA_PROMPT_PRICE_USD')
        text = {'type': 'text', 'supported_inputs': {
            'max_context_length': {'value': int(env('SAINA_MAX_LENGTH', '8192')), 'unit': 'token'}}}
        if price:
            text['pricing'] = [{'type': 'prompt', 'unit': 'token', 'cost_usd': price}]
        return {'data': [{
            'schema_version': '2.4',
            'id': 'saina-helm-0.8b',
            'name': 'Saina: Helm 0.8B',
            'hugging_face_id': 'run-saina/saina-helm-0.8b',
            'created': 1791331200,
            'quantization': 'bf16',
            'description': ('Saina Helm is an open 0.8B decision model for workflow automation. It reads a state and '
                            'answers typed questions (noul, choice, score) with a probability for every option, '
                            'instead of generating text.'),
            'input_modalities': [text],
            'output_modalities': [{'type': 'decisions', 'supported_parameters': {}}],
            'capacity': [{'type': 'request', 'unit': 'request', 'per': 'minute', 'value': 300}],
            'is_ready': True,
            'datacenters': [{'country_code': env('SAINA_COUNTRY', 'CA')}],
        }]}

    @app.post('/v1/ask', response_model=AskResponse, response_model_exclude_unset=True)
    async def native_questions(body: AskRequest, request: Request, kind: str = Depends(caller)):
        result, headers = await serve(request, kind, prepare_ask(body),
                                      lambda prepared, execution: respond_ask(body, prepared, execution))
        return JSONResponse(AskResponse.model_validate(result).model_dump(exclude_unset=True), headers=headers)

    @app.post('/v1/systemone')
    async def system_one(body: SystemOneRequest, request: Request, kind: str = Depends(caller)):
        # TypeSafe System One wire format, as used by Jev clients and OpenRouter's Decisions API.
        result, headers = await serve(request, kind, prepare_systemone(body),
                                      lambda prepared, execution: respond_systemone(body, prepared, execution))
        return JSONResponse(result, headers=headers)

    @app.delete('/v1/internal/requests/{request_id}', dependencies=[Depends(service_only)])
    def cancel(request_id: str):
        # Drops queued work the gateway abandoned. Running work keeps its slot until it finishes.
        return {'request_id': request_id, 'state': executor.cancel(request_id)}

    return app
