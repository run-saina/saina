"""Authenticated single-worker API serving one or more Helm models. Run with uvicorn's factory option."""
from contextlib import asynccontextmanager
import hmac
import os
from threading import Lock
from fastapi import Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from .errors import InputTooLong, ModelNotServed
from .loader import load_scorer
from .contract import LEGACY_MODEL_ID, AskRequest, AskResponse, ask, model_names
from .jev import SystemOneRequest, evaluate


# Public listing details per released model (OpenRouter provider format).
CATALOG = {
    'saina-helm-0.8b': {'name': 'Saina: Helm 0.8B', 'created': 1791331200,
                        'hugging_face_id': 'run-saina/saina-helm-0.8b'},
    'saina-helm-2-0.8b': {'name': 'Saina: Helm 2 0.8B', 'created': 1791590400,
                          'hugging_face_id': 'run-saina/saina-helm-2-0.8b'},
}


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


def load_models():
    device = os.environ.get('SAINA_DEVICE', 'cpu')
    if os.environ.get('SAINA_MODELS'):
        return [load_scorer(c, revision=r, device=device, max_length=m)
                for c, r, m in parse_models(os.environ['SAINA_MODELS'])]
    return [load_scorer(os.environ['SAINA_CHECKPOINT'], revision=os.environ.get('SAINA_REVISION'), device=device,
                        max_length=int(os.environ['SAINA_MAX_LENGTH']) if os.environ.get('SAINA_MAX_LENGTH') else None)]


def create_app(scorer=None, api_key=None, cors_origins=None, scorers=None):
    key = api_key if api_key is not None else os.environ.get('SAINA_API_KEY', '')
    # One key per caller, comma-separated, so a key can be revoked without rotating the rest.
    keys = [k.strip().encode() for k in key.split(',') if k.strip()]
    if not keys:
        raise RuntimeError('SAINA_API_KEY is required')

    @asynccontextmanager
    async def lifespan(app):
        loaded = list(scorers) if scorers else [scorer] if scorer is not None else load_models()
        ids = [getattr(s, 'model_id', LEGACY_MODEL_ID) for s in loaded]
        if len(set(ids)) != len(ids):
            raise RuntimeError('Each served model must be a different release')
        # The first model is the default: it answers the unversioned `saina-helm` name.
        app.state.scorers, app.state.scorer = loaded, loaded[0]
        app.state.locks = {id(s): Lock() for s in loaded}
        yield

    def pick(name, fallback=False):
        for s in app.state.scorers:
            if name != 'saina-helm' and name in model_names(getattr(s, 'model_id', LEGACY_MODEL_ID)):
                return s
        if name == 'saina-helm' or fallback:
            return app.state.scorer
        raise ModelNotServed('This endpoint serves ' + ', '.join(
            getattr(s, 'model_id', LEGACY_MODEL_ID) for s in app.state.scorers))

    app = FastAPI(title='Saina Helm', version='0.2.0', lifespan=lifespan)
    if cors_origins is None:
        cors_origins = os.environ.get('SAINA_CORS_ORIGINS', 'https://saina.run')
    if isinstance(cors_origins, str):
        cors_origins = [o.strip() for o in cors_origins.split(',') if o.strip()]
    if cors_origins:
        # Lets the hosted playground (saina.run) call this server from the browser.
        # Only the listed origins; no cookies; only the two headers clients send.
        app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=['GET', 'POST'],
                           allow_headers=['Authorization', 'Content-Type'], allow_credentials=False, max_age=600)
    bearer = HTTPBearer(auto_error=False)

    @app.middleware('http')
    async def https_only(request, call_next):
        # Behind a TLS-terminating proxy, refuse plain-HTTP requests before reading any token.
        # Without proxy headers (local use), requests are served as-is.
        proto = request.headers.get('x-forwarded-proto', '').split(',')[0].strip().lower()
        if not proto and '"scheme":"http"' in request.headers.get('cf-visitor', '').replace(' ', ''):
            proto = 'http'
        if proto == 'http':
            return JSONResponse(status_code=426, content={'detail': 'Use https'}, headers={'Upgrade': 'TLS/1.2'})
        response = await call_next(request)
        if proto == 'https':
            response.headers['Strict-Transport-Security'] = 'max-age=15552000'
        return response

    def authorize(auth: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if auth is None or not any(hmac.compare_digest(auth.credentials.encode(), k) for k in keys):
            raise HTTPException(401, 'Invalid bearer token', headers={'WWW-Authenticate': 'Bearer'})

    def serve(name, run, fallback=False):
        # One request at a time per model; errors never carry request content.
        try:
            target = pick(name, fallback)
        except ModelNotServed as e:
            raise HTTPException(404, str(e)) from None
        lock = app.state.locks[id(target)]
        if not lock.acquire(blocking=False):
            raise HTTPException(503, 'Model is busy', headers={'Retry-After': '1'})
        try:
            return run(target)
        except ModelNotServed as e:
            raise HTTPException(404, str(e)) from None
        except InputTooLong:
            raise HTTPException(422, 'Input exceeds model context window') from None
        except ValueError:
            raise HTTPException(422, 'Unsupported input, checkpoint capability, or invalid model output') from None
        except Exception:
            raise HTTPException(500, 'Inference failed') from None
        finally:
            lock.release()

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Validation responses must not echo customer context or input values.
        return JSONResponse(status_code=422, content={'detail': [
            {'loc': e['loc'], 'type': e['type'], 'msg': e['msg']} for e in exc.errors()]})

    @app.get('/healthz')
    def health():
        # The pinned revision is public, so callers can verify which weights answered them.
        default = app.state.scorer
        health = {'status': 'ready', 'model': getattr(default, 'model_id', LEGACY_MODEL_ID),
                  'revision': getattr(default, 'revision', None) or os.environ.get('SAINA_REVISION') or None}
        if len(app.state.scorers) > 1:
            health['models'] = [{'model': getattr(s, 'model_id', LEGACY_MODEL_ID),
                                 'revision': getattr(s, 'revision', None)} for s in app.state.scorers]
        return health

    @app.get('/v1/models')
    def models():
        # Model listing in OpenRouter's provider format (schema 2.4). Public: no secrets in it.
        # SAINA_PROMPT_PRICE_USD is the per-token input price as a string; unset means unpriced.
        price = os.environ.get('SAINA_PROMPT_PRICE_USD')

        def listing(scorer):
            # The limit the model enforces: the checkpoint's unless configured lower.
            limit = getattr(scorer, 'max_length', None) or int(os.environ.get('SAINA_MAX_LENGTH', '8192'))
            text = {'type': 'text', 'supported_inputs': {
                'max_context_length': {'value': limit, 'unit': 'token'}}}
            if price:
                text['pricing'] = [{'type': 'prompt', 'unit': 'token', 'cost_usd': price}]
            model_id = getattr(scorer, 'model_id', LEGACY_MODEL_ID)
            known = CATALOG.get(model_id, {'name': 'Saina: ' + model_id, 'created': 0,
                                           'hugging_face_id': 'run-saina/' + model_id})
            return {
                'schema_version': '2.4',
                'id': model_id,
                **known,
                'quantization': 'bf16',
                'description': ('Saina Helm is an open 0.8B decision model for workflow automation. It reads a state and '
                                'answers typed questions (noul, choice, score) with a probability for every option, '
                                'instead of generating text. Zero retention: request and response bodies are never '
                                'stored (https://saina.run/api/#data).'),
                'input_modalities': [text],
                'output_modalities': [{'type': 'decisions', 'supported_parameters': {}}],
                'capacity': [{'type': 'request', 'unit': 'request', 'per': 'minute', 'value': 300}],
                'is_ready': True,
                'datacenters': [{'country_code': os.environ.get('SAINA_COUNTRY', 'CA')}],
            }
        return {'data': [listing(s) for s in app.state.scorers]}

    @app.post('/v1/ask', response_model=AskResponse, response_model_exclude_unset=True,
              dependencies=[Depends(authorize)])
    def native_questions(request: AskRequest):
        return serve(request.model, lambda scorer: ask(request, scorer))

    @app.post('/v1/systemone', dependencies=[Depends(authorize)])
    def system_one(request: SystemOneRequest):
        # TypeSafe System One wire format, as used by Jev clients and OpenRouter's Decisions API.
        # System One callers send their own model labels; unknown labels reach the default model.
        return serve(request.model, lambda scorer: evaluate(request, scorer), fallback=True)

    return app
