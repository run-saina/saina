"""Authenticated single-worker API. Run with uvicorn's factory option."""
from contextlib import asynccontextmanager
import hmac
import os
from threading import Lock
from fastapi import Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from .errors import InputTooLong
from .loader import load_scorer
from .contract import AskRequest, AskResponse, ask
from .jev import SystemOneRequest, evaluate


def create_app(scorer=None, api_key=None, cors_origins=None):
    key = api_key if api_key is not None else os.environ.get('SAINA_API_KEY', '')
    # One key per caller, comma-separated, so a key can be revoked without rotating the rest.
    keys = [k.strip().encode() for k in key.split(',') if k.strip()]
    if not keys:
        raise RuntimeError('SAINA_API_KEY is required')
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.scorer = scorer or load_scorer(os.environ['SAINA_CHECKPOINT'],
            revision=os.environ.get('SAINA_REVISION'), device=os.environ.get('SAINA_DEVICE', 'cpu'),
            max_length=int(os.environ.get('SAINA_MAX_LENGTH', '8192')))
        yield

    app = FastAPI(title='Saina Helm', version='0.1.2', lifespan=lifespan)
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

    def serve(run):
        # One request at a time on the single model; errors never carry request content.
        if not lock.acquire(blocking=False):
            raise HTTPException(503, 'Model is busy', headers={'Retry-After': '1'})
        try:
            return run(app.state.scorer)
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
        return {'status': 'ready', 'model': 'saina-helm-0.8b',
                'revision': os.environ.get('SAINA_REVISION') or None}

    @app.get('/v1/models')
    def models():
        # Model listing in OpenRouter's provider format (schema 2.4). Public: no secrets in it.
        # SAINA_PROMPT_PRICE_USD is the per-token input price as a string; unset means unpriced.
        price = os.environ.get('SAINA_PROMPT_PRICE_USD')
        text = {'type': 'text', 'supported_inputs': {
            'max_context_length': {'value': int(os.environ.get('SAINA_MAX_LENGTH', '8192')), 'unit': 'token'}}}
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
                            'instead of generating text. Zero retention: request and response bodies are never '
                            'stored (https://saina.run/api/#data).'),
            'input_modalities': [text],
            'output_modalities': [{'type': 'decisions', 'supported_parameters': {}}],
            'capacity': [{'type': 'request', 'unit': 'request', 'per': 'minute', 'value': 300}],
            'is_ready': True,
            'datacenters': [{'country_code': os.environ.get('SAINA_COUNTRY', 'CA')}],
        }]}

    @app.post('/v1/ask', response_model=AskResponse, response_model_exclude_unset=True,
              dependencies=[Depends(authorize)])
    def native_questions(request: AskRequest):
        return serve(lambda scorer: ask(request, scorer))

    @app.post('/v1/systemone', dependencies=[Depends(authorize)])
    def system_one(request: SystemOneRequest):
        # TypeSafe System One wire format, as used by Jev clients and OpenRouter's Decisions API.
        return serve(lambda scorer: evaluate(request, scorer))

    return app
