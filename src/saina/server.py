"""Authenticated single-worker API. Run with uvicorn's factory option."""
from contextlib import asynccontextmanager
import hmac
import os
from threading import Lock
from fastapi import Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from .errors import InputTooLong
from .loader import load_scorer
from .contract import AskRequest, AskResponse, ask


def create_app(scorer=None, api_key=None):
    key = api_key if api_key is not None else os.environ.get('SAINA_API_KEY', '')
    if not key:
        raise RuntimeError('SAINA_API_KEY is required')
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.scorer = scorer or load_scorer(os.environ['SAINA_CHECKPOINT'],
            revision=os.environ.get('SAINA_REVISION'), device=os.environ.get('SAINA_DEVICE', 'cpu'),
            max_length=int(os.environ.get('SAINA_MAX_LENGTH', '8192')))
        yield

    app = FastAPI(title='Saina Helm', version='0.1.0', lifespan=lifespan)
    bearer = HTTPBearer(auto_error=False)

    def authorize(auth: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if auth is None or not hmac.compare_digest(auth.credentials.encode(), key.encode()):
            raise HTTPException(401, 'Invalid bearer token', headers={'WWW-Authenticate': 'Bearer'})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Validation responses must not echo customer context or input values.
        return JSONResponse(status_code=422, content={'detail': [
            {'loc': e['loc'], 'type': e['type'], 'msg': e['msg']} for e in exc.errors()]})

    @app.get('/healthz')
    def health():
        return {'status': 'ready'}

    @app.post('/v1/ask', response_model=AskResponse, response_model_exclude_unset=True,
              dependencies=[Depends(authorize)])
    def native_questions(request: AskRequest):
        if not lock.acquire(blocking=False):
            raise HTTPException(503, 'Model is busy', headers={'Retry-After': '1'})
        try:
            return ask(request, app.state.scorer)
        except InputTooLong:
            raise HTTPException(422, 'Input exceeds model context window') from None
        except ValueError:
            raise HTTPException(422, 'Unsupported input, checkpoint capability, or invalid model output') from None
        except Exception:
            raise HTTPException(500, 'Inference failed') from None
        finally:
            lock.release()

    return app
