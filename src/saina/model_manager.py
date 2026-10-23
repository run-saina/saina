"""Authenticated browser-driven model selection; one active model per process."""
import asyncio
import hmac
import os
from contextlib import asynccontextmanager


class EmptyScorer:
    model_id = 'saina-helm'
    supported_modes = ()
    inputs = ('text',)


def smoke(scorer):
    from .contract import AskRequest, prepare_ask, billable_ask
    from .prepare import execute
    request = AskRequest.model_validate({'model':'saina-helm','state':'The customer was charged twice.',
        'questions':{'route':{'type':'single_choice','question':'Which team should handle this?',
                             'options':{'billing':'Billing','shipping':'Shipping'}}}})
    execute(prepare_ask(request), scorer, billable_ask(request))


class ModelManager:
    def __init__(self, backend):
        self.backend = backend
        self.phase = 'idle'
        self.selected = None
        self.error = None
        self.busy = False
        self.inflight = 0
        self.task = None
        self.active = None

    def listing(self):
        from .releases import catalog, plan_release
        models=[]
        for key in catalog(self.backend):
            p=plan_release(self.backend, model_key=key)
            models.append({'key':key,'repo':p['repo'],'revision':p['revision'],'backend':self.backend,
                           'cached':not p['missing_files'],'download_bytes':p['download_bytes'],
                           'total_bytes':p['total_bytes']})
        return {'models':models,'selected':self.selected,'phase':self.phase,'busy':self.busy,'error':self.error}

    async def load(self, key, app):
        from .releases import plan_release, stage_release, dependencies
        from .backends import load_backend
        new = None
        try:
            dependencies(self.backend)
            plan = plan_release(self.backend, model_key=key)
            # Admission is blocked while busy. All prior inference finished before switching.
            old = self.active
            self.active = None
            app.state.scorers = [EmptyScorer()]
            app.state.scorer = app.state.scorers[0]
            if old is not None and hasattr(old,'close'):
                old.close()
            del old
            self.selected = None
            self.phase = 'downloading' if plan['missing_files'] else 'verifying'
            root = await asyncio.to_thread(stage_release, plan, download=True,
                offline=os.environ.get('HF_HUB_OFFLINE','').upper() in ('1','TRUE','YES','ON'))
            self.phase = 'loading'
            new = await asyncio.to_thread(load_backend, str(root), self.backend, plan['revision'])
            self.phase = 'checking'
            await asyncio.to_thread(smoke, new)
            self.active = new
            app.state.scorers, app.state.scorer = [new], new
            self.selected, self.phase = plan['key'], 'ready'
        except Exception as exc:
            if new is not None and hasattr(new,'close'):
                new.close()
            self.error, self.phase = str(exc), 'error'
        finally:
            self.busy = False

    def attach(self, app, token):
        from fastapi import Request
        from fastapi.responses import JSONResponse
        def authorized(request):
            value=request.headers.get('authorization','')
            return hmac.compare_digest(value.encode(), ('Bearer '+token).encode())

        @app.get('/local/models')
        async def models(request: Request):
            if not authorized(request):
                return JSONResponse({'detail':'Invalid bearer token'},status_code=401)
            return await asyncio.to_thread(self.listing)

        @app.post('/local/models/select')
        async def select(request: Request):
            if not authorized(request):
                return JSONResponse({'detail':'Invalid bearer token'},status_code=401)
            from .releases import resolve_key
            try:
                body=await request.json()
                if not isinstance(body,dict) or set(body)!={'key'} or not isinstance(body['key'],str):
                    raise ValueError('Send a model key')
                key=resolve_key(self.backend,body['key'])
            except ValueError as exc:
                return JSONResponse({'detail':str(exc)},status_code=400)
            stats=app.state.executor.stats()
            if self.busy or self.inflight or stats['running'] or stats['queued']:
                return JSONResponse({'detail':'Wait for the current model operation or inference to finish'},status_code=409)
            self.busy, self.error, self.phase = True, None, 'preparing'
            self.task=asyncio.create_task(self.load(key,app))
            return JSONResponse({'status':'preparing'},status_code=202)

        original=app.router.lifespan_context
        @asynccontextmanager
        async def lifespan(application):
            async with original(application):
                try:
                    yield
                finally:
                    if self.task is not None:
                        await asyncio.shield(self.task)
                    if self.active is not None and hasattr(self.active,'close'):
                        self.active.close()
        app.router.lifespan_context=lifespan
