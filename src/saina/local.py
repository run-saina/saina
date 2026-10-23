"""Loopback-only launcher and offline hosting for the shared console build."""
import json
from pathlib import Path
import secrets
import socket
import threading
import webbrowser


def create_local_app(scorer, token, ui_dir=None, port=8765, backend='native', manager=None):
    from fastapi.responses import JSONResponse, Response
    from starlette.staticfiles import StaticFiles
    from .server import create_app
    app = create_app(scorer=scorer, api_key=token, service_token='', cors_origins=[], execution_slots=1)
    origins = {f'http://127.0.0.1:{port}', f'http://localhost:{port}'}
    hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}

    @app.middleware('http')
    async def local_boundary(request, call_next):
        # Host validation prevents DNS rebinding; foreign web pages cannot use the local API.
        if request.headers.get('host') not in hosts or request.headers.get('origin', next(iter(origins))) not in origins:
            return JSONResponse({'detail': 'Local origin required'}, status_code=403)
        inference = manager is not None and request.method == 'POST' and request.url.path.startswith(('/v1/', '/api/v1/'))
        if inference and (manager.busy or manager.active is None):
            return JSONResponse({'detail':'Select a ready model in the playground'},status_code=503)
        if inference:
            manager.inflight += 1
        try:
            response = await call_next(request)
        finally:
            if inference:
                manager.inflight -= 1
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; font-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        return response

    @app.get('/console/runtime-config.js')
    def runtime():
        return Response('window.SainaRuntime=' + json.dumps({'mode': 'local', 'backend': backend,
            'manage_models':manager is not None, 'model': scorer.model_id, 'revision': getattr(scorer, 'revision', None)}) + ';', media_type='text/javascript')

    if manager is not None:
        manager.attach(app, token)
    if ui_dir is not None:
        root = validate_ui(ui_dir)
        app.mount('/', StaticFiles(directory=root, html=True), name='playground')
    return app


def validate_ui(ui_dir):
    root = Path(ui_dir).expanduser().resolve()
    from .artifacts import verify_artifacts
    verify_artifacts(root, manifest='ui-manifest.json', required=['console/index.html', 'console/console.js', 'console/ui-release.json'])
    release = json.loads((root / 'console/ui-release.json').read_text())
    if release.get('api_contract') != 1:
        raise ValueError('Shared UI/API contract mismatch; rebuild or update the SDK')
    return root


def run_local(args):
    import os
    from .backends import select_backend, load_backend
    backend = select_backend(args.backend)
    from .releases import choose_release, stage_release, dependencies
    plan = None
    managed = args.command == 'playground'
    if not managed and not args.checkpoint:
        plan = choose_release(backend, args.model_key)
        print(f"Model: {plan['repo']}@{plan['revision']}\nBackend: {backend}\nModel bytes: {plan['total_bytes']:,}\nDownload bytes: {plan['download_bytes']:,}\nFree cache bytes: {plan['free_bytes']:,}", flush=True)
    if not managed:
        dependencies(backend)
    import uvicorn
    if not 1 <= args.port <= 65535:
        raise ValueError('Port must be between 1 and 65535')
    # Reserve the port before downloading/loading a model; keep it until the server exits.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    scorer = None
    try:
        sock.bind(('127.0.0.1', args.port))
        ui = Path(args.ui_dir) if args.ui_dir else Path(__file__).parent / 'ui'
        if not args.api_only and not (ui / 'ui-manifest.json').is_file():
            raise ValueError('Shared UI assets are missing. Supply --ui-dir from build-local-ui.mjs, or reinstall the SDK wheel')
        if not args.api_only:
            ui = validate_ui(ui)
        if plan:
            args.checkpoint = str(stage_release(plan, download=True, offline=os.environ.get('HF_HUB_OFFLINE', '').upper() in ('1', 'TRUE', 'YES', 'ON')))
            args.revision = plan['revision']
        from .model_manager import ModelManager, EmptyScorer, smoke
        manager = ModelManager(backend) if managed else None
        if managed:
            scorer = EmptyScorer()
        else:
            print(f'Loading {args.checkpoint} with {backend}; local inference only.', flush=True)
            scorer = load_backend(args.checkpoint, backend, args.revision, args.max_length,
                                  args.runner, args.gguf_file, args.threads)
            smoke(scorer)
        token = secrets.token_urlsafe(32)
        app = create_local_app(scorer, token, None if args.api_only else ui, args.port, backend, manager)
        server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=args.port,
                                proxy_headers=False, access_log=False, log_level='warning'))
        stop = threading.Event()
        def ready():
            while not stop.wait(.1):
                if server.started:
                    base = f'http://127.0.0.1:{args.port}'
                    # Fragments never reach HTTP logs; no persistent browser token storage.
                    url = base + '/console/#local-token=' + token
                    print('Ready: ' + (base if args.api_only else url), flush=True)
                    if args.api_only:
                        print('Local API bearer token: ' + token, flush=True)
                    elif not args.no_browser:
                        webbrowser.open(url)
                    return
        thread = threading.Thread(target=ready, daemon=True)
        thread.start()
        try:
            server.run(sockets=[sock])
        finally:
            stop.set()
            thread.join(timeout=1)
    finally:
        sock.close()
        if scorer is not None and hasattr(scorer, 'close'):
            scorer.close()
    return 0
