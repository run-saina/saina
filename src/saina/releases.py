"""Pinned, package-owned local release selection and download preflight."""
import importlib.util
import json
import os
from pathlib import Path
import shutil


def catalog(backend):
    releases = json.loads(Path(__file__).with_name('local-releases.json').read_text())
    return {key:variants[backend] for key,variants in releases.items() if backend in variants}


def resolve_key(backend, key):
    models = catalog(backend)
    for name, release in models.items():
        if key in (name, release['repo'], release['repo'].split('/')[-1]):
            return name
    raise ValueError(f'Unknown model {key!r} for {backend}. Available keys: ' + ', '.join(models))


def default_release(backend, model_key='helm'):
    return catalog(backend)[resolve_key(backend, model_key)]


def hub_cache():
    home = Path(os.environ.get('HF_HOME', str(Path(os.environ.get('XDG_CACHE_HOME', str(Path.home()/'.cache'))) / 'huggingface')))
    return Path(os.environ.get('HF_HUB_CACHE', os.environ.get('HUGGINGFACE_HUB_CACHE', str(home/'hub')))).expanduser()


def plan_release(backend, cache=None, model_key='helm'):
    key = resolve_key(backend, model_key)
    release = default_release(backend, key)
    cache = Path(cache) if cache is not None else hub_cache()
    snapshot = cache / ('models--' + release['repo'].replace('/', '--')) / 'snapshots' / release['revision']
    missing = []
    for name, info in release['files'].items():
        path = snapshot/name
        if not path.is_file() or path.stat().st_size != info['size']:
            missing.append(name)
    parent = cache
    while not parent.exists():
        parent = parent.parent
    return {'key':key, 'backend':backend, 'repo':release['repo'], 'revision':release['revision'],
            'cache':str(cache), 'snapshot':str(snapshot), 'files':release['files'],
            'total_bytes':sum(f['size'] for f in release['files'].values()),
            'download_bytes':sum(release['files'][f]['size'] for f in missing),
            'missing_files':missing, 'free_bytes':shutil.disk_usage(parent).free}


def dependencies(backend):
    modules = ['uvicorn', 'fastapi', 'pydantic'] + {
        'native':['torch','transformers','safetensors','huggingface_hub'],
        'mlx':['mlx_lm','huggingface_hub'], 'gguf':['numpy','transformers','safetensors']}[backend]
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        extra = 'local' if backend == 'native' else backend
        raise ImportError('Missing '+', '.join(missing)+f"; install 'saina[server,{extra}]' from the supplied wheel")


def stage_release(plan, *, download=False, offline=False):
    if plan['missing_files']:
        if offline:
            raise ValueError('Pinned release is not completely cached; connect once and run saina serve')
        if not download:
            raise ValueError(f"Download needs {plan['download_bytes']:,} bytes. Run saina serve to fetch the pinned release")
        if plan['free_bytes'] < plan['download_bytes'] + 128*1024*1024:
            raise ValueError('Not enough free disk space for the download plus 128 MiB working space')
        from huggingface_hub import snapshot_download
        # The Hub manages atomic blobs and resumable .incomplete downloads. Never expose a
        # partial snapshot to the loader; every file is verified below before model allocation.
        snapshot_download(plan['repo'],revision=plan['revision'],cache_dir=plan['cache'],
                          allow_patterns=list(plan['files']))
    from .artifacts import verify_files
    verify_files(plan['snapshot'], {name:info['sha256'] for name,info in plan['files'].items()})
    return Path(plan['snapshot'])


def choose_release(backend, key=None, *, interactive=None, allow_download=True, reader=input):
    import sys
    if key:
        return plan_release(backend, model_key=key)
    plans = [plan_release(backend, model_key=k) for k in catalog(backend)]
    cached = [p for p in plans if not p['missing_files']]
    if len(cached) == 1:
        return cached[0]
    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        raise ValueError('Specify a model key in a non-interactive shell. Available: ' + ', '.join(p['key'] for p in plans))
    choices = cached + ([p for p in plans if p['missing_files']] if allow_download else [])
    if not choices:
        raise ValueError('No compatible models cached. Run saina playground to choose and download one.')
    print('Choose a model:')
    for i,p in enumerate(choices, 1):
        status = 'cached' if not p['missing_files'] else f"download {p['download_bytes']/1e9:.2f} GB"
        print(f"  {i}. {p['key']} — {status}")
    while True:
        try:
            selected = reader(f'Model [1–{len(choices)}, q to cancel]: ').strip()
        except EOFError:
            raise ValueError('Model selection cancelled') from None
        if selected.lower() in ('q','quit'):
            raise ValueError('Model selection cancelled')
        if selected.isdigit() and 1 <= int(selected) <= len(choices):
            return choices[int(selected)-1]
        print('Enter one of the listed numbers, or q to cancel.')
