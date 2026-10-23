"""Local diagnostics; intentionally usable without inference dependencies."""
import argparse
import json
import re
import sys

from . import __version__

from .artifacts import verify_artifacts, verify_files


def main(argv=None):
    parser = argparse.ArgumentParser(prog='saina', description='Run Saina on this device and verify model artifacts.')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    commands = parser.add_subparsers(dest='command', required=True)
    doctor = commands.add_parser('doctor', help='Check local artifact integrity without loading a model')
    doctor.add_argument('model_key', nargs='?', help='Model key to verify; otherwise select a cached model')
    doctor.add_argument('--checkpoint', help='Local bundle directory; omit to verify the pinned default cache')
    doctor.add_argument('--manifest', default=None, help='Manifest path relative to the bundle')
    doctor.add_argument('--backend', choices=['auto', 'native', 'mlx', 'gguf'], default='auto')
    doctor.add_argument('--require', action='append', default=[], metavar='FILE',
                        help='Additional required artifact; repeat for each file')
    doctor.add_argument('--json', action='store_true', help='Emit machine-readable diagnostics')
    local = commands.add_parser('serve', help='Start the local inference API server',
                                formatter_class=argparse.HelpFormatter)
    local.add_argument('model_key', nargs='?', help='Model key; omit to select from compatible cached or downloadable models')
    local.add_argument('--checkpoint', help='Local bundle or Hub repository; omit for the pinned default model')
    local.add_argument('--revision', help='40-character commit for --checkpoint; required for a Hub repository')
    local.add_argument('--backend', choices=['auto', 'native', 'mlx', 'gguf'], default='auto', help='auto selects MLX on Apple Silicon, native CPU elsewhere')
    local.add_argument('--port', type=int, default=8765, help='Loopback server port (1–65535)')
    local.add_argument('--max-length', type=int, help='Positive model context limit')
    local.add_argument('--runner', help='GGUF helm-hidden executable')
    local.add_argument('--gguf-file', help='GGUF bundle filename (default: helm-trunk-f32.gguf)')
    local.add_argument('--threads', type=int, help='GGUF CPU threads, 1–16 (default: 8)')
    local.set_defaults(api_only=True, no_browser=True, ui_dir=None)
    playground = commands.add_parser('playground', help='Open the playground to select, download and run models')
    playground.add_argument('--backend', choices=['auto', 'native', 'mlx'], default='auto')
    playground.add_argument('--port', type=int, default=8765)
    playground.add_argument('--ui-dir', help='Verified shared UI asset directory')
    playground.add_argument('--no-browser', action='store_true', help='Print the URL without opening a browser')
    playground.set_defaults(checkpoint=None, model_key=None, revision=None, max_length=None, runner=None,
                            gguf_file=None, threads=None, api_only=False)
    args = parser.parse_args(argv)
    if args.command in ('serve', 'playground'):
        from .backends import select_backend
        try:
            args.backend = select_backend(args.backend)
        except ValueError as exc:
            parser.error(str(exc))
        if not 1 <= args.port <= 65535:
            parser.error('--port must be between 1 and 65535')
        if args.max_length is not None and args.max_length <= 0:
            parser.error('--max-length must be positive')
        if args.revision and (not args.checkpoint or not re.fullmatch(r'[0-9a-f]{40}', args.revision)):
            parser.error('--revision requires --checkpoint and a full 40-character lowercase commit')
        if args.model_key and args.checkpoint:
            parser.error('Use a model key or --checkpoint, not both')
        if args.checkpoint:
            from pathlib import Path
            path = Path(args.checkpoint).expanduser()
            if path.is_dir():
                args.checkpoint = str(path)
            elif not args.revision:
                parser.error('--checkpoint is not a local directory; a Hub repository requires --revision')
        if args.api_only and args.ui_dir:
            parser.error('--ui-dir is unused with --api-only')
        if args.backend != 'gguf' and any(x is not None for x in (args.runner, args.gguf_file, args.threads)):
            parser.error('--runner, --gguf-file and --threads require --backend gguf')
        if args.backend == 'gguf':
            if not args.checkpoint or not args.runner:
                parser.error('--backend gguf requires --checkpoint and --runner')
            if args.threads is not None and not 1 <= args.threads <= 16:
                parser.error('--threads must be between 1 and 16')
        args.gguf_file = args.gguf_file or 'helm-trunk-f32.gguf'
        args.threads = args.threads if args.threads is not None else 8
        try:
            from .local import run_local
            return run_local(args)
        except ImportError as exc:
            from .backends import select_backend
            backend = select_backend(args.backend)
            extra = 'local' if backend == 'native' else backend
            print(f"Missing runtime dependency: {exc}. Reinstall your Saina wheel with [server,{extra}], or use '.[server,{extra}]' from a source checkout.", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print('Cancelled.', file=sys.stderr)
            return 130
        except (ValueError, OSError, RuntimeError) as exc:
            print(f'Local setup failed: {exc}', file=sys.stderr)
            return 1
    if not args.checkpoint and args.manifest:
        parser.error('--manifest requires --checkpoint')
    try:
        from .backends import select_backend
        backend = select_backend('auto') if args.backend == 'auto' else args.backend
        if args.model_key and args.checkpoint:
            parser.error('Use a model key or --checkpoint, not both')
        if not args.checkpoint:
            from .releases import choose_release
            plan = choose_release(backend, args.model_key, interactive=False if args.json else None, allow_download=False)
            files = verify_files(plan['snapshot'], {n:f['sha256'] for n,f in plan['files'].items()}, required=args.require)
        else:
            files = doctor_bundle(args, backend)
        result = {'status': 'integrity_verified', 'files': sorted(files),
                  'message': 'Declared files match the manifest. Authenticity, compatibility and inference readiness are not checked.'}
        code = 0
    except (ValueError, OSError) as exc:
        result = {'status': 'failed', 'message': str(exc)}
        code = 1
    if args.json:
        print(json.dumps(result))
    else:
        print(f"{result['status']}: {result['message']}", file=sys.stderr if code else sys.stdout)
    return code


def doctor_bundle(args, backend):
    manifest = args.manifest or ('gguf-manifest.json' if backend == 'gguf' else 'artifact-manifest.json')
    required = ['head-config.json', 'config.json'] if backend == 'mlx' else ['head-config.json', 'head.safetensors']
    return verify_artifacts(args.checkpoint, manifest=manifest, required=[*required, *args.require])
