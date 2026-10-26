"""Offline file integrity checks for native and experimental GGUF manifests.

Hashes establish consistency with the supplied manifest, not release authenticity,
checkpoint compatibility, or inference readiness. No model dependencies required.
"""
import hashlib
import json
import re
from pathlib import Path, PurePosixPath


class ArtifactError(ValueError):
    """A bundle cannot be verified against its manifest."""


# Model cards and version notes describe a release; they never change what the model computes, and they are
# often edited after a release is packaged. Manifest checks skip them (present or not); every other file is checked.
DOCUMENTATION_FILES = frozenset({'README.md', 'VERSIONS.md'})


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f'Duplicate manifest key: {key}')
        result[key] = value
    return result


def _name(name):
    # Manifest paths are portable relative POSIX paths. HF cache symlinks are
    # intentional; check the logical filename, not the resolved cache blob path.
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name or
            any(ord(c) < 32 for c in name) or PurePosixPath(name).is_absolute() or
            any(part in ('', '.', '..') for part in name.split('/'))):
        raise ArtifactError(f'Invalid artifact path: {name!r}')
    return name


def verify_artifacts(root, *, manifest='artifact-manifest.json', required=()):
    """Stream-check all declared assets and return their verified SHA256 values.

    Accept the native flat mapping or the GGUF mapping under ``files``. Callers
    must supply the assets their backend requires; an omitted asset is an error.
    Documentation files (``DOCUMENTATION_FILES``) are not checked.
    This function does not download, repair, execute, or rewrite anything.
    """
    root = Path(root)
    manifest_path = root / _name(manifest)
    try:
        data = json.loads(manifest_path.read_text(encoding='utf-8'), object_pairs_hook=_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f'Cannot read manifest {manifest}: {exc}') from exc
    files = data.get('files', data) if isinstance(data, dict) else None
    if isinstance(files, dict):
        files = {name: digest for name, digest in files.items() if name not in DOCUMENTATION_FILES}
    return verify_files(root, files, required=required)


def verify_files(root, files, *, required=()):
    """Verify hashes supplied by package-owned release metadata or a parsed manifest."""
    root = Path(root)
    if not isinstance(files, dict) or not files:
        raise ArtifactError('Manifest must contain a nonempty filename-to-SHA256 mapping')
    for name, expected in files.items():
        _name(name)
        if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
            raise ArtifactError(f'Invalid SHA256 for artifact: {name}')
    for name in required:
        _name(name)
        if name not in files:
            raise ArtifactError(f'Required artifact missing from manifest: {name}')
    for name, expected in files.items():
        path = root / name
        if not path.is_file():
            raise ArtifactError(f'Missing artifact: {name}; restore the pinned bundle')
        digest = hashlib.sha256()
        try:
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
        except OSError as exc:
            raise ArtifactError(f'Cannot read artifact {name}: {exc}') from exc
        if digest.hexdigest() != expected:
            raise ArtifactError(f'Artifact hash mismatch: {name}; restore it from the pinned release')
    return dict(files)
