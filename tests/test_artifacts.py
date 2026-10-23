import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from saina.artifacts import ArtifactError, verify_artifacts
from saina.cli import main


@pytest.fixture
def bundle(tmp_path):
    files = {'head-config.json': b'{}', 'head.safetensors': b'head',
             'merged-hf/tokenizer.json': b'tokenizer'}
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    mapping = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    (tmp_path / 'artifact-manifest.json').write_text(json.dumps(mapping))
    return tmp_path, mapping


def test_native_and_gguf_manifest_compatibility(bundle):
    root, mapping = bundle
    assert verify_artifacts(root, required=['head.safetensors']) == mapping
    (root / 'gguf-manifest.json').write_text(json.dumps({
        'files': mapping, 'status': 'private-prototype-not-release-approved',
        'llama_cpp_commit': 'a' * 40}))
    assert verify_artifacts(root, manifest='gguf-manifest.json') == mapping


def test_corruption_and_missing_asset(bundle):
    root, _ = bundle
    (root / 'head.safetensors').write_bytes(b'corrupted')
    with pytest.raises(ArtifactError, match='hash mismatch: head.safetensors'):
        verify_artifacts(root)
    (root / 'head.safetensors').unlink()
    with pytest.raises(ArtifactError, match='Missing artifact: head.safetensors'):
        verify_artifacts(root)


def test_required_asset_cannot_be_omitted(bundle):
    root, mapping = bundle
    del mapping['head.safetensors']
    (root / 'artifact-manifest.json').write_text(json.dumps(mapping))
    with pytest.raises(ArtifactError, match='missing from manifest: head.safetensors'):
        verify_artifacts(root, required=['head.safetensors'])


@pytest.mark.parametrize('name', ['../outside', '/absolute', 'C:/drive', 'folder\\file',
                                  'a/../b', 'a//b', './head', 'bad\x00name'])
def test_reject_nonportable_or_escaping_paths(bundle, name):
    root, _ = bundle
    (root / 'artifact-manifest.json').write_text(json.dumps({name: 'a' * 64}))
    with pytest.raises(ArtifactError, match='Invalid artifact path'):
        verify_artifacts(root)


@pytest.mark.parametrize('data', ['[]', '{}', '{"files": []}', '{bad',
                                   '{"head": "bad"}', '{"head": 123}',
                                   '{"head": "a", "head": "b"}'])
def test_bad_manifest_is_actionable(bundle, data):
    root, _ = bundle
    (root / 'artifact-manifest.json').write_text(data)
    with pytest.raises(ArtifactError):
        verify_artifacts(root)


def test_huggingface_cache_symlinks_are_supported(bundle, tmp_path):
    root, mapping = bundle
    blob = tmp_path / 'cache-blob'
    blob.write_bytes((root / 'head.safetensors').read_bytes())
    (root / 'head.safetensors').unlink()
    (root / 'head.safetensors').symlink_to(blob)
    assert verify_artifacts(root) == mapping


def test_doctor_success_is_only_an_integrity_claim(bundle, capsys):
    root, mapping = bundle
    assert main(['doctor', '--checkpoint', str(root), '--json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'integrity_verified'
    assert result['files'] == sorted(mapping)
    assert 'readiness are not checked' in result['message']
    assert main(['doctor', '--checkpoint', str(root), '--require', 'helm-hidden', '--json']) == 1
    assert 'helm-hidden' in json.loads(capsys.readouterr().out)['message']


def test_doctor_missing_manifest_fails_without_traceback(tmp_path, capsys):
    assert main(['doctor', '--checkpoint', str(tmp_path), '--json']) == 1
    assert 'Cannot read manifest' in json.loads(capsys.readouterr().out)['message']


def test_module_entrypoint_needs_no_site_packages(bundle):
    root, _ = bundle
    source = Path(__file__).resolve().parents[1] / 'src'
    # -S excludes installed inference packages; the complete command must remain offline.
    result = subprocess.run([sys.executable, '-S', '-m', 'saina', 'doctor',
                             '--checkpoint', str(root), '--json'], cwd=source,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['status'] == 'integrity_verified'
