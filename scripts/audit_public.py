"""Fail closed on unexpected release files and common secret/private-host patterns.

Heuristic guard, not proof of absence of secrets. Review every public diff too.
Only paths are printed on failure, never matching secret values.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    rb'(?:hf_|npm_|gh[pousr]_)[A-Za-z0-9]{20,}',
    rb'sk-[A-Za-z0-9_-]{20,}',
    rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    rb'(?:/Users/|/home/)[A-Za-z0-9_.-]+/',
    rb'\b(?:192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|100\.107\.\d+\.\d+)\b',
]
BANNED = {'.env', '.npmrc', '.pypirc', 'credentials', 'auth.json', 'trainer.pt'}
BAD_SUFFIXES = {'.safetensors', '.pt', '.pth', '.onnx', '.gguf', '.jsonl', '.pem', '.key'}

def check(path, data):
    parts = Path(path).parts
    if any(x in BANNED or x.startswith('.env.') for x in parts):
        raise ValueError(f'Forbidden credential/config path: {path}')
    if Path(path).suffix in BAD_SUFFIXES:
        raise ValueError(f'Model/data/private-key artifact: {path}')
    if any(re.search(pattern, data) for pattern in PATTERNS):
        raise ValueError(f'Possible sensitive content: {path}')

def main():
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for path in filter(None, paths):
        target = ROOT / path
        if target.is_symlink():
            raise ValueError(f'Symlink needs explicit review: {path}')
        check(path, target.read_bytes())
    # Check committed history too; deleting a leaked secret does not remove it.
    commits = subprocess.check_output(['git', 'rev-list', '--all'], cwd=ROOT).decode().split()
    for commit in commits:
        tree = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit], cwd=ROOT).decode().splitlines()
        for path in tree:
            check(path, subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT))
    print('Public tree and committed history passed heuristic checks. Manual review still required.')

if __name__ == '__main__':
    main()
