"""Local adapters for the existing Helm runtimes. Optional engines load lazily."""
import json
import math
import platform
from pathlib import Path
from .prompt import answer_codes, encode_agentic_prompt


def select_backend(requested='auto', system=None, machine=None):
    if requested not in ('auto', 'native', 'mlx', 'gguf'):
        raise ValueError('Backend must be auto, native, mlx or gguf')
    apple = (system or platform.system()) == 'Darwin' and (machine or platform.machine()) == 'arm64'
    if requested == 'mlx' and not apple:
        raise ValueError('MLX requires Apple Silicon; use --backend native or gguf here')
    return ('mlx' if apple else 'native') if requested == 'auto' else requested


class TextScorer:
    inputs = ('text',)
    model_id = 'saina-helm-0.8b'

    def configure(self, root, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.config = json.loads((Path(root) / 'head-config.json').read_text())
        if self.config.get('prompt') != 'AgenticScorer.encode_prompt':
            raise ValueError('This backend requires the Helm AgenticScorer prompt')
        self.codes, self.code_ids = answer_codes(tokenizer)
        if self.codes != self.config.get('codes') or self.code_ids != self.config.get('code_ids'):
            raise ValueError('Checkpoint answer codes do not match the tokenizer')
        self.supported_modes = self.config.get('supported_modes', ['single_label'])
        limit = self.config.get('max_length', 8192)
        self.max_length = min(limit, max_length) if max_length is not None else limit
        if type(self.max_length) is not int or self.max_length <= 0:
            raise ValueError('Context limit must be a positive integer')

    def count_text(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False)) if text else 0

    def encode(self, context, question, choices, mode='single_label'):
        if mode not in self.supported_modes:
            raise ValueError('Unsupported prediction mode')
        return encode_agentic_prompt(self.tokenizer, self.codes, context, question, list(choices), self.max_length, mode)

    def validate_forward(self, ids, count, mode, temperature):
        from .errors import InputTooLong
        if not ids or len(ids) > self.max_length:
            raise InputTooLong('Input exceeds model context window')
        if mode not in self.supported_modes or not 2 <= count <= len(self.codes):
            raise ValueError('Unsupported mode or option count')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be positive and finite')

    def predict(self, context, question, choices, temperature=1., mode='single_label'):
        return self.forward(self.encode(context, question, choices, mode), len(choices), mode, temperature)


class MLXScorer(TextScorer):
    """Same last-hidden-state/head-row readout as integrations/mlx/score.py."""
    def __init__(self, root, max_length=None):
        import mlx.core as mx
        from mlx_lm import load
        config = json.loads((Path(root) / 'config.json').read_text())
        if config.get('quantization') or config.get('quantization_config'):
            raise ValueError('Use the unquantized FP16 MLX build; quantization changes the decision head')
        self.mx = mx
        self.model, tokenizer = load(str(root))
        self.configure(root, tokenizer, max_length)
        head = self.model.language_model.lm_head
        if max(self.code_ids) >= head.weight.shape[0]:
            raise ValueError('MLX head does not contain every answer code')

    def forward(self, ids, count, mode='single_label', temperature=1.):
        self.validate_forward(ids, count, mode, temperature)
        mx = self.mx
        h = self.model.model(mx.array([list(ids)]))[:, -1]
        rows = self.model.language_model.lm_head.weight[mx.array(self.code_ids[:count])]
        z = (h @ rows.T)[0].astype(mx.float32) / temperature
        return (mx.sigmoid(z) if mode == 'multi_label' else mx.softmax(z)).tolist()


class GGUFScorer(TextScorer):
    """Persistent helm-hidden runner, using the existing FP32 external-head algorithm."""
    def __init__(self, root, runner, filename='helm-trunk-f32.gguf', threads=8, max_length=None):
        import subprocess
        import threading
        import numpy as np
        from safetensors.numpy import load_file
        from transformers import AutoTokenizer
        from .artifacts import verify_artifacts
        if platform.system() == 'Windows':
            raise ValueError('The GGUF pipe runner is currently supported on POSIX only')
        if type(threads) is not int or not 1 <= threads <= 16:
            raise ValueError('Threads must be between 1 and 16')
        if Path(filename).name != filename:
            raise ValueError('GGUF filename must be a basename')
        root = Path(root)
        verify_artifacts(root, manifest='gguf-manifest.json', required=[filename, 'head-config.json',
            'head.safetensors', 'merged-hf/config.json', 'merged-hf/tokenizer.json',
            'merged-hf/tokenizer_config.json', 'merged-hf/chat_template.jinja'])
        tokenizer = AutoTokenizer.from_pretrained(root / 'merged-hf', trust_remote_code=False, local_files_only=True)
        self.configure(root, tokenizer, max_length)
        self.max_length = min(self.max_length, 8192)
        self.np = np
        self.weight = load_file(str(root / 'head.safetensors'))['weight']
        if self.weight.ndim != 2 or self.weight.shape[0] != len(self.codes) or not np.isfinite(self.weight).all():
            raise ValueError('Invalid GGUF decision head')
        self.lock = threading.Lock()
        self.helper = subprocess.Popen([str(Path(runner).resolve(strict=True)), str(root / filename), str(threads)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def forward(self, ids, count, mode='single_label', temperature=1.):
        import selectors
        self.validate_forward(ids, count, mode, temperature)
        with self.lock:
            if self.helper.poll() is not None:
                raise RuntimeError('GGUF runner has exited; restart saina serve')
            self.helper.stdin.write(' '.join(map(str, ids)) + '\n')
            self.helper.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(self.helper.stdout, selectors.EVENT_READ)
                if not selector.select(timeout=300):
                    self.helper.kill()
                    self.helper.wait()
                    raise TimeoutError('GGUF inference exceeded 300 seconds')
            line = self.helper.stdout.readline()
        np = self.np
        hidden = np.fromstring(line, sep=' ', dtype=np.float32)
        if hidden.shape != (self.weight.shape[1],) or not np.isfinite(hidden).all():
            raise RuntimeError('GGUF runner returned an invalid hidden state')
        logits = (self.weight[:count] @ hidden) / temperature
        if mode == 'multi_label':
            p = np.exp(-np.logaddexp(0, -logits))
        else:
            p = np.exp(logits - logits.max())
            p /= p.sum()
        return p.tolist()

    def close(self):
        import subprocess
        with self.lock:
            if self.helper.poll() is None:
                self.helper.stdin.close()
                try:
                    self.helper.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.helper.kill()
                    self.helper.wait()
            self.helper.stdout.close()


def load_backend(checkpoint, backend, revision=None, max_length=None, runner=None, filename='helm-trunk-f32.gguf', threads=8):
    from .loader import load_scorer, staged
    if backend not in ('native', 'mlx', 'gguf'):
        raise ValueError('Unknown local backend')
    if backend == 'native':
        root = staged(checkpoint, revision)
        if (root / 'artifact-manifest.json').is_file():
            from .artifacts import verify_artifacts
            verify_artifacts(root, required=['head-config.json', 'head.safetensors'])
        scorer = load_scorer(str(root), revision=revision, device='cpu', max_length=max_length)
        return scorer
    if backend == 'gguf':
        if not Path(checkpoint).is_dir() or not runner:
            raise ValueError('GGUF needs a local --checkpoint bundle and --runner executable')
        scorer = GGUFScorer(checkpoint, runner, filename, threads, max_length)
    else:
        scorer = MLXScorer(staged(checkpoint, revision), max_length)
    scorer.revision = revision
    return scorer
