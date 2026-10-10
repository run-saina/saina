"""Exact input-token counting without torch (`pip install 'saina[tokenize]'`).

The counter loads the tokenizer, chat template and answer codes pinned by a staged Helm
checkpoint and encodes prompts byte-for-byte like the inference server, so a gateway can
price a request before any inference runs.
"""
import hashlib
import json
import re
from pathlib import Path
from .prompt import PROMPT_FORMATS, SHARED_PROMPT, answer_codes, encode, encode_shared_prompt
from .loader import LEGACY_MODEL_ID, checkpoint_model_id
from .prepare import bill, billable_ask, billable_systemone, prepare_ask, prepare_systemone, encode_all

TOKENIZER_FILES = ['*.json', '*.txt', '*.model', '*.jinja', '*.tiktoken']


def _digest(root):
    """SHA-256 over every tokenizer-relevant file, so a manifest can pin what was counted."""
    h = hashlib.sha256()
    for name in sorted(p.name for p in root.iterdir() if p.is_file()):
        if name.endswith(('.json', '.txt', '.model', '.jinja', '.tiktoken')) and name != 'config.json':
            h.update(name.encode() + b'\0' + (root / name).read_bytes() + b'\0')
    return h.hexdigest()


class TokenCounter:
    def __init__(self, tokenizer, codes, prompt_format, supported_modes, max_length=8192,
                 revision=None, tokenizer_sha256=None, model_id=LEGACY_MODEL_ID):
        if prompt_format not in PROMPT_FORMATS:
            raise ValueError('Unsupported prompt format')
        self.tokenizer, self.codes, self.prompt_format = tokenizer, list(codes), prompt_format
        # Shared-pass checkpoints read every question from one model input (see prepare.encode_all).
        self.shared_trunk_pass = prompt_format == SHARED_PROMPT
        self.supported_modes = tuple(supported_modes)
        self.max_length, self.revision, self.tokenizer_sha256 = max_length, revision, tokenizer_sha256
        self.model_id = model_id  # the release this counter prices, as the server names it

    @classmethod
    def from_checkpoint(cls, checkpoint, revision=None, max_length=None):
        from transformers import AutoTokenizer
        root = Path(checkpoint)
        if not root.is_dir():
            if not revision or not re.fullmatch(r'[0-9a-f]{40}', revision):
                raise ValueError('Pin remote checkpoints to an immutable Hub commit')
            from huggingface_hub import snapshot_download
            root = Path(snapshot_download(checkpoint, revision=revision, allow_patterns=TOKENIZER_FILES))
        config = json.loads((root / 'head-config.json').read_text())
        tokenizer = AutoTokenizer.from_pretrained(str(root), trust_remote_code=False)
        codes, code_ids = answer_codes(tokenizer)
        if codes != config['codes'] or code_ids != config['code_ids']:
            raise ValueError('Checkpoint answer codes do not match the pinned tokenizer')
        # Default to the checkpoint's own context limit, like the inference server.
        max_length = max_length if max_length is not None else config.get('max_length', 8192)
        return cls(tokenizer, codes, config['prompt'], config.get('supported_modes', ['single_label']),
                   max_length=max_length, revision=revision, tokenizer_sha256=_digest(root),
                   model_id=checkpoint_model_id(root))

    def encode(self, context, question, choices, mode='single_label'):
        if mode not in self.supported_modes:
            raise ValueError('Checkpoint does not declare trained multi-label support')
        return encode(self.prompt_format, self.tokenizer, self.codes, context, question, list(choices),
                      self.max_length, mode)

    def encode_shared(self, context, questions):
        if any(mode not in self.supported_modes for _, _, mode in questions):
            raise ValueError('Checkpoint does not declare trained multi-label support')
        return encode_shared_prompt(self.tokenizer, self.codes, context,
                                    [(q, list(c), m) for q, c, m in questions], self.max_length)

    def count_text(self, text):
        """Tokens in one billable text part, without special tokens or prompt formatting."""
        return len(self.tokenizer.encode(text, add_special_tokens=False)) if text else 0

    def bill_ask(self, request):
        """Billable input: the context once plus each question's text, options or levels as sent. The same
        for every model, whatever prompt format or number of passes it uses."""
        return bill(billable_ask(request), self)

    def bill_systemone(self, request):
        return bill(billable_systemone(request), self)

    def count_prepared(self, prepared):
        """Model-input length per forward pass: one per question, or one for a shared-pass checkpoint
        (context-window limit and metering check);
        raises before returning if any question is invalid."""
        return [len(ids) for ids in encode_all(prepared, self)]

    def count_ask(self, request):
        return self.count_prepared(prepare_ask(request))

    def count_systemone(self, request):
        return self.count_prepared(prepare_systemone(request))

    def manifest(self):
        return {'model': self.model_id, 'prompt_format': self.prompt_format, 'shared_trunk_pass': self.shared_trunk_pass,
                'revision': self.revision,
                'tokenizer_sha256': self.tokenizer_sha256, 'max_length': self.max_length,
                'supported_modes': list(self.supported_modes)}
