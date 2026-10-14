"""Exact input-token counting without torch (`pip install 'saina[tokenize]'`).

The counter loads the tokenizer, chat template and answer codes pinned by a staged Helm
checkpoint and encodes prompts byte-for-byte like the inference server, so a gateway can
price a request before any inference runs.
"""
import hashlib
import json
import re
from pathlib import Path
from .prompt import (MULTIMODAL_PROMPT, PROMPT_FORMATS, SHARED_PROMPTS, answer_codes, encode, encode_shared_prompt,
                     insert_images)
from .loader import LEGACY_MODEL_ID, checkpoint_inputs, checkpoint_model_id
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
                 revision=None, tokenizer_sha256=None, model_id=LEGACY_MODEL_ID, inputs=('text',), image_policy=None):
        if prompt_format not in PROMPT_FORMATS:
            raise ValueError('Unsupported prompt format')
        self.tokenizer, self.codes, self.prompt_format = tokenizer, list(codes), prompt_format
        # Shared-pass checkpoints read every question from one model input (see prepare.encode_all).
        self.shared_trunk_pass = prompt_format in SHARED_PROMPTS
        self.supported_modes = tuple(supported_modes)
        self.max_length, self.revision, self.tokenizer_sha256 = max_length, revision, tokenizer_sha256
        self.model_id = model_id  # the release this counter prices, as the server names it
        self.inputs = tuple(inputs)
        if len(self.inputs) > 1 and (prompt_format != MULTIMODAL_PROMPT or image_policy is None):
            raise ValueError('Images and audio need a multimodal checkpoint')
        self.image_policy = image_policy  # saina.media.ImagePolicy: image geometry, without torch

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
        inputs, policy = checkpoint_inputs(root), None
        if config['prompt'] == MULTIMODAL_PROMPT:
            from .media import IMAGE_MAX_PIXELS, IMAGE_MIN_PIXELS, ImagePolicy
            processor = json.loads((root / 'preprocessor_config.json').read_text())
            policy = ImagePolicy(processor['patch_size'], processor['merge_size'], IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS)
        return cls(tokenizer, codes, config['prompt'], config.get('supported_modes', ['single_label']),
                   max_length=max_length, revision=revision, tokenizer_sha256=_digest(root),
                   model_id=checkpoint_model_id(root), inputs=inputs, image_policy=policy)

    def encode(self, context, question, choices, mode='single_label'):
        if mode not in self.supported_modes:
            raise ValueError('Checkpoint does not declare trained multi-label support')
        return encode(self.prompt_format, self.tokenizer, self.codes, context, question, list(choices),
                      self.max_length, mode)

    def encode_shared(self, context, questions, media=None):
        if any(mode not in self.supported_modes for _, _, mode in questions):
            raise ValueError('Checkpoint does not declare trained multi-label support')
        ids = encode_shared_prompt(self.tokenizer, self.codes, context,
                                   [(q, list(c), m) for q, c, m in questions], self.max_length)
        if not media:
            return ids
        return insert_images(self.tokenizer, ids, [self.image_policy.tokens(*image.size) for image in media.images],
                             self.max_length)

    def count_text(self, text):
        """Tokens in one billable text part, without special tokens or prompt formatting."""
        return len(self.tokenizer.encode(text, add_special_tokens=False)) if text else 0

    def media_tokens(self, media):
        """Input tokens of the images and audio spectrograms the model reads (see saina.media.decode_media)."""
        return sum(self.image_policy.tokens(*image.size) for image in media.images) if media else 0

    def media(self, request):
        """Validated, decoded media of a request (raises saina.media.MediaError), or None without media.
        Needs `pip install 'saina[media]'` when the request has images or audio."""
        from .media import decode_media, request_media
        raw = request_media(request, self)
        return decode_media(raw) if raw else None

    def bill_ask(self, request, media=None):
        """Billable input: the context once plus each question's text, options or levels as sent, plus the input
        tokens of its images and audio. Text is billed the same for every model, whatever prompt format or number
        of passes it uses. Pass `media` from `self.media(request)` to avoid decoding the request twice."""
        return bill(billable_ask(request), self, media if media is not None else self.media(request))

    def bill_systemone(self, request, media=None):
        return bill(billable_systemone(request), self, media if media is not None else self.media(request))

    def count_prepared(self, prepared, media=None):
        """Model-input length per forward pass: one per question, or one for a shared-pass checkpoint
        (context-window limit and metering check);
        raises before returning if any question is invalid."""
        return [len(ids) for ids in encode_all(prepared, self, media)]

    def count_ask(self, request, media=None):
        return self.count_prepared(prepare_ask(request), media if media is not None else self.media(request))

    def count_systemone(self, request, media=None):
        return self.count_prepared(prepare_systemone(request), media if media is not None else self.media(request))

    def manifest(self):
        manifest = {'model': self.model_id, 'prompt_format': self.prompt_format,
                    'shared_trunk_pass': self.shared_trunk_pass, 'revision': self.revision,
                    'tokenizer_sha256': self.tokenizer_sha256, 'max_length': self.max_length,
                    'supported_modes': list(self.supported_modes), 'inputs': list(self.inputs)}
        if self.image_policy is not None:
            p = self.image_policy
            manifest['image_policy'] = {'patch_size': p.patch_size, 'merge_size': p.merge_size,
                                        'min_pixels': p.min_pixels, 'max_pixels': p.max_pixels}
        return manifest
