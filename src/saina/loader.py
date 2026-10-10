"""Load saved training artifacts without a random or untrained fallback head."""
import json
import math
import re
from pathlib import Path
from .prompt import (SHARED_PROMPT, answer_codes, encode, encode_agentic_prompt,  # noqa: F401 (re-exported)
                     encode_head_prompt, encode_shared_prompt, shared_positions)

CHECKPOINT_FORMATS = (1, 2)
LEGACY_MODEL_ID = 'saina-helm-0.8b'  # checkpoints from before releases named themselves


class FrozenHeadScorer:
    prompts = ('BaseScorer.encode_prompt', 'AgenticScorer.encode_prompt')

    def __init__(self, path, device='cuda', max_length=None):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        root = Path(path)
        config = json.loads((root / 'head-config.json').read_text())
        # Default to the context the checkpoint was trained for; operators may set a lower serving limit.
        max_length = max_length if max_length is not None else config.get('max_length', 8192)
        if type(max_length) is not int or max_length <= 0:
            raise ValueError('Context limit must be a positive integer')
        self.torch, self.device, self.max_length = torch, device, max_length
        # Released checkpoints ship merged backbone weights beside the head. A head-only
        # checkpoint without weights reuses the pinned, frozen base model instead.
        merged = (root / 'model.safetensors').is_file() or (root / 'model.safetensors.index.json').is_file()
        if (config.get('prompt') not in self.prompts or
                (not merged and config.get('frozen_backbone') is not True) or
                not re.fullmatch(r'[0-9a-f]{40}', config.get('revision', ''))):
            raise ValueError('Unsupported or unpinned head configuration')
        source, revision = (str(root), None) if merged else (config['model'], config['revision'])
        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=revision, trust_remote_code=False)
        self.codes, code_ids = answer_codes(self.tokenizer)
        if self.codes != config['codes'] or code_ids != config['code_ids']:
            raise ValueError('Checkpoint answer codes do not match the pinned tokenizer')
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(source,
            revision=revision, dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32,
            trust_remote_code=False).to(device).eval().requires_grad_(False)
        self.supported_modes = config.get('supported_modes', ['single_label'])
        self.prompt_format = config['prompt']
        self.head = torch.nn.Linear(self.model.lm_head.weight.shape[1], len(self.codes),
                                    bias=False, device=device, dtype=torch.float32)
        self.head.load_state_dict(load_file(str(root / 'head.safetensors')))
        self.head.eval().requires_grad_(False)
        self.restored_head = True

    def count_text(self, text):
        """Tokens in one billable text part (see prepare.bill)."""
        return len(self.tokenizer.encode(text, add_special_tokens=False)) if text else 0

    def encode(self, context, question, choices, mode='single_label'):
        """Token IDs of the exact model input; their length is checked against the metering header."""
        if mode not in self.supported_modes:
            raise ValueError('This legacy checkpoint was not trained for multi-label prediction')
        return encode(self.prompt_format, self.tokenizer, self.codes, context, question, list(choices),
                      self.max_length, mode)

    def forward(self, ids, count, mode='single_label', temperature=1.0):
        """Probabilities for `count` options from already-encoded input IDs."""
        if mode not in self.supported_modes:
            raise ValueError('This legacy checkpoint was not trained for multi-label prediction')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be positive and finite')
        torch = self.torch
        # For callers' information only; metering uses the encoded inputs, never this attribute.
        self.last_input_tokens = len(ids)
        ids = torch.tensor([list(ids)], device=self.device)
        # No per-request feature cache: a serving process must not retain customer text.
        with torch.inference_mode():
            h = self.model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                use_cache=False, return_dict=True).last_hidden_state[:, -1]
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=h.is_cuda):
                logits = self.head(h).float()[0, :count]
            scaled = logits / temperature
            return (scaled.sigmoid() if mode == 'multi_label' else scaled.softmax(0)).cpu().tolist()

    def predict(self, context, question, choices, temperature=1.0, mode='single_label'):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be positive and finite')
        return self.forward(self.encode(context, question, choices, mode), len(choices), mode, temperature)


class SharedHeadScorer(FrozenHeadScorer):
    """Shared-pass checkpoints: one backbone call answers every question in a request."""
    prompts = (SHARED_PROMPT,)
    shared_trunk_pass = True

    def __init__(self, path, device='cuda', max_length=None):
        root = Path(path)
        config = json.loads((root / 'head-config.json').read_text())
        if config.get('head_type') not in (None, 'routed_residual_v1'):
            raise ValueError('Unsupported shared scoring head')
        if not (root / 'model.safetensors').is_file() and not (root / 'model.safetensors.index.json').is_file():
            raise ValueError('Shared-pass checkpoints must ship merged backbone weights')
        super().__init__(path, device=device, max_length=max_length)
        if config.get('head_type') == 'routed_residual_v1':
            from safetensors.torch import load_file
            from .routed import RoutedResidualHead
            progress = config.get('route_progress')
            if type(progress) not in (int, float) or not math.isfinite(progress) or not 0 <= progress <= 1:
                raise ValueError('Invalid saved routing progress')
            self.head = RoutedResidualHead(self.head, config['experts'], config['bottleneck'], float(progress))
            self.head.load_state_dict(load_file(str(root / 'experts.safetensors')))
            self.head.eval().requires_grad_(False)

    def encode_shared(self, context, questions):
        """Token IDs of the one model input for every question: [(question, choices, mode)] in order."""
        if any(mode not in self.supported_modes for _, _, mode in questions):
            raise ValueError('Checkpoint does not support a requested mode')
        return encode_shared_prompt(self.tokenizer, self.codes, context,
                                    [(q, list(c), m) for q, c, m in questions], self.max_length)

    def forward_shared(self, ids, counts, modes, temperature=1.0):
        """Probability vectors, one per question slot, from one already-encoded shared input."""
        if len(counts) != len(modes) or not counts:
            raise ValueError('Expected one option count and mode per question')
        if any(mode not in self.supported_modes for mode in modes):
            raise ValueError('Checkpoint does not support a requested mode')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be positive and finite')
        torch = self.torch
        positions = shared_positions(self.tokenizer, len(ids), len(counts))
        # For callers' information only; metering uses the encoded inputs, never this attribute.
        self.last_input_tokens = len(ids)
        ids = torch.tensor([list(ids)], device=self.device)
        # No per-request feature cache: a serving process must not retain customer text.
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=ids.is_cuda):
            hidden = self.model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                use_cache=False, return_dict=True).last_hidden_state[0]
            logits = self.head(hidden[positions]).float()
            out = []
            for row, count, mode in zip(logits, counts, modes):
                scaled = row[:count] / temperature
                out.append((scaled.sigmoid() if mode == 'multi_label' else scaled.softmax(0)).cpu().tolist())
        return out

    def forward(self, ids, count, mode='single_label', temperature=1.0):
        return self.forward_shared(ids, [count], [mode], temperature)[0]

    def predict_questions(self, context, questions, temperature=1.0):
        """questions: [{'question': str, 'choices': [str, ...], 'mode': ...}] in caller order."""
        items = [(q['question'], q['choices'], q['mode']) for q in questions]
        return self.forward_shared(self.encode_shared(context, items), [len(c) for _, c, _ in items],
                                   [m for _, _, m in items], temperature)


def checkpoint_format(path):
    """Reject artifacts this runtime does not understand instead of misreading them."""
    config = json.loads((path / 'head-config.json').read_text())
    package = json.loads((path / 'config.json').read_text()) if (path / 'config.json').is_file() else {}
    version = package.get('saina_checkpoint_format', 1)
    if type(version) is not int or version not in CHECKPOINT_FORMATS:
        raise ValueError(f'Unsupported Helm checkpoint format {version!r}; upgrade the saina package')
    shared = config.get('prompt') == SHARED_PROMPT
    for field, expected in (('saina_prompt_format', config.get('prompt')),
                            ('saina_head_type', config.get('head_type', 'linear')),
                            ('shared_trunk_pass', shared)):
        if field in package and package[field] != expected:
            raise ValueError('Conflicting checkpoint architecture metadata: ' + field)
    if shared != (version == 2):
        raise ValueError('Checkpoint format does not match its prompt format')
    return version


def checkpoint_model_id(path):
    # Releases name themselves; checkpoints from before naming are the original Helm 0.8B.
    package = json.loads((path / 'config.json').read_text()) if (path / 'config.json').is_file() else {}
    model_id = package.get('saina_model', LEGACY_MODEL_ID)
    if not isinstance(model_id, str) or not re.fullmatch(r'saina-helm(-[0-9]+)?-[0-9.]+b', model_id):
        raise ValueError('Invalid saina_model name in checkpoint config')
    return model_id


def load_scorer(checkpoint, revision=None, device='cpu', max_length=None):
    path = Path(checkpoint)
    if not path.is_dir():
        if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError('Set SAINA_REVISION to an immutable Hub commit for remote checkpoints')
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(checkpoint, revision=revision,
            allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', '*.tiktoken']))
    if (path / 'head-config.json').is_file():
        model_id = checkpoint_model_id(path)
        scorer_class = SharedHeadScorer if checkpoint_format(path) == 2 else FrozenHeadScorer
        scorer = scorer_class(path, device=device, max_length=max_length)
        scorer.model_id, scorer.revision = model_id, revision
        return scorer
    raise ValueError('Expected a staged Helm checkpoint with head-config.json')
