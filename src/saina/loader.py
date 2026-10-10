"""Load saved training artifacts without a random or untrained fallback head."""
import json
import math
import re
from itertools import product
from pathlib import Path
from .errors import InputTooLong

SHARED_PROMPT = 'shared_context_query_slots_v1'
CHECKPOINT_FORMATS = (1, 2)


def encode_head_prompt(tokenizer, codes, context, question, choices, max_length):
    if not 2 <= len(choices) <= len(codes):
        raise ValueError('Unsupported option count')
    # Keep byte-for-byte aligned with BaseScorer.encode_prompt used during training.
    content = ('Answer the multiple-choice question using the supplied context. '
               'Choose exactly one of the provided options. Reply with only its code, '
               'with no leading whitespace, explanation, or punctuation.\n' +
               json.dumps({'context': context, 'question': question, 'options': [
                   {'code': codes[i], 'text': text} for i, text in enumerate(choices)
               ]}, ensure_ascii=False))
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if len(ids) > max_length:
        raise InputTooLong('Input exceeds model context window; shorten it explicitly')
    return ids


def encode_agentic_prompt(tokenizer, codes, context, question, choices, max_length, mode='single_label'):
    if mode == 'single_label':
        return encode_head_prompt(tokenizer, codes, context, question, choices, max_length)
    if mode != 'multi_label' or not 2 <= len(choices) <= len(codes):
        raise ValueError('Invalid mode or option count')
    content = ('Evaluate each option independently using the supplied context. '
               'Select all applicable options; zero, one, or multiple may apply.\n' +
               json.dumps({'mode': mode, 'context': context, 'question': question,
                   'options': [{'code': codes[i], 'text': v} for i, v in enumerate(choices)]}, ensure_ascii=False))
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_length: raise InputTooLong('Input exceeds model context window')
    return ids


def encode_shared_prompt(tokenizer, codes, context, questions, max_length):
    """One prompt for every question, followed by one fixed readout slot per question."""
    if not questions:
        raise ValueError('Expected at least one question')
    items = []
    for i, q in enumerate(questions):
        if not 2 <= len(q['choices']) <= len(codes):
            raise ValueError('Unsupported option count')
        if q['mode'] not in ('single_label', 'multi_label'):
            raise ValueError('Unsupported mode')
        items.append({'id': f'q{i}', 'question': q['question'], 'mode': q['mode'],
                      'options': [{'code': codes[j], 'text': text} for j, text in enumerate(q['choices'])]})
    # Keep byte-for-byte aligned with the training encoder (shared_pass_model.encode_shared).
    content = ('Use the shared context to answer every question. For single_label choose one option; '
               'for multi_label assess each option independently.\n' +
               json.dumps({'context': context, 'questions': items}, ensure_ascii=False))
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    positions = []
    for i in range(len(items)):
        ids.extend(tokenizer.encode(f'\nDecision q{i}:', add_special_tokens=False))
        positions.append(len(ids) - 1)
    if len(ids) > max_length:
        raise InputTooLong('Input exceeds model context window; shorten it explicitly')
    return ids, positions


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
        self.codes, code_ids = [], []
        for width in (1, 2):
            for letters in product('ABCDEFGHIJKLMNOPQRSTUVWXYZ', repeat=width):
                code = ''.join(letters)
                ids = self.tokenizer.encode(code, add_special_tokens=False)
                if len(ids) == 1:
                    self.codes.append(code)
                    code_ids.append(ids[0])
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

    def predict(self, context, question, choices, temperature=1.0, mode='single_label'):
        if mode not in self.supported_modes:
            raise ValueError('This legacy checkpoint was not trained for multi-label prediction')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be positive and finite')
        torch = self.torch
        if self.prompt_format == 'AgenticScorer.encode_prompt':
            ids = encode_agentic_prompt(self.tokenizer, self.codes, context, question, choices, self.max_length, mode)
        else:
            ids = encode_head_prompt(self.tokenizer, self.codes, context, question, choices, self.max_length)
        self.last_input_tokens = len(ids)
        ids = torch.tensor([ids], device=self.device)
        # No per-request feature cache: a serving process must not retain customer text.
        with torch.inference_mode():
            h = self.model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                use_cache=False, return_dict=True).last_hidden_state[:, -1]
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=h.is_cuda):
                logits = self.head(h).float()[0, :len(choices)]
            scaled = logits / temperature
            return (scaled.sigmoid() if mode == 'multi_label' else scaled.softmax(0)).cpu().tolist()


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

    def predict_questions(self, context, questions, temperature=1.0):
        """questions: [{'question': str, 'choices': [str, ...], 'mode': ...}] in caller order."""
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be positive and finite')
        if any(q['mode'] not in self.supported_modes for q in questions):
            raise ValueError('Checkpoint does not support a requested mode')
        torch = self.torch
        ids, positions = encode_shared_prompt(self.tokenizer, self.codes, context, questions, self.max_length)
        self.last_input_tokens = len(ids)
        ids = torch.tensor([ids], device=self.device)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=ids.is_cuda):
            hidden = self.model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                use_cache=False, return_dict=True).last_hidden_state[0]
            logits = self.head(hidden[positions]).float()
            out = []
            for row, q in zip(logits, questions):
                scaled = row[:len(q['choices'])] / temperature
                out.append((scaled.sigmoid() if q['mode'] == 'multi_label' else scaled.softmax(0)).cpu().tolist())
        if len(out) != len(questions):
            raise ValueError('Model returned the wrong number of answers')
        return out

    def predict(self, context, question, choices, temperature=1.0, mode='single_label'):
        return self.predict_questions(context, [{'question': question, 'choices': choices, 'mode': mode}],
                                      temperature)[0]


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
    model_id = package.get('saina_model', 'saina-helm-0.8b')
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
