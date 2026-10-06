"""Load saved training artifacts without a random or untrained fallback head."""
import json
import math
import re
from itertools import product
from pathlib import Path
from .errors import InputTooLong


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


class FrozenHeadScorer:
    def __init__(self, path, device='cuda', max_length=8192):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        self.torch, self.device, self.max_length = torch, device, max_length
        root = Path(path)
        config = json.loads((root / 'head-config.json').read_text())
        # Released checkpoints ship merged backbone weights beside the head. A head-only
        # checkpoint without weights reuses the pinned, frozen base model instead.
        merged = (root / 'model.safetensors').is_file() or (root / 'model.safetensors.index.json').is_file()
        if (config.get('prompt') not in ('BaseScorer.encode_prompt', 'AgenticScorer.encode_prompt') or
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


def load_scorer(checkpoint, revision=None, device='cpu', max_length=8192):
    path = Path(checkpoint)
    if not path.is_dir():
        if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError('Set SAINA_REVISION to an immutable Hub commit for remote checkpoints')
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(checkpoint, revision=revision,
            allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', '*.tiktoken']))
    if (path / 'head-config.json').is_file():
        return FrozenHeadScorer(path, device=device, max_length=max_length)
    raise ValueError('Expected a staged Helm checkpoint with head-config.json')
