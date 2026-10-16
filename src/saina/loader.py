"""Load saved training artifacts without a random or untrained fallback head."""
import json
import math
import re
from pathlib import Path
from .prompt import (MULTIMODAL_PROMPT, SHARED_PROMPT, VISION_TOKENS, answer_codes,  # noqa: F401 (re-exported)
                     encode, encode_agentic_prompt, encode_head_prompt, encode_shared_prompt, insert_images,
                     shared_positions)

CHECKPOINT_FORMATS = (1, 2, 3)
# Format 1: one prompt per question. 2: one shared pass for every question. 3: a shared pass that also reads
# images (and audio, as spectrogram images).
FORMAT_PROMPTS = {2: SHARED_PROMPT, 3: MULTIMODAL_PROMPT}
INPUTS = ('text', 'image', 'audio')
LEGACY_MODEL_ID = 'saina-helm-0.8b'  # checkpoints from before releases named themselves
MODEL_NAME = r'saina-helm(-[0-9]+)?(-[a-z]+)?-[0-9.]+b'
SPECIALIST_NAME = r'[a-z][a-z0-9-]{0,31}'
SPECIALIST_FORMATS = (1,)


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


class MultimodalHeadScorer(SharedHeadScorer):
    """Shared-pass checkpoints with a vision tower: images (and audio clips, read as log-mel spectrogram
    images) go before the text of the prompt, and every question sees all of them."""
    prompts = (MULTIMODAL_PROMPT,)

    def __init__(self, path, device='cuda', max_length=None, inputs=('text', 'image')):
        from . import media
        super().__init__(path, device=device, max_length=max_length)
        root = Path(path)
        config = json.loads((root / 'head-config.json').read_text())
        self.inputs = tuple(inputs)
        self.image_processor = load_image_processor(root)
        p = self.image_processor
        policy = config.get('image_policy') or {}
        for name in ('patch_size', 'merge_size', 'temporal_patch_size'):
            if name in policy and policy[name] != getattr(p, name):
                raise ValueError('The image processor does not match the checkpoint image policy')
        # Every image is read at up to IMAGE_MAX_PIXELS (and at least IMAGE_MIN_PIXELS), keeping its aspect ratio.
        self.image_policy = media.ImagePolicy(p.patch_size, p.merge_size, media.IMAGE_MIN_PIXELS,
                                              media.IMAGE_MAX_PIXELS)
        self.vision_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in VISION_TOKENS]
        model_config = self.model.config
        if [getattr(model_config, k, None) for k in ('vision_start_token_id', 'image_token_id', 'vision_end_token_id')
            ] != self.vision_ids:
            raise ValueError('Checkpoint vision tokens do not match its tokenizer')

    def media_tokens(self, media):
        """Input tokens of every image the model reads for this media (placeholders, without markers)."""
        return sum(self.image_policy.tokens(*image.size) for image in (media.images if media else ()))

    def encode_shared(self, context, questions, media=None):
        ids = super().encode_shared(context, questions)
        return insert_images(self.tokenizer, ids, [self.image_policy.tokens(*image.size) for image in media.images]
                             if media else [], self.max_length)

    def forward_shared(self, ids, counts, modes, temperature=1.0, media=None):
        if not media:
            return super().forward_shared(ids, counts, modes, temperature)
        if len(counts) != len(modes) or not counts:
            raise ValueError('Expected one option count and mode per question')
        if any(mode not in self.supported_modes for mode in modes):
            raise ValueError('Checkpoint does not support a requested mode')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be positive and finite')
        torch, policy = self.torch, self.image_policy
        grids = [policy.grid(*image.size) for image in media.images]
        pad = self.vision_ids[1]
        if list(ids).count(pad) != sum(t * h * w // policy.merge_size ** 2 for t, h, w in grids):
            raise ValueError('Encoded input does not match its images')
        positions = shared_positions(self.tokenizer, len(ids), len(counts))
        self.last_input_tokens = len(ids)
        pixels = self.image_processor(images=list(media.images), return_tensors='pt', size={
            'shortest_edge': policy.min_pixels, 'longest_edge': policy.max_pixels})
        if pixels['image_grid_thw'].tolist() != grids:
            raise ValueError('Image preprocessing does not match the encoded input')
        ids = torch.tensor([list(ids)], device=self.device)
        # No per-request feature cache: a serving process must not retain customer text or media.
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=ids.is_cuda):
            hidden = self.model.model(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                pixel_values=pixels['pixel_values'].to(self.device),
                image_grid_thw=pixels['image_grid_thw'].to(self.device),
                mm_token_type_ids=(ids == pad).to(torch.int32),
                use_cache=False, return_dict=True).last_hidden_state[0]
            logits = self.head(hidden[positions]).float()
            out = []
            for row, count, mode in zip(logits, counts, modes):
                scaled = row[:count] / temperature
                out.append((scaled.sigmoid() if mode == 'multi_label' else scaled.softmax(0)).cpu().tolist())
        return out

    def predict_questions(self, context, questions, temperature=1.0, media=None):
        items = [(q['question'], q['choices'], q['mode']) for q in questions]
        return self.forward_shared(self.encode_shared(context, items, media), [len(c) for _, c, _ in items],
                                   [m for _, _, m in items], temperature, media)


def load_image_processor(root):
    """The image processor a multimodal checkpoint was trained with (PIL backend, as in training)."""
    from transformers import AutoImageProcessor
    if not (Path(root) / 'preprocessor_config.json').is_file():
        raise ValueError('Multimodal checkpoints must ship preprocessor_config.json')
    try:
        return AutoImageProcessor.from_pretrained(str(root), backend='pil')
    except (TypeError, ValueError):
        return AutoImageProcessor.from_pretrained(str(root), use_fast=False)


def checkpoint_format(path):
    """Reject artifacts this runtime does not understand instead of misreading them."""
    config = json.loads((path / 'head-config.json').read_text())
    package = json.loads((path / 'config.json').read_text()) if (path / 'config.json').is_file() else {}
    version = package.get('saina_checkpoint_format', 1)
    if type(version) is not int or version not in CHECKPOINT_FORMATS:
        raise ValueError(f'Unsupported Helm checkpoint format {version!r}; upgrade the saina package')
    prompt = config.get('prompt')
    shared = prompt in FORMAT_PROMPTS.values()
    for field, expected in (('saina_prompt_format', prompt),
                            ('saina_head_type', config.get('head_type', 'linear')),
                            ('shared_trunk_pass', shared)):
        if field in package and package[field] != expected:
            raise ValueError('Conflicting checkpoint architecture metadata: ' + field)
    if FORMAT_PROMPTS.get(version) != (prompt if shared else None):
        raise ValueError('Checkpoint format does not match its prompt format')
    return version


def checkpoint_inputs(path):
    """Inputs a checkpoint was trained to read: always text; images and audio only for format 3."""
    package = json.loads((path / 'config.json').read_text()) if (path / 'config.json').is_file() else {}
    inputs = package.get('saina_inputs', ['text'])
    if (not isinstance(inputs, list) or not inputs or inputs[0] != 'text' or len(set(inputs)) != len(inputs)
            or set(inputs) - set(INPUTS)):
        raise ValueError('Invalid saina_inputs in checkpoint config')
    if len(inputs) > 1 and package.get('saina_checkpoint_format') != 3:
        raise ValueError('Only format 3 checkpoints read images or audio')
    return tuple(inputs)


def checkpoint_model_id(path):
    # Releases name themselves; checkpoints from before naming are the original Helm 0.8B.
    package = json.loads((path / 'config.json').read_text()) if (path / 'config.json').is_file() else {}
    model_id = package.get('saina_model', LEGACY_MODEL_ID)
    if not isinstance(model_id, str) or not re.fullmatch(MODEL_NAME, model_id):
        raise ValueError('Invalid saina_model name in checkpoint config')
    return model_id


def staged(checkpoint, revision):
    path = Path(checkpoint)
    if not path.is_dir():
        if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError('Set SAINA_REVISION to an immutable Hub commit for remote checkpoints')
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(checkpoint, revision=revision,
            allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', '*.tiktoken']))
    return path


def low_rank_delta(torch):
    class LowRankDelta(torch.nn.Module):
        """A frozen linear layer plus a specialist's low-rank delta, computed in float32 as in training."""
        def __init__(self, base, down, up, scale):
            super().__init__()
            self.base, self.scale = base, scale
            self.register_buffer('down', down.float().to(base.weight.device), persistent=False)
            self.register_buffer('up', up.float().to(base.weight.device), persistent=False)

        def forward(self, x):
            y = self.base(x)
            return y + (x.to(self.down.dtype) @ self.down.t() @ self.up.t()).to(y.dtype) * self.scale
    return LowRankDelta


def load_specialist(scorer, checkpoint, revision=None):
    """A copy of a multimodal `scorer` that answers as one of its domain specialists.

    A specialist ships low-rank weight deltas (`<module>.down`, `<module>.up`) for one exact base release. The copy
    adds scale * up @ down to every listed module's output (kept separate from the bf16 weights so serving matches
    training exactly) and keeps the base's frozen answer head. The base scorer is left unchanged, so one process
    can serve the base and its specialists side by side."""
    import copy
    from safetensors.torch import load_file
    torch = scorer.torch
    path = staged(checkpoint, revision)
    config = json.loads((path / 'specialist-config.json').read_text())
    if config.get('saina_specialist_format') not in SPECIALIST_FORMATS:
        raise ValueError('Unsupported specialist format; upgrade the saina package')
    name = config.get('specialist')
    if not isinstance(name, str) or not re.fullmatch(SPECIALIST_NAME, name):
        raise ValueError('Invalid specialist name')
    if not isinstance(scorer, MultimodalHeadScorer):
        raise ValueError('Specialists need a multimodal base checkpoint')
    if (config.get('base_model') != scorer.model_id or
            config.get('base_model_safetensors_sha256') != getattr(scorer, 'weights_sha256', None)):
        raise ValueError(f'Specialist {name} was trained for different base weights than {scorer.model_id}')
    inputs = config.get('inputs')
    if not isinstance(inputs, list) or 'text' not in inputs or set(inputs) - set(scorer.inputs):
        raise ValueError('Invalid specialist inputs')
    scale = config.get('scale')
    if type(scale) not in (int, float) or not math.isfinite(scale):
        raise ValueError('Invalid specialist scale')
    deltas = load_file(str(path / 'specialist.safetensors'))
    modules = config.get('modules')
    if not isinstance(modules, list) or set(deltas) != {f'{m}.{k}' for m in modules for k in ('down', 'up')}:
        raise ValueError('Specialist weights do not match its module list')
    special = copy.copy(scorer)
    special.model = copy.deepcopy(scorer.model)
    LowRankDelta = low_rank_delta(torch)
    for m in modules:
        linear = special.model.get_submodule(m)
        down, up = deltas[f'{m}.down'], deltas[f'{m}.up']
        if (not isinstance(linear, torch.nn.Linear) or up.shape[1] != down.shape[0] or
                (up.shape[0], down.shape[1]) != tuple(linear.weight.shape)):
            raise ValueError('Specialist weights do not match the base model')
        parent, _, child = m.rpartition('.')
        setattr(special.model.get_submodule(parent), child, LowRankDelta(linear, down, up, float(scale)))
    special.model.eval().requires_grad_(False)
    special.inputs = tuple(i for i in scorer.inputs if i in inputs)
    special.specialists = {}
    special.specialist, special.specialist_revision = name, revision
    special.specialist_description = config.get('description') if isinstance(config.get('description'), str) else None
    return special


def load_scorer(checkpoint, revision=None, device='cpu', max_length=None):
    path = staged(checkpoint, revision)
    if (path / 'head-config.json').is_file():
        model_id, inputs = checkpoint_model_id(path), checkpoint_inputs(path)
        version = checkpoint_format(path)
        if version == 3:
            scorer = MultimodalHeadScorer(path, device=device, max_length=max_length, inputs=inputs)
        else:
            scorer = (SharedHeadScorer if version == 2 else FrozenHeadScorer)(path, device=device,
                                                                              max_length=max_length)
            scorer.inputs = inputs
        scorer.model_id, scorer.revision = model_id, revision
        # Specialists name the exact base weights they were trained on (see load_specialist).
        manifest = path / 'artifact-manifest.json'
        scorer.weights_sha256 = (json.loads(manifest.read_text()).get('model.safetensors')
                                 if manifest.is_file() else None)
        return scorer
    raise ValueError('Expected a staged Helm checkpoint with head-config.json')
