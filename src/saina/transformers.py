"""Installed-code Transformers integration; no Hub remote code execution."""
import math
from pathlib import Path
from transformers import AutoConfig, AutoModel, Pipeline, PretrainedConfig, PreTrainedModel
from transformers.pipelines import PIPELINE_REGISTRY


class HelmConfig(PretrainedConfig):
    model_type = 'saina_helm'


class HelmModel(PreTrainedModel):
    config_class = HelmConfig

    def __init__(self, config, scorer):
        super().__init__(config)
        self.scorer = scorer
        self.backbone = scorer.model
        self.probability_head = scorer.head

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, config=None,
                        revision=None, device='cpu', max_length=16384, **kwargs):
        from .loader import load_scorer
        if args:
            raise ValueError('Positional model arguments are not supported')
        if kwargs.get('trust_remote_code'):
            raise ValueError('Remote code is not needed or supported')
        if kwargs.get('device_map') is not None:
            raise ValueError('Use a single device; device_map is not supported')
        scorer = load_scorer(str(pretrained_model_name_or_path), revision=revision,
                             device=str(device), max_length=max_length)
        return cls(config or HelmConfig(), scorer).eval()

    def forward(self, request):
        self.scorer.device = str(self.device)
        return {'probabilities': self.scorer.predict(
            request['input'], request['task'], request['choices'], mode=request['mode'])}

    def save_pretrained(self, *args, **kwargs):
        raise NotImplementedError('Use the Helm checkpoint staging exporter; generic saving would lose adapter metadata')


class SainaClassificationPipeline(Pipeline):
    _load_tokenizer = False
    _load_processor = False
    _load_feature_extractor = False
    _load_image_processor = False

    def _sanitize_parameters(self, **kwargs):
        if kwargs:
            raise ValueError('Put task, input, choices and mode in the request dictionary')
        return {}, {}, {}

    def preprocess(self, inputs):
        if not isinstance(inputs, dict) or set(inputs) - {'task', 'input', 'choices', 'mode'}:
            raise ValueError('Expected a Saina request dictionary')
        request = dict(inputs)
        request.setdefault('mode', 'single_label')
        if not all(isinstance(request.get(k), str) for k in ('task', 'input')):
            raise ValueError('task and input must be strings')
        choices = request.get('choices')
        if not isinstance(choices, list) or len(choices) < 2 or not all(isinstance(x, str) for x in choices):
            raise ValueError('choices must contain at least two strings in caller order')
        if request['mode'] not in ('single_label', 'multi_label'):
            raise ValueError('mode must be single_label or multi_label')
        return {'request': request}

    def _forward(self, model_inputs):
        result = self.model(**model_inputs)
        return {**result, 'count':len(model_inputs['request']['choices']),
                'mode':model_inputs['request']['mode']}

    def postprocess(self, model_outputs):
        values = model_outputs['probabilities']
        if len(values) != model_outputs['count'] or not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
            raise ValueError('Invalid probability output')
        if model_outputs['mode'] == 'single_label' and not math.isclose(sum(values), 1, abs_tol=1e-4):
            raise ValueError('Single-label probabilities must sum to one')
        return values

    def __call__(self, inputs, **kwargs):
        # Variable option counts/modes: serial list processing, not padded GPU batching.
        if isinstance(inputs, list):
            return [self.run_single(x, {}, {}, {}) for x in inputs] if not kwargs else self._reject_kwargs(kwargs)
        if kwargs:
            self._reject_kwargs(kwargs)
        return self.run_single(inputs, {}, {}, {})

    @staticmethod
    def _reject_kwargs(kwargs):
        raise ValueError('Pipeline call options are unsupported; pass a request dictionary')


def register():
    AutoConfig.register('saina_helm', HelmConfig, exist_ok=True)
    AutoModel.register(HelmConfig, HelmModel, exist_ok=True)
    if 'saina-classification' not in PIPELINE_REGISTRY.get_supported_tasks():
        PIPELINE_REGISTRY.register_pipeline('saina-classification',
            pipeline_class=SainaClassificationPipeline, pt_model=HelmModel, type='text')
