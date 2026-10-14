"""Build a multimodal Helm package (checkpoint format 3) from an adapter-format training checkpoint.

    python scripts/build_multimodal_package.py CHECKPOINT OUTPUT --model-name NAME [--inputs text,image]

CHECKPOINT holds ``adapter/`` (a LoRA adapter over the language trunk), ``head.safetensors``,
``experts.safetensors``, ``head-config.json`` (prompt ``shared_multimodal_native_processor_v2``) and
``image_processor/``. The base model named and pinned by ``head-config.json`` is read from the Hugging Face
cache or Hub. OUTPUT gets the same layout as a Helm 2 text package plus the image processor:

    config.json                 Qwen3.5 config + saina_checkpoint_format 3, saina_model, saina_inputs, ...
    model.safetensors           merged backbone in bfloat16, language model and vision tower
    head.safetensors            answer-code head
    experts.safetensors         routed residual experts (head_type routed_residual_v1)
    head-config.json            answer codes, prompt format, routing and image policy
    preprocessor_config.json    the image processor the model was trained with
    tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json
    merge-provenance.json       base model, revision and hashes of the merged inputs
    artifact-manifest.json      SHA-256 of every other file

Requires torch, transformers, safetensors and peft. Nothing is uploaded.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

PROMPT = 'shared_multimodal_native_processor_v2'
FORMAT = 3
INPUTS = ('text', 'image', 'audio')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def merged_backbone(base, revision, adapter, rank, alpha):
    """The base Qwen3.5 model with the language-trunk adapter merged in float32."""
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import Qwen3_5ForConditionalGeneration
    model = Qwen3_5ForConditionalGeneration.from_pretrained(base, revision=revision, dtype=torch.float32,
                                                            trust_remote_code=False)
    # The same targets the adapter was trained with: every linear projection of the language trunk.
    targets = [name for name, module in model.model.named_modules()
               if name.startswith('language_model.layers.') and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise ValueError('No language-trunk projections found in the base model')
    peft = get_peft_model(model.model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias='none',
                                                  target_modules=targets))
    state = load_file(str(adapter / 'adapter_model.safetensors'))
    result = set_peft_model_state_dict(peft, state)
    if result.unexpected_keys:
        raise ValueError(f'Adapter has {len(result.unexpected_keys)} weights the base model does not')
    missing = [k for k in result.missing_keys if 'lora_' in k]
    if missing:
        raise ValueError(f'Adapter is missing {len(missing)} LoRA weights')
    model.model = peft.merge_and_unload()
    return model.to(torch.bfloat16).eval()


def build(checkpoint, output, model_name, inputs):
    from huggingface_hub import hf_hub_download
    checkpoint, output = Path(checkpoint), Path(output)
    config = json.loads((checkpoint / 'head-config.json').read_text())
    if config.get('prompt') != PROMPT:
        raise ValueError(f'Expected a {PROMPT} checkpoint')
    if config.get('requires_adapter') != 'adapter' or not (checkpoint / 'adapter').is_dir():
        raise ValueError('Expected an adapter-format checkpoint')
    if not re.fullmatch(r'[0-9a-f]{40}', config.get('revision', '')):
        raise ValueError('The checkpoint does not pin its base model revision')
    if config.get('head_type') not in (None, 'routed_residual_v1'):
        raise ValueError('Unsupported head type')
    if not re.fullmatch(r'saina-helm(-[0-9]+)?(-[a-z]+)?-[0-9.]+b', model_name):
        raise ValueError('Model names look like saina-helm-2-0.8b')
    if 'text' not in inputs or set(inputs) - set(INPUTS):
        raise ValueError(f'Inputs are text plus any of {INPUTS[1:]}')
    if not (checkpoint / 'image_processor' / 'preprocessor_config.json').is_file():
        raise ValueError('Missing image_processor/preprocessor_config.json')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} is not empty')
    output.mkdir(parents=True, exist_ok=True)

    model = merged_backbone(config['model'], config['revision'], checkpoint / 'adapter',
                            config['lora_rank'], config['lora_alpha'])
    model.save_pretrained(str(output), safe_serialization=True)
    del model

    for name in ('head.safetensors', 'experts.safetensors'):
        if (checkpoint / name).is_file():
            shutil.copyfile(checkpoint / name, output / name)
    shutil.copyfile(checkpoint / 'image_processor' / 'preprocessor_config.json', output / 'preprocessor_config.json')
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        shutil.copyfile(hf_hub_download(config['model'], name, revision=config['revision']), output / name)

    head = {k: v for k, v in config.items() if k not in ('requires_adapter', 'base_weights_frozen')}
    head.update(merged_adapter=True)
    write_json(output / 'head-config.json', head)

    package = json.loads((output / 'config.json').read_text())
    package.update(saina_checkpoint_format=FORMAT, saina_model=model_name, saina_prompt_format=PROMPT,
                   saina_head_type=config.get('head_type', 'linear'), shared_trunk_pass=True,
                   saina_inputs=list(inputs))
    write_json(output / 'config.json', package)

    write_json(output / 'merge-provenance.json', {
        'source': 'training checkpoint',
        'base_model': config['model'],
        'base_revision': config['revision'],
        'merge': 'language-trunk LoRA merged in float32 into the base model, saved bfloat16 with its vision '
                 'tower (lm_head tied to the input embeddings)',
        'source_hashes': {name: sha256(checkpoint / name) for name in
                          ('adapter/adapter_model.safetensors', 'adapter/adapter_config.json', 'head.safetensors',
                           'experts.safetensors', 'head-config.json', 'image_processor/preprocessor_config.json')
                          if (checkpoint / name).is_file()},
    })
    write_json(output / 'artifact-manifest.json', {
        p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'artifact-manifest.json'})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('checkpoint')
    parser.add_argument('output')
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--inputs', default='text,image',
                        help='comma-separated inputs the model was trained for (text, image, audio)')
    args = parser.parse_args()
    print(build(args.checkpoint, args.output, args.model_name, tuple(args.inputs.split(','))))


if __name__ == '__main__':
    main()
