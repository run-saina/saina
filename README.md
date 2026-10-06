# Saina

Saina's HTTP SDK, shared decision contract, and optional Helm local inference.
Saina is operated by **Rama Labs Inc.** Website: https://saina.run.

**Pre-release source:** packages and model weights are not published yet.
The current private model candidate is not cleared for commercial distribution.
The Apache-2.0 code license does not grant rights to third-party weights or data.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/run-saina/saina/blob/main/examples/saina_colab.ipynb)
Try Helm in [`examples/saina_colab.ipynb`](examples/saina_colab.ipynb): run the
[Hugging Face weights](https://huggingface.co/run-saina/saina-helm-0.8b) locally through the SDK pipeline,
or call [Replicate](https://replicate.com/run-saina/saina-helm-0.8b) or your own endpoint.

## Install from source

```sh
pip install .                # dependency-free HTTP client
pip install '.[local,server]' # local inference and authenticated server
```

```python
from saina import Saina

client = Saina('https://your-endpoint.example', api_key='YOUR_KEY')

result = client.ask(
    model='helm-0.8b',
    state='I was charged twice.',
    questions={
        'issue': {
            'type': 'single_choice',
            'question': 'Identify the banking issue.',
            'options': {
                'duplicate': 'duplicate charge',
                'delivery': 'card delivery',
            },
        },
    },
)
```

Keep credentials in environment variables or a secret manager, not source files.

## Transformers / local inference

```python
from saina import register_transformers
from transformers import pipeline

register_transformers()

helm = pipeline(
    'saina-classification',
    model='/path/to/staged/checkpoint',
    device='cuda:0',
    model_kwargs={'device': 'cuda:0'},
)

probabilities = helm({
    'task': 'Identify the banking issue.',
    'input': 'I was charged twice.',
    'choices': ['duplicate charge', 'card delivery'],
    'mode': 'single_label',
})
```

Use `multi_label` for independent memberships; output order always follows choices.
Lists of requests are supported serially, not optimized tensor batching. Hub loading
requires an immutable revision and installed registration code, not remote code.
CPU operation is supported by the loader but is not an optimized ONNX release.

## JavaScript / TypeScript

Source: `clients/javascript`; planned registry package: `@run-saina/sdk`.
Exports `Saina` and `SainaError`, retaining `SainaHelm` aliases for compatibility.

## Serving and contracts

Set `SAINA_API_KEY`, `SAINA_CHECKPOINT`, `SAINA_DEVICE` and optionally
`SAINA_REVISION` (pinned Hub commit), then:

```sh
uvicorn saina.server:create_app --factory --host 127.0.0.1 --port 8000
```

Native typed requests use `/v1/ask`. The `saina.contract` module owns their schema.
`saina.jev` provides the Jev/System One adapter; endpoint wiring is separate.
Do not expose a plain HTTP server publicly; place it behind authenticated TLS.

To run the server in Docker or deploy it to a cloud, see
[run-saina/deploy](https://github.com/run-saina/deploy).

## Development

`PYTHONPATH=src python -m unittest discover -s tests`
and `cd clients/javascript && npm test`.

No training datasets, private history, model weights, or credentials are included.

## Developer contact

For integration support and developer enquiries, email [dev@saina.run](mailto:dev@saina.run).
