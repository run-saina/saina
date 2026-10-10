# Saina

Saina's HTTP SDK, shared decision contract, and optional Helm local inference.
Saina is operated by **Rama Labs Inc.** Website: https://saina.run.

The current model is not cleared for commercial distribution.
The Apache-2.0 code license does not grant rights to third-party weights or data.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/run-saina/saina/blob/main/examples/saina_colab.ipynb)
Models: [Helm 2 0.8B](https://huggingface.co/run-saina/saina-helm-2-0.8b) (current) and the original
[Helm 0.8B](https://huggingface.co/run-saina/saina-helm-0.8b). Try Helm in
[`examples/saina_colab.ipynb`](examples/saina_colab.ipynb), run the weights locally through the SDK,
or call [Replicate](https://replicate.com/run-saina/saina-helm-2-0.8b) or your own endpoint.

## Install

```sh
pip install saina                  # dependency-free HTTP client
pip install 'saina[local,server]'  # local inference and authenticated server
```

From a checkout, use `pip install .` or `pip install '.[local,server]'`.

```python
from saina import Saina

client = Saina('https://your-endpoint.example', api_key='YOUR_KEY')

result = client.ask(
    model='saina-helm',  # whichever Helm the endpoint serves; or pin 'helm-2-0.8b'
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

## Model names

`saina-helm` selects whichever Helm the endpoint serves. Versioned names (`helm-2-0.8b` /
`saina-helm-2-0.8b`, `helm-0.8b` / `saina-helm-0.8b`) only reach that exact model; an endpoint
serving a different version answers `404`. Responses always name the model that answered.

## Local inference

`saina[local]` loads released checkpoints at a pinned Hub commit. Helm 2 checkpoints answer every
question in a request from one backbone pass; the original Helm runs one pass per question.

```python
from saina.transformers import HelmModel

helm = HelmModel.from_pretrained('run-saina/saina-helm-2-0.8b', revision='<40-hex commit>', device='cuda:0')
answers = helm.ask(model='saina-helm', state='I was charged twice.', questions={
    'refund': {'type': 'yes_no', 'question': 'Is a refund requested?'},
    'issue': {'type': 'single_choice', 'question': 'Identify the banking issue.',
              'options': {'duplicate': 'duplicate charge', 'delivery': 'card delivery'}},
})
```

The context limit defaults to the checkpoint's trained limit; pass `max_length=` (or set
`SAINA_MAX_LENGTH` for the server) to serve a lower limit that fits your GPU. Longer inputs are
rejected, never truncated. Checkpoint formats newer than the installed package are refused with
a request to upgrade.

## Transformers pipeline

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
`/v1/systemone` accepts the System One wire format (`noul`, `choice`, `score`) used by Jev
clients and OpenRouter's Decisions API; `saina.jev` holds that adapter. `GET /v1/models` lists
the served model in OpenRouter's provider format, and `GET /healthz` reports the pinned revision.

`SAINA_API_KEY` may hold several comma-separated keys, one per caller, so a single key can be
revoked by removing it and restarting. Behind a TLS-terminating proxy (`X-Forwarded-Proto` or
Cloudflare's `CF-Visitor`), plain-HTTP requests are refused with `426` and HTTPS responses carry HSTS.
Run uvicorn with `--no-access-log` if request metadata must not be logged.
Do not expose a plain HTTP server publicly; place it behind authenticated TLS.

Browsers may call the server from the origins in `SAINA_CORS_ORIGINS` (comma-separated).
The default, `https://saina.run`, lets the hosted [playground](https://saina.run/playground)
talk to your server. Set it to an empty string to disable cross-origin access.

To run the server in Docker or deploy it to a cloud, see
[run-saina/deploy](https://github.com/run-saina/deploy).

## Development

`PYTHONPATH=src python -m unittest discover -s tests`
and `cd clients/javascript && npm test`.

No training datasets, private history, model weights, or credentials are included.

## Developer contact

For integration support and developer enquiries, email [dev@saina.run](mailto:dev@saina.run).
