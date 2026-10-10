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
    model='saina-helm',  # whichever Helm the endpoint serves; or pin 'saina-helm-2-0.8b'
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
The base URL defaults to Saina's hosted API, `https://api.saina.run`; pass your own for a self-hosted server.

### Hosted API: idempotency, retries, errors, credits

```python
import os
from saina import Saina, new_idempotency_key
from saina.client import InsufficientCredits

client = Saina(api_key=os.environ['SAINA_API_KEY'], auto_idempotency_key=True)
result, meta = client.ask_with_metadata(model='saina-helm', state='...', questions={...})
meta.credits_charged, meta.balance, meta.request_id, meta.replayed  # ints, str, bool
client.balance()['available']                                       # int
client.usage(from_='2026-10-01'); client.requests(limit=50)
```

- **Idempotency.** Inference is retried automatically only with an `Idempotency-Key` (a UUIDv7): pass
  `idempotency_key=new_idempotency_key()` or set `auto_idempotency_key=True`. Every retry of a call reuses its
  key and identical body bytes, so it cannot run or charge twice; the client never makes a new key for a retry.
- **Retries** (`max_retries=2`, total wait capped by `max_retry_delay=30` s) only on connection loss or an
  uncertain response, `accounting_unavailable`, `request_in_progress`, `rate_limited`, and
  `overloaded`/`inference_unavailable` when the request was not admitted, honoring `Retry-After`.
- **Errors.** `SainaError` carries `status`, `code`, `request_id`, `admitted`, `state`, `retry`, `retry_after`
  and `suspension`, with a subclass per code (`InsufficientCredits`, `RateLimited`, `IdempotencyConflict`, ...;
  see `saina.client.ERROR_CLASSES`).
- **Credits** are 64-bit integers sent as decimal strings; Python returns `int`, JavaScript `BigInt`.
  `ask` still returns the same response dict; billing headers are in `ask_with_metadata` and `client.last_metadata`.

## Model names

`saina-helm` selects whichever Helm the endpoint serves. Release names (`saina-helm-2-0.8b`,
`saina-helm-0.8b`) only reach that exact model; an endpoint serving a different release answers `404`.
Responses always give the release name of the model that answered.

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
Exports `Saina` and `SainaError`, retaining `SainaHelm` aliases for compatibility. Same idempotency, retry,
error and metadata API (`askWithMetadata`, `systemOne`, `balance`, `usage`, `requests`); credits are `BigInt`.

## Serving and contracts

Set `SAINA_API_KEY`, `SAINA_CHECKPOINT`, `SAINA_DEVICE` and optionally
`SAINA_REVISION` (pinned Hub commit), then:

```sh
uvicorn saina.server:create_app --factory --host 127.0.0.1 --port 8000
```

Native typed requests use `/v1/ask`. The `saina.contract` module owns their schema.
`/v1/systemone` accepts the System One wire format (`noul`, `choice`, `score`) used by Jev
clients and OpenRouter's Decisions API; `saina.jev` holds that adapter. `GET /v1/models` lists
the served models in OpenRouter's provider format, `GET /healthz` reports the pinned revisions and queue state, and `GET /livez` reports process liveness.

One server can serve several releases: set `SAINA_MODELS` to comma-separated
`checkpoint@revision[:max_length]` entries instead of `SAINA_CHECKPOINT`/`SAINA_REVISION`. The first is the
default and answers `saina-helm`; versioned names reach only their model, and an unserved name answers
`404 not_found` (System One labels the server does not know go to the default). All models share the same
execution slots and queue.

`SAINA_API_KEY` may hold several comma-separated keys, one per caller, so a single key can be
revoked by removing it and restarting. Behind a TLS-terminating proxy (`X-Forwarded-Proto` or
Cloudflare's `CF-Visitor`), plain-HTTP requests are refused with `426` and HTTPS responses carry HSTS.
Run uvicorn with `--no-access-log` if request metadata must not be logged.
Do not expose a plain HTTP server publicly; place it behind authenticated TLS.

Browsers may call the server only from origins listed in `SAINA_CORS_ORIGINS` (comma-separated).
The default is empty: no cross-origin access. Set it if you build your own browser UI for your server.

### Queueing and overload

Requests wait in a bounded FIFO queue for a fixed number of execution slots:

| Variable | Default | Meaning |
|---|---|---|
| `SAINA_EXECUTION_SLOTS` | `1` | Concurrent model executions. Raise only after measuring that it is safe on your GPU. |
| `SAINA_QUEUE_DEPTH` | `4` | Requests that may wait for a slot. When full, new requests get `429 overloaded` immediately. |
| `SAINA_QUEUE_TIMEOUT` | `30` | Seconds a request may wait; after that it is dropped before running and gets `503 overloaded`. |

A request takes its queue place before it is tokenized, so a full queue rejects without encoding anything.
Every question is then validated and encoded before the request waits for a slot, so an invalid
later question fails the whole request without running the earlier ones. A slot is released only when
the model work actually finishes, even if the client has disconnected.

Errors are JSON: `{"error": {"code": "...", "message": "..."}, "detail": "..."}`. Codes include
`invalid_api_key`, `invalid_request`, `overloaded`, and `inference_failed`. Messages never echo inputs.
`overloaded` errors also carry `"admitted": false, "state": "not_admitted", "retry": "same_operation"`:
the request never ran, so the clients retry it after `Retry-After`.

### Usage metering

`usage.input_tokens` counts the text you send, the same way for every model: the `state` once, plus each question
and its options or levels (keys and descriptions). Strings count as sent; objects and arrays count as compact JSON in the
order sent; each part is tokenized on its own with the pinned tokenizer. Prompt formatting the server adds,
and re-reading the `state` for each question, are not counted. The responses' `X-Saina-Billable-Tokens`
header repeats this number; `X-Saina-Input-Tokens` is the length of the token sequences actually fed to the
model (one sequence per question for the original Helm, one for the whole request for Helm 2).
`pip install 'saina[tokenize]'` installs `saina.tokenize.TokenCounter`, which computes both without
torch:

```python
from saina.tokenize import TokenCounter
from saina.contract import AskRequest

counter = TokenCounter.from_checkpoint('run-saina/saina-helm-0.8b', revision='<pinned commit>')
request = AskRequest.model_validate(body)
counter.bill_ask(request).total  # usage.input_tokens
counter.count_ask(request)       # model-input length per forward pass (one for Helm 2)
```

### Behind a gateway

A gateway authenticates with `SAINA_SERVICE_TOKEN` (comma-separated tokens allowed) instead of a caller key.
Only that token may send `X-Saina-Request-Id`, `X-Saina-Expected-Input-Tokens` (model-input tokens; the server
refuses with `409 metering_mismatch` before running if its own count differs) and `X-Saina-Deadline-Ms` (absolute Unix
milliseconds after which queued work is dropped), and call `DELETE /v1/internal/requests/{request_id}` to drop
queued work it abandoned. Bind the server to a private interface in this setup.

Saina's hosted API at `https://api.saina.run` runs this server behind a separate gateway that handles accounts,
credits, and rate limits. Those do not apply to self-hosted servers: a self-hosted server has only the
keys you configure and the queue limits above.

To run the server in Docker or deploy it to a cloud, see
[run-saina/deploy](https://github.com/run-saina/deploy).

## Development

`PYTHONPATH=src python -m unittest discover -s tests`
and `cd clients/javascript && npm test`.

No training datasets, private history, model weights, or credentials are included.

## Developer contact

For integration support and developer enquiries, email [dev@saina.run](mailto:dev@saina.run).
