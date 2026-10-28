# Saina

Saina's HTTP SDK, shared decision contract, and optional Helm local inference.
Saina is operated by **Rama Labs Inc.** Website: https://saina.run.

The Helm weights are released under Apache-2.0, including for commercial use.
Their training data was limited to permissively licensed sources.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/run-saina/saina/blob/main/examples/saina_colab.ipynb)
Models: [Helm 2 0.8B](https://huggingface.co/run-saina/saina-helm-2-0.8b) (current),
[Helm 3 0.8B](https://huggingface.co/run-saina/saina-helm-3-0.8b) (a native multimodal decision model)
and the original [Helm 0.8B](https://huggingface.co/run-saina/saina-helm-0.8b). Try Helm in
[`examples/saina_colab.ipynb`](examples/saina_colab.ipynb), run the weights locally through the SDK,
or call [Replicate](https://replicate.com/run-saina/saina-helm-2-0.8b) or your own endpoint.

## Install

```sh
pip install saina                  # dependency-free HTTP client
pip install 'saina[local,server]'  # local inference and authenticated server
pip install 'saina[media]'         # decoding images and audio (server, local inference, token counting)
```

From a checkout, use `pip install .` or `pip install '.[local,server]'`.

### Offline artifact diagnostics

From this checkout, check an already downloaded bundle without loading a model:

```sh
PYTHONPATH=src python3 -m saina doctor --checkpoint /path/to/checkpoint --json
```

After installing this checkout, the same command is available as `saina doctor`.
It checks every file in `artifact-manifest.json` except the model card and version notes
(`README.md`, `VERSIONS.md`), which loading skips as well, and requires the trained
`head.safetensors` and `head-config.json` to be listed. Missing manifests, omitted
required files and checksum mismatches exit with status 1. Use `--require FILE`
repeatedly for additional backend assets. For an experimental GGUF bundle, pass
`--manifest gguf-manifest.json`.

This command checks consistency against the supplied manifest only. It does not
authenticate a release, validate tensor or tokenizer compatibility, verify that a
runner can start, or establish model readiness. It performs no downloads or repairs.
For guided launch and the browser interface, see Local playground below.

```python
from saina import Saina

# Set SAINA_API_KEY in the environment.
client = Saina()
ticket = {'id': 'ticket-123', 'text': 'I was charged twice. Please refund the duplicate charge.'}
answers = client.ask(
    model='saina-helm-2-0.8b',
    state=ticket,
    questions={
        'team': {
            'type': 'single_choice',
            'question': 'Which team should handle this?',
            'options': ['billing', 'support', 'legal'],
        },
    },
)
team = answers['team'].select(threshold=0.9)

# Application-provided actions:
if team.accepted:
    assign_ticket(ticket, team=team.value)
else:
    queue_for_review(ticket)
```

Keep credentials in environment variables or a secret manager, not source files.
The base URL resolves from `SAINA_BASE_URL`, defaulting to `https://api.saina.run`. Explicit configuration
wins over environment variables. `SAINA_TIMEOUT` is in seconds (default 60); non-loopback HTTP URLs require
`allow_http=True`. Self-hosted servers must provide request IDs for typed inference responses.
See [migration notes](MIGRATION.md) for the 0.9.3-to-1.0 changes and server requirements.

### Hosted API: idempotency, retries, errors, credits

```python
import os
from saina import Saina, new_idempotency_key
from saina.client import InsufficientCredits

client = Saina(api_key=os.environ['SAINA_API_KEY'])
result, meta = client.ask_with_metadata(
    model='saina-helm', state='Refund my payment.',
    questions={'refund': {'type': 'yes_no', 'question': 'Is a refund requested?'}},
)
meta.credits_charged, meta.balance, meta.request_id, meta.replayed  # ints, str, bool
client.balance()['available']                                       # int
client.usage(from_='2026-10-01'); client.requests(limit=50)
```

- **Idempotency.** Inference is retried automatically only with an `Idempotency-Key` (a UUIDv7):
  one key is generated per logical call by default. A caller-provided `idempotency_key` takes precedence.
  Set `auto_idempotency_key=False` to disable generation; inference then retries only with an explicit key.
  Every retry reuses its key and identical body bytes. Duplicate execution and billing protection require
  server support; see [migration and server compatibility](MIGRATION.md).
- **Retries** (`max_retries=2`, total wait capped by `max_retry_delay=30` s) only on connection loss or an
  uncertain response, `accounting_unavailable`, `request_in_progress`, `rate_limited`, and
  `overloaded`/`inference_unavailable` when the request was not admitted, honoring `Retry-After`.
- **Errors.** `SainaError` carries `status`, `code`, `request_id`, `admitted`, `state`, `retry`, `retry_after`
  and `suspension`, with a subclass per code (`InsufficientCredits`, `RateLimited`, `IdempotencyConflict`, ...;
  see `saina.ERROR_CLASSES`). Broad categories such as `AuthenticationError`, `RateLimitError`,
  `IdempotencyError`, and `ServerError` can catch related codes. Unknown codes use the HTTP status category;
  `.fields` and `.raw` preserve the complete server error.
- **Credits** are 64-bit integers sent as decimal strings; Python returns `int`, JavaScript `BigInt`.
  `ask` returns an immutable `Answers` mapping from question ID to typed answer with `.metadata`, `.raw`, `to_dict()` and `to_json()`.
  Python serialization emits integer credits; `.raw` retains the original decimal strings.
  `ask_with_metadata()` remains available and returns `(answers, answers.metadata)`.

## Model names

`saina-helm` selects whichever Helm the endpoint serves. Release names (`saina-helm-2-0.8b`,
`saina-helm-3-0.8b`, `saina-helm-0.8b`) only reach that exact model; an endpoint serving a different release answers `404`.
Responses always give the release name of the model that answered.

## Images and audio

Models that read images or audio say so in `inputs` (`/healthz`, `/v1/models`, `HelmModel.inputs`):
`["text", "image", "audio"]`. A text-only model, such as `saina-helm-2-0.8b`, refuses a request with images or
audio with `invalid_request` (`saina-helm-2-0.8b accepts text only`); videos are not supported yet.

```python
from saina import Saina

client = Saina('https://your-endpoint.example', api_key='YOUR_KEY')

# Text and an image (or several)
result = client.ask(
    model='saina-helm',
    state='The customer says the blender arrived cracked and attached the receipt.',
    images=['receipt.png'],  # local paths, http(s) URLs, bytes, PIL images or base64 data URLs
    questions={
        'route': {'type': 'single_choice', 'question': 'Which team should handle this request?',
                  'options': {'returns': 'Refunds, replacements and damaged deliveries',
                              'billing': 'Payments, invoices and charges',
                              'technical': 'Product setup and faults'}},
        'on_receipt': {'type': 'yes_no', 'question': 'Does the receipt list the blender?'},
    },
)

# Text and an audio clip (or several)
result = client.ask(
    model='saina-helm',
    state='A voicemail left on the support line.',
    audios=['voicemail.wav'],  # local paths, http(s) URLs, bytes, (samples, sample_rate) arrays or data URLs
    questions={'upset': {'type': 'yes_no', 'question': 'Does the caller sound upset?'}},
)
```

`HelmModel.ask` takes the same `images=` and `audios=` values. Over HTTP, send base64 data URLs:

```bash
curl -s https://your-endpoint.example/v1/ask -H "Authorization: Bearer $SAINA_API_KEY" \
  -H 'content-type: application/json' --data-binary @- <<EOF
{
  "model": "saina-helm",
  "state": "What does the attached image show?",
  "images": ["data:image/png;base64,$(base64 -w0 chart.png)"],
  "questions": {
    "kind": {"type": "single_choice", "question": "What kind of image is this?",
             "options": {"chart": "A chart or plot", "photo": "A photograph", "document": "A document"}}
  }
}
EOF
```

- **Images:** `data:image/png`, `image/jpeg` or `image/webp`; at most 8 per request, each at most 10,000,000
  bytes. Each is read at up to 1.6 megapixels (and at least 65,536 pixels), keeping its aspect ratio, and goes
  before the text of the request.
- **Audio:** `data:audio/wav`, `audio/x-wav`, `audio/flac`, `audio/mpeg`, `audio/ogg` or `audio/webm`; at most 4
  per request, each at most 10,000,000 bytes and 10.0 seconds. Longer clips are refused, never cut. Each clip is
  read as mono 16 kHz audio and turned into a log-mel spectrogram image that follows the images.
  `saina.media.mel_spectrogram_png(waveform, sample_rate)` renders exactly what the model reads: 128 mel bands
  from 0 to 8 kHz (Slaney scale), n_fft 1024, hop 160 (100 pixels per second), the top 80 dB as 8-bit
  grayscale, low frequencies at the bottom; clips shorter than 1 s are padded with silence.
- Every question of the request sees every image and clip. Their input tokens count in `usage.input_tokens`
  (see [Usage metering](#usage-metering)).
- WebM and some other compressed audio is decoded with `ffmpeg` when the server has it installed.
  Decoding stops just beyond the duration limit so long clips can be rejected without expanding the whole
  recording. Clips exceeding the 64 MiB decoded-audio budget are also refused.

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
a request to upgrade. Helm 3 checkpoints (format 4) need saina 0.9.0 or later: their prompt reads in both
directions, so an older runtime refuses them rather than reading them as a different model.

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

Source: `clients/javascript`; registry package: `@run-saina/sdk`.
Exports `Saina` and `SainaError`, retaining `SainaHelm` aliases for compatibility. Same idempotency, retry,
error and metadata API (`askWithMetadata`, `systemOne`, `decisions`, `balance`, `usage`, `requests`); credits are `BigInt`.
`images` and `audios` take data URLs, http(s) URLs, `Blob`/`File` objects, `ArrayBuffer`s or typed arrays
(`Buffer`); the client sends them as data URLs.

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

`POST /v1/decisions` (also `/api/v1/decisions`) takes either a System One body (`state` plus a `questions`
object, answered as on `/v1/systemone`) or the OpenAI Decisions request shape (`input` plus a `questions` array of
`predicate`, `choice` and `score`), told apart by the body; `saina.decisions` holds the translation. An OpenAI
request is translated to the native `/v1/ask` request, so it is scored and billed exactly as that request, and the
answers come back in the shape the official OpenAI SDKs read. Model names are exact on this route (`gpt-6-luna` or
`jev-1.14` is a 404, never the default model). Point an OpenAI client at your server's `/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=SAINA_API_KEY)
decision = client.decisions.create(model="saina-helm", input="I was charged twice for my order.", questions=[
    {"type": "choice", "name": "team", "instructions": "Which team?",
     "choices": [{"value": "billing"}, {"value": "technical"}, {"value": "other"}]}])
```

or use `Saina.decisions(model=..., input=..., questions=[...])` (Python) and `decisions({...})` (JavaScript).
Saina's decision policy is an extension (`mode="decision"`, `threshold`, `min_margin`; `extra_body` in the OpenAI
SDKs): an answer it does not accept becomes a billed `refusal` entry with `reason` and the distribution. Base64
`input_image` parts are read as `images` by models that read images; image URLs, images in question instructions
and images sent to a text-only model are refused with a 400 before anything runs. `multi_choice` stays on `/v1/ask`.
See the [hosted API docs](https://saina.run/api/docs/#decisions) for the full mapping. `scripts/acceptance/run.sh`
checks pinned official OpenAI and TypeSafe clients against a local server with mocked inference.

Request bodies may be compressed with `Content-Encoding: gzip` or `zstd` (the TypeSafe SDKs compress bodies of 1 KiB
or more by default); at most `SAINA_MAX_DECODED_BODY_BYTES` (default 64 MiB) are decoded.

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
Images and audio clips add the input tokens the model reads for them: one per 32 × 32 pixels of each image after
it is resized as above, and the same for each clip's spectrogram (a 10-second clip is 124 tokens). The
`X-Saina-Media-Tokens` header gives that part on its own.
`pip install 'saina[tokenize]'` installs `saina.tokenize.TokenCounter`, which computes both without
running the model (add `saina[media]` for requests with images or audio):

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


### Request extensions

The `extra` dictionary is flattened into each outgoing question or the top-level request. For example,
`{"type": "single_choice", "question": "Team?", "options": ["billing", "support"],
"extra": {"new_server_feature": True}}` sends an options dictionary and `new_server_feature` at the
question level. No nested `extra` is sent. Extensions cannot override known fields, including omitted
optional fields; collisions raise `ValueError` before media downloads or API requests.

`decisions()` and `decisions_with_metadata()` call `/v1/decisions` with `input` and an ordered list of
`predicate`, `choice`, or `score` questions, following the repository's Decisions shim contract. They use
the same idempotency and retry defaults. This endpoint requires a server that implements that shim.


### Typed answers and local selection

`Answers` is an immutable mapping: `answers['team']`, `len(answers)`, and `answers.items()` operate on
questions. Request details are attributes: `.request_id`, `.model`, `.specialist`, `.usage`, and `.metadata`.
`dict(answers)` contains typed values; `to_dict()` recursively serializes them. `.raw` returns an independent
copy of the original response, retaining unknown fields.

Call `answer.select(threshold=..., min_margin=...)` or `answers.select(...)` to apply local requirements.
Check `.accepted` and read `.value` and `.reason`. Threshold is checked before ties and margins. Rating
and multi-choice ignore margins; ratings return integer indices and `.label`, while multi-choice returns
an ordered tuple and uses `no_options_selected` when nothing qualifies. Native `ask()` always requests
distributions; the existing remote `decisions()` endpoint retains its server behavior.

### Durable external logs

```python
from saina import Traces

with Traces() as traces:
    trace_id = traces.log(task='route ticket', inputs={'ticket': 'Charged twice'},
                          outcome={'team': 'billing'}, metadata={'application': 'support'})
    delivery = traces.flush(timeout=5)
```

A returned ID means SQLite persistence completed; default warn-mode storage failure returns `None`.
Records remain until server acknowledgment, with rejected records quarantined for explicit recovery.
The matching backend exposes `POST /v1/traces`. The existing `TraceClient` execution-scope API remains
available. See [durability, paths, recovery and lifecycle](docs/durable-traces.md) and the
[agreed SDK contract](docs/sdk-contract.md). Inference creates no background worker or spool.
### Local playground

```sh
saina playground
```

Opens the shared browser UI before loading a model. Select a downloaded model or
choose Download in the UI. Download size, verification and loading status are
shown there. Cached models work offline. Apple Silicon defaults to the original
Helm FP16 MLX backend; native CPU offers Helm, Helm 2, Helm 2 Omni and Helm 2 Routing.
Each pinned release is hash-verified and runs a startup check before inference.

```sh
saina serve helm     # Inference API only; download missing pinned files
saina serve          # Use the sole cached model, otherwise show a terminal picker
saina doctor helm    # Verify artifacts without loading weights
saina doctor helm --json
saina --version
```

Without a model key, ambiguous selection in a noninteractive shell fails with
available keys. Playground takes no model key: selection and download happen in
its UI. There is no `local` alias or plan/download/offline flag.

Install `saina[server,local]` for native CPU or `saina[server,mlx]` on Apple Silicon.
From a source checkout use `pip install -e '.[server,local]'`, substituting the
backend extra as appropriate. GGUF uses the experimental `server,gguf` extras
and the existing original-Helm custom runner with separate head:

```sh
saina playground --port 8766 --no-browser
saina serve --backend native --checkpoint /path/to/native-bundle
saina serve --backend mlx --checkpoint /path/to/helm-mlx
saina serve --backend gguf --checkpoint /path/to/gguf-bundle --runner /path/to/helm-hidden
```

Remote custom checkpoints require `--revision` with a full immutable Hub commit.
The default port is 8765. Playground's `--no-browser` prints the connection URL.
Serve prints the API URL and bearer token. No hosted inference fallback occurs.
The browser connection token stays in memory: reopen the printed URL after a
refresh. Export data before closing or switching models; drafts/results are in memory.

Both playgrounds share CSV/JSONL import, nested field mapping, preview, question
catalog references, optional evaluation, sequential execution, cancellation and
partial export. Hosted runs retain quote and idempotency controls.

The SDK packages the exact hosted UI assets. Refresh them from the sibling
website with `node build-console.mjs` and `node build-local-ui.mjs`; playground's
`--ui-dir` selects another verified build. File hashes and UI/API compatibility
are checked before loading models.

Exit codes: 0 success, 1 setup/integrity/runtime failure, 2 invalid arguments,
130 cancelled. See [validation](docs/local-validation.md) and [Mac instructions](docs/mac-test.md).

## Decision trace collection

Use `TraceClient` to capture explicit decision inputs and answers from your own code,
observe them in Studio, and prepare datasets from collection evidence. Python supports
sync/async context managers; TypeScript provides an async callback scope.
`invoke_endpoint` / `invokeEndpoint` calls a Saina Decision Endpoint with one server-side trace.
See [decision trace collection](docs/tracing.md).
