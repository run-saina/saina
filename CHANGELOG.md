# Changelog

## 0.9.0

- **Checkpoint format 4.** A format 4 checkpoint is a format 3 checkpoint (shared multimodal prompt, routed head,
  text, image and audio inputs) whose prompt reads in both directions in the full-attention layers: every token
  before the readout slots (the context, every question with its options, and any images or audio) attends to every
  other prompt token. The readout slots stay causal, and so do the linear-attention layers. Its `config.json` sets
  `saina_checkpoint_format: 4` and `saina_prompt_attention: "bidirectional_prefix"`; `head-config.json` sets
  `prompt_attention`. Earlier saina versions refuse format 4 checkpoints, and this version refuses a prompt
  attention it does not know. Format 4 checkpoints serve, count tokens and accept images and audio exactly as
  format 3 does.
- **Saina Helm 3 (Preview)**, `saina-helm-3-0.8b`: a native multimodal decision model, shipped as a format 4
  checkpoint (`run-saina/saina-helm-3-0.8b`). `/v1/models` lists it by name when served.
- `TokenCounter.from_checkpoint` checks the checkpoint format and refuses formats this version does not know.

## 0.7.0 (unreleased)

- **`POST /v1/decisions`** (and the alias `/api/v1/decisions`) accepts two request shapes, told apart by the body:
  System One bodies (`state` plus a `questions` object, as on `/v1/systemone`) and OpenAI Decisions bodies
  (`input` plus a `questions` array of `predicate`, `choice` and `score` questions, as sent by the official OpenAI
  SDKs' `client.decisions.create`). OpenAI requests are translated to the native `/v1/ask` request, so they are
  prepared, scored and billed exactly as the same native request; answers come back in request order in the
  OpenAI response shape. A body that matches both shapes or neither is a 400; `X-Saina-Dialect: openai|systemone`
  may confirm the shape and is a 400 when it disagrees. Responses carry `X-Saina-Dialect`.
- On `/v1/decisions` model names are exact: a name this server does not serve (such as `gpt-6-luna` or `jev-1.14`) is
  a 404, never another model. `/v1/systemone` is unchanged and still answers unknown System One labels with the
  default model.
- OpenAI dialect details: Saina's decision policy (`mode: "decision"`, `threshold`, `min_margin`) is accepted as an
  extension; answers it does not accept become `refusal` entries that keep `reason` and the distribution, and are
  billed. Base64 `input_image` parts are read as native `images` by models that read images; image URLs,
  images inside question instructions, a `detail` other than `auto`, and images sent to a text-only model are
  refused with a 400 naming the part. Sampling parameters (`temperature`, `top_p`, `seed`, ...) are refused.
  `prompt_cache_key`, `prompt_cache_options`, `safety_identifier` and `metadata` are accepted and ignored (there is no
  prompt cache; usage reports zero cached tokens). `multi_choice` stays native-only.
- Request bodies compressed with `Content-Encoding: gzip` or `zstd` are decoded (at most
  `SAINA_MAX_DECODED_BODY_BYTES`, default 64 MiB). The official TypeSafe SDKs zstd-compress bodies of 1 KiB or more
  by default; such requests to `/v1/systemone` used to fail with 400. The `server` extra adds `backports-zstd` on
  Python before 3.14.
- Clients: `Saina.decisions()` (Python) and `decisions()` (JavaScript, `@run-saina/sdk` 0.4.0).
- `scripts/acceptance/` checks pinned official OpenAI and TypeSafe SDKs against a local server with mocked inference.

## 0.6.1

- Documentation: the README now states the Helm weights licence (Apache-2.0, including commercial use). No code changes.

## 0.6.0

- **Standalone specialists.** A checkpoint can carry a specialist merged into its weights: its `config.json` names
  the model it belongs to (`saina_model`) and the specialist (`saina_specialist`, with an optional
  `saina_specialist_description`). It needs no base checkpoint and answers only as that specialist:
  `{"model": "saina-helm-2-omni-0.8b", "specialist": "mechanical"}`; a request without `specialist` is a 404. Serve it
  with `SAINA_MODELS` like any model, on its own server or beside its base, so a model and its specialists can be
  deployed and updated independently. Local use: `HelmModel.from_pretrained(<merged checkpoint>)` and
  `helm.ask(specialist=...)`.

## 0.5.0

- **Domain specialists.** A specialist is a model trained further for one domain, shipped as a small set of
  weights for one exact base release (checked against the base's `model.safetensors` hash). Select one per request
  with `"specialist": "<name>"` on `/v1/ask` and `/v1/systemone`; responses then carry `specialist` beside `model`.
  Without the field the model answers as before. `GET /v1/specialists` lists them; `/v1/models` lists each model's
  `specialists` and `/healthz` includes them. An unknown specialist is a 404.
- Servers load specialists with `SAINA_SPECIALISTS=checkpoint@revision,...`; each attaches to the served model it
  was trained for. Local use: `HelmModel.from_pretrained(..., specialists={checkpoint: revision})` and
  `helm.ask(specialist=...)`, or `saina.loader.load_specialist`. The clients take `specialist=`.
- First specialist: `mechanical` for Saina Helm 2 Omni 0.8B
  ([run-saina/saina-helm-2-omni-0.8b-mechanical](https://huggingface.co/run-saina/saina-helm-2-omni-0.8b-mechanical)):
  machine condition and likely fault from a recording.

## 0.4.0

- **Saina Helm 2 Omni 0.8B** (`saina-helm-2-omni-0.8b`, [run-saina/saina-helm-2-omni-0.8b](https://huggingface.co/run-saina/saina-helm-2-omni-0.8b)):
  the first Helm that reads text, images and audio, listed in the server's model catalog.

- **Images and audio.** `/v1/ask` and `/v1/systemone` take `images` (PNG, JPEG or WebP) and `audios` (WAV, FLAC,
  MP3, Ogg or WebM) as base64 data URLs: at most 8 images and 4 clips per request, each at most 10,000,000 bytes;
  clips at most 10 s (longer clips are refused, never cut). Images are read at up to 1.6 megapixels before the
  text; each clip is read as a log-mel spectrogram image after the images
  (`saina.media.mel_spectrogram_png`). The Python and JavaScript clients and `HelmModel.ask` also accept local
  paths (Python), http(s) URLs, bytes, PIL images, `(samples, sample_rate)` arrays, `Blob`s and `ArrayBuffer`s,
  and send data URLs. `videos` are refused with `invalid_request` (not supported yet).
- **Models declare their inputs.** `/healthz` and `/v1/models` list `inputs` (`["text"]`, or with `"image"` and
  `"audio"`); `/v1/models` adds image and audio entries to `input_modalities`. Text-only models such as
  `saina-helm-2-0.8b` refuse images and audio with `invalid_request` (`saina-helm-2-0.8b accepts text only`).
- **Image and audio tokens are counted.** `usage.input_tokens`, `X-Saina-Billable-Tokens`,
  `X-Saina-Input-Tokens` and `TokenCounter.bill_*`/`count_*` include the input tokens of every image and
  spectrogram; `X-Saina-Media-Tokens` gives that part alone. Text-only requests count as before.
- Checkpoint format 3: shared-pass models with a vision tower (`saina_inputs` in `config.json`, the image
  processor in `preprocessor_config.json`); `scripts/build_multimodal_package.py` builds one.
- New `saina[media]` extra (Pillow, NumPy, SciPy, soundfile) for decoding images and audio.

## 0.3.2

- **Helm 2 is billed for the text sent, like the original Helm.** In 0.3.0 and 0.3.1, a Helm 2 request was
  billed for its one model input, prompt formatting included, so the same request cost about 7× more on
  Helm 2 than on the original Helm. Now `usage.input_tokens`, `X-Saina-Billable-Tokens` and
  `TokenCounter.bill_ask`/`bill_systemone` count the state once plus each question's text as sent, for every
  model. `X-Saina-Input-Tokens` and `TokenCounter.count_*` still report model-input lengths for the metering
  check.

## 0.3.1

- **Apps embedded without the server's lifespan work again.** In 0.3.0, building `saina.server.create_app`
  without running its lifespan (a bare `TestClient`, another ASGI framework) failed with `AttributeError`,
  because the served models were only set there. Models passed to `create_app(scorer=...)` or
  `create_app(scorers=[...])`, or set on `app.state.scorer` as in 0.2, are now used directly. With no model at
  all, requests get `503 inference_unavailable` (not admitted, safe to retry) instead of a server error.
- The JavaScript client is also 0.3.1: its types list release names only (`saina-helm-2-0.8b`, not
  `helm-2-0.8b`).

## 0.3.0

Helm 2 support. Requires no changes from 0.2.0 callers of the original Helm.

- **Helm 2 checkpoints** (`run-saina/saina-helm-2-0.8b`): every question in a request is answered from one
  backbone pass over one model input, with routed expert heads. Checkpoint formats newer than the installed
  package are refused with a request to upgrade.
- **Helm 2 billing:** a Helm 2 request is billed for the one model input it reads, so `usage.input_tokens`,
  `X-Saina-Billable-Tokens` and `X-Saina-Input-Tokens` agree. The original Helm is still billed for the text
  sent. `TokenCounter` prices both the same way the server does.
- **Model names:** `saina-helm` reaches whichever Helm an endpoint serves; release names (`saina-helm-2-0.8b`,
  `saina-helm-0.8b`) only reach that model, otherwise `404 not_found`. Responses give the release name of the
  model that answered: `/v1/ask` now returns `saina-helm-0.8b` where it returned `helm-0.8b`. The old request
  name `helm-0.8b` is still accepted for the original Helm.
- **Several models per server:** `SAINA_MODELS=checkpoint@revision[:max_length],...`; the first is the
  default. All models share the bounded queue.
- **Context limit** defaults to the checkpoint's trained limit (262,144 tokens for Helm 2, 8,192 for the
  original Helm) instead of a fixed 8,192; `SAINA_MAX_LENGTH` or `max_length=` sets a lower serving limit.

## 0.2.0

Behavior changes for self-hosted servers:

- **Concurrent requests queue instead of failing.** A request that arrives while the model is busy now waits
  in a bounded queue (`SAINA_QUEUE_DEPTH`, default 4) for an execution slot (`SAINA_EXECUTION_SLOTS`,
  default 1). When the queue is full the server returns `429 overloaded` with `Retry-After`, instead of
  the previous immediate `503 Model is busy`. A request that waits longer than `SAINA_QUEUE_TIMEOUT`
  (default 30 s) is dropped before running and gets `503 overloaded`. Both carry
  `"admitted": false, "state": "not_admitted", "retry": "same_operation"` (nothing ran), so the Python
  and JavaScript clients retry them automatically like the hosted API's.
- **`https://saina.run` is no longer a default CORS origin.** The hosted playground no longer calls
  self-hosted servers. If you built your own browser UI, set `SAINA_CORS_ORIGINS` explicitly.
- **Error bodies are typed.** Errors are `{"error": {"code", "message"}, "detail"}`; `detail` keeps the
  previous message for older clients.

New:

- Explicit per-question metering. `usage.input_tokens` is the length of the encoded inputs actually passed
  to the model; it is never read from mutable scorer state, and a scorer without `encode`/`forward` is refused.
- Every question is validated and encoded before any inference runs, so a request is all-or-nothing.
- `saina[tokenize]` extra and `saina.tokenize.TokenCounter`: exact token counting with the pinned tokenizer
  and prompt format, without torch. Shared prompt formatting lives in `saina.prompt` and `saina.prepare`.
- `SAINA_SERVICE_TOKEN` for a gateway in front of the server, with `X-Saina-Request-Id`,
  `X-Saina-Expected-Input-Tokens`, `X-Saina-Deadline-Ms` and `DELETE /v1/internal/requests/{id}`.
- `GET /livez`, and queue state in `GET /healthz`.

Clients (Python `saina.client` and JavaScript `@run-saina/sdk` 0.2.0):

- **Default base URL** is the hosted API, `https://api.saina.run`; a positional `base_url` still works.
- **Idempotency keys.** `new_idempotency_key()` / `newIdempotencyKey()` generate RFC 9562 UUIDv7 keys;
  `ask(..., idempotency_key=)` / `ask(request, {idempotencyKey})` send `Idempotency-Key`, and
  `auto_idempotency_key` / `autoIdempotencyKey` generate one key per logical call.
- **Bounded retries, only with a key.** Retries reuse the same key and identical body bytes and happen only on
  connection loss or an uncertain response, `accounting_unavailable`, `request_in_progress`, `rate_limited`,
  and `overloaded`/`inference_unavailable` with `admitted=false`. `Retry-After` is honored, otherwise
  exponential backoff with jitter; `max_retries` (default 2) and a total-wait cap (`max_retry_delay`).
  Calls without a key are still sent exactly once.
- **Typed errors.** `SainaError` (alias `SainaHelmError`, `status` unchanged) adds `code`, `request_id`,
  `admitted`, `state`, `retry`, `retry_after` and `suspension`, with a subclass per contract code
  (`InsufficientCredits`, `RateLimited`, `IdempotencyConflict`, ...) and `SainaConnectionError`. Old
  `{"detail"}` and non-JSON error bodies still raise `SainaError` with `status`.
- **Billing metadata.** `ask_with_metadata()` / `askWithMetadata()` return the unchanged response plus
  `request_id`, `credits_charged`, `balance`, `price_version`, `replayed`, `idempotency_key`; also on
  `last_metadata` / `lastMetadata`.
- `system_one()` / `systemOne()` for `POST /v1/systemone`.
- Read-only account calls with the API key: `balance()`, `usage()`, `requests()`. Credit strings become
  Python `int` and JavaScript `BigInt` (never a JavaScript number).
