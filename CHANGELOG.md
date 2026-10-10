# Changelog

## 0.3.0

Helm 2 support. Requires no changes from 0.2.0 callers of the original Helm.

- **Helm 2 checkpoints** (`run-saina/saina-helm-2-0.8b`): every question in a request is answered from one
  backbone pass over one model input, with routed expert heads. Checkpoint formats newer than the installed
  package are refused with a request to upgrade.
- **Helm 2 billing:** a Helm 2 request is billed for the one model input it reads, so `usage.input_tokens`,
  `X-Saina-Billable-Tokens` and `X-Saina-Input-Tokens` agree. The original Helm is still billed for the text
  sent. `TokenCounter` prices both the same way the server does.
- **Model names:** `saina-helm` reaches whichever Helm an endpoint serves; versioned names (`helm-2-0.8b`,
  `saina-helm-2-0.8b`, `helm-0.8b`, `saina-helm-0.8b`) only reach that model, otherwise `404 not_found`.
  Responses name the model that answered.
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
