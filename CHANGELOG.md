# Changelog

## 0.2.0

Behavior changes for self-hosted servers:

- **Concurrent requests queue instead of failing.** A request that arrives while the model is busy now waits
  in a bounded queue (`SAINA_QUEUE_DEPTH`, default 4) for an execution slot (`SAINA_EXECUTION_SLOTS`,
  default 1). When the queue is full the server returns `429 overloaded` with `Retry-After`, instead of
  the previous immediate `503 Model is busy`. A request that waits longer than `SAINA_QUEUE_TIMEOUT`
  (default 30 s) is dropped before running and gets `503 overloaded`.
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
