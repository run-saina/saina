# Migrating from 0.9.x to 1.0

The migration baseline is Python `saina==0.9.3` and the corresponding JavaScript SDK, as specified in
[the agreed contract](docs/sdk-contract.md). The older package version in this checkout does not change
that baseline. These changes have not advanced package versions or published a release.

## Typed answers and local selection

Native `ask()` returns immutable `Answers`, a mapping from question IDs to typed answers in Python.
Use `answers['team'].probabilities`, `answers.model`, `answers.usage.input_tokens` and `answers.metadata`.
`dict(answers)` contains typed answer objects. `answers.raw` returns an independent original-body copy,
including unknown fields. `answers.to_dict()` and `answers.to_json()` serialize the typed envelope.

JavaScript uses `result.answers.team`, `result.select({threshold: 0.9})`, `result.toObject()` and
`JSON.stringify(result)`. Individual answers and selections expose the same serialization conventions.
Results and their nested collections are immutable.

For arbitrary JavaScript question IDs, use `selections.get(id)`. This supports names such as `toObject`
and `toJSON` without shadowing serialization methods. Non-colliding IDs retain direct property access.
Bulk serialization includes all IDs; unknown IDs return `undefined` from `get()`.

Replace `ask(mode='decision', threshold=..., min_margin=...)` with distribution `ask()` followed by
`answers.select(threshold=..., min_margin=...)` or an individual answer's `select()`. Remove redundant
`mode='distribution'`. JavaScript uses `minMargin` for local selection. Selection is synchronous and
makes no network calls. The threshold is required; both numbers must be finite and in `[0, 1]`.

Use `.accepted`, not the truthiness of `.value`: `False` and rating index `0` can be accepted values.
Ratings use integer probability keys in Python and integer selected indices; `.label` exposes the
selected description. Serialized rating keys are strings. Rating and multi-choice ignore valid margins
and report an effective margin of zero. Multi-choice preserves option order and returns a tuple in Python.

Local selection tests threshold first, then ties (within `1e-12`), then minimum margin where applicable.
Its reasons are `accepted`, `below_threshold`, `tie`, `below_margin`, and `no_options_selected`.
Existing server decision-mode reasons and precedence remain unchanged.

Typed answers require a non-empty body `request_id` or `X-Request-Id` header; supplied IDs must agree.
Older servers omitting both require an upgrade. Invalid derived confidence, rating expectations or echoes,
probabilities, labels, and answer shapes raise `ProtocolError` and are not automatically retried.

## Configuration and compatibility methods

`Saina()` resolves `SAINA_API_KEY`, `SAINA_BASE_URL` and `SAINA_TIMEOUT` at construction; explicit arguments
win. The timeout environment variable is in seconds. JavaScript keeps its existing millisecond constructor
units. Localhost and loopback HTTP URLs work by default; other HTTP endpoints require `allow_http=True`
(JavaScript `allowHttp: true`). Constructing `Saina` creates no worker or spool.

`ask_with_metadata()` returns `(answers, answers.metadata)`; JavaScript retains `askWithMetadata()`.
Python `last_metadata` is thread-local. Prefer response-bound metadata for concurrent code.
`system_one()` and `decisions()` retain wire-shaped dictionary results and their metadata variants,
media, specialist selection, and existing Decisions extension keywords. Account calls retain dictionary
access and existing filters and pagination shapes.

## Idempotency and server compatibility

`max_retries=2` and `max_retry_delay=30` remain unchanged. `auto_idempotency_key` now defaults to `True`;
JavaScript uses `maxRetries=2`, `maxRetryDelay=30000` and `autoIdempotencyKey=true`.
Pass `False` / `false` explicitly to preserve the previous default. Without a generated or explicit key,
inference does not retry. An explicit key always wins. All retries reuse the same UUIDv7 and serialized
bytes. The server-aware retry policy is preserved; terminal and protocol failures are not retried.

Idempotency requires server replay, admission, and accounting support. The bundled self-hosted server
in SDK 0.3.2 does **not** implement idempotency storage or replay. Disable generated keys/retries for it
if repeating inference is unacceptable. No self-hosted release with verified idempotency guarantees is
identified by a qualified release version; a supported version matrix is still required before promising
those guarantees. The [backend audit](docs/trace-backend-audit.md) confirms the implementation in hosted
gateway source revision `a3d63b3`, including replay and recovery fences. A self-hosted gateway stack at
that revision is distinct from the bundled standalone model server; its unchanged `0.1.0` package version
is insufficient qualification. The `/v1/decisions` shim also requires server support.

## Errors, credits, and extensions

Code-specific errors inherit broad categories: catch either `RateLimited` or `RateLimitError`, and either
`Overloaded`, `ModelBusyError`, or `ServerError`. Recognized codes take precedence over status. Otherwise,
401/403/404/422/429/503 map to authentication, permission, not-found, invalid-request, rate-limit and
model-busy categories. Other 5xx responses use `ServerError`; other unsuccessful statuses use `HTTPError`.
`SainaHelmError is SainaError` and `SainaConnectionError is TransportError`. Transport failures expose
`timed_out` (JavaScript `timedOut`). Complete server fields remain in `.fields` and `.raw`.

Python credits remain arbitrary-precision integers in parsed data and serialization. `.raw` retains wire
decimal strings. JavaScript uses `BigInt`, serialized as decimal strings. Do not round-trip arbitrary Python
credit integers through ordinary JavaScript number parsing.

Move unknown native request keywords to explicit `extra={...}`. Ordinary unknown fields fail validation.
`extra` at request or question level is flattened; extensions cannot override known fields, omitted
optional fields, or SDK-only argument names. Validation happens before media downloads or API requests.
Options lists become mappings with null descriptions; labels and ordering are preserved.

## Durable external logging

`Traces.log()` and matching `POST /v1/traces` are now implemented as additive APIs.
The newer staged `TraceClient` (Python 0.10.2 / JavaScript 0.5.2) is preserved with
its original collection-based execution-scope protocol and memory queue.

The new recorder commits locally to SQLite before returning an ID. Default warn
mode returns null/None on failed admission, with one warning per failure episode;
strict mode raises. Storage is isolated by service and credential hashes. Default
capacity is 64 MiB, including quarantine; records are limited to 1 MiB. No worker
or spool is created by inference-only `Saina`.

`spool_path` (JavaScript `spoolPath`) overrides the full filename. Default roots:
Linux uses `$XDG_STATE_HOME/saina` or `$HOME/.local/state/saina`; macOS uses
`$HOME/Library/Application Support/Saina`; Windows uses `%LOCALAPPDATA%/Saina`.
Each appends `traces/<service-sha256>/<credential-sha256>/spool.sqlite3`.
Explicit overrides still enforce tenant/configuration binding. The
[durable logging guide](docs/durable-traces.md) specifies fallback paths,
permissions, physical-disk headroom and credential-rotation recovery.

Flush/close default to five seconds and never discard unacknowledged records.
Flush returns an immutable complete/pending/rejected snapshot; local storage
errors raise, network failures return incomplete. After close, logging, flushing,
pending and recovery operations raise; repeated valid close is harmless.
Quarantined records count in capacity and pending, but do not block healthy
records. Export/retry/discard are explicit operator-directed recovery operations.

JavaScript durability requires Node.js 22.13+ with builtin SQLite, while browser
inference retains its existing support. Await `traces.pending`; flush/close trace
timeouts are in seconds. Explicit undefined is not JSON data. No browser or
memory-only durability fallback is used. Ephemeral/serverless handlers must await
flush before returning; deleting their filesystem can still lose queued records.

The backend accepts logging facts directly, without fabricated status/timing,
and keeps account-lifetime deduplication receipts after content cleanup. Native
background capture remains separate and best effort. These sources are implemented
and tested, but the matching backend/SDK packages have not been deployed or
published by this task.
