# @run-saina/sdk

JavaScript client for Saina, the decision layer for software systems. Saina builds specialist AI decision models and the developer platform around them. Developers use Saina to turn model predictions into versioned decision endpoints, observe how those decisions perform in production, turn failures into test cases, and improve their models or policies without breaking existing behavior. Saina works with open models and external inference providers, so teams can use their own infrastructure.

This package calls Helm decision models over HTTP (hosted at `api.saina.run` or self-hosted)
and sends decision traces to Saina Studio. It is a pre-release.

```sh
npm install @run-saina/sdk@next
```

```js
import { Saina } from '@run-saina/sdk';

// baseUrl defaults to https://api.saina.run; pass your own for a self-hosted server.
const client = new Saina({
  apiKey: process.env.SAINA_API_KEY,
  autoIdempotencyKey: true, // one UUIDv7 key per call, so the call can be retried safely
});

const result = await client.ask({
  model: 'saina-helm',
  state: 'I was charged twice.',
  questions: {
    issue: {
      type: 'single_choice',
      question: 'Identify the banking issue.',
      options: {
        duplicate: 'duplicate charge',
        delivery: 'card delivery',
      },
    },
  },
});
```

## Images and audio

For a model that reads them (see `inputs` in `/v1/models`), pass `images` and `audios`: data URLs, http(s)
URLs, `Blob`/`File` objects, `ArrayBuffer`s or typed arrays such as a Node `Buffer`. The client sends them as
base64 data URLs; every question sees all of them.

```js
const photo = document.querySelector('input[type=file]').files[0];
const result = await client.ask({
  model: 'saina-helm',
  state: 'The customer attached a photo of the delivery and a voice note.',
  images: [photo],      // PNG, JPEG or WebP; at most 8, each at most 10,000,000 bytes
  audios: [voiceNote],  // a Blob: WAV, FLAC, MP3, Ogg or WebM; at most 4, each at most 10 s
  questions: {damaged: {type: 'yes_no', question: 'Is the item damaged?'}},
});
```

A text-only model refuses images and audio with `InvalidRequest`; `videos` are not supported yet.
`imageDataUrl(value)` and `audioDataUrl(value)` convert one input ahead of time.

## Idempotency and retries

Inference (`ask`, `systemOne`, `decisions`, and their metadata variants) is retried automatically only when the call carries an `Idempotency-Key`:
one UUIDv7 key is generated per logical call by default. Pass `{idempotencyKey: newIdempotencyKey()}`
as the second argument to supply your own key. Set `autoIdempotencyKey: false` to disable generation;
inference then retries only with an explicit key. Retries reuse the key and identical bytes. Protection
against duplicate execution and billing requires server-side idempotency support; see
[server compatibility](../../MIGRATION.md).

Retries (`maxRetries`, default 2; total wait capped by `maxRetryDelay`, default 30000 ms) happen only on
connection loss or an uncertain response, `accounting_unavailable`, `request_in_progress`, `rate_limited`,
and `overloaded`/`inference_unavailable` when the server says the request was not admitted. `Retry-After`
is honored; otherwise backoff is exponential with jitter.

## Errors

Every failure is a `SainaError` (`SainaHelmError`) with `status`, `code`, `requestId`, `admitted`, `state`,
`retry`, `retryAfter` (seconds) and `suspension`. Each contract code has a subclass, such as
`InsufficientCredits`, `RateLimited`, `IdempotencyConflict` or `InferenceFailed` (see `ERROR_CLASSES`);
connection failures are `SainaConnectionError`. A `retry` of `new_operation` means only a new key can run it again.

## Billing metadata and account

```js
const {result, metadata} = await client.askWithMetadata(request);
metadata.creditsCharged; // BigInt, e.g. 1234n
metadata.balance;        // BigInt after settlement, or null
metadata.requestId; metadata.replayed; metadata.idempotencyKey;

const {available} = await client.balance();             // BigInt
await client.usage({from: '2026-10-01', channel: 'api'});
await client.requests({limit: 50, status: 'settled'});
```

Credit amounts are 64-bit integers sent as decimal strings; the client parses every credit field
(`balance`, `reserved`, `available`, `overdraft_allowance`, `debt`, `credits`, `credits_charged`,
`balance_after`) to `BigInt` and never through a JavaScript number. Native `ask()` returns typed answers;
`client.lastMetadata` remains a compatibility property. Prefer `result.metadata` for concurrent calls.

This client calls an existing endpoint; it does not run weights in JavaScript.
Apache-2.0. Saina is operated by Rama Labs Inc.


Response envelopes expose `.metadata` and the original wire response in `.raw`. Parsed credit fields
are `BigInt`; `toObject()` and `toJSON()` serialize them as decimal strings.
`JSON.stringify(result)` uses the same representation. `stringifyCredits(value)` also serializes account
results containing `BigInt`. Python serializes credits as JSON integers; ordinary JavaScript number
parsing cannot round-trip arbitrary Python credit amounts without precision loss.

Question-level and top-level `extra` objects are flattened into outgoing fields. Collisions with known
fields throw `TypeError` before network I/O. Options may be dictionaries or lists of strings.
Errors inherit broad categories such as `AuthenticationError`, `RateLimitError`, `IdempotencyError`,
`ServerError`, and `ModelBusyError`. `.fields` preserves all server error fields, including unknown ones.
`SainaHelmError` is exactly `SainaError`; `SainaConnectionError` is exactly `TransportError`.


## Typed answers and local selection

`new Saina()` resolves `SAINA_API_KEY`, `SAINA_BASE_URL`, and `SAINA_TIMEOUT` at construction. Explicit
arguments win. Environment timeout is in seconds; the constructor's `timeout` remains in milliseconds.
Non-loopback HTTP requires `allowHttp: true`.

```js
const selections = result.select({threshold: 0.9, minMargin: 0.1});
const issue = selections.issue;
if (issue.accepted) {
  console.log(issue.value);
}
```

`result.answers.issue` is an immutable typed answer and also supports `.select()`. Selection is synchronous
and makes no network calls. `.accepted` distinguishes accepted false/zero values from rejection. Rating
probability keys are numeric indices; rating selections expose `.label`. Multi-choice preserves option
order and returns `no_options_selected` if nothing qualifies. Threshold checks precede ties and margins;
margin is ignored for rating and multi-choice.

Use `selections.get(questionId)` for arbitrary question IDs, including names such as `toObject`,
`toJSON`, or `get` that collide with methods. It returns `undefined` for an unknown ID. Direct properties
such as `selections.issue` remain available when the ID does not collide. `selections.toObject()` and
`JSON.stringify(selections)` include every question, including colliding names.

Options supplied as lists retain their declared order in the wire request, including numeric-looking
labels such as `['10', '2']`.

Native `ask()` always requests distributions. Move unknown ordinary fields into `extra`, and replace
`mode: 'decision'` and server thresholds with local selection. Responses require a body `request_id` or
`X-Request-Id` header; if both exist, they must agree. Metadata is bound to each result for concurrent code.
System One and Decisions compatibility methods retain their wire-shaped results.

## Durable external logs (Node.js 22.13+)

```js
import {Traces} from '@run-saina/sdk';
const traces = new Traces({onStorageError: 'warn'});
const id = await traces.log({task: 'route ticket', inputs: {ticket: 'Charged twice'}, outcome: null});
const delivery = await traces.flush({timeout: 5});
await traces.close({timeout: 5});
```

A returned ID acknowledges local SQLite persistence. `await traces.pending` counts unacknowledged
records including quarantine. Storage errors during admission warn once per episode and resolve null;
`onStorageError: 'raise'` raises. Flush returns a frozen complete/pending/rejected snapshot. Trace
flush/close timeouts use seconds. The `spoolPath` override is a complete filename; default namespace
paths use service and credential hashes. Browser inference imports remain compatible; durable browser
recording is not provided. The existing `TraceClient` is preserved. See
[storage, protocol, recovery and lifecycle](../../docs/durable-traces.md).
## Decision trace collection

Use `TraceClient` to capture explicit decision inputs and answers from your own code,
observe them in Studio, and prepare datasets from collection evidence. Python supports
sync/async context managers; TypeScript provides an async callback scope.
`invoke_endpoint` / `invokeEndpoint` calls a Saina Decision Endpoint with one server-side trace.
Business outcomes record the business value of a decision, or a proxy for it (resolved,
revenue, cost, CSAT), not a label. `reportBusinessOutcome` posts them to
`/v1/studio/business-outcomes`. `reportOutcome` remains a deprecated alias and warns once.
See [decision trace collection](https://github.com/run-saina/saina/blob/staging/docs/tracing.md).
