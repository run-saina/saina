# @run-saina/sdk

Pre-release Saina HTTP client.

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
  model: 'helm-0.8b',
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

## Idempotency and retries

Inference (`ask`, `systemOne`) is retried automatically only when the call carries an `Idempotency-Key`:
pass `{idempotencyKey: newIdempotencyKey()}` as the second argument, or construct the client with
`autoIdempotencyKey: true`. Every retry of one call reuses that key and the identical body, so a retry
never runs or charges twice; the client never generates a new key for a retry.

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
`balance_after`) to `BigInt` and never through a JavaScript number. `ask` still returns the same response
object as before; `client.lastMetadata` holds the metadata of the last inference call.

This client calls an existing endpoint; it does not run weights in JavaScript.
Apache-2.0. Saina is operated by Rama Labs Inc.
