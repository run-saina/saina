# Trace backend and recent-fix audit

Historical review, 2026-10-27. **Follow-up:** the user subsequently authorized
resolving and implementing the remaining requirements. The new additive endpoint
and durable recorder are specified in [durable-traces.md](durable-traces.md);
the gaps below describe the backend before that implementation.

Reviewed 2026-10-27. Remote `staging` refs were fetched before this audit. This is
source and local-test evidence, not verification of a running deployment.

## Reviewed revisions

| Repository | Staging revision | Relevant surface |
|---|---|---|
| website | `a3d63b343e64ed814b375303ac227ca2ba725671` | Authenticated Studio route, journaled Postgres persistence, bounded native background delivery, CORS fix |
| decision-platform | `c8e581a4293b86a489ec96bb9d5dd3051005a004` | External trace validator, ingestion, identity, receipts, outcomes and observability |
| saina SDK | `1df3df03293e0701fb2d9fb731dadd1232740d6b` | Python 0.10.2 / JavaScript 0.5.2 `TraceClient`, delivery and acknowledgment fixes |

The working SDK implementation in this chat is based on an older `main`. The newer
staged `TraceClient` is real functionality and must not be silently removed when
integrating 1.0. It is distinct from the agreed new `Traces.log()` API.

## Resolved: the existing external-capture protocol

The backend already implements:

```text
POST /v1/studio/data/collections/{collection_id}/traces
Authorization: Bearer <account API key>    # hosted gateway
Content-Type: application/json

{"source": "application", "traces": [<record>, ...]}
```

The body has exactly `source` and `traces`. A missing collection is created on
successful ingestion; an existing collection's configuration is preserved.
`collection_id`, `source`, `decision_id`, and `task` use identifiers matching
`[A-Za-z0-9_.-]{1,120}`. This is narrower than the new logging API's non-empty task
name. Standalone Studio uses the operator's service origin and its own auth policy;
this audit does not equate it with the account-authenticated hosted route.

A record has these required fields:

| Field | Existing contract |
|---|---|
| `schema_version` | Exactly `"1.0"`; this is the record schema version, not the SDK release |
| `decision_id` | Stable caller or SDK-generated identifier |
| `task` | Stable identifier for the operation |
| `inputs` | Finite JSON data |
| `recorded_at`, `completed_at` | Timezone-aware timestamps; completion cannot precede start |
| `status` | `completed`, `error`, or `incomplete` |
| `timing` | Exactly `{total_ms: <finite nonnegative number>}` |

Optional fields are `answer`, `metadata` (an object), and `error`. `completed`
requires an explicit `answer`; JSON null counts as an answer. `error` status
requires exactly `{type: <nonempty string, at most 200 characters>}`; error data is
forbidden for other statuses. Exception messages and stacks are not part of this
record. Unknown record fields are rejected.

Limits are implemented and tested:

- 1–100 records per batch.
- At most 1 MiB per record, measured as compact UTF-8 JSON.
- At most 8 MiB for the sum of compact record encodings in a batch.
- The hosted parser separately bounds the JSON envelope to 8 MiB + 64 KiB, checks
  duplicate keys, finite values and nesting depth (64); its transport body cap is
  25 MiB. The 8 MiB trace-content cap remains the effective batching limit.

Sources: platform `packages/contracts/saina_decision_contracts/decision_traces.py`,
`media.py`, and `apps/api/saina_decision_service/decision_traces.py`; website
`gateway/src/saina_gateway/routes/studio.py`.

## Resolved: acknowledgments, duplicates and rejection

Successful ingestion returns HTTP 201 and a receipt containing `decision_ids` in
submitted order, `record_ids`, `collection_id`, `count`, `inserted_count`,
`deduplicated_count`, `status: "succeeded"`, ingest `id`, digest, source, channel,
and creation time. An identical batch replay additionally reports `duplicate: true`.

The staged SDK validates that `decision_ids` exactly matches its submitted ordered
list before removing that batch from its queue. A missing, malformed or mismatched
acknowledgment retains the records for retry.

Per-record identity is account-scoped and derived as:

```text
record_id = "trc_" + sha256(collection + NUL + source + NUL + decision_id).hexdigest()[:48]
```

The batch ingest key is server-generated as `traces_` plus a truncated canonical
payload digest including collection, source and traces. There is no required
client `Idempotency-Key` header for this route. Stable record identity also makes
regrouping records into different batches safe after a lost acknowledgment.
Identical evidence deduplicates; changed evidence under the same identity returns
409 `record_conflict`. A conflicting explicit generic ingest key instead returns
409 `idempotency_conflict`.

**Batches are atomic. There is no partial-acceptance response.** Validation and the
repository transaction reject the batch rather than acknowledging selected rows.
A permanently invalid/conflicting record can therefore prevent delivery of the
whole retained batch. The backend's atomic behavior is known; a future durable
client's quarantine/export/recovery policy is not implemented by that fact.

Deduplication currently relies on stored record and ingest rows, not a separately
specified TTL. No guaranteed deduplication interval after retention cleanup,
collection/account deletion or disaster recovery is established by these sources.
Do not infer indefinite deduplication from the absence of an expiry column.

Sources: platform `apps/api/saina_decision_service/data.py`, `decision_traces.py`,
and `apps/api/tests/test_decision_traces.py`; SDK `src/saina/traces.py` and
`clients/javascript/traces.js`.

## Confirmed recent fixes

- SDK `473f086`: closes retry HTTP error responses and checks acknowledgments before
  dropping queued evidence.
- SDK `cc4a3dc`: reuses a bounded HTTP connection pool and reduces capture overhead.
  Explicit native endpoint invocation is never automatically retried.
- SDK `2966199`: JavaScript failed full batches back off instead of spinning.
- SDK `1df3df0`: outcome reporting retries temporary 404 while native evidence is
  still being persisted; retrying an outcome does not invoke the model again.
- Platform `dee938e` and website `a62df9b`: batch durable Studio writes and paginate
  observability in storage rather than scanning unrelated evidence.
- Platform `31d0ba0` and website `d462197`: move native evidence persistence after
  HTTP response transmission. Stable IDs prevent duplicate evidence on retries.
- Website `a3d63b3`: allows SDK headers from the configured console origin. This
  does not authorize exposing account credentials in browser code or add durable
  browser storage.

External capture retries delivery of evidence, using the original IDs; it does
not execute the external operation again. Existing capture retries transport
failures and HTTP 408/429/500/502/503/504 with bounded per-attempt backoff. These are
legacy trace delivery rules, distinct from inference's server-aware retry policy.

## Native background evidence is best effort

The hosted writer uses a separate database pool, up to 16 workers, 256 waiting
jobs, and 32 MiB of pending response-size estimates. Capacity exhaustion drops
jobs; storage failures retry four total attempts and then drop evidence. Graceful
shutdown attempts a drain for 30 seconds. The standalone writer has 8 workers and
128 admission slots; it is also best effort.

A native inference response is **not** a durable trace acknowledgment. Native
observability is eventually consistent. A crash before persistence can lose the
evidence. Client `flush()` does not drain the server's native queue. These fixes
must not be described as a durable SQLite spool or at-least-once crash recovery.

## What still blocks the new Traces.log API

The existing protocol cannot directly implement the agreed logging contract:

| New 1.0 requirement | Existing staged capture |
|---|---|
| `Traces.log(task, inputs, outcome?, metadata?)` | `TraceClient.decision(...)` scope and mutable `set_answer()` |
| No required collection/source | Collection in route; source required in body and identity |
| Logging timestamp; no invented operation facts | Execution start/end, duration and status required |
| Arbitrary non-empty task name | Restricted stable identifier |
| Return an ID only after local durable commit | Enqueue in memory; no SQLite spool |
| Bounded `flush(timeout)` returning an immutable snapshot | Flush delivers queued records or raises; no such result type |
| Closed log/flush/pending consistently raise | Legacy lifecycle and pending-count behavior differ |
| Thread-safe warning episodes and recovery counts | Per-failure warning/callback behavior |

Consequently the existing schema resolves the **legacy wire-protocol questions**,
but does not settle a `/v1/traces` record format for the new API. Translating the
new API by fabricating completion time, duration, status, collection or source
would contradict the agreed contract. A separately specified additive route/schema
or an explicit public-contract revision is still necessary. Do not restore the
superseded `{collection, source, runs}` envelope.

The remaining SDK design work is concrete: platform spool paths/override; lazy
SQLite initialization; URL/hashed-credential isolation; cross-process ownership;
logical/physical capacity and maximum-record accounting; permanent rejection and
recovery; concurrent flush/close semantics; storage failures during flush/close;
flush-result name/default timeout; and supported JavaScript durable runtimes.
None is answered by a bounded in-memory queue.

## Inference idempotency: confirmed implementation, separate version qualification

The hosted gateway at audited website revision `a3d63b3` implements account-scoped
UUIDv7 idempotency for native inference, including request fingerprint conflicts,
request-in-progress, retained result replay, expired-result responses and recovery
fences. New keys must be no more than 24 hours old or 5 minutes in the future;
existing keys are resolved before the new-key age check. Default replay-body
retention is 24 hours and is configurable. An expired replay must not silently run
the operation again. This path is separate from trace record deduplication.

A self-hosted deployment of the **gateway stack at that revision**, with its
configured database, journal and replay services, has that implementation. The
standalone model server shipped in this working checkout does not. Gateway
`0.1.0` is not sufficient version evidence: the package version is unchanged over
many commits. Qualify a supported self-hosted release using an immutable release
manifest/source revision and integration tests; this audit does not claim a live
production verification or a supported semver range.

Sources: website `gateway/src/saina_gateway/admission.py`, `config.py`,
`recovery.py`, and `gateway/tests/test_inference.py`.

## Local verification

The following targeted suites were executed against the reviewed source:

- Platform external-capture suite: **11 passed** (SQLite-backed service tests,
  including batch boundaries, atomic rejection, conflicts, tenant separation,
  observability and deferred native capture).
- Gateway background-delivery suite: **2 passed** (bounded retry/drain and response
  transmission preceding storage).
- Staged Python trace SDK suite: **11 passed** (including real local HTTP retry,
  queue retention after invalid acknowledgment, native connection reuse and
  eventual outcome reporting).
- Staged JavaScript trace SDK suite: **11 passed** (including backoff after full
  batch failure, concurrent flush, invalid acknowledgment and eventual outcomes).

The gateway's paid-inference PostgreSQL/recovery suite and live deployed services
were inspected/documented, not exercised by this audit. No application/backend
code, running service, or published package was changed.
