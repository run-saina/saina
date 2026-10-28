# Durable logging protocol and storage decisions

This resolves the durable `Traces` release questions under the user's 2026-10-27
implementation request. Implementation spans the public SDK, decision-platform
service/contracts, and the hosted website gateway. It is additive to the existing
`TraceClient` execution-scope API and its collection-based endpoint. These source
changes have not been deployed or published to package registries.

## Public use

```python
from saina import Traces

with Traces() as traces:  # resolves SAINA_API_KEY, SAINA_BASE_URL, SAINA_TIMEOUT
    trace_id = traces.log(
        task="route ticket",
        inputs={"ticket": "Charged twice"},
        outcome={"team": "billing"},
        metadata={"application": "support", "environment": "production"},
    )
    delivery = traces.flush(timeout=5)
    if not delivery.complete:
        print(delivery.pending, delivery.rejected)
```

A non-null returned ID means a successful local SQLite commit, not server receipt.
Default warn mode returns `None` for failed storage admissions; strict mode is
`on_storage_error="raise"`. Invalid arguments always raise. There is no provider
interception, operation timing or inferred status. Omitted outcome differs from
explicit null. Values are snapshotted before storage. Tasks are nonblank names of
at most 1024 Unicode characters; JSON nesting is limited to 60 within a record.

JavaScript durable recording requires **Node.js 22.13 or newer with node:sqlite**.
There is no durable browser mode. Inference retains its existing runtime support;
Node-only modules load lazily when durable storage is used.

```javascript
const traces = new Traces({onStorageError: "warn"});
const id = await traces.log({task: "route ticket", inputs: {ticket: "Charged twice"}, outcome: null});
const delivery = await traces.flush({timeout: 5});
const pending = await traces.pending;
await traces.close({timeout: 5});
```

JavaScript's `pending` getter returns a promise because initialization is lazy.
Constructor `timeout` retains milliseconds; flush/close timeout is in seconds, as
in the agreed JavaScript trace examples. Explicit `undefined` is not JSON data.
The existing `TraceClient` and its asynchronous scope helpers are preserved.

## Lazy storage and platform paths

Python override: `spool_path=<file>`; JavaScript: `spoolPath: <file>`.
An override is the complete SQLite filename; relative overrides resolve against
the working directory. A recorder never uses an in-memory fallback.

Defaults are:

| Platform | Default filename |
|---|---|
| Linux | `${XDG_STATE_HOME:-$HOME/.local/state}/saina/traces/<service>/<credential>/spool.sqlite3` |
| macOS | `$HOME/Library/Application Support/Saina/traces/<service>/<credential>/spool.sqlite3` |
| Windows | `%LOCALAPPDATA%/Saina/traces/<service>/<credential>/spool.sqlite3` |

Relative `XDG_STATE_HOME` is ignored. Windows falls back to the user's
`AppData/Local` directory when `LOCALAPPDATA` is absent.

`service` is the complete SHA-256 hex digest of the configured base URL with
lowercase scheme/authority and trailing path slashes removed. `credential` is
the complete SHA-256 digest of the API key. Raw API keys never enter spool paths,
filenames, SQLite metadata or diagnostic messages. A second binding hash of the
two hashes is stored in SQLite and checked even for explicit overrides. Reusing
a spool with different credentials/origin or capacity raises a storage error;
it never silently mixes records between tenants. Python and JavaScript share this
format and can drain one another's spool with identical configuration. Node sends
stored payload text directly, preserving arbitrary Python integers and numeric
representations rather than passing them through JavaScript number parsing.

Construction creates no file or worker. The first successful logging operation
initializes storage. Empty `pending`, `flush` and `close` do not create a missing
spool. Reading/flushing an existing spool opens it and can resume background
delivery; after a restart call `flush()` to drain without logging a new operation.
On POSIX, new namespace directories use 0700 and new files 0600; Windows inherits
the user's filesystem ACLs. Existing unrelated databases are refused. Replacement
or deletion of an opened spool is a storage error, not an empty successful flush.

Credential rotation deliberately selects another default spool. Drain before
revoking an old key. An operator recovering records after revocation must verify
the destination account and replay exported stable IDs through the endpoint;
there is no automatic credential rebinding or cross-account retargeting.

## Capacity, transactions and process ownership

Default `capacity` is **64 MiB**. It counts compact UTF-8 payload bytes plus
**256 bytes per row**, including quarantined records. SQL triggers maintain this
counter in the same transactions as insertion/deletion. Capacity check and admission
are atomic across threads/processes. No record is evicted to admit another.
Record payloads are limited to **1 MiB**; oversize arguments fail validation
before I/O rather than being treated as storage failure.

SQLite uses 4096-byte pages, DELETE journaling, `synchronous=FULL`, and FULL
auto-vacuum. A page-count cap limits the main file to logical capacity plus 2 MiB;
SQLite metadata can cause an earlier full condition. Rollback journals temporarily
need additional space of roughly one database file, plus journal/filesystem
headers. Provision approximately twice capacity plus 4 MiB **and filesystem
margin**. Capacity is a logical admission budget, not a hard filesystem quota.
No unbounded WAL is created. Acknowledgment deletes reclaim pages.

Python opens and closes a connection on the thread performing each transaction;
no SQLite connection crosses threads. JavaScript owns its synchronous connection
on its event loop. Lock waits are bounded at 250 ms. Concurrent processes may
append and acknowledge the same spool. Workers can send the same immutable record
concurrently; server identity/digest deduplication makes that safe. SQL updates
commute and an ambiguous network failure does not remove records. No long-lived
exclusive spool lock or crash lease delays restart recovery.

## Protocol: POST /v1/traces

The hosted route requires a bearer account API key and existing tenant binding
and security-suspension checks; it ignores console cookies for authentication.
It is registered with Studio enabled and uses the existing fenced/journaled
PostgreSQL repository. Standalone Studio exposes the same endpoint under its
existing operator-owned, unauthenticated single-workspace deployment model.
It never invokes a model or submits native inference as external evidence.

```json
{
  "schema_version": "1.0",
  "traces": [{
    "trace_id": "01890a5d-ac96-774b-bcce-b302099a8057",
    "recorded_at": "2026-10-27T12:00:00+00:00",
    "task": "route ticket",
    "inputs": {"ticket": "Charged twice"},
    "outcome": null,
    "metadata": {"application": "support"}
  }]
}
```

`trace_id` is a canonical RFC UUID (SDKs generate UUIDv7); recording time includes
a timezone. The record has exactly the fields above, with `outcome` optional.
Metadata is an object, defaulting to `{}` in the SDK. No collection, source,
execution start/end, duration, inferred status, or nested `extra` is introduced.

Limits: 1–100 records, at most 1 MiB per compact UTF-8 record, 8 MiB summed record
content, and 8 MiB + 64 KiB for the full envelope. Duplicate or uncorrelatable IDs,
unknown envelope versions, malformed JSON/duplicate keys, or oversized envelopes
fail the entire request without acknowledgment. Individual invalid records with
valid unique IDs are rejected separately. No HTTP `Idempotency-Key` is required;
the record IDs themselves provide replay identity.

HTTP 200 acknowledges the complete partition of submitted IDs. For a two-record
batch, one possible response is:

```json
{
  "schema_version": "1.0",
  "accepted": ["01890a5d-ac96-774b-bcce-b302099a8057"],
  "rejected": [{
    "trace_id": "01890a5d-ac96-774b-bcce-b302099a8058",
    "code": "trace_conflict",
    "retryable": false
  }]
}
```

Each ID appears exactly once across the two arrays. SDKs validate the schema,
partition, uniqueness and exact submitted identity set before modifying storage.
Missing, extra, duplicate, malformed or ambiguous acknowledgments retain the
entire batch. Response bodies are bounded to 2 MiB. A transport or storage failure
returns no receipt; retries use the identical stored records, including their IDs
and timestamps. Regrouping batches is safe.

Identity is **authenticated account + trace_id**. The backend stores the immutable
record and a canonical payload-digest receipt in one repository transaction.
Exact replays are acknowledged as accepted without another stored record. Changed
content returns `trace_conflict`; invalid record fields return `invalid_trace`.
Rejections do not echo payloads. Other valid records in a mixed batch commit
normally. A receipt is returned only after the durable transaction completes.

Digest receipts have **no time expiry** and survive content-retention cleanup.
They remain for the account's lifetime; explicit account deletion removes them.
This is the deduplication retention contract, subject to the gateway's existing
journal/backup recovery guarantees. Backups/receipts must be recovered together.
No promise survives operator destruction of durable account state.

The legacy collection-based trace endpoint remains atomic and unchanged. Its
schemas, execution facts, scoped IDs and memory queue are not translated into the
new logging protocol. Native background capture remains best effort.

## Delivery, quarantine and explicit recovery

A worker starts lazily after admission or an explicit operation on an existing
spool. Delivery is at least once. It sends up to 100 records/8 MiB per batch and
retries failures with 100 ms exponential backoff capped at 30 seconds. Retry-After
(seconds or HTTP date) is honored, bounded to one day; explicit flush can request
an immediate new attempt. A request has at most a five-second socket/fetch timeout,
or the smaller configured timeout. Network errors never throw asynchronously into
the application. `last_error` / `lastError` reports a sanitized last failure.

Accepted IDs are removed transactionally. Permanently rejected IDs move to
`rejected` with the server reason. They still count in `pending` and capacity,
are never retried automatically, and do not block later healthy records.

Recovery is operator-directed and works only on rejected records:

```python
traces.export_rejected("rejected.jsonl")  # new 0600 file; fsynced; no overwrite
traces.retry_rejected([trace_id])        # unchanged IDs/payloads after fixing cause
traces.discard_rejected([trace_id])      # explicit deletion after export/review
```

JavaScript uses `await exportRejected(path)`, `retryRejected(ids)`, and
`discardRejected(ids)`. Export is JSONL containing `record` and `reason`; Node preserves raw stored JSON
numbers when exporting Python records. Export never
removes records. Recovery ID sets must be unique and entirely quarantined or the
transaction fails. Pending records cannot be discarded through this API. Corrected
content must be logged as a new record/ID; the original is preserved until explicit
discard. No server-side trace search/edit API is added.

## Lifecycle, snapshots and storage failures

Python `TraceFlushResult` and the equivalent frozen JavaScript result contain:
`complete`, `pending`, `rejected`. Flush targets the persisted sequence watermark
at entry. Later admissions do not expand it. `complete` means every targeted record
was acknowledged; a targeted quarantine makes it false. Counts are transactionally
read snapshots of **all** unacknowledged rows at return, including newer admissions
and rejected rows. Previously returned snapshots do not change.

Concurrent flushes share the worker but retain independent watermarks/deadlines.
Network failure or deadline returns an incomplete result; a local storage error
raises `TraceStorageError`/`SpoolFullError` regardless of warn mode. `pending` also
surfaces storage errors; no successful admission is implied for warn-mode failures.

`close(timeout=5)` atomically stops new admission, attempts a bounded flush,
cancels network work, stops worker scheduling, and releases resources. Undelivered
records remain on disk. Admissions already in progress finish their commit or
fail without returning an ID. Concurrent close calls share shutdown; valid repeated
close calls are harmless, even after a storage-error shutdown. A concurrent flush
ends with its acknowledged/incomplete snapshot when close stops delivery. If JavaScript
shutdown wins before an initial flush snapshot exists, that flush raises a closed-recorder
error instead of claiming completion. After
close, `log`, `flush`, `pending` and recovery operations raise, regardless of storage
mode. Python `with` manages resources and preserves an active application exception.

Timeouts bound delivery/waiting, not an unresponsive filesystem or OS DNS resolver;
SQLite lock acquisition can additionally take up to its 250 ms bound. Connected
HTTP bodies and fetches are interrupted on close. Python's atexit and Node's
beforeExit hooks attempt a five-second best-effort close; forced termination has no
hook guarantee. In ephemeral/serverless environments explicitly flush before
returning and inspect the result. Durable means a committed record survives
process failure on the same durable filesystem, not destruction of that filesystem.

Warn mode emits once per storage-failure category episode per recorder. Repeated
failures increment the lost-admission count and return null/None. A category change
starts a new episode. The next successful commit reports recovery and that episode's
lost count at informational level. Python state is protected by the recorder lock;
JavaScript updates it serially on its event loop. Failed admissions were never queued.
