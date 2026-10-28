# Collect decision traces

`TraceClient` records decisions made by your own application into Saina collections.
Use an explicit Python context manager or a TypeScript callback scope. Declare the
inputs and answer you want to keep; database handles, model clients, credentials
and other function arguments are not inspected. Capture does not run inference.

A collection is the growing container of evidence. Observability reads those same
records. A dataset version is a frozen selection prepared for training or evaluation;
capturing an answer does not approve it as a training target or start fine-tuning.

## Python

```python
import os
from saina import TraceClient

traces = TraceClient(
    collection="support-routing",
    source="support-agent",
    api_key=os.environ["SAINA_API_KEY"],
    base_url="https://api.saina.run",
)

def route_ticket(ticket, db, model_client, config):
    context = db.customer_context(ticket["customer_id"])
    with traces.decision(
        task="route-ticket",
        inputs={"message": ticket["message"], "context": context},
        metadata={"app_version": "2026-10"},
    ) as trace:
        answer = model_client.route(ticket, context, config)
        trace.set_answer(answer)
    db.save_route(ticket["id"], answer, decision_id=trace.decision_id)
    return answer

# When the business value is known, reuse a stable event ID on retries.
# A business outcome is resolved, revenue, cost, CSAT, or another proxy — not a label.
traces.report_business_outcome(
    "trace_returned_decision_id",
    values={"resolved": True},
    event_id="ticket-48213-resolved",
)
# At graceful process shutdown, wait for acknowledgement of pending traces.
traces.close()
```

Inputs and answers must be finite JSON data. They are copied when supplied, so later
mutation of application objects does not change the evidence. Optional `recorded_at`
replays a benchmark on a simulated clock. It is an ISO 8601 timestamp with a
timezone, such as `2020-01-02T03:04:05+00:00` or `2020-01-02T03:04:05.250Z`, and is
kept exactly. A naive timestamp is rejected. When omitted, the scope records the
clock at entry. JavaScript uses the same `recorded_at` option. `invoke_endpoint` /
`invokeEndpoint` does not send it, and native `decide` keeps server time.

An explicit `None` answer is valid; omitting `set_answer` produces an `incomplete` trace. Application
exceptions propagate unchanged; the trace records the exception type, without its
message or stack. Execution timing covers the scope, excluding background delivery.
Redact sensitive content before supplying it; explicit inputs, answers and metadata
are stored in your collection. Use references for assets, not raw media bytes.

For async applications, use `async with traces.decision(...)`, `await traces.aflush()`,
`await traces.areport_business_outcome(...)`, and `await traces.aclose()`. The client itself
also supports `with TraceClient(...)` and `async with TraceClient(...)` to flush on exit.

## TypeScript / JavaScript

```typescript
import {TraceClient} from "@run-saina/sdk";

const traces = new TraceClient({
  collection: "support-routing",
  source: "support-agent",
  apiKey: process.env.SAINA_API_KEY,
  baseUrl: "https://api.saina.run",
});

async function routeTicket(ticket, db, modelClient, config) {
  const context = await db.customerContext(ticket.customerId);
  let decisionId = "";
  const answer = await traces.decision(
    {task: "route-ticket", inputs: {message: ticket.message, context}},
    async trace => {
      decisionId = trace.decisionId;
      const answer = await modelClient.route(ticket, context, config);
      trace.setAnswer(answer);
      return answer;
    },
  );
  await db.saveRoute(ticket.id, answer, {decisionId});
  return answer;
}

await traces.reportBusinessOutcome("trace_returned_decision_id", {
  eventId: "ticket-48213-resolved", values: {resolved: true},
});
await traces.close();
```

`report_business_outcome` / `areport_business_outcome` and `reportBusinessOutcome` send
`POST /v1/studio/business-outcomes` with `source`, `collection_id`, `decision_id`,
`event_id`, and `values`. Studio still accepts `POST /v1/studio/outcomes` as a deprecated
server alias; this client does not call it. `report_outcome`, `areport_outcome`, and
`reportOutcome` remain deprecated aliases that call the business-outcome methods. Python
aliases emit `DeprecationWarning`. JavaScript `reportOutcome` is marked `@deprecated` and
warns once on the console.

`decision()` preserves the callback's return value and propagates its thrown value.
Only `setAnswer()` defines the recorded answer; the return value is not captured
implicitly. Missing answers are incomplete, including callbacks that return a value
without calling `setAnswer`. `null` is a valid explicit answer. Never embed an account
API key in a browser bundle; use your application's server to deliver authenticated
traces. Browser capture uses memory only, with no localStorage or IndexedDB writes.

## Call a Saina Decision Endpoint

Calling an endpoint is an explicit inference operation. The proxy returns the answer
without waiting for trace storage. Native evidence is captured asynchronously into
the configured collection; do not wrap this call in an extra external trace scope.
The result includes the native `decision_id` and `collection_id`.

Native evidence and observability are eventually consistent. A response does not
acknowledge durable trace storage: a crash before persistence, queue exhaustion or
exhausted storage retries can lose evidence. Gateway metrics expose persisted and
dropped trace counts, queued jobs and pending bytes. Graceful shutdown drains pending
work within the shutdown deadline. Each persisted trace uses its original decision
ID, so background retries do not duplicate evidence or repeat inference.

Business outcome reporting retries temporary not-found responses while native evidence is
being persisted. If retries are exhausted, retry the business outcome with the same event ID.
`flush()` on the client waits for external SDK captures, not the server's native queue.

```python
decision = traces.invoke_endpoint(
    "support-triage",
    {"state": {"message": "I was charged twice"}},
    version="production",  # or an immutable positive endpoint version
)
traces.report_business_outcome(
    decision["decision_id"], values={"resolved": True},
    event_id="ticket-48213-resolved",
)
```

```typescript
const decision = await traces.invokeEndpoint(
  "support-triage",
  {state: {message: "I was charged twice"}},
  {version: "production"},
);
await traces.reportBusinessOutcome(decision.decision_id, {
  eventId: "ticket-48213-resolved", values: {resolved: true},
});
```

Python also exposes `await traces.ainvoke_endpoint(...)`. Native invocation is never
retried by the trace client. An invocation timeout can have an unknown outcome;
inspect Observability before deciding whether to execute again. Calling the native
inference SDK's `Saina.ask()` inside a capture scope also works: your code calls the
model and passes its answer to `set_answer`; capture itself only sends evidence.

## Delivery and shutdown

Constructing a client and entering/exiting a decision scope perform no synchronous
network request. Pending traces are sent in background batches (default 100 traces
or every 5 seconds), bounded to 8 MiB of trace content per request. Each trace is at
most 1 MiB. The backend creates a missing collection using its ID as the name and
preserves existing collection configuration. Choose a collection ID of 1–120 letters,
digits, underscores, dots or hyphens.

Delivery retries transient transport errors and HTTP 408/429/500/502/503/504 with
bounded backoff. Stable collection/source/decision identities make acknowledgement
loss and regrouped retries safe. Changed evidence under the same identity conflicts.
Set `decision_id` / `decisionId` to a durable application identifier when replaying
an event after restarting the process. Different sources have separate identities.

Call `flush()` (Python async: `aflush()`) to wait for queued evidence. `report_business_outcome`
/ `reportBusinessOutcome` flushes pending evidence before sending the business outcome. A failed
batch stays queued; retry `flush()` using the same client. `close()` stops background
delivery and flushes; retry it if delivery fails. Complete active scopes before closing.

The queue is bounded, in memory and not crash durable. Default capacity is 1,000
pending traces. If the process is killed before acknowledgement, unsent data can be
lost. Serverless handlers must await flush before returning. Applications requiring
crash durability should write their own durable outbox and reuse stable decision IDs
when replaying through the SDK.

Capture/delivery failures do not replace application exceptions or callback results.
A full queue or oversized trace is dropped with an error notification. Inspect
`pending_count`, `dropped_count`, `last_error` (TS: `pendingCount`, `droppedCount`,
`lastError`) and configure `on_error` / `onError` for monitoring. Without a handler,
failures emit a warning containing no payload. Explicit flush, close, and business-outcome calls
raise delivery errors. Scope-construction and `set_answer` validation errors raise
immediately; supply only valid JSON.

Options: Python `batch_size`, `flush_interval` (seconds), `max_pending`, `timeout`
(seconds), `max_retries`, `max_connections` (32 by default, 1–256); TypeScript `batchSize`, `flushInterval` (milliseconds),
`maxPending`, `timeoutMs`, `maxRetries`. Transient delivery retries default to two,
with a maximum of five. Never rely on background delivery without monitoring.

Reuse one client per service origin and collection. Python reuses a bounded pool of
HTTP connections for delivery and explicit endpoint calls; TypeScript uses the
runtime's fetch connection pool. Full batches are scheduled immediately. Capture
scopes perform JSON snapshots and enqueue locally; they never wait for HTTP.
Payload size still affects CPU time: prefer small explicit fields and asset references.

For standalone Studio, set `base_url` / `baseUrl` to the operator's service origin
and omit the API key. Hosted clients use an account API key and the account-scoped
API service. Use `https://api-staging.saina.run` for staging.

## Observe and prepare training examples

Open **Studio → Observability → Decisions** to inspect captured external decisions,
inputs, explicit answers, timing, errors, metadata and observed business outcomes. The trace
links to its collection. External traces are marked as external and never presented
as native model predictions with invented probabilities. Incomplete traces remain
visible without counting as completed decisions.

In **Data → Collections**, the original trace is immutable, and business outcomes and labels
are attached separately. A business outcome is the business value of a decision, or a proxy
for it (resolved, revenue, cost, CSAT). It is not a label. Approve or correct targets through the existing labeling
workflow, then create an immutable dataset version. Map `prompt` to a suitable
`raw.inputs` text field and `completion` to an approved `labels` field. Preserve
related workflow groups in the same split. Dataset versions retain source record
IDs and digests and feed the existing provider fine-tuning workflow. Capture alone
neither treats model output as ground truth nor authorizes paid training.
