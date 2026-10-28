# Saina SDK 1.0 contract

Status: agreed public API and behavior. Durable logging storage/lifecycle and the
additive backend protocol are resolved and implemented in these working changes;
see [the implementation decisions](durable-traces.md). No deployment or package
publication is implied.

Migration baseline: Python `saina==0.9.3` and the corresponding JavaScript SDK.
The version of a local checkout does not change that baseline.

This document supersedes the earlier SDK draft using `Estimates`, `Policy`,
`decide()`, mutable trace runs, and required collection/source fields.

**Trace protocols:** the existing `TraceClient` execution-scope protocol and
collection endpoint are preserved. New durable `Traces.log()` uses the additive
`POST /v1/traces` logging protocol, which records no invented execution facts.
The user's subsequent implementation request authorized resolving these remaining
decisions. [Durable logging](durable-traces.md) specifies the implemented wire
schema, SQLite accounting, isolation, recovery, lifecycle and Node runtime support.
The superseded `{collection, source, runs}` envelope is not used.

## 1. Scope

The SDK supports:

1. Asking Saina models questions and receiving typed answers.
2. Selecting options locally using probability thresholds.
3. Calling existing System One and Decisions compatibility endpoints.
4. Reading account balance, usage, and request history.
5. Recording external operations and delivering those records to Saina, when
   tracing's server and storage requirements are complete.

```python
from saina import Saina, Traces
```

- `Saina` owns inference and account operations.
- `Traces` records external operations.

They share configuration conventions but have independent lifecycles. There is
no implicit `client.traces` property or global client.

This contract specifies the HTTP clients. Existing local-model, tokenization,
media, and server extras are separate package capabilities and are not removed.

## 2. Quickstart

Set `SAINA_API_KEY`, then:

```python
from saina import Saina

saina = Saina()

ticket = {
    "id": "ticket-123",
    "text": "I was charged twice. Please refund the duplicate charge.",
}

answers = saina.ask(
    model="saina-helm-2-0.8b",
    state=ticket,
    questions={
        "team": {
            "type": "single_choice",
            "question": "Which team should handle this?",
            "options": ["billing", "support", "legal"],
        },
    },
)

team = answers["team"].select(threshold=0.9)

# Application-provided actions:
if team.accepted:
    assign_ticket(ticket, team=team.value)
else:
    queue_for_review(ticket)
```

The default service URL is `https://api.saina.run`.

`ask()` calls the model. `select()` performs a local calculation.

## 3. Vocabulary

| Term | Meaning |
|------|---------|
| State | Information submitted to the model. |
| Question | Something the model evaluates about that state. |
| Question ID | Caller-provided key identifying a question. |
| Answer | Model probabilities for a question. |
| Selection | Result of applying acceptance requirements to an answer. |
| Threshold | Minimum probability required for selection. |
| Minimum margin | Required lead over the second-highest probability, where applicable. |
| Request ID | Server-generated identifier for an inference request. |
| Idempotency key | Caller-supplied or SDK-generated identifier reused when retrying the same operation. |
| Trace | Record of an external operation. |
| Trace ID | SDK-generated identifier for that record. |
| Task | Caller-defined name for what the external operation was doing. |
| Inputs | Recorded information about what entered the operation. |
| Outcome | What the operation produced or did. |
| Metadata | Additional information describing the operation. |
| Spool | Persistent local storage for traces awaiting delivery. |

There is no public `Policy` class, `decide()` method, or `peak` property.

`Saina.decisions()` is an existing remote compatibility operation, distinct
from local `answer.select()`.

## 4. Creating an inference client

```python
Saina(
    base_url: str | None = None,
    *,
    api_key: str | None = None,
    timeout: float | None = None,
    max_retries: int = 2,
    max_retry_delay: float = 30,
    auto_idempotency_key: bool = True,
    allow_http: bool = False,
)
```

The positional `base_url` form remains supported.

### Configuration resolution

| Argument | Environment fallback | Default |
|----------|----------------------|---------|
| `api_key` | `SAINA_API_KEY` | Required |
| `base_url` | `SAINA_BASE_URL` | `https://api.saina.run` |
| `timeout` | `SAINA_TIMEOUT` | 60 seconds |

Explicit arguments take precedence. Environment variables are read at
construction, never at import time.

Invalid configuration raises `TypeError` or `ValueError` before network I/O.
Validation includes a non-empty API key, a valid URL, no embedded credentials,
query or fragment, a positive finite timeout, a non-negative integer retry
count, and a non-negative finite retry-delay budget.

### HTTP configuration

HTTPS is accepted by default. HTTP on localhost or loopback addresses is
accepted for local development. Other HTTP endpoints require explicit opt-in:

```python
saina = Saina(
    base_url="http://saina:8700",
    allow_http=True,
)
```

The SDK does not infer trust from DNS names or private address ranges.

### Thread safety

`Saina` is safe to share across threads. Per-request configuration, metadata,
and retries must not leak between concurrent calls. It starts no background
worker.

## 5. Asking questions

```python
answers = saina.ask(
    model="saina-helm-2-0.8b",
    state={
        "ticket": "Refund my order — I was charged twice!",
        "plan": "free",
    },
    questions={
        "refund": {
            "type": "yes_no",
            "question": "Is a refund being requested?",
        },
        "team": {
            "type": "single_choice",
            "question": "Which team should handle this?",
            "options": {
                "billing": "Charges, refunds, and invoices",
                "support": "Product questions and troubleshooting",
                "legal": "Contracts and compliance",
            },
        },
        "severity": {
            "type": "rating",
            "question": "How severe is this?",
            "levels": [
                "Minor inconvenience",
                "Work is impeded",
                "Money or data is at risk",
                "Urgent harm or outage",
            ],
        },
        "tags": {
            "type": "multi_choice",
            "question": "Which tags apply?",
            "options": ["refund", "duplicate_charge", "account_access"],
        },
    },
)
```

Signature:

```python
saina.ask(
    *,
    model: str,
    state,
    questions: Mapping,
    images=None,
    audios=None,
    specialist: str | None = None,
    idempotency_key: str | None = None,
    extra: Mapping | None = None,
) -> Answers
```

The native wire contract remains owned by the server request schema and its
question-types documentation. The SDK requests distribution mode. Thresholds
and margins are applied through `select()`, not `ask()`.

### Question types

| Type | Meaning |
|------|---------|
| `yes_no` | Probabilities for yes and no. |
| `single_choice` | Distribution over mutually exclusive options. |
| `rating` | Distribution over ordered levels. |
| `multi_choice` | Independent membership probabilities for options. |

`state` and each question's `question` accept the text and structured content
permitted by the server contract. Structured content must be JSON-compatible.

### Option definitions

Single-choice and multi-choice questions accept a list:

```python
"options": ["billing", "support", "legal"]
```

or a mapping:

```python
"options": {
    "billing": "Charges, refunds, and invoices",
    "support": "Product questions and troubleshooting",
    "legal": None,
}
```

List entries become labels without additional descriptions. Mapping values
provide optional descriptions; `None` means no additional description. Existing
structured description values permitted by the wire contract remain supported.

Lists normalize to the existing wire mapping representation. Order is
preserved; labels are not sorted or renamed.

Ratings use an ordered list of descriptions. Index zero is the first and
lowest level. Descriptions may repeat, so numeric indices remain their identity.

## 6. Media and specialists

Existing image, audio, and specialist arguments remain supported:

```python
answers = saina.ask(
    model="saina-helm-2-omni-0.8b",
    state="Classify this recording.",
    audios=["recording.wav"],
    specialist="mechanical",
    questions={
        "normal": {
            "type": "yes_no",
            "question": "Does the machine sound normal?",
        },
    },
)
```

Preserve the media input forms supported by 0.9.x, including data URLs and the
documented path, URL, byte, image, and audio forms. Optional conversions may
require existing media extras.

Unsupported media is not silently dropped; oversized inputs are not silently
truncated. Model capabilities and server limits remain authoritative. Responses
identify the answering model and, when present, specialist.

## 7. Request validation and extensions

Before sending, validate known structure: required fields, supported question
types, type-specific fields, question and option counts, non-blank and unique
labels, rating-level limits, JSON compatibility, and finite numbers.

Unknown ordinary fields are rejected to catch mistakes. The explicit `extra`
escape hatch permits new server features without an SDK release:

```python
answers = saina.ask(
    model="saina-helm-2-0.8b",
    state=state,
    questions=questions,
    extra={"new_server_feature": {"enabled": True}},
)
```

The SDK checks JSON compatibility but not extension semantics. The server
validates their meaning.

Top-level extension contents are flattened into the outgoing request. They
cannot override SDK-owned fields, including `model`, `state`, `questions`,
`mode`, `threshold`, `min_margin`, `images`, `audios`, and `specialist`.
SDK-only argument names such as `extra` and `idempotency_key` are reserved too.
Collision attempts raise `ValueError` before I/O.

### Question extensions

The literal key `extra` is reserved by the SDK inside question dictionaries.
Its contents are flattened into the wire question, not sent as a nested object.

Input:

```python
questions={
    "team": {
        "type": "single_choice",
        "question": "Which team?",
        "options": ["billing", "support"],
        "extra": {"new_server_feature": True},
    },
}
```

Outgoing question:

```json
{
  "type": "single_choice",
  "question": "Which team?",
  "options": {"billing": null, "support": null},
  "new_server_feature": true
}
```

Question extensions cannot override known question fields, including optional
fields omitted by the caller. Collisions raise before I/O.

Unknown response fields are retained in `.raw`. The known typed answer
contract is still validated.

## 8. Reading answers

`Answers` is an immutable mapping from question ID to typed answer:

```python
answers["team"]
len(answers)
"team" in answers

for question_id, answer in answers.items():
    ...
```

Request information is available as attributes:

```python
answers.request_id
answers.model
answers.specialist
answers.usage.input_tokens
answers.usage.output_tokens
answers.metadata
```

`specialist` is `None` when absent.

| Type | Fields |
|------|--------|
| `YesNoAnswer` | `yes`, `no`, `confidence` |
| `SingleChoiceAnswer` | `probabilities`, `confidence` |
| `RatingAnswer` | `probabilities`, `levels`, `expected_level`, `confidence` |
| `MultiChoiceAnswer` | `memberships` |

Single-choice probabilities use option labels as keys. Typed rating
probabilities use integer keys:

```python
answers["team"].probabilities["billing"]
answers["severity"].probabilities[0]
answers["severity"].levels[0]
answers["tags"].memberships["refund"]
```

`expected_level` is the probability-weighted index `Σ index × probability`.
It is an expectation, not a selected level.

Multi-choice memberships are independent and need not sum to one. Multi-choice
has no aggregate `confidence`; callers use individual membership probabilities.

Typed answers, request metadata, and nested collections are immutable.

## 9. Response and billing metadata

New code obtains metadata from the result:

```python
answers = saina.ask(...)
metadata = answers.metadata
```

`ask_with_metadata()` is retained as a compatibility method:

```python
answers, metadata = saina.ask_with_metadata(...)
```

It returns the same typed `Answers` result and equivalent metadata.

`ResponseMetadata` includes:

```python
metadata.request_id
metadata.credits_charged
metadata.balance
metadata.price_version
metadata.replayed
metadata.idempotency_key
metadata.attempts
```

Billing fields may be `None` when not supplied by an endpoint. Python credit
amounts are integers, never floating-point values.

### Request IDs

Accept a request ID from the response body or `X-Request-Id` header. If both
exist, they must agree. Typed `Answers` requires a non-empty request ID.

Older self-hosted servers supplying neither must be upgraded for the typed
1.0 inference path.

`client.last_metadata` remains as a thread-local compatibility property. For
concurrent code, prefer result-bound metadata.

## 10. Confidence

Confidence summarizes a distribution. It is not the probability checked by
`select()`.

- Yes/no: `2 × highest_probability − 1`.
- Single choice, with `n` options:
  `(highest_probability − 1/n) / (1 − 1/n)`.
- Rating: concentration around the highest-probability level, penalizing
  probability mass farther from it. The exact distance-weighted formula follows
  the existing question-types wire contract.
- Multi choice: no aggregate confidence field.

For example, `yes=0.9`, `no=0.1` gives confidence `0.8`. A threshold of `0.9`
compares against probability `0.9`.

Confidence does not establish calibration or accuracy on an application task.

## 11. Selecting options

Selection is optional and local:

```python
selections = answers.select(threshold=0.9, min_margin=0.1)
team = answers["team"].select(threshold=0.9, min_margin=0.1)
```

Both expose the parameters:

```python
select(*, threshold: float, min_margin: float = 0.0)
```

`Answers.select()` returns `Selections`, an immutable mapping from question ID
to typed selection. Individual answer selection returns one typed selection.

Both numbers must be finite and in `[0, 1]`. Invalid types raise `TypeError`;
invalid values raise `ValueError`. The threshold is required.

Selection performs no network I/O and does not modify answers.

Margin applies only to yes/no and single choice. For rating and multi choice,
it is ignored consistently in individual and bulk calls. Their returned
selections report `min_margin == 0.0`.

## 12. Reading selections

Every selection exposes:

```python
selection.value
selection.accepted
selection.reason
selection.answer
selection.threshold
selection.min_margin
```

| Type | `.value` |
|------|----------|
| `YesNoSelection` | `bool | None` |
| `SingleChoiceSelection` | `str | None` |
| `RatingSelection` | `int | None` |
| `MultiChoiceSelection` | `tuple[str, ...]` |

Rating selections also expose `.label`: the selected level description, or
`None` on rejection.

Use `.accepted` to test acceptance. `False` and rating index `0` can both be
accepted values; their truthiness must not be used to detect rejection.

```python
SelectionReason = Literal[
    "accepted",
    "below_threshold",
    "tie",
    "below_margin",
    "no_options_selected",
]
```

`accepted` is true exactly when `reason == "accepted"`. Not meeting selection
requirements is an ordinary result, not an exception.

## 13. Selection rules

### Yes/no and single choice

Apply checks in this order:

1. Highest probability below `threshold`: `below_threshold`.
2. Top two probabilities differ by at most `1e-12`: `tie`.
3. Top-two difference below `min_margin`: `below_margin`.
4. Otherwise: select the highest-probability option with `accepted`.

On rejection, `value` is `None` and `accepted` is false. Equality with the
threshold or margin passes, subject to the tie rule.

For `0.5 / 0.5`, threshold `0.9` reports `below_threshold`; threshold `0.4`
reports `tie`. Threshold comes first because it identifies whether any option
reached the caller's minimum requirement.

### Rating

Check threshold first, then tie. Margin does not apply. On acceptance, `value`
is the integer index and `label` the corresponding description. On rejection,
both are `None`.

### Multi choice

Select every label whose membership is at least the threshold, preserving
option order.

- At least one qualifies: `accepted=True`, reason `accepted`.
- None qualify: `value=()`, `accepted=False`, reason `no_options_selected`.

An empty tuple is a valid filtering result. It establishes neither uncertainty
nor a confident prediction that no options apply. Applications needing the
latter must define an appropriate rule using the membership probabilities.

### Server decision mode

These are local SDK 1.0 rules. Existing server decision mode remains unchanged,
including tie-first precedence and its multi-choice reason names. Do not claim
exact reason equivalence between the two APIs.

The intended direction is convergence on the local vocabulary and precedence
through a separately versioned server change. This SDK release does not
silently alter existing server responses or commit to a server release date.

## 14. Serialization and credit representations

Answers, individual answers, selections, and individual selections expose
`to_dict()` and `to_json()`:

```python
answers.to_dict()
answers.to_json()
answers["team"].to_dict()
answers["team"].to_json()
selections.to_dict()
selections.to_json()
selections["team"].to_dict()
selections["team"].to_json()
```

`to_dict()` returns an independent JSON-compatible representation. Typed
objects become dictionaries, tuples become lists, and rating probability keys
become strings. Mutations to the copy do not change typed results.

The `Answers` envelope contains `request_id`, `model`, `specialist`, `usage`,
`metadata`, and `answers` keyed by question ID. Individual answers include
their type and public data fields.

Selections serialize type, value, accepted, reason, original answer, threshold,
effective margin, and rating label where applicable. `Selections.to_dict()`
is keyed by question ID.

`to_json()` returns a JSON string for that representation. There are no `.json`
properties or `as_dict()` / `as_json()` aliases.

### Python credits

Parsed, typed, and serialized Python credit amounts are integers everywhere:

```python
saina.balance()["available"]                 # int
answers.metadata.balance                    # int | None
answers.to_dict()["metadata"]["balance"]     # int | None
```

`to_json()` emits JSON integers. Do not convert credit amounts to floats.

### Mapping conversion and raw responses

`dict(answers)` follows normal mapping behavior and contains typed answer
objects; it is not recursive serialization.

`answers.raw` returns an independent copy of the original parsed response body,
including unknown fields, stringified rating indices, and wire decimal-string
credits. Headers are represented by `answers.metadata`, not `.raw`.

### JavaScript credits

JavaScript parses wire credit amounts as `BigInt` and serializes them as decimal
strings because JSON has no BigInt type. Python and JavaScript representations
preserve the same information but intentionally differ for serialized credits.
Ordinary JavaScript number parsing must not round-trip arbitrary Python credit
integers.

## 15. Response validation

Before constructing typed answers, validate:

- Question IDs and answer types.
- Option label sets.
- Finite probabilities in `[0, 1]`.
- Exclusive sums within `1e-5` of one, including yes/no.
- Independent multi-choice membership validity.
- Derived fields within `1e-9` of wire formulas.
- Rating-level echoes.
- Request ID presence and consistency.

Violations raise `ProtocolError`. Do not repair invalid responses or return
partially validated results.

## 16. Idempotency and retries

The 1.0 defaults are:

```python
Saina(
    max_retries=2,
    max_retry_delay=30,
    auto_idempotency_key=True,
)
```

Automatic idempotency is enabled for `ask()`, `system_one()`, `decisions()`,
and their metadata variants.

Each logical operation gets one generated UUIDv7 key unless the caller supplies
a key. A caller-provided key always takes precedence. All retries reuse the
same key and identical serialized request bytes. Never generate a new key for
a retry.

```python
from saina import new_idempotency_key

key = new_idempotency_key()
answers = saina.ask(..., idempotency_key=key)
```

The quickstart therefore has automatic retries for eligible failures.
`max_retries=2` permits at most two additional attempts; it does not make every
failure retryable.

### Retry policy

Inference retries require an idempotency key. Preserve the server-aware policy:

- Connection loss or uncertain response, using the same key.
- `accounting_unavailable`.
- `request_in_progress`.
- `rate_limited`.
- `overloaded` or `inference_unavailable` only when not admitted.

Honor `Retry-After` and the `max_retry_delay` budget. Do not replace this policy
with blanket status-based retries.

Read-only account calls may retry under the established policy without a key.

Validation, protocol failures, conflicts, expired results, and unverifiable
keys are not automatically retried as new operations.

```python
saina = Saina(auto_idempotency_key=False)
```

With generated keys disabled, inference has no automatic retries unless a
caller key is supplied. An application retry without a key can repeat inference
and incur another charge.

Idempotency guarantees require server support. Supported self-hosted server
versions must be documented rather than inferred from accepting a header.

## 17. Compatibility inference endpoints

Existing remote compatibility methods remain public.

### System One

```python
result = saina.system_one(
    model="saina-helm-2-0.8b",
    state=state,
    questions=system_one_questions,
)
```

Calls `POST /v1/systemone`, preserving existing question schemas, media,
specialist and idempotency arguments, and wire-shaped dictionary results.

### Decisions

```python
result = saina.decisions(
    model="saina-helm-2-0.8b",
    input=input,
    questions=decision_questions,
)
```

Calls `POST /v1/decisions`, preserving its existing request shape, extension
keywords, and wire-shaped dictionary results. This is a network operation,
unlike `select()`.

Retain `system_one_with_metadata()` and `decisions_with_metadata()`, returning
`(result, ResponseMetadata)`. New native examples use `ask()` and optional
local `select()`.

## 18. Account operations

Existing account methods remain available with established dictionary shapes.

```python
balance = saina.balance()
available = balance["available"]

usage = saina.usage(
    from_="2026-10-01",
    to="2026-10-27",
    channel="api",
)

page = saina.requests(limit=50, cursor=cursor, channel="api")
```

- `balance()`: `GET /v1/account/balance`, including balances, reservations,
  debt, and suspension information supplied by the server.
- `usage()`: `GET /v1/account/usage`; retain `from_`, `to`, and `channel`.
- `requests()`: `GET /v1/account/requests`; retain `from_`, `to`, `key_id`,
  `channel`, `status`, `cursor`, and `limit`.

`from_` maps to the `from` query parameter. Requests returns one server page;
the SDK does not automatically fetch all pages.

Known credit fields are Python integers. Future typed account models must not
silently break dictionary access.

Self-hosted servers may omit account endpoints. Surface their errors normally;
do not fabricate balances or usage.

## 19. Error hierarchy

Broad categories are base classes. Server code-specific errors subclass the
appropriate category:

```text
SainaError
├── HTTPError
│   ├── AuthenticationError
│   │   ├── InvalidApiKey
│   │   ├── KeyRevoked
│   │   └── FreshAuthRequired
│   ├── PermissionDeniedError
│   │   ├── PermissionDenied
│   │   └── AccountSuspended
│   ├── NotFoundError
│   │   └── NotFound
│   ├── InvalidRequestError
│   │   └── InvalidRequest
│   ├── RateLimitError
│   │   └── RateLimited
│   ├── InsufficientCredits
│   ├── IdempotencyError
│   │   ├── IdempotencyConflict
│   │   ├── RequestInProgress
│   │   ├── IdempotencyResultExpired
│   │   └── IdempotencyUnverifiable
│   └── ServerError
│       ├── InferenceFailed
│       ├── AccountingUnavailable
│       └── ModelBusyError
│           ├── Overloaded
│           └── InferenceUnavailable
├── TransportError
├── ProtocolError
├── SpoolFullError
└── TraceStorageError
```

Compatibility names are exact aliases:

```python
SainaHelmError = SainaError
SainaConnectionError = TransportError
```

Recognized server codes determine code-specific classes. Otherwise use status:

| Status | Fallback |
|--------|----------|
| 401 | `AuthenticationError` |
| 403 | `PermissionDeniedError` |
| 404 | `NotFoundError` |
| 422 | `InvalidRequestError` |
| 429 | `RateLimitError` |
| 503 | `ModelBusyError` |
| Other 5xx | `ServerError` |
| Other unsuccessful status | `HTTPError` |

Preserve `status`, `code`, `request_id`, `admitted`, `state`, `retry`,
`retry_after`, and `suspension` when supplied; absent values are `None`.
`TransportError` also exposes `timed_out`.

Preserve server validation detail and do not infer which resource was missing
from status alone. Callers may catch specific codes or their broad categories.
Local argument failures use `TypeError` or `ValueError`.

## 20. Recording external operations

`Traces` records external operations. Saina inference uses existing server-side
accounting and request recording and must not be submitted again through
`Traces`.

This does not imply storing inference request or response bodies. Explicit
trace ingestion is a separate data submission.

```python
traces = Traces()  # on_storage_error="warn" by default
```

Authentication, base URL, and HTTP opt-in follow `Saina` conventions. There is
no required collection or source. The instance owns its spool and delivery
worker; create it once and reuse it. There is no static `Traces.log()` or
module-level logging singleton.

```python
trace_id = traces.log(
    task="route-ticket",
    inputs={"prompt": prompt},
    outcome={"text": response.output_text},
    metadata={
        "application": "ticket-router",
        "environment": "production",
        "provider": "openai",
        "model": response.model,
        "provider_response_id": response.id,
        "status": "ok",
    },
)
```

Signature:

```python
traces.log(
    *,
    task: str,
    inputs,
    outcome=UNSET,
    metadata: Mapping[str, JSONValue] | None = None,
) -> str | None
```

`UNSET` denotes omission; callers need not import it. Successful logging
validates, copies, assigns a stable UUID, and commits locally before returning
the ID. Returned IDs never represent unpersisted records. Return does not
imply remote delivery.

`task` is a non-empty operation name. Inputs and outcome are caller-defined;
omitted outcome differs from explicit `None`. There is no mutable run or
separate outcome, metadata, or link mutation. Provider IDs belong in metadata.

The SDK records logging time. It does not infer operation status, start time,
duration, or provider identifiers, or intercept external provider calls.

## 21. Metadata conventions

These optional names are reserved for consistent interpretation and grouping:

| Key | Meaning |
|-----|---------|
| `application` | Reporting application or service. |
| `environment` | Deployment environment, such as production or staging. |
| `provider` | External provider that performed the operation. |
| `model` | External model identifier. |
| `provider_request_id` | External request identifier, when available. |
| `provider_response_id` | External response identifier, when available. |
| `status` | `ok` or `error`, describing the operation. |
| `duration_ms` | Caller-measured duration in milliseconds. |
| `error_type` | Error class or application error category. |

Use `environment`, not `deployment`, for the standard environment field.
Additional keys are permitted. These are caller-supplied facts, not automatic
instrumentation.

## 22. Recording failures

External exceptions require explicit recording. Recommended pattern:

```python
from time import perf_counter

prompt = "Summarize this document."
started = perf_counter()

try:
    response = external_client.generate(prompt)
except Exception as exc:
    traces.log(
        task="summarize-document",
        inputs={"prompt": prompt},
        metadata={
            "application": "document-service",
            "environment": "production",
            "provider": "external-provider",
            "status": "error",
            "error_type": type(exc).__name__,
            "duration_ms": (perf_counter() - started) * 1000,
        },
    )
    raise
else:
    traces.log(
        task="summarize-document",
        inputs={"prompt": prompt},
        outcome={"text": response.text},
        metadata={
            "application": "document-service",
            "environment": "production",
            "provider": "external-provider",
            "status": "ok",
            "duration_ms": (perf_counter() - started) * 1000,
        },
    )
```

`external_client` is application-provided pseudocode. Use default warn mode so
storage failures do not replace external exceptions. Invalid trace arguments
still raise. No timing helper or operation context manager is implied.

## 23. Trace validation and storage-error handling

Inputs, outcomes and metadata must be JSON-compatible and copied during `log()`.
Reject non-finite numbers, cycles, non-string keys, unsupported objects, invalid
Unicode, nesting beyond 60 and records over 1 MiB before I/O. Tasks are nonblank
names of at most 1024 Unicode characters. Oversize arguments raise `ValueError`;
unsupported types raise `TypeError`.

`on_storage_error="warn"` is the default: capacity/persistence failures return
`None` and warn once per failure-category episode. Strict `"raise"` mode raises
`SpoolFullError` or `TraceStorageError`. Initialization is lazy, so first-admission
storage failures follow this policy. It never suppresses invalid arguments or
use-after-close errors. Failed admissions were never queued.

Warning state is serialized per recorder. Same-category failures increment the
lost-admission count; a changed category starts a new episode. A successful commit
logs recovery and the episode's lost-record count at informational level.

## 24. Trace thread and process safety

Python uses thread-owned, short-lived SQLite connections; JavaScript owns its
connection on its Node event loop. Transactions coordinate capacity/admission and
acknowledgment. Lock waits are bounded at 250 ms. Concurrent processes can share a
spool with matching service/credential/capacity binding. Workers may redeliver the
same immutable record concurrently; tenant/trace-ID digest deduplication makes
this safe. There is no crash lease delaying recovery or exclusive process owner.

## 25. Persistence, delivery and recovery

Use a persistent SQLite spool with synchronous FULL commits, DELETE journaling,
4096-byte pages and FULL auto-vacuum. Default logical capacity is 64 MiB, counting
UTF-8 payload bytes plus 256 bytes per record, including quarantine. SQL triggers
maintain capacity counters transactionally. Main-file pages are capped at capacity
plus 2 MiB; rollback journals need additional disk space. This is a logical
admission budget, not a filesystem quota. See [physical-space details](durable-traces.md).

`traces.pending` counts all persisted records awaiting acknowledgment, including
rejected records. Nothing is silently evicted. Accepted records are removed only
after exact receipt validation; permanent per-record rejections are quarantined
and do not block healthy later records. Ambiguous receipts retain the entire batch.
Delivery is at least once; retry backoff is capped at 30 seconds, honors Retry-After
up to one day, and never re-executes external operations.

Recovery is explicit: `export_rejected(path)` writes a new private, fsynced JSONL
file without deleting anything; `retry_rejected(ids)` retries unchanged quarantined
records after the cause is fixed; `discard_rejected(ids)` explicitly removes only
quarantined records after operator export/review. ID sets must be unique and
entirely quarantined or the transaction fails. Corrected content uses a new ID.
These are local outbox maintenance operations, not server search/edit APIs.

## 26. Flush and close

`flush(timeout=5)` returns immutable `TraceFlushResult(complete, pending, rejected)`.
It targets the sequence watermark at entry; later admissions do not expand that
snapshot. `complete` means all targeted records were acknowledged. Counts describe
all unacknowledged records at the final transactional read, including newer or
quarantined records. Previously returned results remain valid snapshots.

Concurrent flushes share delivery but retain independent snapshots/deadlines.
Network failures and deadlines return incomplete results; local storage failures
raise in both warn and strict modes. Close stops new admission, attempts bounded
delivery, interrupts network work, stops worker scheduling, and releases resources.
Unacknowledged records stay on disk. Concurrent close calls share shutdown;
repeated valid closes are harmless. Concurrent flushes finish with snapshots;
if shutdown prevents a JavaScript flush from obtaining its initial snapshot, it
raises a closed-recorder error instead of claiming completion.

After close, `log()`, `flush()`, `pending` and recovery operations raise
`RuntimeError` (JavaScript Error), regardless of storage mode. Local spool errors
during close still raise after cleanup, and repeated close remains harmless.
Python context management preserves an already-active application exception.

Delivery/flush/close timeouts do not bound an unresponsive filesystem or OS DNS
resolver; SQLite lock waits can additionally take up to 250 ms. Connected HTTP
bodies and fetches are interrupted on close. Python atexit and Node beforeExit
attempt five-second best-effort shutdown; forced termination has no hook guarantee.

## 27. Filesystems and serverless environments

The override is the complete `spool_path` filename (JavaScript `spoolPath`). Defaults:

- Linux: `${XDG_STATE_HOME:-$HOME/.local/state}/saina/traces/<service>/<credential>/spool.sqlite3`.
- macOS: `$HOME/Library/Application Support/Saina/traces/<service>/<credential>/spool.sqlite3`.
- Windows: `%LOCALAPPDATA%/Saina/traces/<service>/<credential>/spool.sqlite3`.

Service and credential components are full SHA-256 digests; raw API keys never
enter paths, stored binding metadata or diagnostics. Explicit overrides still
check the service/credential binding and capacity. Python and Node share the
SQLite format. New POSIX namespace directories/files use 0700/0600; Windows uses
the user's filesystem ACLs. See [normalization/fallback details](durable-traces.md).

Storage initializes lazily on admission. Empty flush/pending/close do not create
a missing spool. After restart call flush or log to resume existing records.
Credential rotation deliberately chooses a different default namespace; drain
before revoking the old key or perform operator-directed stable-ID recovery.
Constructing `Saina` creates no spool or worker. `Traces` requires writable durable
storage; it cannot survive destruction of ephemeral filesystems. Serverless code
must flush before returning and inspect the result.

## 28. JavaScript

Use equivalent concepts, selection rules, and reasons:

```javascript
const saina = new Saina();
const ticket = { id: "ticket-123", text: "I was charged twice." };

const result = await saina.ask({
  model: "saina-helm-2-0.8b",
  state: ticket,
  questions: {
    team: {
      type: "single_choice",
      question: "Which team should handle this?",
      options: ["billing", "support", "legal"],
    },
  },
});

const selections = result.select({ threshold: 0.9 });
if (selections.team.accepted) {
  assignTicket(ticket, selections.team.value);
}
```

`ask()` is asynchronous; `select()` is synchronous. Application actions and
their arguments in examples are supplied by the caller.

Python accesses `answers["team"]`; JavaScript accesses `result.answers.team`.
JavaScript nesting avoids collisions between IDs and result methods/metadata.

For bulk selections, `selections.get(questionId)` accesses any question ID and
returns `undefined` for an unknown ID. Direct properties such as `selections.team`
remain available for IDs that do not collide with methods or inherited names.
IDs such as `toObject`, `toJSON`, or `get` are not rejected or overwritten: use
`get()` to access their selections. Serialization includes every question ID.

Preserve `askWithMetadata`, `systemOne`, `systemOneWithMetadata`, `decisions`,
`decisionsWithMetadata`, `balance`, `usage`, and `requests`. Use existing
camelCase configuration names. `autoIdempotencyKey` defaults to true; preserve
existing JavaScript units for other configuration values.

```javascript
result.toObject();
JSON.stringify(result);
```

`toObject()` returns an independent JSON-compatible representation. `toJSON()`
supplies it to `JSON.stringify()` and returns an object, not a string.
Individual answers and selections follow the same convention. Credit BigInts
serialize as decimal strings.

Durable Node.js trace API:

```javascript
const traces = new Traces({ onStorageError: "warn" });
const traceId = await traces.log({
  task: "route-ticket",
  inputs: { prompt },
  outcome: { text },
  metadata: {
    application: "ticket-router",
    environment: "production",
    provider: "external-provider",
    status: "ok",
  },
});

const delivery = await traces.flush({ timeout: 5 });
await traces.close({ timeout: 5 });
```

A returned ID means persistence completed; warn-mode failure resolves to `null`.
Omitted outcome differs from `null`; explicit `undefined` is not JSON data.
Durable tracing requires Node.js 22.13+ with node:sqlite; browser inference remains
supported but browser durable recording is not introduced. `await traces.pending`
reads the lazy asynchronous pending getter. Flush/close timeouts use seconds.
Recovery methods are `exportRejected`, `retryRejected`, and `discardRejected`.

## 29. Migration from 0.9.x

| 0.9.x surface | 1.0 behavior |
|---------------|--------------|
| `Saina` | Retained. |
| `SainaHelm` | Deprecated compatibility alias retained through 1.x. |
| `SainaError` | Retained. |
| `SainaHelmError` | Exact alias of `SainaError`, retained through 1.x; deprecated. |
| `SainaConnectionError` | Exact alias of `TransportError`. |
| Default hosted URL | Preserved: `https://api.saina.run`. |
| Positional `base_url` | Preserved. |
| `max_retries=2` | Name and default preserved. |
| `max_retry_delay=30` | Name and retry-delay budget preserved. |
| `auto_idempotency_key=False` | Default changes to `True`; pass `False` explicitly for the old behavior. |
| `new_idempotency_key()` | Preserved. |
| Dictionary-returning native `ask()` | Returns typed `Answers`; `.raw` is the original body and `.to_dict()` is typed serialization. |
| `ask(mode="distribution")` | Omit the redundant argument. |
| `ask(mode="decision", threshold=..., min_margin=...)` | Use `ask()`, then local `.select(...)`. |
| `ask_with_metadata()` | Retained as a compatibility method; result becomes typed `Answers`. |
| `last_metadata` | Thread-local compatibility access; prefer result-bound metadata. |
| `system_one()` | Retained, including wire-shaped results. |
| `system_one_with_metadata()` | Retained. |
| `decisions()` | Retained, including extension keywords and wire-shaped results. |
| `decisions_with_metadata()` | Retained. |
| `balance()` | Retained with dictionary access and integer credits. |
| `usage()` | Retained with existing filters and response shape. |
| `requests()` | Retained with filters and pagination shape. |
| Media inputs | Preserved. |
| Specialist selection | Preserved. |
| Native `ask()` extension keywords | Move to explicit `extra={...}`. |
| Error codes and server metadata | Preserved, with explicit broad base classes. |

Release notes must identify typed answers, local selection, threshold-first
reasons, `no_options_selected`, integer rating access, `.value`/`.accepted`,
and automatic idempotency's new default.

### Tracing migration notes

Tracing remains opt-in; inference-only users acquire no worker or spool. Existing
staged `TraceClient` functionality is preserved alongside the new API. See the
[durable logging decisions](durable-traces.md) for exact paths, overrides, capacity,
warn/recovery behavior, shutdown and ephemeral-filesystem limitations.

## 30. Additive backend logging protocol

The existing collection-based execution protocol remains unchanged. New logging
uses `POST /v1/traces` with `schema_version="1.0"` and a `traces` array. Each record
contains canonical UUID `trace_id`, timezone-aware logging `recorded_at`, `task`,
`inputs`, optional `outcome`, and object `metadata`. It has no required collection,
source, execution status or duration. SDKs generate UUIDv7 IDs before local commit.

Limits are 100 records, 1 MiB per compact UTF-8 record, 8 MiB summed record content,
and 8 MiB + 64 KiB per full envelope. Malformed envelopes or duplicate/uncorrelatable
IDs receive no acknowledgment. Validly identified record errors are rejected
separately. HTTP 200 receipts partition every submitted ID exactly once into
`accepted` and `rejected`; rejected entries contain `trace_id`, `code`
(`invalid_trace` or `trace_conflict`) and `retryable=false`.

The backend commits valid records and canonical digest receipts in one durable
account-scoped transaction before returning acceptance. Exact replays deduplicate,
including regrouped batches; changed content under an existing account/ID is
rejected. Digest tombstones have no time expiry, survive content-retention cleanup,
and last until account deletion, subject to the existing durable recovery boundary.
There is no batch Idempotency-Key requirement. Ambiguous storage/network failures
are replayed using the original record IDs and payloads.

The hosted endpoint uses existing account-key authentication, tenant binding,
security suspension, rate limiting and fenced/journaled PostgreSQL repository.
Standalone Studio exposes the same route under its existing single-workspace auth
model. Native background capture remains best effort and is separate from these
explicitly acknowledged durable logs. [Protocol examples and implementation details](durable-traces.md)
resolve the former release blockers; [the earlier audit](trace-backend-audit.md)
records why adapting legacy execution records would have invented operation facts.

## 31. Public surface and dependencies

Python package: `saina`.

Public names include `Saina`, `Answers`, the four answer types, `Selections`,
the four selection types, `SelectionReason`, `ResponseMetadata`,
`new_idempotency_key`, errors and compatibility aliases, and `Traces`, `TraceFlushResult`, and its local quarantine-recovery methods.

The base HTTP client and tracing implementation use the standard library.
Existing model, media, tokenization, and server dependencies remain optional.

## 32. Not introduced by this contract

- `Policy`, `decide()`, `peak`, or `top`.
- Mutable run objects.
- Separate `record_outcome()`, `set_metadata()`, or `link()` operations.
- Required collection or source fields.
- Streaming inference or a Python async HTTP client.
- Automatic provider instrumentation or execution timing.
- Client-side inference caching.
- Trace search, editing, or retrieval.
- Browser tracing without a specified persistence model.

Existing account and compatibility methods are not removed.

## 33. Storage and lifecycle decisions resolved

The following are implemented by these changes and specified in
[durable-traces.md](durable-traces.md):

1. Exact per-platform paths and complete-filename override.
2. Full service/credential hashes and persisted binding checks.
3. Transactional multi-process sharing with safe duplicate delivery.
4. Logical capacity accounting, physical SQLite cap and 1 MiB record limit.
5. Matching additive upload schemas, limits, partial acceptance and exact receipts.
6. Permanent rejection quarantine, export, unchanged retry and explicit discard.
7. Account-lifetime digest tombstones, independent of content retention.
8. Snapshot-based concurrent flush and bounded, idempotent close.
9. Explicit local storage errors during flush/shutdown, after cleanup.
10. Immutable TraceFlushResult and five-second default shutdown timeout.
11. Node.js 22.13+ SQLite durability; no browser persistence fallback.

Legacy trace capture is preserved. No production/staging activation or registry
publication is implied by implementing these source changes; release manifests
must identify the matching gateway, platform and SDK revisions before rollout.
