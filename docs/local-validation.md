# Guided local setup validation

Implementation checkpoint: 2026-10-23. No deployment or package publication.

- Native: `saina local --backend native --checkpoint <cached Helm 2 routing bundle>
  --offline --no-browser --port 8770` loaded real weights on Linux CPU, completed
  its startup inference and served the shared console. A Chromium browser run
  returned HTTP 200, three answers and 104 input tokens. No browser errors or
  external browser requests. Checkpoint was the local retained
  `benchmark-results/routing-lora-full-2026-10-14/public-release` directory;
  no new weight download. This is integration evidence, not quality benchmarking.
- Python contracts: 140 passed, 39 subtests; includes native API auth/origin
  boundaries, MLX readout math with mock arrays and a GGUF helper subprocess
  exercising the real pipe protocol. One pre-existing Starlette/httpx warning.
- Website contracts: 194 passed. Browser fixture: single inference, mapped
  two-row JSONL run, duplicate IDs, partial export, no external requests and
  no persistent browser storage.
- CrewAI: 100 route, 80 guardrail and 60 score records from the retained evaluation
  snapshot mapped through the importer and passed the SDK request schema.
- Assets: one console build is copied byte for byte into the SDK with SHA256
  hashes, a build ID and API contract version. Runtime configuration is served
  separately. Console uses system font fallbacks without external stylesheets.

Reproduce:

```sh
# SDK, with server/media/test dependencies including safetensors
PYTHONPATH=src python -m pytest -q tests/test_local.py tests/test_artifacts.py \
  tests/test_server.py tests/test_decisions.py tests/test_billing.py \
  tests/test_client.py tests/test_media.py
# Website
node build-console.mjs
node build-local-ui.mjs
npm test
node test/local-playground.browser.mjs
```

Still unqualified: MLX startup on Apple Silicon, GGUF real-runner startup through
this launcher, fresh-install/platform coverage, pinned default release selection
and download/disk preflight, and broader hosted batch billing/cancellation browser
coverage. Existing MLX and GGUF inference pilot results remain valid prior evidence.


## Follow-up qualification

- Existing Helm 0.8B GGUF F16 bundle and `/home/dev/helm-gguf-export-v1/helm-hidden`
  passed startup through `saina local`, then a real browser request returned
  HTTP 200 with three answers and 104 input tokens; no browser errors or external
  requests. The entire supplied manifest was verified before starting the runner.
- Pinned default native release `e82055b348d02bb43552ccba88eda24835ba6baf` passed
  offline startup both from the development checkout and from an installed wheel
  in a fresh Python 3.12 environment. Installation resolved entirely from cached
  dependencies; loading used existing verified model cache.
- Package-owned native/MLX release locks now record immutable revisions, exact
  sizes and SHA256 hashes. `--plan`, explicit `--download`, disk checks, dependency
  checks and offline cache verification are implemented. Tests exercise interrupted
  downloads, partial snapshots, offline cache misses, corruption and low disk.
- Hosted dataset browser scenarios passed: authentication loss, credit exhaustion,
  quote expiry/resuming remaining rows, cancellation and identical-key/body retries.
- Per-question evaluation now separates labeled matches, unlabeled rows, errors,
  unsupported evaluations and not-run rows.
- Current totals: 143 Python tests plus 39 subtests; 195 frontend unit tests; local
  and five hosted browser scenarios. The wheel contains 27 UI assets identical
  to the hosted build. Test package version: 0.7.0+local.1.
- Mac MLX acceptance is handed to the user with docs/mac-test.md. The MLX extra
  pins mlx-lm 0.32.0, matching the existing package's recorded tested version.
  Windows execution remains unverified; no Windows support claim is made.


## Mac acceptance — user reported success
After receiving the Mac test wheel and instructions, the user reported: "it worked".
Record this as successful user acceptance of the Mac setup, not independently
observed logs or separate confirmation of every optional checklist step.
Implementation acceptance is complete. Publication/deployment was not requested
and has not occurred; Windows remains outside verified coverage.


## CLI review — simplified public commands
User selected `saina serve` and rejected setup flags and a compatibility alias.
Current commands: `saina serve`, `saina doctor`, `saina --version`. No `local`
alias or --plan/--download/--offline flags remain. Serve automatically uses the
verified default cache or downloads missing pinned files on first run, then opens
the shared UI. A complete default cache needs no network. Advanced backend,
checkpoint, port and browser options remain available. Doctor can verify the
default pinned cache without a separate manifest or loading model frameworks.

Review fixes: reject ignored or conflicting backend/revision/UI options and
invalid numeric arguments before setup; send runtime errors to stderr with
wheel-appropriate dependency instructions; verify UI hashes/API compatibility
before any model download or loading. Added command-level regression tests.
Test build is 0.7.0+local.2; Mac instructions and SHA256SUMS updated. Prior Mac
acceptance was for 0.7.0+local.1; no new Mac hardware execution is claimed.

## CLI and shared playground follow-up — local.3

API-only `saina serve [model-key]` now selects the sole cached model or presents a
terminal picker. `saina playground` opens before loading and offers authenticated
browser selection/download of compatible pinned models. No model argument on
playground and no legacy local/plan/download/offline interface. Shared UI now has
model cards, loading/error states, responsive layout and a numbered dataset workflow.

Verified: 174 Python tests + 39 subtests, 195 frontend unit tests, local batch/export
browser test and all five hosted batch scenarios. Real native Helm selection through
the browser completed startup and inference. Desktop/mobile screenshots checked;
zero browser errors or horizontal overflow. New manager tests cover unauthorized
access, invalid keys, concurrent selection, inference admission and failed loading.
Mac acceptance remains user-reported for local.1; local.3 Mac run not performed.
