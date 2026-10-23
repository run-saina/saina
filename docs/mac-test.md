# Mac acceptance test — Saina local 0.7.0+local.3

Use an Apple Silicon Mac running macOS 14 or newer with an ARM-native Python environment, `uv` installed,
and Tailscale connected to the development machine. This is an unpublished test
wheel, not the public PyPI release. It serves the same UI build as the hosted
console and uses the existing Helm 0.8B FP16 MLX model.

## Copy and install

```sh
mkdir -p ~/saina-local-test
cd ~/saina-local-test
scp 'dev@100.107.48.126:/home/dev/Projects/saina/artifacts/guided-local-setup/*' .
shasum -a 256 -c SHA256SUMS
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python './saina-0.7.0+local.3-py3-none-any.whl[server,mlx]'
```

## Inspect, download, run

```sh
.venv/bin/saina playground
```

The browser opens with model selection before loading weights. Choose **Use model**
for cached Helm, or **Download** to fetch the pinned MLX release (about 2.03 GB).
The UI shows download, verification, loading and startup-check phases. Keep the
terminal running while using the UI. Only original Helm is currently offered for MLX.

1. Run the default example. Confirm answer probabilities appear without a login
   or credit prompt. The page should identify the MLX backend.
2. Under **Try your own data**, import `sample.jsonl`, select `/state` as model input,
   `/id` as optional row ID and `/expected` as optional expected answer. Keep the
   default Inbound lead questions. Inspect the preview: expected answers must
   appear separately from the request. Confirm the mapping and run the dataset.
3. Confirm both rows finish and CSV/JSONL export works. Sample labels only exercise
   the evaluation UI; they are not a model-quality benchmark.
4. Stop the server with Ctrl-C. Disconnect Wi-Fi, then run:

```sh
.venv/bin/saina playground
```

Choose the downloaded model again. The UI should open and still answer a request and run the dataset
without internet access. A browser refresh drops the in-memory connection token;
reopen the Ready URL printed in the terminal to reconnect. Dataset state is also
in memory: export results before closing the tab.

Stop with Ctrl-C when finished. If port 8765 is occupied, append `--port 8766`.
Report the exact failing step and error if any. The connection URL contains a
session bearer token; exclude that URL from shared logs.

## Already checked on the development machine

Native CPU and the existing GGUF helper both loaded real weights and returned
three-question browser results offline. Tests cover authenticated loopback serving,
shared UI asset integrity, dataset mapping, hosted quote expiry, credit exhaustion,
cancellation, authentication loss and retry operation-key reuse. The Mac run is
needed to verify this launcher against real MLX/Metal execution.

The earlier local.1 Mac build was accepted by user report. This local.3 build adds
browser model management, revised commands and UI polish; it has not yet been
retested on a Mac. For API-only use: `.venv/bin/saina serve helm`.
