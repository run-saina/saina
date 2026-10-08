# Releasing saina 0.2.0

Prepared on branch `mems/hosted-billing` (2026-10-08). Publishing is an owner decision because it changes
self-hosted behavior (see CHANGELOG 0.2.0). Checks already run on this branch:

- `uv build`: `saina-0.2.0-py3-none-any.whl` and `saina-0.2.0.tar.gz`; `twine check` passes for both.
- A clean install of `saina[tokenize,server]` from the wheel imports without torch and counts tokens with the
  pinned tokenizer.
- `npm pack --dry-run` in `clients/javascript`: `@run-saina/sdk@0.2.0`, 6 files.
- `python scripts/audit_public.py`: passes (manual review still required).
- Unit tests: `python -m unittest discover -s tests` and `cd clients/javascript && npm test`.
- Real-model parity: identical answers and usage to 0.1.2 on revision e82055b (GPU), and golden answers
  recorded for `saina-verify` in the gateway package.

Steps:

1. Merge `mems/hosted-billing` into `main` and tag `v0.2.0`.
2. `uv build && uvx twine upload dist/saina-0.2.0*` (PyPI token from the password manager).
3. `cd clients/javascript && npm publish --access public`.
4. In run-saina/deploy, switch the inference images from the wheel to `saina==0.2.0` and rebuild them.
5. In the training repository, change the `serve` extra from the git URL to `saina[server]>=0.2.0`.
6. Self-hosters: the CHANGELOG lists the behavior changes (queue with `429 overloaded`, no default CORS origin,
   typed error bodies).
