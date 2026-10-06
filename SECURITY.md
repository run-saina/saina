# Public release boundary

This repository is a separately reviewed SDK/inference snapshot, not a mirror of
the private training repository or its git history.

Do not commit or publish credentials, environment files, account configuration,
customer inputs, datasets, teacher caches, optimizer states, model weights,
private host addresses, or workstation paths. Use CI/registry secret stores;
never put credentials in examples, Docker build arguments, or notebook outputs.

Before each release:

1. Review the complete diff and every new file, including notebook outputs.
2. Run `python scripts/audit_public.py` (heuristic, not a comprehensive secret scanner).
3. Build artifacts and inspect their complete file lists. The Python wheel must
   contain only the `saina` package and distribution metadata; npm must contain
   only its explicit `files` allowlist, package metadata and license notices.
4. Confirm examples contain invented text, not private training/evaluation records.
5. Publish only to the intended namespace and tag, with explicit release approval.

Model revisions are a separate release. A successful code/package audit does not
clear the weights or their training data for commercial distribution.

If a credential leaks, revoke/rotate it immediately. Removing the file or rewriting
history alone cannot make a disclosed credential secret again.

## Developer contact

Report security concerns privately to [dev@saina.run](mailto:dev@saina.run).
