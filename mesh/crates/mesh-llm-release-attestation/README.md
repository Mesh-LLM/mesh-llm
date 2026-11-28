# mesh-llm-release-attestation

Command-line tool that signs a packaged `mesh-llm` binary and checks the
signature afterwards.

It owns the attestation claims (version, commit, target, artifact digest), the
Ed25519 key files, and the `generate-keypair`, `stamp` and `inspect` commands.
The byte layout of the footer appended to the binary belongs to
[`mesh-llm-release-footer`](../mesh-llm-release-footer/README.md), which the
host also uses to read its own attestation. This crate is never linked into the
host and is not published.

Its callers are the packaging scripts
[`package-release.sh`](../../scripts/package-release.sh) and
[`package-release.ps1`](../../scripts/package-release.ps1), and the release
workflow, which inspects the host inside each archive before publishing it.

## Usage

```sh
cargo run -p mesh-llm-release-attestation -- generate-keypair \
  --private-key-out private.json --public-key-out public.json

cargo run -p mesh-llm-release-attestation -- stamp \
  --binary mesh-llm --signing-key-file private.json \
  --require-source-commit --commit "$(git rev-parse HEAD)" \
  --target-triple aarch64-apple-darwin

cargo run -p mesh-llm-release-attestation -- inspect \
  --binary mesh-llm --public-key-file public.json --json
```

`inspect` reports `valid`, `missing` or `invalid`, plus the signed commit and
target. With `--require-source-commit`, `stamp` refuses to sign unless it has a
full 40-character commit and an explicit `--target-triple`; without that flag
it fills both in from the local machine, which is only suitable for local
experiments.
