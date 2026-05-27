# mesh-llm-ui

`mesh-llm-ui` owns the Mesh LLM React console and the Rust asset surface used
by the host API server and SDK console packages.

The TypeScript/Vite app builds into `dist/`. The Rust crate embeds that
directory when it exists and exposes helpers for serving the console index and
static assets. If `dist/` is absent, the crate embeds an empty fallback
directory so Rust-only checks can still run without building the UI. SDKs can
use the filesystem-backed asset provider to serve packaged console resources
without compiling those resources into the native FFI library.

## Responsibilities

- React console source, package metadata, and Vite configuration.
- UI unit tests and type checks.
- Built console asset ownership.
- Rust helpers that map asset paths to bytes, content type, and cache policy.
- Embedded and filesystem-backed console asset providers.

## Common commands

From the repository root:

```bash
just ui-dev
just ui-test
just clean-ui
scripts/build-ui.sh crates/mesh-llm-ui
```

The host crate depends on this crate for embedded assets; it should not depend
on the React source layout directly.
