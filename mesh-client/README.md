# mesh-client

`mesh-client` is the low-level Rust client implementation crate for embedded
Mesh integrations.

This crate owns client-side protocol, transport, and runtime behavior used by
higher-level SDK surfaces. It is not intended to be the primary application
integration boundary.

Most consumers should depend on:

- `mesh-api/` for the public Rust client SDK

Language bindings should generally reach this crate through:

- `mesh-api/`
- `mesh-api-ffi/`

Keep this crate implementation-focused. Public, app-facing ergonomics should be
added in `mesh-api/`, not here.
