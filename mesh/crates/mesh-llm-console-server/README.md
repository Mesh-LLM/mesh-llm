# mesh-llm-console-server

`mesh-llm-console-server` serves packaged MeshLLM console assets for SDK
bindings.

The CLI keeps using embedded console assets through `mesh-llm-host-runtime`.
SDK packages should keep console files as optional package resources, resolve
those resources in the language wrapper, and pass the resource directory to
this crate through UniFFI or N-API.

This crate intentionally serves static console files only. It does not own the
full CLI management API state.
