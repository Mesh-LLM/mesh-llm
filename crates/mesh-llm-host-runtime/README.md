# mesh-llm-host-runtime

`mesh-llm-host-runtime` composes the host-side mesh node runtime. It wires
model resolution, local serving, discovery, networking, runtime state, plugins,
the management API, and the shipped CLI entrypoint used by the `mesh-llm`
binary.

This crate is being split so reusable CLI, TUI, SDK, and embeddable runtime
surfaces can be published and consumed independently.
