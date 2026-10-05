# mesh-llm-node

`mesh-llm-node` contains embeddable node primitives shared by the Rust SDK,
native SDK bindings, and the host runtime as MeshNode support is extracted out
of the CLI and management API.

This crate owns SDK-safe building blocks for:

- model catalog search and recommendations
- installed model discovery
- model detail and capability inspection
- model download, delete, cleanup, and derived-cache pruning
- in-process serving control traits and serving status types

It should not own CLI parsing, terminal UI behavior, local REST route handling,
or process-global host runtime state. Those layers should call into this crate
or implement its traits.

## Serving Boundary

Serving control is modeled as an in-process `ServingController` trait. SDK
surfaces such as `MeshNode::serving().load()` should call this boundary
directly when embedded serving is enabled.

The host runtime's `MeshApi` is the reference implementation of this trait. It
adapts SDK serving calls onto the existing runtime-control loop, so embedded
load/unload uses the same path as local operator control without making REST
requests back into the process.

The serving contract uses explicit model refs for load, explicit
model-or-instance targets for unload, and rich `ServedModel` status records
with model ref, runtime identity, state, backend, capabilities, context length,
and error fields. Runtime adapters should preserve typed serving errors where
possible.

The local REST management API remains useful for controlling an external
`mesh-llm` daemon, but it is not the primary serving SDK implementation.

## Model Boundary

Model APIs in this crate are deliberately independent of Clap command types and
terminal output. They return structured data that higher layers can expose
through Rust, FFI, CLI, REST, or UI adapters.
