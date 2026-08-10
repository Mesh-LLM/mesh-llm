# Mesh native serving plugin API

This crate defines Mesh's versioned C-compatible ABI for native plugins that
observe Skippy generation lifecycle events and supply speculative proposals.
Mesh owns model execution, tokenization, verification, and the absolute
proposal deadline. Plugins exchange only fixed-layout values, borrowed slices,
opaque handles, and host-owned output buffers across the dynamic-library
boundary.

Every host-provided event or query starts with `struct_size`. Plugins must
validate that the supplied size covers the last field they read and ignore any
trailing fields. This lets a newer host append fields without changing the
layout of the prefix understood by an older plugin. Removing or reordering
fields, changing a field's type, or changing a callback signature requires a
new versioned ABI table and continued support for the previous table.

The V2 activation contract lends a model-bound tokenizer capability through
`ActivationContext`. The capability includes the opaque native tokenizer
inventory, the model identity and binding digest, explicit input/output
limits, and a bounded encode callback for tagged ordinary-byte and opaque
control pieces. Mesh does not know or own Rosetta vocabulary: a consumer
builds any translation from the inventory and calls the capability outside the
proposal deadline. Unsupported or non-lossless input is rejected; it is never
decoded with replacement semantics. A plugin copies or transforms the
inventory before returning from `activate` and must not call the encode
callback from proposal callbacks.

The V2 table is a breaking replacement for V1. Plugins must resolve
`mesh_native_serving_plugin_v2` and validate the V2 ABI/version and structure
sizes before use.

Each proposal query carries the capacity Skippy can verify at that exact decode
position. The native host adapter also applies a bounded implementation cap so
a request cannot force an unbounded host allocation.
