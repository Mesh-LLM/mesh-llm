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

Each proposal query carries the capacity Skippy can verify at that exact decode
position. The native host adapter also applies a bounded implementation cap so
a request cannot force an unbounded host allocation.
