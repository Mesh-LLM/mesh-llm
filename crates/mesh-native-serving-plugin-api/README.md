# Mesh native serving plugin API

This crate defines Mesh's versioned C-compatible ABI for native plugins that
observe Skippy generation lifecycle events and supply speculative proposals.
Mesh owns model execution, tokenization, verification, and the absolute
proposal deadline. Plugins exchange only fixed-layout values, borrowed slices,
opaque handles, and host-owned output buffers across the dynamic-library
boundary.

The ABI deliberately has no fixed proposal-token limit. Each query carries the
capacity Skippy can verify at that exact decode position.
