# Threat model (pointer)

The threat model these documents cite ("TRUST-MODEL.md §…") is not in this
repository yet. Until it moves here, it is published in the capsule-emit-mesh
repository at
<https://github.com/action-state-group/capsule-emit-mesh/blob/main/docs/TRUST-MODEL.md>,
with a plain-language version at
<https://github.com/action-state-group/capsule-emit-mesh/blob/main/docs/CAN-YOU-TRUST-A-STRANGER.md>.
Section numbers cited here refer to that document.

## Hardware key custody: research, not shipped

The node's signing key is a file under `<data dir>/keys/`. Keeping it in
hardware instead (the Secure Enclave on Apple machines) was prototyped and is
not part of this plugin. That work is preserved, unmerged, at the tag
`archive/rung3b-sep-key` (commit `b66096867472845ba4d9001dd9409e53d65d10e6`)
in <https://github.com/action-state-group/capsule-emit-mesh/tree/archive/rung3b-sep-key>.
