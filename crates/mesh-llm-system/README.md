# mesh-llm-system

`mesh-llm-system` owns machine-local concerns for mesh-llm.

This crate includes:

- backend flavor and binary/device helpers
- hardware discovery and GPU identity/facts
- process liveness and PID validation helpers
- local benchmark fingerprinting and prompt corpus import support
- release target and self-update plumbing

Keep distributed mesh membership, request routing, API routes, CLI dispatch, and
host runtime orchestration outside this crate. Those layers may consume system
facts, but this crate should stay focused on local platform behavior.

Default native-runtime startup can use a composed product's runtime release
instead of the build's fallback release. The product manifest must sit beside
its discovered `native-runtimes` directory, identify that exact bundle, and
agree with its manifest digest, payload checksums and compiled host ABI.
Conflicting products or inconsistent metadata fail selection. An explicit
runtime release pin bypasses this default policy; ordinary resolver checks
still apply. Without a product manifest, the Skippy runtime version metadata
supplies the default. This does not change the Mesh product version.
