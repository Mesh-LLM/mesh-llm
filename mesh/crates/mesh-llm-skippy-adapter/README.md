# Mesh Skippy adapter

Translates Mesh model/configuration policy into Skippy stage, runtime and OpenAI options. The host supplies model selection, resolved artifact paths, available memory, and request defaults. Skippy owns model preparation, inference, transport execution and serving lifecycle.

The adapter owns Mesh configuration precedence, validation, speculative selection, cache defaults, device/load option translation, advisory digest-cache location and Mesh checkpoint notices. It depends on Skippy APIs and Mesh configuration/event types, never on the host runtime. Host orchestration retains client configuration, model handles, operational events, membership and plugin lifetime. The adapter translates Skippy package progress into Mesh terminal/dashboard output.

Resolver tests live beside the implementation. Host hook/receipt integration stays in the host runtime. `test-support` disables the implicit home-directory hash cache in dependent host tests; explicit `MESH_LLM_HASH_CACHE_DIR` remains supported.

`package` applies Mesh digest-cache policy to Skippy full-package and metadata-only identity verification. Schema probing and verification remain in Skippy; the host resolves remote package references before calling the adapter.

`package::inspect_local_stage_package` applies the same Mesh cache policy to Skippy inspection; the host resolves remote references once before invoking it.

`readiness` exposes the Skippy startup contract to Mesh stage orchestration; no Mesh load-request or coordinator identity enters the serving probe.

`package::acquisition` supplies the existing Mesh hub/package-integrity cache directories and progress observer to Skippy acquisition. The host supplies its existing lazy client factory, preserving endpoint/token/TLS policy. Local validation and snapshot selection use the same explicit integrity-cache policy.
