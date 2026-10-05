# Mesh management control API

Owns management request trust classification/header validation and the `/health` status-view policy. The host injects cached peer connectivity, local runtime/process observations, plugin endpoint inventory and startup roles. Health rendering performs no I/O, never probes a runtime and preserves the answering-process liveness contract.

The host still owns the TCP listener, request lifecycle/correlation, response framing and collection from its concrete membership, plugin and runtime handles. The production server uses this crate's access checks before dispatch; the health route uses its rendered payload. Existing host HTTP tests exercise that composition; policy and serialization tests live here.

This is the first control-API extraction slice. The larger `/api/status`, runtime control, discovery, logs and plugin route families still depend on host-local types and remain in the host until their explicit handle boundaries are extracted. The crate has no dependency on the host runtime or Skippy lifecycle.

The status slice also owns runtime daemon/capability/intent/activity DTOs, model/process DTOs and deterministic process ordering, and cached metrics/slot snapshots with the management `/api/runtime/llama` projection. The host collects observations and maps concrete lifecycle/activity types into API labels; those conversions remain beside host orchestration. Metrics monitoring and publication still run in the host, with the extracted snapshot structs as their input/output contract.

Management HTTP framing now lives here too: JSON/error/byte responses, bounded opaque upstream headers and request-ID replacement. A request-scoped `ResponseObserver` is injected by the host lifecycle owner. HTTP code reports a status only after the corresponding write succeeds; the host retains admission, terminalization and cancellation semantics. Console assets still come from the host so feature/build embedding behavior is unchanged.
