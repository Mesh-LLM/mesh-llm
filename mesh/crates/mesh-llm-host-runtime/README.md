# mesh-llm-host-runtime

Mesh discovers peer-evidence verifiers from authenticated versioned manifests before accepting peers. Installed package metadata makes verifier plugins required and eager, including bare store entries that otherwise start optionally. The live declaration must match the installed namespace, handler and version. Command-only verifiers must use eager, non-optional startup. The declaration owns the namespace and handler; no evidence configuration grant is needed. Unsupported declarations, conflicting namespaces and unauthenticated lifecycle handlers fail startup. A plugin can publish or withdraw its own bounded reference through `PublishPeerEvidence`, with identity from existing APIs and mesh ID from snapshots/events. Starting a plugin never creates collateral.

Verifiers receive authenticated identity and optional evidence from the origin-signed record. Missing evidence means no published proof and is passed as null; the plugin decides whether that provider qualifies. Every discovered verifier must accept a provider before propagation or selection. Core interprets no chain, token, amount or collateral policy.

The host advertises `peer_evidence.manifest_enforcement.v1` alongside the evidence interface capability. Plugins depending on mandatory manifest-owned enforcement require both, so an older host with discretionary filtering cannot silently skip their policy. Successful verifier startup queues retained providers through the existing refresh lane after snapshot delivery; unrelated plugins and cache eviction do not trigger it.

One bounded Mesh cache owns accepted and rejected decisions, including those for absent evidence. Keys bind verifier, mesh, provider and evidence presence/content. Concurrent misses coalesce; fresh cache hits require no plugin call. The verifier chooses `valid_until` without a core cap, or explicitly returns `cache_until_restart: true` for policy-only decisions. Cache eviction prefers rejected and least-recently-used decisions; a full cache cannot refuse a new acceptance. Wall and monotonic deadlines prevent clock rollback from extending validity. Verifier shutdown clears only its own publications and cached/in-flight decisions. Authenticated membership is independent of provider eligibility. Gossip and heartbeats retain their original direct admission proofs and do not wait for evidence verification. A single background refresh lane verifies retained providers, including initially rejected candidates, without an extra pending-record cache. Signature-valid records are retained under Mesh's existing ordering rules regardless of eligibility: relays cannot replace a held record with an older one, while authenticated direct origins retain restart sequence-reset behavior. Expiry or rejection blocks provider propagation, routing-table responses, selection and new streams while retaining bounded signed records for heartbeat revalidation. Revalidation projects supply from the current held record only after its ordinary mesh-policy, version and owner checks pass. Verified cache-affinity metadata is preserved and filtered to that supply. Legacy providers without a signed record are verified with absent evidence and retain the mesh's ordinary compatibility behavior when the plugin accepts absence. Ordinary advertisement freshness and peer-health rules remain independent. Authenticated control-channel messages retain membership transport, independently of provider eligibility. Inference and work streams enforce provider eligibility, including after stream-credit waits. Admission callbacks perform no evidence lookup or additional bond verification.

`Node::peers()` reports admitted membership independently of verifier availability. Provider selection and supply projections use `eligible_peers()` and the routing checks. Discovery counts, bootstrap persistence, startup retry, split recovery and diagnostics retain admitted peers; advertised models, capacity and new work use current eligibility. Context limits, model/target media admission and spillover ranking use the current signed supply even when direct membership still holds an older announcement.

An incomplete installed-store scan fails startup because unreadable metadata could hide a required verifier. Local evidence is snapshotted within the record-signing lock so concurrent gossip builders cannot sign an older publication after a newer withdrawal record.

Publication authority begins after the current generation's authenticated manifest has been validated. Error-marked service responses become finite verifier rejections. A refresh deadline returns a finite rejection without cancelling the actual verifier RPC or releasing its serialization permit. The RPC retains its normal transport timeout and can replace that rejection with a late result in the same generation-checked cache. Authenticated stage stop, status and inventory retain membership transport and their existing receiver-side identity and topology checks; claims, loads and provider work retain eligibility checks.

Authenticated discovery does not wait for evidence verification. The existing heartbeat task wakes the background refresh lane at finite decision deadlines, independently of the gossip interval. Requests arriving during a refresh trigger another pass through that same lane; current cache entries are skipped. Cache-change notifications reschedule the timer and notify provider views without queuing verification, so eviction cannot create a verification loop. Idle maintenance retains only a notification handle, allowing endpoint closure to release the plugin manager. Expiry still blocks provider work until revalidation succeeds. Policy-only decisions schedule no timer. Default realtime model selection reuses its already filtered provider snapshot. Split placement uses the current signed download-refusal flag.

Retained finite deadlines remain pending across notifications, unrelated insertions and delayed scans until a refresh is queued through that monotonic time. Capacity eviction prefers expired entries, then current rejections and least recently used entries. Evicted decisions wait for normal gossip or heartbeat refresh; eviction never queues verification.

Each managed split runtime binds its HTTP port to its actual generation's remote workers and coordinator verification context. Every new local request checks those workers in the existing eligibility cache, including when the runtime reuses persistent stage lanes. Existing request guards continue unchanged. Registration precedes target publication; retired ports deny new work through shutdown, and cleanup matches the retired registration so it cannot remove a later instance on the same port. Forced teardown keeps its deny entry until owner cleanup or replacement registration. The runtime handle owns the verification context; weak registry references avoid a Node ownership cycle and fail closed if that context disappears.

Unsigned legacy providers reach verification only while authenticated direct admission is fresh. The same freshness requirement applies when reusing an absent-evidence verdict or propagating that legacy provider; a relay's fresh mention cannot replace that authority. Signed records authenticate their origins without a direct connection. Discovery remains independent of verifier decisions.

With verifiers configured, known work targets must still advertise provider role or supply. Runtime admission pins the policy-checked records and validates the complete dependency set in one final membership/cache snapshot, with one wall/monotonic time and unchanged verifier generations. No asynchronous suspension separates that snapshot from request-guard creation. Shutdown retains the stopped verifier registry until endpoint closure, so removing the plugin runner cannot turn a failed decision into an absent policy while work entry points remain open.

Known providers can complete on-demand authenticated gossip before the work eligibility check. Local plugin stream attachment has a ten-second deadline and stops when its mesh connection closes; cancellation removes the Unix socket pathname, and byte forwarding retains no Node reference after admission. During verifier discovery, a disconnected ordinary plugin with a retained non-verifier manifest keeps its normal supervisor recovery; unknown manifests and required verifiers fail closed.

Publication must fit the remaining signed node record budget as well as the per-reference limit. An overflowing publication is rejected without replacing existing evidence; withdrawal remains allowed. Later oversized local metadata is logged and never emitted as an invalid signed record. Malformed evidence and unusable verifier deadlines receive finite cached rejections. Dedicated stage dials complete on-demand gossip before the work gate. Mesh ID updates and startup snapshots share ordered delivery to the current plugin process; old events are never replayed into its replacement.

`mesh-llm-host-runtime` composes the host-side mesh node runtime. It wires
model resolution, local serving, discovery, networking, runtime state, plugins,
the management API, and the shipped CLI entrypoint used by the `mesh-llm`
binary.

This crate is being split so reusable CLI, TUI, SDK, and embeddable runtime
surfaces can be published and consumed independently.

## Embedded lifecycle regression tests

From the workspace root, run the real SDK startup-cancellation and join-stop
scenarios (isolated child processes, loopback networking, no model required):

```sh
just with-lld cargo test -p mesh-llm-host-runtime --no-default-features --features dynamic-native-runtime --lib sdk::embedded_lifecycle_tests
```

The opt-in download and inference scenarios also require a compatible installed
native runtime. Set `MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR` when using a prepared
runtime bundle. Download `tinyllamas/stories15M-q4_0.gguf` from the Hugging Face
[`ggml-org/models` repository](https://huggingface.co/ggml-org/models/tree/main/tinyllamas)
and set `MESH_EMBEDDED_TEST_GGUF` to its absolute local path. The test verifies
SHA-256 `66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739`.

```sh
MESH_EMBEDDED_TEST_GGUF=/absolute/path/stories15M-q4_0.gguf \
  just with-lld cargo test -p mesh-llm-host-runtime --no-default-features --features dynamic-native-runtime --lib sdk::embedded_lifecycle_tests -- --include-ignored
```

The download fixture holds an actual HTTP transfer open until cancellation;
the join fixture observes an actual QUIC dial before stopping. Successful stop
must release listeners and permit restart on the same ports. A separate
caller-pool regression holds the caller's blocking worker after the runtime
join and checks that the QUIC port is already free, so deferred socket cleanup
cannot make `stop()` report success early. The inference scenario cancels the
download, restarts serving in the same process, and checks nonempty generated
output through the OpenAI endpoint. Child deadlines turn
leaked runtime threads into failures rather than hanging the test runner.
These tests cover SDK lifecycle and native inference, not the Buzz UI or a
multi-machine community mesh.

Clients still use ephemeral transport identities by default. An explicit, nonempty `MESH_LLM_NODE_KEY_PATH` opts a client into persisted identity, allowing identity-bound obligations and evidence to survive restart. `MESH_LLM_EPHEMERAL_KEY` continues to force an ephemeral key even with an explicit path.
