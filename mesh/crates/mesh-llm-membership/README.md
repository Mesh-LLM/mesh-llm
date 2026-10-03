# mesh-llm-membership

Mesh peer membership: peer identity, admission state and discovery helpers.

Owns *who* is in the mesh, *what* they announce, and *whether* they are
reachable. Does **not** own inference/split policy (that stays in Skippy), the
serving transport, owner control, or the plugin host (those stay in
`mesh-llm-host-runtime` for their own later extractions).

Dependency direction is Mesh → Skippy. This crate imports only Mesh crates
(`mesh-llm-identity`, `mesh-llm-protocol`, `mesh-llm-routing`, and
`mesh-llm-types`) plus iroh/anyhow/serde and
small utility crates — never a `skippy-*` crate and never
`mesh-llm-host-runtime`.

## Extracted so far (slice 1 — leaf foundations)

| Module | What moved | Host consumers rewired |
|--------|-----------|------------------------|
| `types` | `NodeRole` (was in host `peer_state.rs`, duplicated in client `mesh/types.rs`) | host `peer_state.rs`, client `mesh/types.rs` re-export |
| `address` | `is_public_ipv4_candidate`, `is_global_ipv4_candidate` (was in host `connections.rs`) | host `connections.rs` / `stun.rs` / `node_identity.rs` / `advertisement.rs` via `mesh/mod.rs` prelude |
| `identity_persistence` | node/mesh identity persistence (was `mesh/identity_persistence.rs`), gated behind the `host-io` feature | host `mesh/mod.rs` public re-exports; `node_requirements.rs` / `node/startup.rs` import directly |
| `lan_bootstrap` | `lan_ipv4_candidates`, `is_private_lan_ipv4` (was `mesh/lan_bootstrap.rs`) | host `node_identity.rs` |
| `cache_affinity_gossip` | cache-affinity advertisement construction/merge (was `mesh/cache_affinity_gossip.rs`) | host `announcements.rs`, `gossip.rs` |
| `model_identity` | pure leaves — `unknown_identity`, `local_gguf_identity_from_source`, `parse_hf_ref_parts`, `parse_hf_resolve_url_parts`, `format_hf_canonical_ref`, `identity_hash_for` (was in host `model_identity.rs`) | host `model_identity.rs` re-exports for prelude/tests |
| `weights_digest` | `weights_digest_for_file_in`, `file_fingerprint`, single-flight digest cache (was `mesh/weights_digest.rs`); cache directory now injected | host `weights_digest.rs` shim resolves `mesh_llm_cache_dir()/weights-digest/` |

## Extracted so far (slice 2 — the three contracts + admission core)

| Module | What moved | Host consumers rewired |
|--------|-----------|------------------------|
| `release_attestation` | `ReleaseBuildAttestation` (+ `validate`/`canonical_bytes`/`verify`/`to_proto`/`from_proto`), `EmbeddedReleaseAttestation`, `ReleaseAttestationClaims`, `ReleaseSignerTrustStore`/`TrustedReleaseSigner`, `ReleaseAttestationStatus`/`ReleaseAttestationSummary`/`ReleaseAttestationError`, `VerifiedEmbeddedReleaseAttestation`/`LoadedEmbeddedReleaseAttestation`, `release_signer_key_id`, `parse_release_signer_public_key`, `verify_release_attestation`, and the pure tests. Verification/trust behavior preserved verbatim | host `crypto/release_attestation.rs` shim re-exports the contracts and keeps the trust-store path/disk I/O + embedded release-footer loader (which needs `mesh_llm_system`) |
| `advertised_throughput` | `ModelThroughputHint`, `sanitize_model_throughput_hints`, `THROUGHPUT_SCALE_MILLI` + `MAX_ADVERTISED_*` bounds | host `network/metrics.rs` re-exports; gossip `PeerAnnouncement` field now references the moved type |
| `selected_path` | `SelectedPathObservation`, `SplitStagePathKind`, `SplitStagePathSnapshot` (+ constructors/fallbacks), `split_stage_path_snapshot_from_observation` | host `stage_transport.rs` re-exports and keeps the `iroh::endpoint::Connection` walk (`selected_path_observation`), which is a serving-transport effect |
| `requirements` | the full admission policy — `MeshGenesisPolicy`, `SignedMeshGenesisPolicy`, `SignedBootstrapToken`, `DirectNodeAdmissionProof`, `MeshRequirements`, `NodeVersionBounds`, `ProtocolGenerationBounds`, `ReleaseAttestationRequirement`, `MeshRequirementDecision`/`MeshRequirementRejectReason`, rejection taxonomy/event types, `Normalized*` bounds, `peer_release_attestation_status`, `evaluate_direct_peer_admission`, and the direct-proof encoding tests | host `mesh/requirements.rs` shim re-exports; `announcements.rs`/`gossip.rs` reach `current_time_unix_ms` through that re-export |
| `peer_state` | `PeerAnnouncement`; the neutral `PeerInfo` state and its non-routing accessors (`from_announcement`, `is_admitted`, `current_direct_rtt_ms`, `split_stage_path_fallback`, `display_latency`, `is_assigned_model`, `accepts_http_inference`); `DirectLatencyObservation`/`PropagatedLatencyObservation`/`DisplayLatency`/`DisplayLatencySource`; `MeshCatalogEntry`; the pure admission helpers `policy_accepts_peer`, `model_identity_score`, `stream_allowed_before_admission`, `ingest_tunnel_map`, `resolve_peer_leaving`; and the `PEER_STALE_SECS`/`DEAD_PEER_TTL`/`PEER_DOWN_REPORTER_COOLDOWN_SECS` constants. Gated behind `host-io` (it holds `SignedNodeOwnership`) | host `mesh/peer_state.rs` re-exports `PeerInfo` and keeps the serving-routing projections as free functions over `&PeerInfo` (`routable_models`, `routes_model`, `http_routable_models`, `routes_http_model`, `public_model_id_for_routable_model`, `advertised_context_length`), plus `OwnerRuntimeConfig`, `ControlListenerLifecycle`, and host `impl Node` blocks |

`release_attestation`, `advertised_throughput`, `selected_path`, and
`requirements` are available without `host-io`. `peer_state` is gated behind
`host-io` because `PeerAnnouncement.owner_attestation` carries
`mesh_llm_identity::SignedNodeOwnership` (which identity itself exposes only
under `host-io`).

`identity_persistence` is feature-gated behind `host-io` (it touches the node
key/keystore/ownership file I/O that the embedded client must not pull). The
pure modules (`types`, `address`, `lan_bootstrap`, `cache_affinity_gossip`,
`model_identity`, `weights_digest`) are available without `host-io`, so the
client depends on the default feature set and stays embedded-client-pure.

`test-support` (implies `host-io`) is the explicit `MESH_LLM_TEST_HOME`
override, enabled only by the host crate's dev-dependency. It exists because a
bare `#[cfg(test)]` guard is compiled out when the host exercises this crate as
a normal dependency (where the dependency's `cfg(test)` is false) — the exact
regression the host's `public_identity_tests` and admission tests were hitting.
See `mesh/crates/mesh-llm-host-runtime/tests/membership_test_home_isolation.rs` for
the executable cross-crate proof. Production builds never enable
`test-support`, so the env-var override stays out of library consumers.

## Peer health orchestration

`peer_health` owns heartbeat failure thresholds, relay health observations,
reconnect selection/cooldowns, home-relay status transitions and peer-down
report/removal decisions. It is available without `host-io`: callers provide
path observations, request activity and time; the controller returns decisions
without opening connections, emitting product events or invoking plugins.
The host heartbeat loop uses these decisions and retains transport effects,
peer snapshots, logging and plugin notifications. Existing policy tests live
beside the controller in `peer_health/tests.rs`.

## Adopted membership

`adopted_membership` owns the requirement-aware membership state, conflict
checks, signed-policy enrichment and persisted adopted-membership format.
The host supplies the path and retains locking, bootstrap orchestration and
plugin notifications. Loading and persistence verify signed policy identity;
writes use the existing identity crate's atomic keystore writer. This module
is gated behind `host-io`.

## Live membership state and connection admission

`state::MembershipState` owns peer records, control connections, pending
handshakes, tunnel maps, peer-death/reporter cooldowns and admission rejection
history. `connection_reservation` owns single-owner handshake reservations,
waiter completion and owner-scoped cleanup. Both are gated behind `host-io`;
Tokio synchronization is enabled only with that feature.

The host `Node` composes this state under its existing mutex and delegates
handshake lifecycle operations. Mesh plugin duplicate suppression has a separate
host-owned state and mutex; neither plugin state nor plugin types cross into
membership. The host still performs gossip I/O and admission-side effects.

## Gossip state transitions

`announcements` owns direct and transitive peer updates, advertised-version
and idle-client filters, meaningful-change detection, peer rebroadcast projection
and stale-peer selection over `MembershipState`. Existing host integration tests
still exercise these operations through real `Node` gossip; version-policy tests
live with membership. The host composes its local model/plugin advertisement,
backfills legacy model descriptors and retains network I/O and notifications.

## Remaining boundary (not yet moved)

The heavier membership modules still live in `mesh-llm-host-runtime/src/mesh/`
until their host-service dependencies are injected. Each is listed with the
host surface it reaches today:

- `peer_state.rs` (remaining) — the neutral `PeerInfo` state moved; host keeps
  `OwnerRuntimeConfig`, `ControlListenerLifecycle`, and the
  `impl Node` blocks. `PeerInfo`'s serving-routing projections
  (`routable_models`, `routes_model`, `http_routable_models`,
  `routes_http_model`, `public_model_id_for_routable_model`,
  `advertised_context_length`) are host-owned free functions over `&PeerInfo`.
  They call `public_model_id_from_identity`/`canonical_demand_model_ref`, which
  depend on `skippy_model_ref` (a `skippy-*` crate) and the host `models`
  catalog, so they cannot live in this crate (no Skippy dependency) and the
  orphan rule forbids `impl PeerInfo` in the host once `PeerInfo` moved. The
  free functions keep the resolution concrete — no trait, no injected
  abstraction.
- `node_requirements.rs` (remaining) — the `impl Node` / plugin-side of the
  admission policy (`MeshConfig`, `GpuAssignment`, `mesh_event` projection,
  `crate::system::hardware`). The policy (`requirements`), adopted persistence and state transitions moved.
- `node_identity.rs` — node/admission identity; `merge_public_addr_into_advertisement`
  plus release-attestation + plugin `MeshPeer`/`MeshEvent` projection.
- `model_identity.rs` (remaining) — `identity_from_model_source` and
  `infer_remote_served_descriptors` stay in host. The former calls
  `skippy_model_ref::ModelRef::parse` (a `skippy-*` crate), which is the
  actual blocker — not the `ModelCapabilities` mapping, which already resolves
  to `mesh_llm_types::models::capabilities::ModelCapabilities`. Moving them
  needs an injected model-ref parse operation across the Skippy boundary.
- `gossip.rs` / `heartbeat.rs` / `announcements.rs` — the heartbeat policy and
  relay controller have moved; the remaining host effects reach the plugin
  `mesh_event` projection and the release-attestation verifier; also depend on
  `peer_state` types above.
- `stun.rs` / `capacity.rs` / `connectivity.rs` /
  `direct_path.rs` / `direct_rescue.rs` / `host_role_claims.rs` — `stun` tests
  depend on `node_identity::merge_public_addr_into_advertisement`; the remaining host modules add `impl Node` blocks. Connection reservation
  state and lifecycle have moved into membership.
- `node.rs` — the `Node` struct. Reaches runtime orchestration
  (`crate::runtime`, `crate::runtime_data`, `crate::inference::skippy`,
  `crate::network::metrics`, `crate::capture`, `crate::plugin::PluginManager`).
  This is the last module to move; every other membership module's `impl Node`
  blocks must be split first.

The serving transport (`connections.rs` + `connections/`, `stage_transport*`,
`stage_proto`, `stage_artifacts`), owner control (`owner_control*`,
`owner_lifecycle_cache`), plugin host (`plugin_*`), operational logging, and
`artifact_transfer_io` remain in host-runtime and are not part of this crate.

## Peer table transitions

`transitions` owns accepted direct-peer update/insertion and transitive-peer
merge/insertion, including admission provenance, dead-peer suppression, mention
age, and the resulting count/change decisions. The host verifies announcement
policy, supplies peer identity, and publishes the returned observations and
plugin events after releasing the membership lock. Announcement-only removals
preserve connection and rejection state. Broader admission and removal lifecycle
orchestration remains in the host.

General peer removal also belongs to `transitions`: it clears admission rejection
history and returns the removed peer/count snapshot while preserving independent
connection and quarantine lifetimes. Stale-peer selection and heartbeat cooldown
retention run over membership state; the host supplies the direct-path cooldown
and handles capture, network cleanup and plugin notifications.
