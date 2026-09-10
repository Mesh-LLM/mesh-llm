# VRAM accounting

mesh-llm uses three VRAM concepts:

- **Rated VRAM**: user-facing capacity class, such as `32 GB`, derived from
  the system-reported byte count when no explicit rated value exists.
- **System-reported VRAM**: raw total bytes reported by the driver/runtime. This
  remains the source for internal calculations.
- **Reserved VRAM**: true driver/runtime reserved or unavailable bytes, when the
  platform reports them. Live used-memory counters are not reserved VRAM.

Internal fit decisions should use `system_reported_bytes - reserved_bytes`
where a true reserved value is available. Per-GPU labels should show the
rated capacity class.

Node and mesh totals are a different case. A node advertises one capacity to
the mesh (`PeerAnnouncement.vram_bytes`, reported by `/api/status` as
`my_vram_gb` and `peers[].vram_gb`), and that is the number the scheduler,
`doctor split` (`aggregate_capacity_bytes`), and model-target advice sum. Any
surface that presents a node or mesh total (the dashboard `Mesh Capacity` tile, the
peer table `VRAM` column, the chat header) must use that advertised figure so
the console, the CLI, and the API agree. Client-role nodes advertise capacity
but never serve, and the scheduler excludes them from the aggregate, so totals
exclude them as well. Summing rated classes across GPUs is display-only and
overstates schedulable capacity; the UI helpers fall back to allocatable, then
rated, inventory only for legacy payloads that carry no advertised value, never
as the primary source of a total (see #1656 for the itemized total / reserved /
usable breakdown that will replace the single value).

![Dashboard showing Mesh Capacity 115.4 GB from the advertised capacity](assets/vram-dashboard-advertised.png)

![Chat header showing 1 node and 115.4 GB from live status](assets/vram-chat-advertised.png)

| Location | Value source | Classification | Current use |
|---|---|---|---|
| `crates/mesh-llm-system/src/hardware/mod.rs` | platform tools, Skippy devices, system RAM fallback | internal source | Builds `HardwareSurvey.vram_bytes`, per-GPU `gpu_vram`, and `gpu_reserved`. |
| `crates/mesh-llm-system/src/hardware/enrichers.rs` | CUDA/NVML | internal source | Enriches NVIDIA totals and true NVML reserved memory. |
| `crates/mesh-llm-system/src/vram.rs` | system-reported bytes plus optional reserved bytes | shared semantic utility | Computes rated capacity, decimal reported GB, and allocatable bytes. |
| `crates/mesh-llm-commands/src/gpus.rs` | `HardwareSurvey.gpus` | user-facing CLI and machine JSON | Human CLI displays rated VRAM; JSON keeps raw `vram_bytes` and adds rated/allocatable fields. |
| `crates/mesh-llm/src/commands/models/formatters.rs` | `hardware::survey().vram_bytes` | mixed | Model search fit hints use reported capacity. Human summary still reports effective available capacity. |
| `crates/mesh-llm-host-runtime/src/mesh/mod.rs` | `HardwareSurvey` startup snapshot | internal and protocol | Stores node `vram_bytes`, `gpu_vram`, and `gpu_reserved_bytes` for runtime, gossip, and status. |
| `crates/mesh-llm-host-runtime/src/protocol/convert.rs` | peer announcements and protobuf GPU fields | protocol/internal | Preserves additive per-GPU totals and reserved bytes across mixed-version gossip. |
| `crates/mesh-llm-host-runtime/src/api/status.rs` | node fields and GPU CSV fields | API for user-facing console | Emits raw `vram_bytes`, `reserved_bytes`, rated VRAM, and allocatable VRAM per GPU. |
| `crates/mesh-llm-host-runtime/src/runtime/local.rs` | startup model specs and pinned GPU targets | internal | Skippy fit targets use allocatable capacity for pinned GPUs. |
| `crates/mesh-llm-host-runtime/src/runtime/split_planning.rs` | participant `vram_bytes` | internal | Plans splits from advertised capacity, with separate runtime headroom. |
| `crates/mesh-llm-host-runtime/src/runtime/context_planning.rs` | local/split capacity bytes | internal | Computes KV/context budget from capacity after model bytes. |
| `crates/mesh-llm-host-runtime/src/api/model_target_capacity.rs` | local and peer `vram_bytes` | internal/API advice | Computes fit summaries and capacity advice. |
| `crates/mesh-llm-host-runtime/src/runtime_data/collector.rs` | peer `vram_bytes` | API/user-facing aggregate | Produces mesh and peer VRAM summaries for status views. |
| `crates/mesh-llm-ui/src/lib/vram.ts` | status GPU and node fields | shared UI semantic utility | Computes rated, system-reported, reserved, and allocatable values per GPU, and advertised node and mesh totals (`nodeAdvertisedVramGB`, `meshAdvertisedVramGB`). |
| `crates/mesh-llm-ui/src/features/network/api/status-adapter.ts` | `/api/status` | user-facing dashboard | Node rows and the `Mesh Capacity` tile use advertised capacity (`my_vram_gb` / `vram_gb`), falling back to allocatable then rated inventory only when nothing is advertised. |
| `crates/mesh-llm-ui/src/features/app-shell/lib/status-helpers.ts` | `/api/status` and topology data | user-facing dashboard helpers | Formats GPU inventory with rated capacity; node and mesh totals (`displayVramGb`, `meshGpuVram`) use advertised capacity. |
| `crates/mesh-llm-ui/src/features/chat/lib/live-chat-metrics.ts` | `/api/status` | user-facing chat header | Node count and advertised mesh capacity badges, from the same helper as the dashboard so both tabs agree. |
| `crates/mesh-llm-ui/src/features/configuration/api/config-adapter.ts` | `/api/status.gpus[]` | bridge from API to UI math | Maps rated total, system total, reserved, and allocatable fields into config nodes. |
| `crates/mesh-llm-ui/src/features/configuration/lib/config-math.ts` | config node GPU fields | internal UI calculation | Uses system/allocatable capacity for fit math while preserving rated total for labels. |
| `crates/mesh-llm-ui/src/features/configuration/components/VRAMBar.tsx` | config math props | user-facing and internal UI | Displays total/reserved/free lanes; sizing is driven by system capacity. |
| `crates/mesh-llm-ui/src/features/dashboard/components/details/NodeSidebar.tsx` | node GPU inventory | user-facing console | Displays per-GPU rated capacity. |
| `crates/mesh-llm-ui/src/features/dashboard/components/topology/**` | adapted node VRAM | user-facing visual weighting | Uses adapted display VRAM for labels and node sizing. |
| `crates/mesh-llm-ui/src/features/reserves/**` | wakeable node/status VRAM fields | user-facing reserve planning | Live mesh comparisons use GPU rated capacity when status inventory is available; reserve-provider VRAM still depends on wakeable upstream inventory. |
