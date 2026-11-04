---
title: One model across independent meshes
---

# One model across independent meshes

Use `mesh-llm serve --memberships-config` to load one local GGUF once and
offer it to several independent meshes. Each membership has its own endpoint
identity, peers, trust settings, request cancellation, and native session/cache
namespace. The memberships share the model weights and one execution budget.

Repeated `--join` values supply bootstrap peers for one mesh. To serve
independent meshes, put their token files in a local memberships manifest.
Ordinary `serve` and `client` behavior stays the same without this manifest.

## Prepare the manifest

Obtain one join token per target mesh through its existing
[join workflow](/docs/pages/private-meshes/#discovery-and-admission). Save each
token in a protected local file. A join token supplies bootstrap information;
configure the mesh's owner/trust policy to control admission.

Save this as `memberships.toml` beside the token and trust files:

```toml
version = 1
model_id = "shared-model"
state_dir = "./shared-model-state"
concurrency = 2
queue_capacity = 16
membership_queue_capacity = 8

[[memberships]]
name = "work"
join_token_file = "./work.token"
trust_store = "./work-trust.json"
owner_key = "./owner.json"

[[memberships]]
name = "lab"
join_token_file = "./lab.token"
trust_store = "./lab-trust.json"
owner_key = "./owner.json"
```

The example expects existing trust stores and an owner keystore in their normal
Mesh formats. `trust_store` and `owner_key` are optional, but omitting
`trust_store` selects `Off`: peer owner trust is not enforced. Every attach and
reconnect reports the effective trust policy and warns when it is `Off`.
These settings do not inherit from the ordinary single-mesh configuration.

Paths inside the manifest resolve relative to the manifest's directory.
The `--gguf` path resolves relative to the shell's working directory.
The manifest accepts 1–16 uniquely named memberships and is limited to 64 KiB.
Keep credentials in their files; inline tokens and unknown fields are rejected.

## Validate and start

```sh
mesh-llm config validate --memberships-config ./memberships.toml
# Config valid: ./memberships.toml

mesh-llm serve --gguf /models/model.gguf \
  --memberships-config ./memberships.toml
```

Validation checks the schema and limits without reading credentials, creating
state, or loading the model. Startup checks the model and membership inputs
before allocating the backend. Successful validation does not prove admission
to a target mesh.

This launch requires exactly one local `--gguf`. `--ctx-size` and `--parallel`
override the manifest's `context_size` and `concurrency`. The other accepted
runtime flags are `--config`, `--log-format`, and `--debug`. Configure topology,
GPU offload, and the management port in the manifest. Other explicit runtime
flags, including `--join`, `--auto`, `--port`, and `--console`, are rejected.
`--config` still selects ordinary logging/analytics settings, but its model,
topology, and GPU settings do not configure this host.

| Setting | Default and meaning |
| --- | --- |
| `context_size` | `4096` tokens; at least `512`, shared across memberships |
| `concurrency` | `1`; 1–64 native execution lanes for the whole host |
| `queue_capacity` | `32`; 0–4096 waiting requests across all memberships |
| `membership_queue_capacity` | `16`; 0–4096 waiting requests per membership |
| `gpu_layers` | `-1` for all supported layers; `0` for CPU |
| `management_port` | `0` to choose an available loopback port |
| `lan_only` | `false`; `true` disables public relay and STUN |
| `bind_ip` | Unset; optionally select a local QUIC interface |

Concurrency is a host-wide limit. Do not multiply it by the membership count.
Ready queues take round-robin turns, with FIFO ordering within a membership.
An admitted generation keeps its lane until it finishes or actually cancels.
Full or paused execution queues return HTTP 429 with `Retry-After: 1`.
The separate ingress limit can return HTTP 503 before execution admission.

Use a dedicated `state_dir`, separate from `~/.mesh-llm`. It holds saved
membership identities, the host lock, and the management token. Startup creates
private state and rejects unsafe existing permissions or symlinks. On Unix,
the directory must have mode `0700`; the host lock and management token must
have mode `0600`. On Windows, state requires a protected owner-only DACL.
Preserve the saved node keys to retain membership identities across restarts.

## Observe and control memberships

Startup reports a loopback management URL and the path to `management.token`.
All control requests require the token as an `Authorization: Bearer` header.
Keep its contents private; a fresh token is generated on every process start.
This host uses its own local control API rather than the ordinary web console.

| Request | Behavior |
| --- | --- |
| `GET /api/shared-serving` | Shared limits and active/queued counts, plus each membership's state, identity, and peers |
| `POST /api/shared-serving/memberships/work/pause` | Withdraw the offer and reject queued/new work; admitted requests may finish |
| `POST /api/shared-serving/memberships/work/resume` | Readvertise on the existing connected membership |
| `POST /api/shared-serving/memberships/work/leave` | Cancel that membership's requests and close its connection |
| `POST /api/shared-serving/memberships/work/reconnect` | Re-read token/trust files and join with the saved endpoint identity and a fresh execution scope |

Replace `work` in these paths with the manifest's membership name.
If all initial joins fail, startup exits unsuccessfully and releases the model,
state lock, and control listener. If at least one joins, serving continues and
failed memberships can be retried with `reconnect`. The announced management
URL alone does not mean a mesh admitted the host.

Edit the manifest and restart to change memberships or host limits. Reconnect
only re-reads that membership's token and trust inputs. Shutdown withdraws all
memberships and waits for admitted native work to drain.

## Isolation and supported use

This mode provides logical isolation within one trusted operator's process.
It shares physical memory, scheduling, and timing. It does not provide a
hostile-tenant sandbox, bandwidth guarantees, or secure cache erasure on
reconnect. A long request can occupy a lane until completion.

Each attachment serves the full local model. It does not merge meshes, route
requests between memberships, participate in split placement, or let mesh peers
resize or unload the shared backend. Multimodal projector configuration,
standalone Whisper/Laya backends, live transcription, MoA, and plugin hooks
are outside this launch mode. Certified-release requirements on the target
mesh still apply.

For the full manifest constraints and implementation details, see the
[shared-serving reference](https://github.com/Mesh-LLM/mesh-llm/blob/main/mesh/docs/SHARED_SERVING.md).
