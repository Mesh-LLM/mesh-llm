# One model, independent mesh memberships

`mesh-llm serve --memberships-config memberships.toml` serves one model to
independent meshes from a local configuration file, for the first
milestone of [#2309](https://github.com/Mesh-LLM/mesh-llm/issues/2309). It loads
one operator-selected local GGUF once and attaches that backend to independent
meshes. It does not merge those meshes or distribute the model between them.
Without `--memberships-config`, `serve` and `client` keep their existing
configuration and lifecycle. Repeated `--join` still supplies bootstrap peers
for **one** mesh; it does not create independent memberships.

## Ownership and isolation

The process-level host owns the native model, context configuration, aggregate
execution budget, queue budget, and state-directory lock. Each membership owns
its QUIC key/endpoint, membership persistence, trust store, peer table, model
advertisement, local ingress, and request cancellation domain. An attachment
never starts model election, split planning, plugin discovery, or model unload
loops. Neither mesh can resize or unload the shared backend.

Each attachment receives an internal execution scope that callers cannot set
through HTTP. That scope participates in native session identity **and** the
content-addressed KV namespace, including requests without `prompt_cache_key`.
Equal client session IDs and equal prompts do not share native sessions,
resident prefixes, or L3 page identities across memberships. Reconnecting keeps
the endpoint key but starts a fresh execution scope. Existing cache entries may
remain until normal bounded eviction; reconnect is not secure memory erasure.

This is logical isolation within one trusted operator's process, not a hostile
tenant sandbox. Weights, physical GPU memory, scheduling and timing are shared.
It does not provide hardware side-channel isolation, per-mesh bandwidth
guarantees, or arbitrary plugin isolation. Local processes with access to the
operator's account are trusted; membership backend ports are loopback-only.

## Start

Create one token file per target mesh using that mesh's existing join workflow.
Protect those files as credentials; do not paste tokens into command arguments,
logs, issues, or chat. Join tokens are bootstrap addresses/policies, not by
themselves an access-control boundary: use each mesh's trust requirements for
private deployments.

Save a separate `memberships.toml` on the serving host:

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

Every path inside this file is relative to the file's directory, unless
absolute. The `--gguf` path follows normal command-line working-directory
rules. Inline join tokens and unknown TOML fields are rejected.

```sh
mesh-llm config validate --memberships-config /private/operator/memberships.toml
# Config valid: /private/operator/memberships.toml

mesh-llm serve --gguf /models/model.gguf \
  --memberships-config /private/operator/memberships.toml
```

Validation checks the version, schema, names, limits and nonempty paths without
reading credentials or creating state. Startup additionally validates the model
file, join tokens and explicit trust/owner files before allocating the backend
or binding endpoints. Correct schema alone does not prove a mesh will admit it.

The manifest is limited to 64 KiB and is local-only, separate from `config.toml`, `MeshConfig`, gossip
and the web config editor. It is read at startup; edit it and restart to change
membership definitions or host limits. Reconnect re-reads token and trust files.
`--config` still selects ordinary logging/analytics settings; its model,
topology and GPU settings do not configure this shared host. Only `--gguf`,
`--memberships-config`, `--ctx-size`, `--parallel`, `--config`, `--log-format`
and `--debug` are accepted for this launch. Other explicit runtime flags,
including `--join`, `--auto`, `--port` and `--console`, fail rather than being
silently ignored. The control port belongs in the manifest.

| Manifest field | Default / constraint |
| --- | --- |
| `version`, `model_id`, `state_dir`, `memberships` | Required; version is `1`, model ID is nonblank and at most 256 bytes, state path is nonempty, 1–16 memberships |
| `concurrency` | `1`; 1–64 native lanes shared by all memberships |
| `queue_capacity`, `membership_queue_capacity` | `32`, `16`; each 0–4096 waiting requests |
| `context_size` | `4096`; at least 512 tokens |
| `gpu_layers` | `-1`; all supported layers, or a nonnegative count (`0` for CPU) |
| `management_port` | `0`; an available loopback port, or explicit 1–65535 |
| `lan_only` | `false`; set `true` to disable relay/public STUN |
| `bind_ip` | Unset; optional local interface address |
| Membership `name`, `join_token_file` | Required; unique local name and nonempty token-file path |
| Membership `trust_store`, `owner_key` | Unset; optional explicit existing files, per membership |

`--ctx-size` and `--parallel` override `context_size` and `concurrency` for
this launch. Invalid manifest values still fail validation even when overridden.

The state directory is created privately on Unix; an existing directory must
have mode `0700` and cannot be a symlink. It must be dedicated to this host, not
the ordinary `~/.mesh-llm` directory. A process lock prevents two hosts using it.
On Unix, the management token and host lock are created with mode `0600`;
existing files with any group or other permissions are rejected before their
contents are changed. Correct their permissions explicitly before restarting.
On Windows, state directories, the management token, host lock and membership
node keys are created with a protected DACL granting full access only to the
current account. Existing objects must have the same owner-only DACL, with
inheritance from their parent disabled. The directory's owner entry must also
apply to new child files and directories. Unsafe existing state is rejected
without rewriting its permissions or contents. Correct its permissions
explicitly before restarting, preserving existing node keys to retain membership
identities. Directory junctions, other reparse points and multiply linked state
files are rejected.

Membership labels accept 1–64 lowercase ASCII letters, digits, `.`, `_` and
`-`, and must be unique. They follow the SDK's label syntax from #2571, with
additional restrictions because they become state directories: no leading or
trailing `.`, and no Windows device names (`con`, `prn`, `aux`, `nul`,
`com1`–`com9`, `lpt1`–`lpt9`, including names followed by extensions).
`host.lock` and `management.token` are also reserved for the host's own files.
These restrictions apply on every OS so moving a manifest to Windows cannot
alias another membership's state. Here names identify local attachments, not
public mesh names; the existing SDK label contract is unchanged.

Use each membership's `trust_store` to supply the existing trust-store format.
Its `owner_key` selects an existing owner keystore; the owner signs each
endpoint separately. Without an
explicit trust store, the existing `off` trust policy applies. These settings
are not inherited from the normal single-mesh config, and are not remotely
synchronized. Once its inputs are loaded, each attachment attempt, including
reconnect, reports its name, effective trust policy and explicit store path
(or `no explicit store`). An
`Off` policy emits a warning that peer owner trust is not enforced. For example:

```text
Shared membership 'work': trust policy Off (no explicit store); peer owner trust is not enforced
Shared membership 'lab': trust policy Allowlist (store: /private/operator/lab-trust.json)
```

Ownership certificates are saved and renewed in each attachment's own state
directory. Renewal does not overwrite another attachment's certificate or the
ordinary single-mesh node's `~/.mesh-llm/node-ownership.json`. If saving a renewed
certificate fails, the node logs a warning and keeps using the renewed proof
in memory; it does not fall back to the global certificate file.

`lan_only = true` disables public relay and STUN; otherwise the usual
public relay transport is used, **without publishing the mesh to discovery**.
Use `bind_ip` to select a local QUIC interface for all attachments, each on
its own ephemeral port. `lan_only = true` with `bind_ip = "127.0.0.1"` keeps a same-machine
test entirely on loopback; use an actual LAN address for other machines.

Each attachment loads the running binary's verified embedded release proof
before joining, including on reconnect. A mesh that requires certified releases
still enforces its genesis policy and allowed release signers: an unsigned,
tampered, or untrusted binary cannot join it. The owner keystore supplied through
`owner_key` is separate from this binary certification.

`gpu_layers = 0` selects CPU. Context size and concurrency are host-wide
native settings, fixed until restart. Increasing either can increase memory
use. Do not multiply the concurrency by the membership count.

## Scheduling and overload

There is one aggregate native concurrency limit and one bounded waiting budget.
Each membership also has a waiting limit. Ready, non-empty queues take
round-robin turns; within one membership, requests are FIFO. This is
non-preemptive request fairness, not token-time fairness: a long admitted
generation occupies its slot until it finishes or actually cancels. It can use
all available lanes while no other membership has waiting work.

Full or paused execution queues reject new requests with HTTP 429 and
`Retry-After: 1`. Queued cancellation releases its waiting slot immediately.
An admitted request keeps its execution lease through blocking preparation and
native execution, even if the outer HTTP future disappears. A streaming
consumer also retains its lease until the stream is released.

QUIC ingress additionally limits in-flight parsing/relay work, before reading
request bodies. Its aggregate limit is concurrency + queue capacity + membership
count; its per-membership limit is concurrency + the smaller queue capacity + 1.
Exhaustion returns HTTP 503 before execution admission. Normal frontend body and
transport timeouts still apply. These bounds do not constitute OS-level memory
or CPU quotas.

## Local control and status

Startup reports a loopback management URL and the path to `management.token` in
the state directory. The token is newly generated at every process start and
is never advertised over gossip. All management requests require it in an
`Authorization: Bearer ...` header. Treat its contents as a secret.

| Method and path | Result |
| --- | --- |
| `GET /api/shared-serving` | Model, native runtime identity/lane count, shared limits and aggregate active/queued counts, plus each membership's state, identity, peers and counts |
| `POST /api/shared-serving/memberships/work/pause` | Withdraw work's model; reject queued/new requests, allow already admitted work to finish |
| `POST /api/shared-serving/memberships/work/resume` | Readvertise on the existing connected membership |
| `POST /api/shared-serving/memberships/work/leave` | Withdraw and close work; cancel only its requests; keep shared weights and other memberships alive |
| `POST /api/shared-serving/memberships/work/reconnect` | Re-read its token/trust inputs and reconnect with its saved endpoint key and a fresh execution scope |

An unavailable mesh is reported as failed without stopping successful joins.
After initial joins finish, the command reports the connected membership count.
If none connected, startup fails with a nonzero exit status, closes the control
listener and releases the model and state lock. Correct the inputs and restart.
With at least one connected membership, serving continues and failed memberships
can be retried through the control API. The control URL announced during startup
indicates management availability, not successful mesh admission.
Admission by the seed is sufficient to start serving; discovery of secondary
peers runs in the background, so a stale peer cannot stall an otherwise valid
attachment.
An invalid replacement token fails closed: the old membership is not silently
reused. An explicitly configured trust store must still exist and parse on
every reconnect; removing it cannot fall back to the `off` policy. Attaching
the same learned mesh ID twice is rejected. Whole-process
shutdown withdraws all memberships and waits for admitted native work to drain.

## Compatibility and deliberate limits

[PR #2571](https://github.com/Mesh-LLM/mesh-llm/pull/2571) provides the separate
SDK `MeshMembershipConfig` / add-remove-list API for a **running node**. It
creates named memberships with an owner allowlist and a per-membership routing
API. This local manifest instead bootstraps a shared full-model server into
existing meshes using token files and explicit trust stores, with one aggregate
admission budget and isolated native execution/cache scopes. It does not
replace or reinterpret that SDK API. In particular, `name` here is a local
alias, and omitting `trust_store` preserves the explicit shared host's `Off`
policy; it does not inherit the SDK's allowlist. Neither API treats multiple
`--join` values as independent meshes.

- The implementation adds no wire fields or stream kinds. Each membership is
  an ordinary full-model host to compatible peers. Its GPU placement capacity
  is zero, and stage execution/artifact-transfer capabilities are not advertised.
- A shared attachment requires successful peer gossip admission before HTTP
  execution. Passive clients that bypass gossip must join first. This stricter
  opt-in mode does not alter ordinary single-mesh ingress.
- Stage, model-lifecycle, configuration, artifact and plugin streams are
  rejected, including from an admitted peer. Requests cannot forward from one
  membership into another.
- Chat/completions/Responses and supported non-chat backend endpoints use the
  same scoped admission. Actual model support remains the native backend's
  responsibility. This first slice does not configure multimodal projectors,
  standalone Whisper/Laya backends, live transcription (`/v1/realtime`),
  split placements, MoA, plugin hooks,
  System One, or token-counting endpoints.
- The control surface is CLI plus authenticated local HTTP, not the ordinary
  web console/config editor. No UI setting or persistent config migration is
  introduced.

## Acceptance evidence

Unit tests cover aggregate/per-membership queue bounds, FIFO/round-robin turns,
cancelled waiters, pause/resume, detach cancellation and leases held by native
workers. Skippy tests verify scoped native IDs and actual resident/L3 cache
identities. Host integration tests use two independent local QUIC meshes and
one recording backend to exercise isolation, lifecycle, reconnect and bad
token handling. Those transport fixtures do not prove native model quality or
cross-machine interoperability; a composed-product real-model smoke and a
released-peer compatibility run are separate validation lanes.

Manifest/CLI regressions also cover unknown fields, secret-safe parse errors,
relative paths, unsupported option combinations and unchanged repeated joins.
The product smoke launches ordinary `serve` with a relative-path manifest and
explicit `--ctx-size`/`--parallel` overrides.

Run the opt-in product test with a freshly composed release host/runtime and an
existing causal GGUF. It creates isolated local seed meshes, state and caches;
it does not download a model or publish a mesh:

```sh
just release-build
MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$PWD/dist/native-runtimes" \
SHARED_SERVING_BINARY="$PWD/target/release/mesh-llm" \
SHARED_SERVING_MODEL=/models/causal-model.gguf \
cargo test -p mesh-llm-host-runtime \
  composed_product_serves_two_meshes_from_one_native_model \
  -- --ignored --test-threads=1
```

The test defaults to CPU (`SHARED_SERVING_GPU_LAYERS=0`); set
`SHARED_SERVING_GPU_LAYERS=-1` to exercise supported GPU offload. Optionally set
`SHARED_SERVING_RELEASED_PEER` to an independently downloaded, checksum-verified
official release binary. That lane starts a client with isolated state, waits
for discovery of the shared offer, and performs real inference through its
public HTTP API. Preserve the tested release version and artifact checksum
with the validation results. This is same-machine mixed-version coverage, not
cross-machine or WAN qualification.
