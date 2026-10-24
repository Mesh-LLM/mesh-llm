# One model, independent mesh memberships

`mesh-llm serve-shared` is an explicit, full-model serving mode for the first
milestone of [#2309](https://github.com/Mesh-LLM/mesh-llm/issues/2309). It loads
one operator-selected local GGUF once and attaches that backend to independent
meshes. It does not merge those meshes or distribute the model between them.
The ordinary `serve` and `client` modes keep their existing configuration and
lifecycle.

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

```sh
mesh-llm serve-shared \
  --model-path /models/model.gguf \
  --model-id shared-model \
  --state-dir /private/operator/shared-model \
  --membership work=/private/operator/work.token \
  --membership lab=/private/operator/lab.token \
  --concurrency 2 \
  --queue-capacity 16 \
  --membership-queue-capacity 8
```

The state directory is created privately on Unix; an existing directory must
have mode `0700` and cannot be a symlink. It must be dedicated to this host, not
the ordinary `~/.mesh-llm` directory. A process lock prevents two hosts using it.
Membership labels accept ASCII letters, digits, `_` and `-` only and must be
unique ignoring case, so their state directories stay distinct on macOS and
Windows too. They identify local attachments, not public mesh names. At most
sixteen are supported.

Use `--membership-trust work=/private/operator/work-trust.json` to supply the
existing trust-store format for that attachment. `--attestation-key` selects an
existing owner keystore; the owner signs each endpoint separately. Without an
explicit trust store, the existing `off` trust policy applies. These settings
are not inherited from the normal single-mesh config, and are not remotely
synchronized. `--lan-only` disables public relay and STUN; otherwise the usual
public relay transport is used, **without publishing the mesh to discovery**.
Use `--bind-ip` to select a local QUIC interface for all attachments, each on
its own ephemeral port. `--lan-only --bind-ip 127.0.0.1` keeps a same-machine
test entirely on loopback; use an actual LAN address for other machines.

Each attachment loads the running binary's verified embedded release proof
before joining, including on reconnect. A mesh that requires certified releases
still enforces its genesis policy and allowed release signers: an unsigned,
tampered, or untrusted binary cannot join it. The owner keystore supplied through
`--attestation-key` is separate from this binary certification.

`--gpu-layers 0` selects CPU. `--context-size` and `--concurrency` are host-wide
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
Admission by the seed is sufficient to start serving; discovery of secondary
peers runs in the background, so a stale peer cannot stall an otherwise valid
attachment.
An invalid replacement token fails closed: the old membership is not silently
reused. An explicitly configured trust store must still exist and parse on
every reconnect; removing it cannot fall back to the `off` policy. Attaching
the same learned mesh ID twice is rejected. Whole-process
shutdown withdraws all memberships and waits for admitted native work to drain.

## Compatibility and deliberate limits

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
