# Experimental Apple runtime

Status: Milestones 0–3 have experimental implementations as of 2026-08-12:
the system-model spike, local REST vertical slice, shared SDK carrier contract,
and private-mesh whole-model routing. Release promotion and public-mesh
advertisement remain gated. Tracking issue:
[#1246](https://github.com/mesh-llm/mesh-llm/issues/1246).

## Outcome

MeshLLM can invoke Apple's on-device system language model from one native Swift
sidecar on Apple silicon, stream generated text, report first-party token usage,
cancel work, use guided generation, and execute tools. The same signed runtime
can be carried unchanged by the CLI and every SDK distribution that can run on
macOS. A per-user `launchd` background process also completed inference.

Core AI Instruments recorded active **Apple Neural Engine** load and prediction
intervals during the request. This is direct evidence that the system model
path uses Apple's native accelerator stack; it is not a CPU-only Swift wrapper.

The experimental sidecar exposes a loopback OpenAI-shaped REST surface. The
Rust host can now resolve and supervise the packaged process and route
`apple/system` through MeshLLM's normal local OpenAI frontend. Private meshes
gossip its observed availability, version, context, and one-slot load; the
provider does not ship in release products or advertise on public meshes.

## Why this is valuable to Apple silicon users

The system-model provider gives MeshLLM a zero MeshLLM-managed-download,
zero-model-management local inference option on eligible Macs. Apple must still
provision the system model on the device. Apple owns model delivery, device
compatibility, quantization, compilation, accelerator selection, and operating
system updates. MeshLLM can therefore offer a useful local model without
shipping another multi-gigabyte checkpoint or linking the neutral Rust host to
an Apple-only ML runtime.

For users, the likely advantages are:

- private on-device inference with no remote model endpoint;
- immediate availability when Apple Intelligence has provisioned the model;
- ANE-backed inference with Apple's power, memory, and thermal scheduling;
- no Hugging Face download, GGUF conversion, or Skippy layer packaging;
- fast whole-request routing to a capable Mac, with no per-layer network hops;
- one model identifier and capability contract across CLI, Rust, Swift,
  Node.js, and JVM clients;
- graceful fallback when the model is unavailable, not installed, or outside
  its supported capability/context envelope.

The product role is complementary to Skippy. `apple/system` should be routed as
a whole-model provider. Skippy remains the path for user-selected large models,
cross-node capacity, and staged execution. There is no reason to introduce
pipeline parallelism for Apple's opaque system model.

## Tested environment

| Component | Observed value |
|---|---|
| Machine | Apple silicon Mac Studio, 128 GiB |
| macOS | Golden Gate 27.0, build 26A5406e |
| Xcode | 27.0 beta, build 27A5194q |
| Swift | 6.4 |
| macOS SDK | 27.0 |
| System model | available |
| Context size | 4,096 tokens |
| Capabilities | guided generation, tool calling, vision |
| Languages | 24 advertised locales |
| Apple-documented model generation | 27.0 |
| Stable checkpoint/build identity | not exposed by the public arm64 beta module |

Apple documents system-model generations aligned with OS release bands: 26.0
through 26.3, 26.4, and 27.0. The runtime therefore exposes `apple/system` as a
rolling alias and `apple/system@27.0` as the resolved route on the tested host,
with `model_version=27.0` and
`version_source=apple_os_release_band`. It does not derive a checkpoint identity
from the OS build. A request for the versioned ID only matches a host currently
running that documented generation; MeshLLM cannot install, pin, or roll back
Apple's system model. Unknown future OS generations remain unversioned until
Apple documents their mapping.

There is intentionally no compatibility lane for an unversioned Apple system
provider. `model_version`, `version_source`, and `versioned_model_id` are
mandatory and must be mutually consistent. The host rejects a sidecar that
omits or contradicts them. An OS generation without a documented mapping
exposes no `apple/system` model rather than guessing an identity.

See [Apple's SystemLanguageModel documentation](https://developer.apple.com/documentation/FoundationModels/SystemLanguageModel)
and [Foundation Models updates](https://developer.apple.com/documentation/updates/foundationmodels).
Production cache identity and reproducibility policy must still account for
Apple updating the system model independently of MeshLLM.

## Implementation shape

The experimental sidecar lives in
[`providers/apple/`](../../providers/apple/). It is a Swift package with two
products:

1. `MeshAppleRuntime`, a reusable library that owns the runtime protocol,
   lifecycle, transports, and Foundation Models providers;
2. `mesh-apple-runtime`, one sidecar executable used by the CLI, REST QA,
   packaging, SDK carriers, and launchd validation.

`apple/system` is the first logical model inside that runtime. Named
`coreai/<artifact-id>` models will be added to the same executable and protocol
rather than shipped through a parallel Core AI sidecar.

The provider currently implements:

- availability and explicit unavailability reasons;
- context size, languages, and capability discovery;
- exact input token preflight for prompts, instructions, tool definitions, and
  guided-generation schemas, including requested response-token headroom;
- session prewarming;
- snapshot-to-delta streaming;
- elapsed time and time to first token;
- input, cached-input, output, and reasoning token accounting;
- guided `@Generable` output;
- a deterministic `Tool` invocation with an auditable recorder;
- cancellation, including Foundation Models' empty-stream cancellation edge;
- normalized retryable/non-retryable error codes.

The sidecar provides a loopback provider REST listener with `/health`,
`/v1/models`, and `/v1/chat/completions`. It supports buffered completions, SSE
streaming, usage, disconnect cancellation, and the deterministic fixture-tool
probe. The Rust host adapter supervises the same executable and registers its
ephemeral loopback port as an ordinary local inference target, so the existing
MeshLLM frontend preserves completion, SSE, errors, usage, tools, and disconnect
cancellation without an Apple-specific proxy. This keeps Foundation Models and
Swift out of the backend-neutral Rust host.

For process hosting, the runtime also accepts a validated `--parent-pid` and
checks parent liveness every 50 ms. A QA supervisor is killed with `SIGKILL`
mid-generation; the runtime exits rather than becoming an orphan. The Rust
supervisor passes its real PID, while the provider's parent watchdog covers an
ungraceful host crash.

### Rust host supervision

Host-capable `mesh-llm serve` processes on macOS now start one experimental
supervisor for the model selected by the packaged Apple provider runtime;
`mesh-llm client` never starts a local provider.
The supervisor:

- resolves a `kind=apple`, protocol `0.1` artifact from a product bundle,
  immutable cache, or explicitly enabled release index, filtering by the
  requested model identity when one is supplied;
- revalidates manifest hashes and executable policy, then verifies the macOS
  code signature, declared signing identity/team, declared entitlements, and
  notarization policy before execution;
- removes common credential environment variables and launches the sidecar
  with an ephemeral loopback port and the host's real parent PID;
- waits for the structured readiness record and probes both `/health` and
  `/v1/models` before registering the normal local inference target;
- registers both the rolling `apple/system` alias and the validated resolved
  generation ID, such as `apple/system@27.0`, against the same whole-model
  provider process;
- exposes PID, port, status, backend, context length, and restart state through
  the existing runtime-process dashboard and management API;
- withdraws the target immediately on unavailability, repeated health failure,
  or process exit, and restarts unexpected exits with bounded backoff; and
- stops ingress first during host shutdown, withdraws the route, sends
  `SIGTERM`, waits up to five seconds, and force-kills a child that does not
  exit.

On private meshes, the supervisor advertises both the packaged model identity
and its resolved generation as ordinary whole-model routes. Additive runtime
fields carry provider kind, generation, maximum concurrency, active requests,
and queued requests. Health loss or process exit withdraws the routes and
triggers fresh gossip; public meshes suppress the advertisement entirely.

Routing prefers an idle provider, then peers without load metadata, then busy
or queued providers. Existing request affinity keeps a session on the selected
Mac while it remains healthy. Failover is allowed only before response headers
or the first stream event; MeshLLM never attempts to splice a generation across
providers. Apple provider identities are explicit-only: they are excluded from
`auto`, the virtual `mesh` model, and MoA worker pools.

## Runtime packaging and SDK ownership

The package output is a carrier-neutral directory:

```text
meshllm-apple-runtime-darwin-arm64/
├── provider-runtime.json
├── README.md
├── Resources/background-inference.entitlements
└── bin/mesh-apple-runtime
```

The manifest declares platform and minimum OS, runtime protocol version,
entrypoint, `apple/system`, features, file hash, build provenance, and signing
metadata. The archive is independent of a particular SDK. QA copies the exact
same signed binary and verifies its hash, signature, availability, and provider
identity in these representative layouts:

| Carrier | Representative location | Result |
|---|---|---|
| CLI | `runtimes/apple/` | pass |
| Swift | macOS-only SwiftPM resource target | pass |
| Node.js | `@mesh-llm/apple-runtime-darwin-arm64` outside Electron ASAR | pass |
| JVM | `meshllm-apple-runtime-macos-arm64` resource JAR | pass |

This establishes the intended SDK contract:

- the host/runtime protocol is the compatibility surface;
- each SDK resolver finds or installs the same platform runtime package;
- SDKs do not bind Foundation Models directly or fork prompt/tool semantics;
- non-macOS SDK builds omit the package and continue using remote/mesh models;
- versioning, checksums, signing, and updates belong to the runtime package;
- the Swift SDK may additionally embed the provider library in an app, but its
  behavior must remain protocol-conformant with the process form.

The CLI release composer now embeds that artifact at
`provider-runtimes/apple/<runtime-id>`, records its tree and manifest digests in
`product-manifest.json`, and preserves it through Unix installation. The host's
adjacent-product discovery requires no provider bundle or index override.

`scripts/package-sdk-provider-runtime.sh` copies one already-signed artifact
into all three platform-gated package-resource layouts. It never rebuilds the
sidecar per SDK, and non-macOS npm, Maven, iOS, and Android artifacts do not
carry the macOS executable.
Swift, Node/Electron, and Kotlin/JVM then enter the same Rust `ProviderHost`
lifecycle through the UniFFI or N-API boundary and expose the same loopback
OpenAI-compatible URL. The accelerator-entitled process remains
`mesh-apple-runtime`; the host-language process supervises it but does not call
Foundation Models directly.

`just apple::sdk-carriers` starts each real language carrier and runs the same
REST assertions for model listing, the rolling and versioned IDs, buffered and
streaming completions, tools, usage, disconnect cancellation, slot release,
and structured errors. Final npm/JAR/XCFramework release publication must
consume the same notarized artifact used by CLI product composition.

## Validation evidence

All repository commands use `just`:

```bash
just apple::build
just apple::test
just apple::live
just apple::mesh
just apple::private-mesh
just apple::product 0.72.1 target/apple-runtime/product
just apple::product-qa 0.72.1 target/apple-runtime/product
just apple::rust-sdk
just apple::sdk-carriers
MESH_APPLE_RUNTIME_CODESIGN_IDENTITY="Mesh-LLM Local Codesign" \
  just apple::rest
MESH_APPLE_RUNTIME_CODESIGN_IDENTITY="Mesh-LLM Local Codesign" \
  just apple::carriers
MESH_APPLE_RUNTIME_CODESIGN_IDENTITY="Mesh-LLM Local Codesign" \
  just apple::launchd
MESH_APPLE_RUNTIME_CODESIGN_IDENTITY="Mesh-LLM Local Codesign" \
  just apple::instruments
MESH_APPLE_RUNTIME_CODESIGN_IDENTITY="Mesh-LLM Local Codesign" \
  just apple::orphan
```

Observed live results:

| Probe | Result |
|---|---|
| Availability and capabilities | pass |
| Prewarm | pass |
| Streaming generation | pass |
| Usage accounting | pass |
| Cancellation | pass |
| Loopback REST model listing/completion/SSE | pass |
| REST tool execution | pass |
| REST client-disconnect cancellation | pass |
| MeshLLM `/v1` completion, SSE, tool, and cancellation | pass |
| Two-node private mesh provider gossip and completion | pass; 2 replicas / 2 total slots |
| Version-resolved `apple/system@27.0` completion | pass |
| Management API provider PID, health, port, and context | pass |
| Unexpected provider exit and supervised restart | pass |
| MeshLLM shutdown and provider-child cleanup | pass |
| Guided-generation API | pass, with quality caveat below |
| Tool invocation | pass; exactly one recorded fixture lookup |
| Local signing and strict verification | pass |
| Four SDK carrier layouts | pass |
| Per-user launchd background generation | pass |
| Foundation Models Instruments trace | pass |
| Core AI Instruments accelerator trace | pass; ANE load/prediction observed |
| Supervisor SIGKILL/orphan prevention | pass; provider child exited |
| Release-shaped CLI product auto-discovery | pass; no bundle/index override |
| Rust SDK typed provider carrier | pass; completion and tool call |

A representative deterministic text request completed in 2.30 seconds with a
1.81-second time to first token, 73 input tokens, and 25 output tokens. A
launchd request completed in 1.78 seconds with a 1.68-second time to first
token. These are smoke observations, not benchmark claims; cold/warm runs,
prompt lengths, OS builds, and system-model updates need a real benchmark
matrix before product comparisons.

The raw `.trace` files stay under ignored `target/apple-runtime/` because the
Foundation Models template records prompts and responses in unencrypted form.
Public evidence should contain only aggregate timings, token counts, and
accelerator classification.

### Quality and context findings

[Apple TN3193](https://developer.apple.com/documentation/technotes/tn3193-managing-the-on-device-foundation-model-s-context-window)
confirms that each language-model session has a 4,096-token context and that
instructions, prompts, tool schemas/inputs/outputs, `Generable` schemas,
transcript entries, and model responses all consume it. The sidecar therefore
uses Apple's `tokenCount(for:)` APIs before generation. It sums the applicable
prompt, instruction, tool, and schema counts, reserves an explicitly requested
response-token limit, and returns `context_exceeded` before starting work when
the request cannot fit. The framework error remains a final safety net because
tool outputs and an unconstrained response can grow after preflight.

TN3193 also changes the product guidance for longer tasks: do not imply that
request routing creates a larger logical context. Use concise instructions,
small tool schemas (Apple recommends offering at most 3–5 tools), retrieval of
only relevant snippets, or application-level chunk/summarize/compact flows that
start a new Apple language-model session. A strict response-token maximum is a
runaway-generation guard, not a general truncation strategy, because it may
produce malformed or incomplete output.

The guided-generation API returned a valid `@Generable` value and usage, but it
misclassified the simple phrase “choose a warm mesh replica” as unrelated with
zero confidence. API conformance is therefore proven; task quality is not.
Routing policy must gate this provider by evaluated workloads rather than by API
availability alone.

Tool schemas consume a material part of the 4,096-token context. The first
verbose tool probe exceeded context at 4,154 tokens. A narrow schema and concise
instructions reduced the request to 329 input tokens and passed. MeshLLM must
budget tool definitions and history against the advertised context before
selecting this provider.

`toolCallingMode.required` ended without a response in this beta. The passing
probe uses `.allowed` and independently verifies that the tool was invoked
exactly once; production capability claims should reflect observed behavior.

## Signing, launchd, and entitlement boundary

Full Xcode installation and selection are now complete, so SwiftPM, Instruments,
release compilation, signing, and launchd testing are unblocked.

A release binary signed with the local `Mesh-LLM Local Codesign` identity ran
successfully as a transient per-user LaunchAgent with `ProcessType=Background`.
No special entitlement was required for that short foreground-style generation
inside a background process.

The entitlement
`com.apple.developer.background-tasks.continued-processing.inference` is a
different distribution capability. Applying it with the local certificate
produced a valid code signature, but launchd terminated the process before
`main` with `OS_REASON_EXEC`; stdout and stderr were empty. The same binary
without the entitlement passed. This isolates the failure to entitlement
authorization/provisioning.

Packaging therefore fails closed: setting `MESH_APPLE_RUNTIME_ENTITLEMENTS`
also requires
`MESH_APPLE_RUNTIME_ENTITLEMENT_PROVISIONING_VALIDATED=1`. That override is only
appropriate after an Apple-issued signing identity/profile explicitly grants
the entitlement and the resulting artifact passes launchd and sandbox QA.

The remaining release gate is an Apple Developer account/certificate/profile
with the approved capability, not local `sudo`, Xcode selection, or model
availability. If ordinary on-demand provider requests do not require continued
processing, ship without the entitlement and add it only to an app/service form
whose lifecycle requires it.

## Safety, privacy, and product policy

Milestone 0 should remain local/private. Before public-mesh advertisement:

- confirm Apple's applicable license and acceptable-use terms for the intended
  hosted/routed product behavior;
- never present an opaque, auto-updated system model as reproducible;
- map guardrail/refusal errors without bypassing them;
- avoid logging prompts, responses, tool arguments, or Instruments captures;
- advertise only runtime-observed capabilities and availability;
- keep `apple/system` opt-in until quality, latency, and policy gates pass;
- bound requests to the 4,096-token context after instructions, tools, and
  history are included;
- route whole requests only; never treat the system model as a Skippy stage.

## Roadmap

### 0. System-model spike — complete

Prove Foundation Models access, streaming, cancellation, usage, guided output,
tools, packaging, signing, launchd behavior, SDK carriers, and Instruments.

### 1. Local `apple/system` vertical slice — experimental implementation complete

One Mac now serves model listing, buffered chat completion, SSE streaming,
server-executed fixture tools, usage, errors, and client-disconnect cancellation
from the shared sidecar. The Rust supervisor now exposes the provider through
MeshLLM's normal local OpenAI frontend and runtime-process management surface.

### 2. All host-capable macOS SDKs

The shared executable-provider manifest, resolver, verified downloader,
immutable cache contract, Rust host supervisor, CLI product composition,
adjacent-product auto-discovery, and typed Rust SDK carrier configuration are
implemented. Swift, Node/Electron, and Kotlin/JVM now bind the same provider-only
host and are certified by one live protocol suite. Provider-only SDK hosts do
not load Skippy. The release lane requires a Developer ID Application
signature, secure timestamp, accepted notarization, and `spctl` assessment
before composition. Remaining promotion work is release-channel publication
of the one signed artifact plus app-sandbox and quarantine validation in a
signed Swift app and packaged Electron app.

### 3. Private-mesh system-model routing — experimental implementation complete

Private peers now exchange additive provider runtime descriptors and route the
rolling or versioned system-model ID as one whole request. The Apple sidecar
owns a one-request FIFO scheduler and reports capacity, active work, and queue
depth. Routing is load-aware, honors normal session/prefix affinity, retries a
different healthy Mac only before streaming begins, and withdraws a provider
after health or process failure. Older peers safely ignore the new fields; a
new peer treats an older provider's missing load as unknown. The management API,
OpenAI model metadata, and console expose the observed provider generation and
load. Public meshes, `auto`, virtual `mesh`, and MoA do not select this model.

### 4. Core AI model providers and workload certification

Add separately versioned provider kinds for user-supplied Core ML/Core AI model
packages. Build an offline import pipeline for supported SafeTensors families:
conversion, quantization/palettization, compilation, validation, and signed
artifact caching. Do not imply that Foundation Models accepts arbitrary
SafeTensors or performs that import automatically. Device-side conversion is a
packaging workflow, not a request-time inference feature.

Start with a narrow certified architecture/quantization matrix. Compare native
whole-model execution with GGUF/Skippy using real latency, memory, energy, and
quality data. Only add distributed placement semantics where the chosen Core AI
runtime actually exposes useful partitioning; whole-model routing remains the
default. Benchmark cold/warm latency, throughput, energy, memory, tool overhead,
structured-output quality, guardrails, and model-update drift before automatic
selection.

## Promotion gates

Milestone 1 should not start as a public default until all of these are true:

- runtime protocol and mixed-version behavior are specified;
- cancellation, process death, backpressure, and restart semantics are tested;
- system-model identity/update drift has a cache and observability policy;
- workload quality and context-budget tests pass;
- packaging works from real CLI/npm/JAR/Swift release artifacts;
- sandbox, signing, notarization, quarantine, and required entitlements pass;
- privacy-safe telemetry is reviewed;
- Apple licensing/acceptable-use review approves the deployment mode.
