# Experimental Apple runtime

Status: Milestone 0 and an experimental local REST vertical slice completed on
2026-08-12; production integration remains gated. Tracking issue:
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

The experimental sidecar exposes a loopback OpenAI-shaped REST surface for local
validation. It does not yet connect that surface to MeshLLM's public OpenAI
frontend or gossip `apple/system` to peers. It intentionally proves the native,
REST, and packaging boundaries first.

## Why this is valuable to Apple silicon users

The system-model provider gives MeshLLM a zero-download, zero-model-management
local inference option on eligible Macs. Apple owns model delivery, device
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
| Stable model identity | not exposed by the public arm64 beta module |

The runtime reports the model as `apple/system` with variant
`system-default-unversioned`. It does not invent a version from the OS build.
Production cache identity and reproducibility policy must account for the fact
that Apple can update the system model independently of MeshLLM.

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
- session prewarming;
- snapshot-to-delta streaming;
- elapsed time and time to first token;
- input, cached-input, output, and reasoning token accounting;
- guided `@Generable` output;
- a deterministic `Tool` invocation with an auditable recorder;
- cancellation, including Foundation Models' empty-stream cancellation edge;
- normalized retryable/non-retryable error codes.

The sidecar provides a loopback diagnostic REST listener with `/health`,
`/v1/models`, and `/v1/chat/completions`. It supports buffered completions, SSE
streaming, usage, disconnect cancellation, and the deterministic fixture-tool
probe. This is experimental transport evidence, not yet the production MeshLLM
frontend. The production host adapter should supervise the same executable over
a versioned local protocol and translate it into the existing MeshLLM inference
interface. This keeps Foundation Models and Swift out of the backend-neutral
Rust host, and gives all SDKs one implementation instead of one implementation
per language.

For process hosting, the runtime also accepts a validated `--parent-pid` and
checks parent liveness every 50 ms. A QA supervisor is killed with `SIGKILL`
mid-generation; the runtime must exit rather than becoming an orphan. The Rust
provider manager must pass its real PID and retain instance-scoped child
tracking so `mesh-llm stop`, host crashes, and orphan reaping remain correct.

## Runtime packaging and SDK ownership

The package output is a carrier-neutral directory:

```text
meshllm-apple-runtime-darwin-arm64/
├── manifest.json
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
| Swift | `MeshLLMAppleRuntime.bundle/Contents/Resources/runtime/` | pass |
| Node.js | `@mesh-llm/apple-runtime-darwin-arm64/runtime/` | pass |
| JVM | `ai/meshllm/apple-runtime/macos-arm64/runtime/` | pass |

This establishes the intended SDK contract:

- the host/runtime protocol is the compatibility surface;
- each SDK resolver finds or installs the same platform runtime package;
- SDKs do not bind Foundation Models directly or fork prompt/tool semantics;
- non-macOS SDK builds omit the package and continue using remote/mesh models;
- versioning, checksums, signing, and updates belong to the runtime package;
- the Swift SDK may additionally embed the provider library in an app, but its
  behavior must remain protocol-conformant with the process form.

The current QA proves the carrier layouts, not final npm/JAR/XCFramework
publication. Those release jobs should consume one already-signed runtime artifact rather
than rebuild it independently.

## Validation evidence

All repository commands use `just`:

```bash
just apple::build
just apple::test
just apple::live
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
| Guided-generation API | pass, with quality caveat below |
| Tool invocation | pass; exactly one recorded fixture lookup |
| Local signing and strict verification | pass |
| Four SDK carrier layouts | pass |
| Per-user launchd background generation | pass |
| Foundation Models Instruments trace | pass |
| Core AI Instruments accelerator trace | pass; ANE load/prediction observed |
| Supervisor SIGKILL/orphan prevention | pass; provider child exited |

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
from the shared sidecar. Production completion still requires the Rust
supervisor and conformance mapping into MeshLLM's public OpenAI frontend.

### 2. All host-capable macOS SDKs

Publish one signed macOS arm64 runtime artifact; teach CLI/Swift/npm/JVM
resolvers to install, locate, launch, monitor, upgrade, and terminate it; add
checksum policy; validate Swift app embedding, sandboxing, notarization,
quarantine, and launchd/service lifecycles; then certify all SDKs against one
protocol test suite.

### 3. Private-mesh system-model routing

Add the Rust runtime supervisor, dynamic availability and withdrawal, load and
queue reporting, whole-model placement, pre-stream failover, request affinity,
mixed-version behavior, and two-node validation. Keep `apple/system` explicit
and private-mesh-only.

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
