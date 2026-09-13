# openai-frontend

Reusable OpenAI-compatible HTTP frontend primitives for mesh and staged runtime
entry points.

This crate owns the public API shapes and route machinery that should not be
duplicated inside `skippy-server`. Stage server code should provide a thin
backend adapter that implements the frontend trait, while this crate handles
request/response JSON, OpenAI-style errors, model discovery, generation,
embedding, rerank, audio, and streaming Server-Sent Events framing. Mesh uses this as the single OpenAI
surface for embedded single-stage and stage-split serving.

Mesh-local compatibility wrappers should stay thin. Request normalization,
Responses API translation, stream-chunk parsing, stream usage conversion, and
upstream error mapping are owned here so the host binary and staged runtimes do
not grow separate OpenAI compatibility layers.

The compatibility target is the mesh OpenAI surface, not llama-server process
compatibility. Request fields such as tools, structured-output shape,
logprobs, and `/v1/responses` are parsed and normalized here; backend support
is advertised or rejected explicitly by the runtime adapter.

For the concrete benchy command and contract, see
[`docs/skippy/LLAMA_BENCHY.md`](../../docs/skippy/LLAMA_BENCHY.md).

## Supported API Surface

| Surface | Status | Notes |
|---|---|---|
| `GET /v1/models` | Supported | Returns backend-provided model objects with opaque ids such as `org/repo:Q4_K_M`. |
| `POST /v1/chat/completions` | Supported | Handles streaming and non-streaming response shapes. |
| `POST /v1/completions` | Supported | Handles streaming and non-streaming response shapes. |
| `POST /v1/responses` | Supported | Adapts OpenAI responses requests onto chat/completion backend calls and preserves response metadata where possible. |
| `POST /v1/embeddings` | Supported | String and token inputs, float or base64 vectors, usage accounting. Runtime support is model-gated. |
| `POST /v1/rerank` | Supported | Cross-encoder query/document scoring with optional document return. Runtime support is model-gated. |
| `POST /v1/audio/speech` | Supported | Binary response with format-specific content type. The native backend currently produces WAV or PCM and accepts only `voice: "default"`; speaker selection fails with a structured unsupported error. |
| `POST /v1/audio/transcriptions` | Supported | Bounded multipart audio upload with JSON or text response. Runtime support is model-gated. |
| `POST /v1/audio/translations` | Supported | Bounded multipart audio upload translated to English, with JSON or text response. Runtime support is model-gated. |
| `GET /health` / `GET /healthz` | Supported | Lightweight liveness probes for hosts and CI smoke tests. |
| `GET /readyz` | Supported | Backend readiness probe that verifies model discovery through `OpenAiBackend::models`. |
| Server-Sent Events | Supported | Emits OpenAI-style JSON chunks and `[DONE]`. |
| `model` | Supported | Opaque exact-match id; Mesh-style refs such as `org/repo:Q4_K_M` pass through without frontend parsing. |
| `messages` | Supported | String and text-part content are parsed; stage backend applies the model chat template through llama.cpp `llama-common`. |
| `prompt` | Supported | String and string-array prompts are accepted; token prompts parse but are rejected until a backend can honor token IDs directly. |
| `max_tokens` / `max_completion_tokens` | Parsed | Backend decides enforcement. |
| `stop` | Parsed | Backend decides enforcement. |
| `n` / `best_of` | Parsed | Single-choice requests are accepted; multi-choice generation is rejected until the response/backend path supports it. |
| `temperature`, `top_p`, `seed` | Supported | Stage backend passes supported sampling controls to llama.cpp sampling. |
| penalties / `logit_bias` | Supported | Presence/frequency/repeat penalties and token-id logit bias are passed to llama.cpp. |
| `logprobs` / `top_logprobs` | Frontend-compatible | Parsed, validated, and preserved. Backend decides whether logits/probability data can be produced. |
| `tools` / `tool_choice` | Frontend-compatible | Parsed, validated, and preserved. Backend decides whether tool-call generation is available. |
| `response_format` | Frontend-compatible | Text and structured-output shapes are parsed and preserved. Backend decides whether constrained decoding is available. |
| OpenAI-style error envelope | Supported | Includes strict `type`, `param`, and `code` fields. |
| OpenAI-style HTTP fallbacks | Supported | Unknown routes, unsupported methods, invalid JSON, and oversized JSON return the shared error envelope. |
| Request body limit | Supported | Configurable via `OpenAiFrontendConfig`; defaults to 4 MiB. |
| Request IDs | Supported | Reuses only a valid UUID `x-request-id`; missing or invalid values are replaced with a generated UUID. Every response returns the canonical hyphenated UUID and the frontend emits a tracing event with method, URI, status, and request ID. |
| Client nonce | Supported | Accepts `x-capsule-client-nonce` only when it is exactly one valid UUIDv4; a missing, invalid, non-UUIDv4, or duplicated value is replaced with a freshly minted UUIDv4. When this frontend mints the value it stamps `x-capsule-nonce-origin: frontend`; a forwarded (client-supplied) nonce carries no origin marker, and any inbound `x-capsule-nonce-origin` is always stripped so a caller cannot forge it. Both headers are echoed on covered responses: the axum router and, via the host runtime's forwarding rebuild, the public `:9337` proxy JSON/SSE paths (including the pipeline/MoA strong-model path and remapped upstream error responses). Locally synthesized error responses (e.g. no-target `503`s and `/v1/models` listings) do not yet carry the headers; threading the request nonce onto those senders is a cross-cutting signature change tracked as a follow-up. The origin marker asserts only that *this* frontend minted the value, not that it is the original ingress for a remote-routed request. |
| Backend timeout | Supported | Configurable via `OpenAiFrontendConfig` or the `MESH_OPENAI_BACKEND_TIMEOUT_SECS` environment variable; defaults to 600 seconds (`0` disables it) and maps timeouts to OpenAI-shaped 504 errors. |
| Agent session header | Supported | Set `MESH_AGENT_SESSION_HEADER` to accept a trusted upstream header as the stable agent-session identity. |
| Vision input | Supported | Preserved through chat/Responses content parts and executed by projector-backed runtimes. |
| Non-chat staging | Fail closed | Embedding, rerank, encoder-decoder, and speech-synthesis models currently require an unsplit full-model runtime. |

## Shape

```mermaid
flowchart TB
    C["OpenAI-compatible client<br/>generation, embeddings, audio"] --> R["openai-frontend<br/>Axum routes"]
    R --> Parse["request parsing<br/>validation<br/>normalization<br/>OpenAI errors"]
    Parse --> B["OpenAiBackend implementation"]
    B --> Local["embedded single-stage<br/>skippy runtime"]
    B --> Chain["mesh stage-0 route<br/>skippy-stage/2 chain"]
    Local --> Resp["OpenAI response JSON"]
    Chain --> Resp
    Resp --> R
    R --> C
```

The backend boundary below is a partial generation example. The complete
[`OpenAiBackend` trait](src/backend.rs) also defines `embeddings`, `rerank`,
`audio_speech`, `audio_transcription`, and `audio_translation`; override their
default unsupported responses to serve the corresponding non-chat endpoints.

```rust
#[async_trait]
pub trait OpenAiBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>>;
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionResponse>;
    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream>;
    async fn completion(
        &self,
        request: CompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionResponse>;
    async fn completion_stream(
        &self,
        request: CompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream>;
}
```

## Model Identity

`openai-frontend` treats model ids as opaque OpenAI-facing strings. For
Mesh-owned routing, the expected user-facing form is a Hugging Face coordinate
plus artifact selector, for example `org/repo:Q4_K_M`. The suffix is an artifact
selector, not a stage-server topology or serving-backend variant.

Backends should advertise the exact ids they accept through `/v1/models` and
perform exact string matching on requests. Resolution to a concrete Hugging Face
revision, GGUF file, split-shard distribution, local runtime, or staged binary
chain remains backend-owned.

## Backend Responsibility Matrix

| Area | Frontend responsibility | Backend responsibility | Status |
|---|---|---|---|
| `/v1/models` | Route shape and model-object serialization | Advertise exact full model refs accepted by the runtime | Supported |
| `/v1/chat/completions` | Parse common and advanced fields, stream/non-stream envelopes | Tokenization, sampling, stop handling, usage, feature execution | Supported with backend feature guards |
| `/v1/completions` | Prompt parsing and response envelopes | Token prompts, sampling, stop handling, usage | Supported with backend feature guards |
| `/v1/responses` | Translate request/response shapes onto the backend contract | Execute the resulting chat/completion request | Supported |
| `/v1/embeddings` | Parse OpenAI input/encoding shapes and serialize vectors | Tokenize or accept token IDs, pool, normalize, and report usage | Supported for compatible local full models |
| `/v1/rerank` | Parse query/documents and serialize ranked results | Execute classifier scoring and report usage | Supported for compatible local full models |
| `/v1/audio/*` | Parse JSON or bounded multipart bodies; return JSON, text, or binary media | Execute TTS or projector-backed speech recognition | Supported for compatible local full models |
| HTTP operations | Health/readiness, fallbacks, payload limits, content-type handling | Model readiness and backend timeouts | Supported |
| Streaming | SSE chunks, `[DONE]`, cancellation context | Produce deltas, usage, and optional logprob/tool metadata | Supported with backend feature guards |
| Chat templates | Preserve OpenAI message shape | Apply model-aware chat templates through the skippy ABI | Backend-owned |
| Sampling | Parse OpenAI sampling fields | Apply supported llama sampling controls and reject unsupported knobs | Backend-owned |
| Stop handling | Preserve stop fields | Hold back streamed text enough to avoid leaking stop strings | Backend-owned |
| Context limits | Carry requested limits | Validate prompt plus generation against `ctx_size` | Backend-owned |
| Concurrency | Request timeout and cancellation | Runtime slots, batching, lane pools, and backpressure | Backend-owned |
| Cache behavior | Request IDs for correlation | Prefix/state cache policy and telemetry | Backend-owned |
| Errors | Shared OpenAI error envelope | Return precise unsupported/runtime errors | Supported |
| Usage accounting | Response field shape | Prompt/completion/total token counts | Backend-owned |
| Multi-choice generation | Parse `n` / `best_of` | Generate or reject multi-choice requests | Backend-owned |
| Logprobs | Parse and preserve request/response shape | Expose logits/probabilities | Frontend ready; backend-gated |
| Tools/function calling | Parse and preserve tool schemas and tool-call response shape | Generate tool calls | Frontend ready; backend-gated |
| JSON schema/grammar | Parse and preserve `response_format` | Constrained decoding | Frontend ready; backend-gated |
| Non-chat workload gates | Preserve request/response contracts and structured errors | Probe the native model class and reject unsupported stage shapes | Supported |
| Metrics | Request IDs and tracing context | Stage/OpenAI telemetry emitted to `metrics-server` | Supported |

## Stage-Server Integration

`skippy-server` and mesh use this crate by implementing `OpenAiBackend` for a
small adapter:

- local text/runtime backend for single-stage smoke tests
- staged chain backend that connects to the first `serve-binary` endpoint

That keeps `serve-openai` and the embedded mesh path thin: parse or build the
runtime config, construct the backend, pass it to `openai_frontend::router`,
and serve the Axum app.

See [`docs/NON_CHAT_MODELS.md`](../../docs/NON_CHAT_MODELS.md) for endpoint
examples, execution boundaries, and certification policy.
