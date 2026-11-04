# Document OCR plugin contract

An OCR provider registers an inference endpoint with protocol `document_ocr_v1` and serves `GET /v1/models` plus `POST /mesh/v1/documents/ocr` on its local HTTP address. The host probes `/v1/models` to discover available model IDs, advertises them as `DocumentOcr` workloads, and routes OCR requests only to endpoints carrying this protocol. A provider can be written in any language; the [ONNX plugin](https://github.com/Mesh-LLM/onnx) is the first implementation, using Indra in Rust with `ort`.

The task lives under `/mesh/v1/` because `/v1/` is reserved for OpenAI-shaped endpoints: an OpenAI client or SDK pointed at Mesh finds the paths it expects, and a Mesh task can never collide with a route OpenAI adds later. The path is a **two-sided contract** — the host admits the request by matching this path to a workload class, then forwards that same path to the provider — so host and provider must agree on the literal string. Changing it on one side is a runtime `404`, not a compile error.

## Request and response

`POST /mesh/v1/documents/ocr` accepts `multipart/form-data` with a required `file` field and a `model` field selecting an advertised ID. The mesh reads `model` from the multipart body to choose a host, then forwards the original request body to the selected provider. The current host limit is 64 MiB for the file plus multipart overhead. The provider owns document decoding and processing errors.

The response representation is negotiated with the standard `Accept` header, not a body field: `application/json` (default when `Accept` is absent or `*/*`), `text/plain`, or `text/markdown`. A provider that cannot produce an acceptable type answers `406` and sets `Vary: Accept`, and the host relays that status unchanged — it is a request the caller can fix, not a provider failure. Task options such as `max_pages` stay in the body, where they belong: they change what is recognized, not how it is encoded.

The JSON response has a `model` ID and a `pages` array. Each page has `page_index`, extracted `text`, `markdown`, `lines` with text and boxes, and ordered layout `blocks`. The provider may add fields such as page rotation. Plain text and Markdown formats return joined page content. Errors use an `error.message` object.

## Sibling task contracts

Four task slots share a protocol label, a literal path, and multipart `file` plus `model` fields. A provider implements only the ones it serves and registers each with its own protocol:

| Task | Protocol | Path | Workload class |
|---|---|---|---|
| Document OCR | `document_ocr_v1` | `POST /mesh/v1/documents/ocr` | `DocumentOcr` |
| Transcription | `transcription_v1` | `POST /v1/audio/transcriptions` | `SpeechRecognition` |
| Image/text similarity | `image_text_similarity_v1` | `POST /mesh/v1/images/similarity` | `ImageTextSimilarity` |
| Relative depth | `depth_estimation_v1` | `POST /mesh/v1/images/depth` | `DepthEstimation` |

Transcription keeps the OpenAI path because that endpoint already exists and an OpenAI client should find it; the other three have no OpenAI equivalent and sit under `/mesh/v1/`. The [ONNX plugin](https://github.com/Mesh-LLM/onnx) documents each one with a configuration example, an HTTP request and its use cases.

Mesh-specific tasks negotiate representations with `Accept`. Transcription also supports OpenAI's multipart `response_format` field, which takes precedence over `Accept`. The ONNX provider supports `json` and `text`; it rejects timing-dependent formats such as `srt`, `vtt` and `verbose_json` because its decoder does not supply timestamps.

## Routing

The host includes healthy OCR plugin models in local discovery and peer announcements with workload class `DocumentOcr`. An ingress node filters candidate targets by that class before choosing a local or remote node. The selected node then chooses an inference endpoint by both model ID and protocol. This keeps an OCR provider and a native or chat model with the same ID from being used interchangeably. Remote document bytes travel through the existing peer HTTP tunnel; an incoming peer request executes locally and is never forwarded to a third node.

The workload enum adds a protobuf value without changing existing values. Older peers can ignore the new value, but they do not know how to route the OCR endpoint. Deploy an OCR-aware ingress node for document requests.

## Provider registration

The Rust plugin SDK can register the endpoint with `mesh_llm_plugin::inference::provider(id, local_url).protocol("document_ocr_v1")` and advertise `endpoint:inference/document_ocr_v1`. The HTTP provider must be ready before registration. Its inventory may advertise a configured model as `loadable` while its engine is unloaded; `ready` and `busy` models also remain routable. The provider loads on demand when a request arrives.

[Managed provider models](managed-models.md) use Mesh configuration and the host's shared asset acquisition service. The provider owns model sessions, preprocessing, inference and residency; it also supplies any PDF rendering dependency. These stay outside Skippy.
