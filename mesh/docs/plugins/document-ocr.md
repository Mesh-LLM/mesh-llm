# Document OCR plugin contract

An OCR provider registers an inference endpoint with protocol `document_ocr_v1` and serves `GET /v1/models` plus `POST /v1/documents/ocr` on its local HTTP address. The host probes `/v1/models` to discover ready model IDs, advertises them as `DocumentOcr` workloads, and routes OCR requests only to endpoints carrying this protocol. A provider can be written in any language; the [ONNX plugin](https://github.com/Mesh-LLM/onnx) is the first implementation, using Indra in Rust with `ort`.

## Request and response

`POST /v1/documents/ocr` accepts `multipart/form-data` with a required `file` field and a `model` field selecting an advertised ID. Providers may accept `response_format` (`json`, `text`, or `markdown`) and `max_pages`. The mesh reads `model` from the multipart body to choose a host, then forwards the original request body to the selected provider. The current host limit is 64 MiB for the file plus multipart overhead. The provider owns document decoding and processing errors.

The JSON response has a `model` ID and a `pages` array. Each page has `page_index`, extracted `text`, `markdown`, `lines` with text and boxes, and ordered layout `blocks`. The provider may add fields such as page rotation. Plain text and Markdown formats return joined page content. Errors use an `error.message` object.

## Routing

The host includes healthy OCR plugin models in local discovery and peer announcements with workload class `DocumentOcr`. An ingress node filters candidate targets by that class before choosing a local or remote node. The selected node then chooses an inference endpoint by both model ID and protocol. This keeps an OCR provider and a native or chat model with the same ID from being used interchangeably. Remote document bytes travel through the existing peer HTTP tunnel; an incoming peer request executes locally and is never forwarded to a third node.

The workload enum adds a protobuf value without changing existing values. Older peers can ignore the new value, but they do not know how to route the OCR endpoint. Deploy an OCR-aware ingress node for document requests.

## Provider registration

The Rust plugin SDK can register the endpoint with `mesh_llm_plugin::inference::provider(id, local_url).protocol("document_ocr_v1")` and advertise `endpoint:inference/document_ocr_v1`. The HTTP provider must be ready before registration: the host probes `/v1/models`, and only healthy returned model IDs enter routing. The model files and any PDF rendering dependency are owned by the provider package, outside Skippy and the mesh host.
