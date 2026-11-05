---
title: Mesh Model Discovery
description: Discover all mesh-advertised models through native SDKs or the Mesh HTTP catalog
---

# Mesh model discovery

Mesh advertises all supported model workloads. Use the native SDK model catalog
or the Mesh HTTP API to discover LLMs, speech recognition/synthesis, embeddings,
rerankers, decisions and document OCR. The catalog contains advertisements seen
by this node in its joined mesh, including its local providers. It is not the
download catalog or a global registry. Availability is an observation, not a
reservation for a later inference call.

The OpenAI and Anthropic compatible model endpoints expose only available,
confirmed LLM offerings and eligible virtual LLM routes. Multimodal LLMs remain
included. Models with missing or unknown workload metadata remain visible in
Mesh discovery but are excluded from compatibility model lists. Older peers
need to advertise a known LLM workload to appear in those lists. Existing
non-LLM inference routes, including `/mesh/v1/documents/ocr`, are preserved.

## HTTP catalog

The inference listener (default port `9337`) serves:

```http
GET /mesh/v1/models
GET /mesh/v1/models?workload_class=document_ocr&available=true
GET /mesh/v1/models/lookup?id=org%2Fmodel%40revision%23profile
```

The lookup uses the exact public serving ID. URL-encode the query value once;
native SDK callers pass the unchanged ID. Distinct quantizations and profiles
retain their serving IDs. Multiple providers of one ID remain under that entry.

Illustrative page:

```json
{
  "data": [{
    "id": "ocr-model",
    "display_name": "ocr-model",
    "kind": "model",
    "workload_class": "document_ocr",
    "availability": "available",
    "operations": ["document_ocr"],
    "providers": [{
      "provider_id": "node-a",
      "local": false,
      "workload_class": "document_ocr",
      "available": true,
      "operations": ["document_ocr"]
    }]
  }],
  "next_cursor": null
}
```

Lookup returns the model object itself. An absent ID returns `404`; invalid
IDs, filters or cursors return `400`. A stopped/unavailable discovery service
returns `503`. Filters combine with AND:

| Parameter | Values |
|---|---|
| `workload_class` | `causal_generation`, `embedding`, `rerank`, `encoder_decoder`, `speech_synthesis`, `decision`, `speech_recognition`, `document_ocr`, `unknown` |
| `kind` | `llm`, `model`, `virtual` |
| `available` | `true` or `false` |
| `limit` | 1–500; default 100 |
| `cursor` | Opaque continuation token returned by a previous page |

Keep filters unchanged when passing `next_cursor`; a changed matching snapshot
expires the cursor and requires restarting the list. A final page has
`next_cursor: null`. SDKs expose the same bounded page and continuation token.

## Metadata and provider observations

Model entries include `id`, `display_name`, `kind`, `availability`, `operations`
and `providers`. Optional `metadata` contains common static facts such as
architecture, parameter count, quantization and native context capacity.
`workload_class` is absent at model level when providers disagree. The operations
list describes what at least one provider advertises, not every replica.

Each provider retains its own optional `workload_class`, `capabilities`,
`metadata`, `runtime`, `thinking` and `payment`, plus plugin/endpoint identifiers
where available. Missing facts mean unknown. In particular:

- `runtime.context_length` is the provider's serving limit; native capacity is
  not a guarantee of that serving limit.
- Thinking controls observed locally do not describe remote replicas.
- Payment offers describe advertised prices and never authorize payment or
  constitute binding quotes.
- An OCR provider and an LLM provider sharing an ID retain separate workload
  evidence. Their operations are not interchangeable.

## Compatibility model endpoints

`GET /v1/models` (also `/models`) defaults to the OpenAI shape:

```json
{"object":"list","data":[{"id":"llm-model","object":"model","created":0,"owned_by":"mesh-llm"}]}
```

With the `anthropic-version` header it returns the Anthropic shape:

```json
{
  "data": [{"id":"llm-model","type":"model","display_name":"llm-model","created_at":"1970-01-01T00:00:00Z","lifecycle":"active"}],
  "has_more": false,
  "first_id": "llm-model",
  "last_id": "llm-model"
}
```

Epoch timestamps explicitly represent unknown release dates. Anthropic supports
`limit`, `before_id` or `after_id`; do not combine both cursor directions.
`GET /v1/models/{encoded-id}` uses the same eligibility rule. A non-LLM model
can exist in Mesh discovery while returning `404` from compatibility lookup.

Mesh-specific `metadata`, capability arrays/status fields, `thinking`,
`virtual_model` and `payment` have moved out of compatibility responses. Read
them through this catalog. `display_name` is standard for Anthropic but is not
part of the OpenAI projection. Internal plugin provider `/v1/models` probes
retain their existing provider contract.

## Native SDK catalog

All five SDKs query the embedded node directly for this catalog, without
fetching the HTTP model list. Existing compatibility inference model-list
methods remain LLM-only. Language-specific guides show equivalent filters,
pages, exact lookup and provider metadata:

- [Rust](/docs/pages/sdk-rust/)
- [Node.js](/docs/pages/sdk-node/)
- [Python](/docs/pages/sdk-python/)
- [Swift](/docs/pages/sdk-swift/)
- [Kotlin](/docs/pages/sdk-kotlin/)

Update every SDK and its generated native bridge together. Python is maintained
in the separate `Mesh-LLM/mesh-llm-python-sdk` repository. Use matching SDK/native
artifacts; this source change alone does not publish new packages.
