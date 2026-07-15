# API Reference

Mesh provides local management APIs and an OpenAI-compatible inference endpoint.

## OpenAI-compatible endpoint

```text
http://localhost:9337/v1
```

Use this endpoint for OpenAI-compatible clients, SDKs, and agent tools.

## Management API

The local management API is served from:

```text
http://localhost:3131
```

Common management surfaces include status, discovery, and mesh join flows.

### Owner-control inventory scan

The management API exposes a loopback-only facade for scanning one explicitly
targeted, owner-attested node:

```http
POST /api/runtime/control/scan-refresh
Content-Type: application/json

{"endpoint":"<control-endpoint>"}
```

Example response:

```json
{
  "target_node_id": "<hex-node-id>",
  "disposition": "executed",
  "inventory": [
    {
      "canonical_model_ref": "owner/model:Q4_K_M",
      "display_name": "model",
      "total_size_bytes": 4294967296
    }
  ]
}
```

Inventory entries are sorted by `canonical_model_ref`. `disposition` is
`executed` when this request ran the scan and `coalesced` when it joined an
in-progress scan. When an older owner-control server returns only the legacy
snapshot, the request still succeeds with `disposition` and `inventory` set to
`null`.

The endpoint token must be obtained locally from the target node's
`GET /api/runtime/control-bootstrap` response and transferred out of band. It
is never inferred from public status, gossip, discovery, or a peer ID. The
controlling node must use a key for the same owner. Non-loopback management API
callers are rejected, and rich inventory results remain on the private
owner-control response path.

The retained `POST /api/runtime/control/refresh-inventory` route is a
compatibility facade with the legacy snapshot-only response shape.

## CLI

For command-line usage, see the [CLI reference](/docs/pages/CLI/).
