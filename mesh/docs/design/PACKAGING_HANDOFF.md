# Packaging handoff (draft)

**Status: proposed, not in force.** The dispatch in `release.yml` is off.
`Mesh-LLM/mesh-packaging` has not agreed to or implemented this. Nothing here
has run against it.

## Why the old contract no longer fits

Until 0.80.0 a MeshLLM release carried one product archive per backend and its
own `native-runtimes.json`. mesh-packaging mirrored those archives into Docker
images, Homebrew and npm. The dispatch named the release by tag, tag commit
and the SHA-256 of that `native-runtimes.json`, and mesh-packaging answered
with a receipt holding the same three values.

A release now carries one host product per platform. The GPU runtimes and
their catalog belong to the Skippy release that MeshLLM pins. One digest can
no longer name everything a package is built from, so the contract names both
releases.

## What MeshLLM hands over

MeshLLM does not build GPU products. mesh-packaging composes each of its
products from the one host archive for the platform and a runtime it has
verified against the Skippy catalog.

The dispatch (`repository_dispatch`, type `mesh-llm-release`) carries:

| Field | Meaning |
|---|---|
| `repository`, `ref`, `version` | The MeshLLM repository, release tag and version. |
| `expected_mesh_sha` | The commit the tag points at. |
| `expected_mesh_assets_sha256` | SHA-256 of the release asset `release-assets.sha256`: one `<sha256>  <name>` line per published asset, sorted by name. |
| `skippy.repository`, `skippy.release` | The Skippy release this MeshLLM was built against. |
| `skippy.expected_catalog_sha256` | SHA-256 of that release's `native-runtimes.json`. |
| `correlation_id` | `mesh-<run id>-<attempt>`, which becomes the packaging run's title. |
| `dry_run`, `publish_images`, `publish_release_assets`, `publish_npm` | As before. |

`expected_manifest_sha256` is retired, not reused. It meant "MeshLLM's
catalog", and giving the same name a new meaning would let an old consumer
accept a digest of the wrong file.

MeshLLM does not publish `release-assets.sha256` yet. It has to before the
dispatch is turned on.

## What mesh-packaging must check

- The tag resolves to `expected_mesh_sha`.
- `release-assets.sha256` has the expected digest, and every MeshLLM asset it
  downloads matches its line.
- The Skippy catalog it downloads has the expected digest, and every runtime
  it packages matches its catalog entry.
- Each host archive holds `host-imports.json` whose `binary_sha256` is the
  SHA-256 of the host in that archive.

## The receipt

Schema `mesh-packaging-readiness-v2`, uploaded as the artifact
`packaging-readiness` holding only `packaging-readiness.json`:

```json
{
  "schema": "mesh-packaging-readiness-v2",
  "status": "success",
  "correlation_id": "mesh-123-1",
  "packaging_run_id": 456,
  "packaging_run_attempt": 1,
  "upstream": {"repository": "Mesh-LLM/mesh-llm", "ref": "v1.2.3", "sha": "<40 hex>", "assets_sha256": "<64 hex>"},
  "skippy": {"repository": "Mesh-LLM/skippy", "release": "v1.2.3", "catalog_sha256": "<64 hex>"},
  "requested": {"publish_images": "true", "publish_release_assets": "true", "publish_npm": "true"},
  "results": {"plan": "success", "...": "success"}
}
```

`upstream` and `skippy` are the values mesh-packaging verified, not an echo of
the dispatch.

## How MeshLLM waits

`scripts/wait-packaging-receipt.sh` does what the removed
`release packaging-receipt` tool did, with the digest check split in two:

- It finds the one `repository_dispatch` run of `images-release.yml` titled
  `Packaging · <correlation_id>` created since the release run started. Two
  such runs is an error.
- A run that concludes anything but `success`, or reports an unknown status,
  is an error. So is any failed GitHub request.
- It reads the receipt of the current attempt and requires every field above
  to match, all three publications to have been requested, and every result
  to be `success`. A `v1` receipt is refused.
- It re-reads the run afterwards and fails if it was re-run meanwhile.
- It gives up after two hours.

`scripts/tests/wait-packaging-receipt-test.sh` covers these with a stub `gh`.
