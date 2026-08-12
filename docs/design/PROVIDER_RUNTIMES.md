# Executable provider runtimes

MeshLLM provider runtimes are signed executable processes that expose models
through a versioned host protocol. They are different from native runtimes:

- a native runtime supplies backend libraries selected against the Skippy ABI;
- a provider runtime supplies an executable, its own protocol version, and one
  or more whole-model provider identities.

Apple's `mesh-apple-runtime` is the first provider runtime. The contract is
provider-neutral so future process-hosted integrations can reuse installation
and lifecycle infrastructure without inheriting Apple or Foundation Models
semantics.

## Bundle contract

Every bundle contains `provider-runtime.json` at its root. Schema version 1
binds these fields:

- immutable artifact ID and semantic version;
- provider kind and host protocol version;
- OS, architecture, optional target triple, and minimum OS version;
- relative executable entrypoint;
- logical model IDs and provider-specific model kinds;
- feature declarations;
- SHA-256 for every installed payload file;
- optional build and code-signing metadata.

Example:

```json
{
  "schema_version": 1,
  "runtime": {
    "id": "meshllm-apple-runtime-darwin-arm64",
    "version": "0.1.0",
    "provider_kind": "apple",
    "protocol_version": "0.1",
    "platform": {
      "os": "macos",
      "arch": "arm64",
      "target": "aarch64-apple-darwin",
      "minimum_os_version": "27.0"
    },
    "entrypoint": "bin/mesh-apple-runtime",
    "models": [{"id": "apple/system", "kind": "system"}],
    "features": ["availability", "streaming", "cancellation"],
    "files": {
      "bin/mesh-apple-runtime": "sha256:..."
    }
  }
}
```

Artifact IDs, versions, provider kinds, protocol versions, and platform names
are data used for selection, not live capability claims. Availability and
observed model capabilities must still come from the running provider.

## Release index and resolution

`provider-runtimes.json` is the carrier-neutral release index. It contains the
same artifact record plus an archive URL and mandatory archive SHA-256 when a
download is offered.

Resolution filters candidates by:

1. host OS, architecture, and minimum OS;
2. requested artifact, provider, protocol, and model identity;
3. newest semantic version;
4. source preference: explicit bundle, installed cache, verified download,
   then an unavailable metadata-only entry.

An SDK may carry the bundle in its own resource layout, but it must hand that
directory to the shared resolver. It must not reinterpret or recreate the
manifest in language-specific code.

## Installation and cache

The shared cache layout is:

```text
<cache>/<artifact-id>/<version>/<os>-<arch>/
```

Installation is staged and then renamed into place. Coordinates are immutable:
installing different metadata or bytes over an existing coordinate fails. An
upgrade installs a new semantic version beside the old version; the resolver
selects the newest compatible version. Pruning old versions is deliberately
outside the version-1 contract so an SDK cannot destroy a runtime that another
host process is still using.

Bundled runtimes can be used in place or copied into the shared cache. Remote
archives are downloaded with a bounded size, checked against the release-index
digest, safely extracted, compared with the selected payload contract, and
then installed through the same cache path.

## Security invariants

The implementation fails closed on:

- absolute paths, `..`, or other unsafe manifest paths;
- payload symlinks or ZIP symlink entries;
- missing, malformed, or mismatched SHA-256 values;
- entrypoints absent from the checked file set;
- non-executable Unix entrypoints;
- duplicate artifact coordinates or model IDs;
- unsupported schema versions;
- downloads without an archive digest;
- ZIP entry-count, compressed-size, or expanded-size limits;
- archives containing zero or multiple provider manifests;
- a downloaded bundle that differs from its release-index artifact;
- attempts to overwrite an installed coordinate with different metadata.

File checksums prove artifact integrity. Platform signature, notarization,
quarantine, and entitlement verification remain additional policy gates owned
by the host supervisor and release packaging. Manifest signature metadata is
descriptive and is never treated as proof by itself.

## Ownership and next layer

`mesh-llm-provider-runtime` owns this data and installation contract. It does
not launch processes, bind ports, route inference, or expose language-specific
APIs.

The next stacked layer will add a Rust host supervisor that resolves one of
these artifacts, validates platform signing policy, launches its entrypoint,
tracks health and availability, forwards OpenAI requests, drains it, and
terminates it with the owning MeshLLM instance.
