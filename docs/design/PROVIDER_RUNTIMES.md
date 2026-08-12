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

## Composed product contract

A macOS arm64 MeshLLM product can carry provider runtimes beside the neutral
host and native runtime:

```text
mesh-bundle/
├── mesh-llm
├── product-manifest.json
├── native-runtimes/<native-runtime-id>/
└── provider-runtimes/
    └── apple/<provider-runtime-id>/
```

`product-manifest.json` attests every embedded provider runtime by ID, semantic
version, provider kind, protocol version, canonical relative path, complete
tree SHA-256, and provider-manifest SHA-256. The Unix installer moves the whole
`provider-runtimes/` tree atomically with the host and native runtime. Because
the Apple supervisor searches `provider-runtimes/apple` adjacent to its own
executable, both an unpacked release and an installed release discover the
provider without `MESH_LLM_PROVIDER_RUNTIME_BUNDLE_DIR` or
`MESH_LLM_PROVIDER_RUNTIME_INDEX`.

Embedding is explicit at product-composition time. Other platforms and normal
MeshLLM products remain unchanged when no provider runtime root is supplied.
An Apple provider may enter a public product only when its executable passes
strict code-signature verification, its manifest declares a real team and
successful notarization, and `spctl` accepts the executable. The local Golden
Gate QA lane has a deliberately named unnotarized exception; that exception is
not release eligible.

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

## Ownership and host layer

`mesh-llm-provider-runtime` owns this data and installation contract. It does
not launch processes, bind ports, route inference, or expose language-specific
APIs. The host runtime now consumes the contract through its experimental Apple
provider supervisor, which owns platform policy, process lifecycle, health, and
local route registration.

Every host-capable macOS SDK packages that same artifact and lifecycle. The
Rust SDK accepts carrier bundle roots, an optional release index/cache, and
explicit download permission through `ProviderHostConfig`. Swift and
Kotlin/JVM call it through UniFFI; Node/Electron calls it through N-API. The
typed configuration flows into the same resolver without mutating or
inheriting process-global provider-discovery environment. Provider-only hosts
also skip the unrelated Skippy native-runtime load.

Carrier resources are deliberately mechanical:

| SDK | Packaged location | Runtime handling |
|---|---|---|
| SwiftPM | macOS-only `MeshLLMAppleProviderResources` target | use the resource directory in place |
| npm / Electron | optional `@mesh-llm/apple-runtime-darwin-arm64` package | ship outside ASAR so the executable remains a file |
| Kotlin/JVM | `meshllm-apple-runtime-macos-arm64` resource JAR | extract the manifest-listed files to an empty private directory and restore the executable bit |

`scripts/package-sdk-provider-runtime.sh` copies the exact signed bytes into
these layouts. `just apple::sdk-carriers` then certifies every wrapper with one
REST suite. SDKs must not reinterpret Foundation Models semantics, fork the
sidecar, or silently fall back to a different model. Platform resource packages
remain separate so Linux/Windows npm installs and iOS/Android artifacts never
carry a macOS executable.
