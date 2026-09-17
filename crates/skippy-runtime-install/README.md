# skippy-runtime-install

`skippy-runtime-install` owns the public native runtime installation flow for
callers supplying a runtime release and catalog policy. Mesh defaults live in
`mesh_llm_system::native_runtime_install`.

It provides:

- release manifest loading from a file, URL, bundled runtime directory, or the
  explicitly configured release catalog
- compatible runtime resolution for the requested release
- cache path selection and installed runtime discovery
- checksum enforcement before installing downloaded archives
- download progress callbacks for SDK and CLI callers
- stale runtime pruning through `NativeRuntimeCache`

Native runtime versions must match the requested release exactly. The
installer rejects incompatible release manifest entries instead of building
native code through Cargo.

## Explicit catalogs

`NativeRuntimeInstallOptions::catalog` and `NativeRuntimeManifestOptions::catalog`
carry a `NativeRuntimeCatalog`: a release URL root and an optional release that
should track its latest catalog. Other releases use pinned version URLs. The
loader uses these inputs; it does not infer release channels from Mesh build
metadata. Explicit manifest files and URLs, environment overrides and download
permissions keep their existing precedence.

For an independently supplied Skippy runtime catalog:

```rust
use skippy_runtime_install::{NativeRuntimeCatalog, NativeRuntimeInstallOptions};

let options = NativeRuntimeInstallOptions::new(
    "1.2.3",
    NativeRuntimeCatalog {
        releases_url: "https://example.invalid/skippy/releases".to_string(),
        rolling_release: None,
    },
);
```

The example URL is illustrative. Options have no `Default` implementation:
callers must provide the release and catalog. Mesh callers use
`mesh_native_runtime_install_options()` or `mesh_native_runtime_manifest_options()`
from the Mesh facade. Standalone Skippy release metadata must be supplied by its
lifecycle layer. Bundle discovery takes an explicit release; existing Mesh path
and environment names remain pending the coordinated packaging migration.
The `mesh_version` field retains its current wire/cache spelling for now.

## Migrating callers

The package replaces `mesh-llm-runtime-install`; Rust imports use
`skippy_runtime_install`. Construct options with `new(release, catalog)` instead
of `Default`. Bundle discovery and `discover_local_native_runtimes` now require
an explicit release argument so discovery and installation select the same
version. These are source API changes.

Mesh callers can use `mesh_llm_system::native_runtime_install` for Mesh policy
defaults and compatibility discovery helpers. The Mesh SDK re-exports those
helpers. This package move does not migrate existing cache data or change
runtime manifest fields, artifact names, or environment variables.

## Example

```rust,no_run
use skippy_runtime_install::{
    NativeRuntimeCatalog, NativeRuntimeInstallOptions, RuntimeSelection, install_native_runtime,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let outcome = install_native_runtime(NativeRuntimeInstallOptions {
        selection: RuntimeSelection::Recommended,
        ..NativeRuntimeInstallOptions::new("1.2.3", NativeRuntimeCatalog {
            releases_url: "https://example.invalid/skippy/releases".to_string(),
            rolling_release: None,
        })
    })
    .await?;

    println!("installed runtime at {}", outcome.runtime.path.display());
    Ok(())
}
```
