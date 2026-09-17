# mesh-llm-runtime-install

`mesh-llm-runtime-install` owns the public native runtime installation flow for
Mesh LLM SDK consumers and command-line tools.

It provides:

- release manifest loading from a file, URL, bundled runtime directory, or the
  default Mesh LLM GitHub release URL
- compatible runtime resolution for the current Mesh LLM version
- cache path selection and installed runtime discovery
- checksum enforcement before installing downloaded archives
- download progress callbacks for SDK and CLI callers
- stale runtime pruning through `NativeRuntimeCache`

Native runtime versions must match the Mesh LLM crate version exactly. The
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
use mesh_llm_runtime_install::{NativeRuntimeCatalog, NativeRuntimeInstallOptions};

let options = NativeRuntimeInstallOptions {
    mesh_version: "1.2.3".to_string(),
    catalog: NativeRuntimeCatalog {
        releases_url: "https://example.invalid/skippy/releases".to_string(),
        rolling_release: None,
    },
    ..Default::default()
};
```

The example URL is illustrative. This intermediate extraction still supplies
Mesh defaults through `Default`; standalone Skippy release metadata, cache and
bundle-discovery defaults must be provided by the standalone lifecycle layer.
The `mesh_version` field retains its current wire/cache spelling for now.

## Example

```rust,no_run
use mesh_llm_runtime_install::{
    NativeRuntimeInstallOptions, RuntimeSelection, install_native_runtime,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let outcome = install_native_runtime(NativeRuntimeInstallOptions {
        selection: RuntimeSelection::Recommended,
        ..Default::default()
    })
    .await?;

    println!("installed runtime at {}", outcome.runtime.path.display());
    Ok(())
}
```

