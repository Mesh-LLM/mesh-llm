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

