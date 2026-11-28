//! Download the Skippy runtime MeshLLM would pick on this machine, for
//! bundling into a release archive.
//!
//!     cargo run --locked -p mesh-llm-system --example fetch-native-runtime -- <cache-dir>
//!
//! It goes through the same installer a user's first `mesh-llm serve` does,
//! so the catalog checksum, the archive checksum, the ABI and the platform
//! are all checked. The runtime directory is printed on standard output.

use mesh_llm_system::native_runtime_install::{
    install_native_runtime, mesh_native_runtime_install_options,
};

fn main() -> anyhow::Result<()> {
    let cache_dir = std::env::args_os()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: fetch-native-runtime <cache-dir>"))?;
    let mut options = mesh_native_runtime_install_options();
    options.cache_dir = Some(cache_dir.into());
    let outcome = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(install_native_runtime(options))?;
    println!("{}", outcome.runtime.path.display());
    Ok(())
}
