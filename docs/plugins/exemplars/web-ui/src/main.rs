#[path = "../manifest.rs"]
mod manifest;

use anyhow::{Context, Result, bail};
use mesh_llm_plugin::{Plugin, PluginRuntime, package_manifest_json};

#[tokio::main]
async fn main() -> Result<()> {
    let plugin = manifest::exemplar_plugin();
    match std::env::args().nth(1).as_deref() {
        Some("--print-package-manifest") => {
            let manifest = plugin.manifest().context("exemplar manifest")?;
            println!("{}", package_manifest_json(&manifest)?);
            Ok(())
        }
        Some(argument) => bail!("unknown option: {argument}"),
        None => PluginRuntime::run(plugin).await,
    }
}
