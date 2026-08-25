use mesh_llm_provider_runtime::ProviderRuntimeManifest;
use serde_json::json;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let bundle = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: inspect <provider-runtime-directory>"))?;
    let manifest = ProviderRuntimeManifest::read_from_dir(&bundle)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "valid",
            "id": manifest.runtime.id,
            "version": manifest.runtime.version,
            "provider_kind": manifest.runtime.provider_kind,
            "protocol_version": manifest.runtime.protocol_version,
            "entrypoint": bundle.join(manifest.runtime.entrypoint),
            "models": manifest.runtime.models,
        }))?
    );
    Ok(())
}
