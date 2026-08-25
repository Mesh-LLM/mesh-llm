use mesh_llm_provider_runtime::ProviderRuntimeReleaseManifest;
use serde_json::json;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: inspect_release <provider-runtimes.json>"))?;
    let manifest = ProviderRuntimeReleaseManifest::read_from_path(&path)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "valid",
            "schema_version": manifest.schema_version,
            "artifacts": manifest.artifacts.iter().map(|artifact| json!({
                "id": artifact.id,
                "version": artifact.version,
                "platform": artifact.platform,
                "downloadable": artifact.url.is_some(),
            })).collect::<Vec<_>>(),
        }))?
    );
    Ok(())
}
