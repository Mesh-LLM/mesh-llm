use mesh_llm_provider_runtime::{
    ProviderRuntimeCache, ProviderRuntimeReleaseManifest, install_provider_runtime_archive,
};
use serde_json::json;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let mut arguments = std::env::args_os().skip(1).map(PathBuf::from);
    let release_manifest_path = arguments.next().ok_or_else(|| {
        anyhow::anyhow!("usage: inspect_archive <provider-runtimes.json> <runtime.zip>")
    })?;
    let archive_path = arguments.next().ok_or_else(|| {
        anyhow::anyhow!("usage: inspect_archive <provider-runtimes.json> <runtime.zip>")
    })?;
    let release_manifest = ProviderRuntimeReleaseManifest::read_from_path(&release_manifest_path)?;
    let artifact = release_manifest
        .artifacts
        .first()
        .ok_or_else(|| anyhow::anyhow!("provider runtime release manifest has no artifacts"))?;
    let archive_sha256 = artifact
        .archive_sha256
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("provider runtime artifact has no archive checksum"))?;
    let workspace = tempfile::tempdir()?;
    let cache = ProviderRuntimeCache::new(workspace.path().join("cache"));
    let (status, installed) =
        install_provider_runtime_archive(&cache, artifact, &archive_path, archive_sha256)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "valid",
            "install_status": status,
            "id": installed.manifest.runtime.id,
            "version": installed.manifest.runtime.version,
            "entrypoint_exists": installed.entrypoint().is_file(),
        }))?
    );
    Ok(())
}
