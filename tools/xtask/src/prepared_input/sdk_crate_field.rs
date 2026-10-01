use super::{Checked, Rejected, positional};
use serde::Deserialize;

#[derive(Deserialize)]
struct CrateMetadata {
    artifact_id: String,
    sdk_version: String,
    platform: String,
    flavor: String,
    target_triple: String,
    backend: String,
}

pub(super) fn run(args: &[String]) -> Checked<String> {
    let [manifest, field] = positional(args, "MANIFEST FIELD")?;
    let bytes = std::fs::read(manifest)
        .map_err(|error| Rejected(format!("cannot read SDK manifest: {error}")))?;
    let metadata: CrateMetadata = serde_json::from_slice(&bytes)
        .map_err(|error| Rejected(format!("invalid SDK crate metadata: {error}")))?;
    let value = match field {
        "artifact_id" => metadata.artifact_id,
        "sdk_version" => metadata.sdk_version,
        "platform" => metadata.platform,
        "flavor" => metadata.flavor,
        "target_triple" => metadata.target_triple,
        "backend" => metadata.backend,
        _ => return Err(Rejected(format!("unsupported SDK crate field: {field}"))),
    };
    if value.is_empty() || value.chars().any(char::is_control) {
        return Err(Rejected(format!("invalid SDK crate field: {field}")));
    }
    Ok(format!("{value}\n"))
}
