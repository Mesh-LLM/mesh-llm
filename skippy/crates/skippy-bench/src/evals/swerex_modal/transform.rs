//! Native admission of immutable external SDK postimages, preserving finite original identities.
use super::*;
pub(super) fn rewrite(destination: &str, bytes: &[u8]) -> Result<Vec<u8>> {
    let mapping = contract::MAPPINGS
        .iter()
        .find(|m| m.destination == destination)
        .ok_or_else(|| anyhow::anyhow!("unknown finite SWE-ReX Modal destination"))?;
    if Some(digest(bytes).as_str()) != mapping.original_sha256 {
        bail!("unknown official SWE-ReX original source");
    }
    let name = format!("swerex-modal/{destination}");
    let replacement = super::super::external_sdk_source::read(&name)?;
    if digest(&replacement) != mapping.replacement_sha256 {
        bail!("external SWE-ReX postimage differs from reviewed identity");
    }
    Ok(replacement)
}
