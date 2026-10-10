use super::super::admission::Artifact as Pin;
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::automation::hf_certify) struct Artifact {
    pub path: PathBuf,
    pub path_in_repo: String,
    pub sha256: String,
    pub byte_size: u64,
}
/// Exact helper-v3 canonical field order. Never serialize credential bytes.
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::automation::hf_certify) struct PublisherInput {
    pub schema_version: u32,
    pub repo: String,
    pub parent_commit: String,
    pub shards: Vec<Artifact>,
    pub sidecars: Vec<Artifact>,
    pub credential_file: Option<PathBuf>,
    pub execution_timeout_ms: u64,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::automation::hf_certify) struct Request {
    pub helper: Pin,
    /// Supplied source-file custody only; not a build attestation linking these bytes to helper.
    pub helper_source: Pin,
    pub input: PublisherInput,
}
pub(super) fn hex(value: &str, size: usize) -> bool {
    value.len() == size
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
impl Request {
    pub(super) fn validate(&self) -> DynResult<()> {
        let input = &self.input;
        for pin in [&self.helper, &self.helper_source] {
            if !pin.path.is_absolute() || !hex(&pin.sha256, 64) {
                return Err("publication helper/source pin refused".into());
            }
        }
        if input.schema_version != 1
            || !(1..=86400000).contains(&input.execution_timeout_ms)
            || input.shards.is_empty()
            || input.shards.len() > 128
            || input.sidecars.len() > 32
            || !hex(&input.parent_commit, 40)
            || input
                .credential_file
                .as_ref()
                .is_none_or(|path| !path.is_absolute())
        {
            return Err(
                "publication input schema/roster/private credential reference refused".into(),
            );
        }
        let pieces = input.repo.split('/').collect::<Vec<_>>();
        if pieces.len() != 2 || pieces.iter().any(|p| !component(p) || p.len() > 96) {
            return Err("publication repo refused".into());
        }
        validate_artifacts(&input.shards, &input.sidecars)
    }
}
pub(in crate::automation::hf_certify) fn validate_artifacts(
    shards: &[Artifact],
    sidecars: &[Artifact],
) -> DynResult<()> {
    if shards.is_empty() || shards.len() > 128 || sidecars.len() > 32 {
        return Err("publication artifact roster refused".into());
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut sidecar_bytes = 0u64;
    for (shard, artifact) in shards
        .iter()
        .map(|a| (true, a))
        .chain(sidecars.iter().map(|a| (false, a)))
    {
        if !artifact.path.is_absolute()
            || !hex(&artifact.sha256, 64)
            || artifact.path_in_repo.len() > 256
            || artifact.path_in_repo.split('/').any(|p| !component(p))
            || !seen.insert(&artifact.path_in_repo)
            || (shard
                && (artifact.byte_size == 0
                    || artifact.byte_size > 1024u64.pow(4)
                    || !artifact.path_in_repo.ends_with(".gguf")))
            || (!shard
                && (artifact.byte_size > 1048576
                    || ![".json", ".md", ".txt"]
                        .iter()
                        .any(|suffix| artifact.path_in_repo.ends_with(suffix))))
        {
            return Err("publication declared artifact refused".into());
        }
        if !shard {
            sidecar_bytes = sidecar_bytes
                .checked_add(artifact.byte_size)
                .ok_or("sidecar byte overflow")?;
        }
    }
    if sidecar_bytes > 8 * 1048576 {
        return Err("publication sidecar total refused".into());
    }
    Ok(())
}

fn component(value: &str) -> bool {
    !value.is_empty()
        && !matches!(value, "." | "..")
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}
