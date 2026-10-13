#[path = "diff_sections.rs"]
mod diff_sections;
#[path = "series.rs"]
mod series;
#[path = "shard_input.rs"]
mod shard_input;

#[cfg(test)]
#[path = "../../../tests/migration_patch_shards/contracts.rs"]
mod tests;

use super::{MailPatch, PatchError, encode_mail_patch};
use serde_json::Value;
use sha2::{Digest, Sha256};
use shard_input::FamilyMap;
use std::collections::BTreeMap;

#[derive(Debug, thiserror::Error)]
pub(in super::super) enum ShardError {
    #[error(transparent)]
    Patch(#[from] PatchError),
    #[error("family source map must use schema_version 1 with a families object")]
    FamilyMapSchema,
    #[error("invalid family source mapping: {0}")]
    FamilyMapping(String),
    #[error("invalid or duplicate model source mapping: {0}")]
    SourceMapping(String),
    #[error("family manifest must contain a models array")]
    ManifestModels,
    #[error("family manifest contains missing, duplicate or unhashable family names")]
    ManifestFamily,
    #[error("family manifest contains a missing or unknown workload class")]
    WorkloadClass,
    #[error("causal family requires a split certification profile")]
    CausalProfile,
    #[error("non-chat family requires a workload profile")]
    WorkloadProfile,
    #[error("family source map does not cover the certification manifest: {0:?}")]
    MissingCoverage(Vec<String>),
    #[error("generated model diff has an unexpected format")]
    DiffFormat,
    #[error("generated model diff repeats source: {0}")]
    RepeatedSource(String),
    #[error("cannot encode family patch series: {0}")]
    SeriesJson(#[from] serde_json::Error),
}

pub(in super::super) struct FamilyShard {
    pub(in super::super) file: String,
    pub(in super::super) families: Vec<String>,
    pub(in super::super) sources: Vec<String>,
    pub(in super::super) sha256: String,
    pub(in super::super) bytes: Vec<u8>,
}

pub(in super::super) struct FamilyPatches {
    pub(in super::super) combined: MailPatch,
    pub(in super::super) shards: Vec<FamilyShard>,
    pub(in super::super) series: Vec<u8>,
    pub(in super::super) series_json: Vec<u8>,
}

/// Encodes captured diff bytes and explicit decoded JSON data without reading or writing files.
///
/// Errors reject invalid maps, certification coverage, diff sections, or UTF-8 before returning
/// any output. JSON decoding and output publication remain the caller's responsibilities.
pub(in super::super) fn encode_family_shards(
    diff: &[u8],
    family_map: &Value,
    certification_manifest: &Value,
) -> Result<FamilyPatches, ShardError> {
    let family_map = FamilyMap::parse(family_map)?;
    family_map.require_coverage(certification_manifest)?;
    let combined = encode_mail_patch("skippy: generate model-family stage controls", diff)?;
    let sections = diff_sections::split(diff)?;
    let mut grouped = BTreeMap::<_, BTreeMap<_, _>>::new();
    for section in sections {
        let owners = family_map.owners(section.source);
        grouped
            .entry((owners.is_empty(), owners))
            .or_default()
            .insert(section.source, section.bytes);
    }
    let mut shards = Vec::with_capacity(grouped.len());
    for (index, ((_, families), sections)) in grouped.into_iter().enumerate() {
        let label = if families.is_empty() {
            "unmapped".to_owned()
        } else {
            families.join("--")
        };
        let file = format!("{:04}-family-{label}.patch", index + 1);
        let shard_diff = sections.values().copied().collect::<Vec<_>>().concat();
        let patch = encode_mail_patch(
            &format!("skippy: annotate {label} graph semantics"),
            &shard_diff,
        )?;
        shards.push(FamilyShard {
            file,
            families,
            sources: sections.keys().map(|source| (*source).to_owned()).collect(),
            sha256: hex::encode(Sha256::digest(&patch.bytes)),
            bytes: patch.bytes,
        });
    }
    let series = shards
        .iter()
        .map(|shard| format!("{}\n", shard.file))
        .collect::<String>()
        .into_bytes();
    let series_json = series::encode(&combined.diff_sha256, &shards)?;
    Ok(FamilyPatches {
        combined,
        shards,
        series,
        series_json,
    })
}
