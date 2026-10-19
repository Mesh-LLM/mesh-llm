//! Per-manifest object footprints, recorded at commit.
//!
//! Budget enforcement and GC need to know which pack each segment lives in to
//! learn what becomes free when a manifest is evicted. Rebuilding that from
//! every manifest and packed index costs O(manifests x segments) per pass;
//! recording it once at commit makes the same decision O(1) per evicted
//! manifest.

use std::collections::HashMap;

use anyhow::Result;

use super::packed::PackedSegmentLocation;
use super::{HandoffManifest, HandoffSegmentStore};

/// What one manifest owns, aggregated by object rather than by segment.
#[derive(Debug, Clone, Default)]
pub(super) struct ManifestFootprint {
    pub(super) model_identity: String,
    pub(super) manifest_bytes: u64,
    /// Pack digest -> (segments referenced, pack size).
    pub(super) packs: HashMap<String, (usize, u64)>,
    /// Segment digest -> (references, segment bytes) for unpacked segments.
    pub(super) loose: HashMap<String, (usize, u64)>,
}

/// Aggregate a manifest's references, resolving pack sizes at most once.
fn build_footprint(
    manifest: &HandoffManifest,
    locations: &[Option<PackedSegmentLocation>],
    manifest_bytes: u64,
    mut pack_bytes: impl FnMut(&str) -> u64,
) -> ManifestFootprint {
    let mut footprint = ManifestFootprint {
        model_identity: manifest.model_identity.clone(),
        manifest_bytes,
        ..ManifestFootprint::default()
    };
    for (segment, location) in manifest.segments.iter().zip(locations) {
        match location {
            Some(location) => {
                let bytes = footprint
                    .packs
                    .get(&location.pack_digest)
                    .map(|(_, bytes)| *bytes)
                    .unwrap_or_else(|| pack_bytes(&location.pack_digest));
                let entry = footprint
                    .packs
                    .entry(location.pack_digest.clone())
                    .or_insert((0, bytes));
                entry.0 += 1;
            }
            None => {
                let entry = footprint
                    .loose
                    .entry(segment.digest.clone())
                    .or_insert((0, segment.bytes));
                entry.0 += 1;
            }
        }
    }
    footprint
}

impl HandoffSegmentStore {
    fn pack_size(&self, digest: &str) -> u64 {
        std::fs::metadata(self.packed.pack_path(digest))
            .map(|metadata| metadata.len())
            .unwrap_or(0)
    }

    /// Record what a committed manifest owns, aggregated by pack.
    pub(super) fn record_footprint(
        &self,
        manifest: &HandoffManifest,
        locations: &[Option<PackedSegmentLocation>],
        manifest_bytes: u64,
    ) {
        let footprint = build_footprint(manifest, locations, manifest_bytes, |digest| {
            self.pack_size(digest)
        });
        self.footprints
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(manifest.payload_digest.clone(), footprint);
    }

    /// The cached footprint for one manifest, rebuilt from disk on a miss.
    pub(super) fn manifest_footprint(&self, key: &str) -> Result<ManifestFootprint> {
        let cache_enabled = std::env::var("SKIPPY_L3_FOOTPRINT_CACHE")
            .map(|value| value != "off" && value != "0")
            .unwrap_or(true);
        let cached = if cache_enabled {
            self.footprints
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(key)
                .cloned()
        } else {
            None
        };
        if let Some(footprint) = cached {
            // Only trust the cache while the manifest on disk still has the
            // size it had when the footprint was recorded. A truncated or
            // replaced manifest must be re-read, so GC fails on it instead of
            // deleting state behind an unreadable entry.
            let on_disk = std::fs::metadata(self.existing_manifest_path(key))
                .map(|metadata| metadata.len())
                .unwrap_or(0);
            if on_disk > 0 && on_disk == footprint.manifest_bytes {
                return Ok(footprint);
            }
            self.footprints
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(key);
        }
        let manifest = self.load_manifest(key)?;
        let locations = self
            .packed
            .load_manifest_index(key, manifest.segments.len())?;
        let manifest_bytes = std::fs::metadata(self.existing_manifest_path(key))
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        let footprint = build_footprint(&manifest, &locations, manifest_bytes, |digest| {
            self.pack_size(digest)
        });
        self.footprints
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(key.to_string(), footprint.clone());
        Ok(footprint)
    }

    /// Drop cached footprints for manifests that no longer exist.
    pub(super) fn forget_footprints(&self, keys: &[String]) {
        if keys.is_empty() {
            return;
        }
        let mut footprints = self
            .footprints
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for key in keys {
            footprints.remove(key);
        }
    }
}
