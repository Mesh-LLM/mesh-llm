//! Key derivation for L3 entries.
//!
//! A composition root binds the ordered segment list and the cutting scheme
//! that produced it. It deliberately excludes entry metadata (`token_count`,
//! `expected_tokens`, `kv_desc_json`, calibration), matching the wire-digest
//! semantics where two entries with identical bytes and different metadata
//! already shared a key.

use anyhow::Result;

use super::manifest_codec::digest_to_bytes;
use super::{CODEC_RAW, CODEC_RAW_VERSION, GeometryKind, HandoffSegmentRef, PayloadGeometry};

/// Identity of a geometry cutting scheme.
///
/// Excludes row count: the scheme describes how a payload is cut, not how long
/// this entry is. Two entries of different lengths cut the same way share an
/// id, which is what lets a longer prefix reuse a shorter one's segments.
pub fn geometry_segmentation_id(geometry: &PayloadGeometry) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"skippy-kv/segmentation/1");
    hasher.update(&geometry.window_rows.to_le_bytes());
    hasher.update(&geometry.tail_bytes.to_le_bytes());
    for block in &geometry.blocks {
        hasher.update(&block.stride.to_le_bytes());
        hasher.update(&[match block.kind {
            GeometryKind::Key => 0,
            GeometryKind::Value => 1,
            GeometryKind::KeyIndex => 2,
        }]);
        hasher.update(&block.layer.to_le_bytes());
        hasher.update(&block.column.to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

/// Identity of the fixed-size fallback cutting scheme.
pub fn fixed_segmentation_id(segment_bytes: u64) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"skippy-kv/segmentation/1");
    hasher.update(b"fixed");
    hasher.update(&segment_bytes.to_le_bytes());
    hasher.finalize().to_hex().to_string()
}

/// Composition root over the ordered segment list.
///
/// Binds offsets, lengths, codec identity and content without hashing payload
/// bytes: every byte is already bound by its segment digest, and order is
/// implied by position.
pub fn composition_root(segmentation_id: &str, segments: &[HandoffSegmentRef]) -> Result<String> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"skippy-kv/root/1");
    hasher.update(segmentation_id.as_bytes());
    hasher.update(&[0u8]);
    for segment in segments {
        hasher.update(&segment.offset.to_le_bytes());
        hasher.update(&segment.bytes.to_le_bytes());
        let (name, version) = segment
            .codec_identity
            .as_ref()
            .map(|identity| (identity.name.as_str(), identity.version))
            .unwrap_or((CODEC_RAW, CODEC_RAW_VERSION));
        hasher.update(name.as_bytes());
        hasher.update(&[0u8]);
        hasher.update(&version.to_le_bytes());
        // Hash the digest's raw bytes, not its hex spelling, so a binary
        // manifest and a JSON manifest produce the same root.
        hasher.update(&digest_to_bytes(&segment.digest)?);
    }
    Ok(hasher.finalize().to_hex().to_string())
}
