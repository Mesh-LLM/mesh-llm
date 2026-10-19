//! Manifest decoding and compatibility validation.
//!
//! Everything here runs before a manifest is trusted: version support, codec
//! identity completeness, tiling, and the offset check used by the opt-in
//! full-wire verification.

use anyhow::{Context, Result, bail};

use super::manifest_codec;
use super::{
    HandoffManifest, HandoffSegmentRef, LEGACY_MANIFEST_VERSION,
    LEGACY_PAYLOAD_CODEC_MANIFEST_VERSION, MANIFEST_VERSION, PayloadCodec, SegmentCodecIdentity,
    segment_digest,
};

/// Verify the assembled bytes at each manifest offset.
///
/// The per-segment digests checked during reads prove the bytes are what they
/// claim; this proves the scatter put them where the manifest says. It is the
/// only check that catches a correct segment copied to the wrong offset, so it
/// is opt-in (`SKIPPY_L3_VERIFY_WIRE`) and exercised by the gate.
pub(super) fn verify_segments_at_offsets(
    payload: &[u8],
    segments: &[HandoffSegmentRef],
) -> Result<()> {
    for segment in segments {
        let start = usize::try_from(segment.offset).context("segment offset exceeds usize")?;
        let len = usize::try_from(segment.bytes).context("segment length exceeds usize")?;
        let end = start.checked_add(len).context("segment range overflows")?;
        let bytes = payload
            .get(start..end)
            .context("segment range exceeds assembled payload")?;
        if segment_digest(bytes) != segment.digest {
            bail!(
                "assembled payload disagrees with segment {} at offset {}",
                segment.index,
                segment.offset
            );
        }
    }
    Ok(())
}

/// Refuse a manifest whose payload codec this build cannot decode, before any
/// segment is read or reassembled. Keeps unknown codecs from being committed
/// (nothing unassemblable is ever persisted) and from being assembled (an
/// unknown codec is a miss, never a silent misinterpretation of the bytes). A
/// missing identity is treated as unsupported, not defaulted to raw.
pub(super) fn reject_unsupported_codec(manifest: &HandoffManifest) -> Result<()> {
    match manifest.codec.as_ref() {
        Some(codec) if codec.is_supported() => Ok(()),
        Some(codec) => bail!(
            "manifest {} uses unsupported codec {}/{}",
            manifest.payload_digest,
            codec.name,
            codec.version
        ),
        None => bail!(
            "manifest {} has no codec identity; this build requires an explicit codec",
            manifest.payload_digest
        ),
    }
}

/// Versions this build can read and write: the current per-segment format
/// and the two legacy formats (pre-codec and payload-level-codec).
fn manifest_version_is_supported(version: u32) -> bool {
    version == MANIFEST_VERSION
        || version == LEGACY_MANIFEST_VERSION
        || version == LEGACY_PAYLOAD_CODEC_MANIFEST_VERSION
}

/// Reject a manifest whose segment codec identities this build cannot decode,
/// before any segment is read or reassembled. Also enforces the per-segment
/// negotiation contract: every identity must be internally consistent with
/// the stored bytes it covers, and a mixed manifest must name the payload
/// codec that describes the assembled whole. Positional errors name the
/// segment index so the offending ref is identifiable in the message.
fn reject_unsupported_segment_codecs(manifest: &HandoffManifest) -> Result<()> {
    let has_native_kv = manifest.uses_native_kv_passthrough();
    let has_cachegen_kv = manifest.uses_cachegen_kv();
    let kv_representation = if has_cachegen_kv {
        "CacheGen KV"
    } else {
        "native KV"
    };
    if has_native_kv && has_cachegen_kv {
        bail!(
            "manifest {} mixes native and CacheGen KV representations",
            manifest.payload_digest
        );
    }
    let envelope_is_cachegen =
        manifest.codec.as_ref() == Some(&PayloadCodec::cachegen_kv_envelope());
    if envelope_is_cachegen != has_cachegen_kv {
        bail!(
            "manifest {} payload codec disagrees with its CacheGen segments",
            manifest.payload_digest
        );
    }
    let has_runtime_descriptor = manifest
        .kv_desc_json
        .as_deref()
        .is_some_and(|descriptor| serde_json::from_str::<serde_json::Value>(descriptor).is_ok());
    if has_native_kv
        && (manifest.version != MANIFEST_VERSION
            || manifest.payload_kind != "kv-recurrent"
            || manifest.kv_bytes == 0
            || !has_runtime_descriptor
            || manifest.kv_bytes.checked_add(manifest.recurrent_bytes)
                != Some(manifest.total_bytes))
    {
        bail!(
            "manifest {} names native KV segments without a current, complete kv-recurrent payload and runtime page descriptor",
            manifest.payload_digest
        );
    }
    if has_cachegen_kv
        && (manifest.version != MANIFEST_VERSION
            || manifest.payload_kind != "kv-recurrent"
            || manifest.kv_bytes == 0
            || manifest.kv_decoded_bytes == 0
            || !has_runtime_descriptor
            || manifest.kv_bytes.checked_add(manifest.recurrent_bytes)
                != Some(manifest.total_bytes)
            || manifest.codec.as_ref() != Some(&PayloadCodec::cachegen_kv_envelope()))
    {
        bail!(
            "manifest {} names CacheGen KV without a current, complete kv-recurrent envelope and runtime page descriptor",
            manifest.payload_digest
        );
    }
    if has_cachegen_kv {
        let mut cachegen_segments = manifest.segments.iter().filter(|segment| {
            segment
                .codec_identity
                .as_ref()
                .is_some_and(SegmentCodecIdentity::is_cachegen_archive)
        });
        let cachegen_segment = cachegen_segments
            .next()
            .context("CacheGen manifest has no archive segment")?;
        let cachegen_identity = cachegen_segment
            .codec_identity
            .as_ref()
            .context("CacheGen archive segment has no codec identity")?;
        if cachegen_segments.next().is_some()
            || cachegen_segment.offset != 0
            || cachegen_segment.bytes != manifest.kv_bytes
            || cachegen_identity.decoded_len != manifest.kv_decoded_bytes
        {
            bail!(
                "manifest {} CacheGen archive must be one complete KV segment with matching encoded and decoded lengths",
                manifest.payload_digest
            );
        }
    }
    for segment in &manifest.segments {
        let Some(identity) = segment.codec_identity.as_ref() else {
            // Legacy formats carry no per-segment identity by construction;
            // the payload gate covers what their segments decode as.
            continue;
        };
        if !identity.is_supported() {
            bail!(
                "manifest {} segment {} uses unsupported codec {}/{}",
                manifest.payload_digest,
                segment.index,
                identity.name,
                identity.version
            );
        }
        if !identity.is_self_consistent(segment.bytes) {
            bail!(
                "manifest {} segment {} codec identity is inconsistent with its stored bytes ({}/{} class, decoded_len {}, encoded_len {})",
                manifest.payload_digest,
                segment.index,
                identity.name,
                identity.version,
                identity.decoded_len,
                segment.bytes
            );
        }
        if has_native_kv || has_cachegen_kv {
            let end = segment
                .offset
                .checked_add(segment.bytes)
                .context("segment range overflows")?;
            let covers_kv = segment.offset < manifest.kv_bytes;
            if covers_kv && end > manifest.kv_bytes {
                bail!(
                    "manifest {} segment {} crosses the {kv_representation} boundary at byte {}",
                    manifest.payload_digest,
                    segment.index,
                    manifest.kv_bytes
                );
            }
            let expected_kv = if has_cachegen_kv {
                identity.is_cachegen_archive()
            } else {
                identity.is_native_kv_page()
            };
            if covers_kv != expected_kv {
                bail!(
                    "manifest {} segment {} representation disagrees with the {kv_representation} boundary at byte {}",
                    manifest.payload_digest,
                    segment.index,
                    manifest.kv_bytes
                );
            }
        }
        // The payload codec describes the assembled envelope. Exact native KV
        // segments remain byte-for-byte members of the raw envelope; a future
        // codec that transforms stored bytes must upgrade that envelope rather
        // than leave it claiming raw.
    }
    Ok(())
}

/// The current format requires explicit per-segment codec identity. Shared
/// by `decode_manifest` (on-disk reads) and
/// `validate_manifest_compatibility` (commit and assembly), so neither path
/// can persist or partially read a v4 manifest whose segment identity was
/// stripped — there is no fallback to the payload codec or to raw.
fn require_v4_segment_identities(manifest: &HandoffManifest) -> Result<()> {
    for segment in &manifest.segments {
        if segment.codec_identity.is_none() {
            bail!(
                "manifest {digest} at version {MANIFEST_VERSION} is missing per-segment codec identity on segment {index}",
                digest = manifest.payload_digest,
                index = segment.index
            );
        }
    }
    Ok(())
}

/// Validate that an in-memory manifest is one this build can both persist and
/// assemble: a supported format version *and* supported codecs at both the
/// payload and every segment. `try_commit` and `assemble` call this before
/// writing or reading any segment, so a manifest that `decode_manifest` would
/// later reject — an unknown version (e.g. a future version carrying a raw
/// codec) or an unsupported codec at either level — cannot be committed to
/// disk or partially assembled from segments.
pub(super) fn validate_manifest_compatibility(manifest: &HandoffManifest) -> Result<()> {
    if !manifest_version_is_supported(manifest.version) {
        bail!(
            "manifest {} has version {} but this build reads {MANIFEST_VERSION} or legacy {LEGACY_MANIFEST_VERSION}/{LEGACY_PAYLOAD_CODEC_MANIFEST_VERSION}",
            manifest.payload_digest,
            manifest.version
        );
    }
    if manifest.version == MANIFEST_VERSION {
        require_v4_segment_identities(manifest)?;
    }
    reject_unsupported_codec(manifest)?;
    reject_unsupported_segment_codecs(manifest)
}

pub(super) fn decode_manifest(payload_digest: &str, bytes: &[u8]) -> Result<HandoffManifest> {
    let mut manifest: HandoffManifest = if manifest_codec::looks_binary(bytes) {
        manifest_codec::decode_manifest(bytes).context("malformed binary manifest")?
    } else {
        // Legacy v2-v4 entries are JSON; keep reading them for one release.
        serde_json::from_slice(bytes).context("malformed manifest")?
    };
    match manifest.version {
        MANIFEST_VERSION => {
            // The current format must stamp both the payload codec and every
            // per-segment codec identity explicitly. A missing field means it
            // was stripped to force a reinterpretation of bytes this build
            // may not decode — reject, never default to raw or to the payload
            // codec.
            if manifest.codec.is_none() {
                bail!(
                    "manifest {payload_digest} at version {MANIFEST_VERSION} is missing its required codec identity"
                );
            }
            require_v4_segment_identities(&manifest)?;
        }
        LEGACY_PAYLOAD_CODEC_MANIFEST_VERSION => {
            // #1750 format: codec identity is carried once at the payload
            // level and every segment implicitly is that codec. The payload
            // codec is required there (a stripped one is rejected, same as
            // current), and no per-segment identity exists to normalize —
            // the payload gate below covers what the segments decode as.
            if manifest.codec.is_none() {
                bail!(
                    "manifest {payload_digest} at version {LEGACY_PAYLOAD_CODEC_MANIFEST_VERSION} is missing its required codec identity"
                );
            }
        }
        LEGACY_MANIFEST_VERSION => {
            // Pre-codec-identity manifests are raw by construction. Accept an
            // absent codec (normalize to raw) or an explicit raw codec; a
            // legacy manifest cannot legitimately name a non-raw codec.
            match manifest.codec {
                None => manifest.codec = Some(PayloadCodec::raw()),
                Some(ref codec) if *codec == PayloadCodec::raw() => {}
                Some(ref codec) => bail!(
                    "legacy manifest {payload_digest} declares non-raw codec {}/{}",
                    codec.name,
                    codec.version
                ),
            }
        }
        other => bail!(
            "manifest {payload_digest} has version {other} but this build reads {MANIFEST_VERSION} or legacy {LEGACY_MANIFEST_VERSION}/{LEGACY_PAYLOAD_CODEC_MANIFEST_VERSION}"
        ),
    }
    // Central capability gate: no load entry point returns a manifest whose
    // codecs this build cannot decode. Startup reconciliation quarantines it,
    // manifest_for_prefix prunes the link and falls back to a shorter prefix,
    // and a direct load fails.
    reject_unsupported_codec(&manifest)?;
    reject_unsupported_segment_codecs(&manifest)?;
    if manifest.payload_digest != payload_digest {
        bail!(
            "manifest key {payload_digest} disagrees with payload digest {}",
            manifest.payload_digest
        );
    }
    Ok(manifest)
}
