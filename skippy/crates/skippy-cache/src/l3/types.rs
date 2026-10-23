//! Manifest, segment and store-policy types.
//!
//! These are the on-disk contract: what a manifest says, what a segment
//! reference binds, and what the store is allowed to occupy.

use serde::{Deserialize, Serialize};

/// On-disk format version stamped into every manifest. A released change to
/// the layout bumps this and makes older entries misses, never migrations.
///
/// Version 4 requires explicit per-segment codec identity; a manifest at this
/// version whose segment lacks it is rejected rather than reinterpreted.
pub const MANIFEST_VERSION: u32 = 4;

/// The pre-codec-identity manifest format. These entries predate codec
/// identity, are always raw by construction, and are decoded as raw through
/// the explicit legacy path in [`decode_manifest`]. New manifests are never
/// written at this version.
pub const LEGACY_MANIFEST_VERSION: u32 = 2;

/// The payload-level-codec manifest format (#1750): codec identity is carried
/// once on the payload and every segment is implicitly that codec. Read for
/// compatibility, never written. New manifests stamp identity per segment.
pub const LEGACY_PAYLOAD_CODEC_MANIFEST_VERSION: u32 = 3;

/// Identity of the payload envelope: raw, uncompressed exact state. Its bytes
/// are the segments verbatim, including segments whose representation is the
/// runtime's native KV page layout.
pub const CODEC_RAW: &str = "raw";
/// Version of the raw codec's on-disk representation. Bumped only if the raw
/// byte layout itself changes; an unknown version is rejected, never migrated.
pub const CODEC_RAW_VERSION: u32 = 1;

/// Exact KV bytes exported by the active native runtime and restored directly
/// into that runtime without a storage transcode.
pub const CODEC_NATIVE_KV_PAGE: &str = "native-kv-page";
/// Version of the native KV segment contract. The concrete runtime ABI,
/// tensor types, geometry, and platform are bound by the exact-state identity
/// and the serialized runtime page descriptor carried by the manifest.
pub const CODEC_NATIVE_KV_PAGE_VERSION: u32 = 1;
/// CacheGen archive followed by an exact recurrent-state tail.
pub const CODEC_CACHEGEN_KV_ENVELOPE: &str = "cachegen-kv-envelope";
pub const CODEC_CACHEGEN_KV_ENVELOPE_VERSION: u32 = 1;

/// The codec used to encode a payload's segment bytes, stamped into the
/// manifest so representations are explicit and negotiable.
///
/// Only [`CODEC_RAW`] is implemented today. The identity is recorded so future
/// lossless or compressed codecs are namespaced rather than guessed, and so an
/// older build refuses a payload it cannot decode instead of returning corrupt
/// state. Backward compatibility is handled by the manifest version, not a
/// serde default: a current-version manifest must stamp its codec explicitly
/// (so the field cannot be stripped to force a raw reinterpretation), while a
/// [`LEGACY_MANIFEST_VERSION`] manifest predates codec identity and is
/// normalized to raw on decode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PayloadCodec {
    /// Codec name, e.g. `raw`.
    pub name: String,
    /// On-disk representation version within `name`.
    pub version: u32,
}

impl PayloadCodec {
    /// The raw, uncompressed codec every current payload uses.
    pub fn raw() -> Self {
        Self {
            name: CODEC_RAW.to_string(),
            version: CODEC_RAW_VERSION,
        }
    }

    pub fn cachegen_kv_envelope() -> Self {
        Self {
            name: CODEC_CACHEGEN_KV_ENVELOPE.to_string(),
            version: CODEC_CACHEGEN_KV_ENVELOPE_VERSION,
        }
    }

    /// Whether this build can assemble a payload encoded with this codec.
    /// Only the exact raw name and version are supported; any other name or a
    /// future raw version is unknown and must be refused before assembly.
    pub fn is_supported(&self) -> bool {
        (self.name == CODEC_RAW && self.version == CODEC_RAW_VERSION)
            || (self.name == CODEC_CACHEGEN_KV_ENVELOPE
                && self.version == CODEC_CACHEGEN_KV_ENVELOPE_VERSION)
    }
}

impl Default for PayloadCodec {
    fn default() -> Self {
        Self::raw()
    }
}

/// What a codec's output promises at assembly time. The class is contract,
/// not documentation: an exact entry can satisfy an exact lookup; a lossy
/// entry can only satisfy a lookup that accepts its codec family and
/// calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodecClass {
    /// Decodes to the exact segment bytes the writer held. Payload digest
    /// verification applies at assembly.
    Exact,
    /// Decodes to a calibrated approximation of the segment bytes. Never
    /// verified against a payload digest; must never satisfy an exact
    /// lookup.
    Lossy,
}

impl CodecClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Lossy => "lossy",
        }
    }
}

/// Per-segment codec identity: which codec, which representation version,
/// what it promises, and (for lossy codecs) which calibration produced it.
/// Required on every segment of a current-version manifest.
///
/// The identity is namespaced per segment rather than per payload so one
/// payload can mix raw and compressed segments (e.g. a grown prefix whose
/// leading windows are compressed and whose new tail is raw), and so a v4
/// manifest with the identity stripped from a segment is detectable and
/// rejected instead of silently inheriting the payload codec or raw.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentCodecIdentity {
    /// Segment codec name, e.g. `raw`. Distinct namespace from the payload
    /// codec so entry formats version independently.
    pub name: String,
    /// On-disk representation version within `name`.
    pub version: u32,
    /// What decoding promises: [`CodecClass::Exact`] entries verify against
    /// the payload digest; [`CodecClass::Lossy`] entries never do.
    pub class: CodecClass,
    /// Byte length the codec decodes to. Must equal the assembled payload
    /// region this segment covers; a mismatch is corruption, not a hint.
    pub decoded_len: u64,
    /// Calibration namespace for lossy codecs (e.g. the CacheGen CDF
    /// calibration digest). Must be empty for exact codecs; lossy lookups
    /// only ever match a payload calibrated with the same digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_digest: Option<String>,
}

impl SegmentCodecIdentity {
    /// Identity for a segment stored verbatim: the exact raw codec at the
    /// store's raw version, decoding to `segment.bytes`.
    pub fn raw(encoded_len: u64) -> Self {
        Self {
            name: CODEC_RAW.to_string(),
            version: CODEC_RAW_VERSION,
            class: CodecClass::Exact,
            decoded_len: encoded_len,
            calibration_digest: None,
        }
    }

    /// Identity for a runtime-native KV page segment stored verbatim. The
    /// store performs no encode/decode step; restore imports these same bytes
    /// through the runtime page descriptor associated with the manifest.
    pub fn native_kv_page(encoded_len: u64) -> Self {
        Self {
            name: CODEC_NATIVE_KV_PAGE.to_string(),
            version: CODEC_NATIVE_KV_PAGE_VERSION,
            class: CodecClass::Exact,
            decoded_len: encoded_len,
            calibration_digest: None,
        }
    }

    pub fn cachegen_archive(decoded_len: u64, calibration_digest: String) -> Self {
        crate::cachegen::lmcache::segment_identity(decoded_len, calibration_digest)
    }

    /// Whether this identity names the supported native KV passthrough
    /// representation. This is stricter than a name check: an identity with
    /// the wrong version, class, or calibration is not native passthrough.
    pub fn is_native_kv_page(&self) -> bool {
        self.name == CODEC_NATIVE_KV_PAGE
            && self.version == CODEC_NATIVE_KV_PAGE_VERSION
            && self.class == CodecClass::Exact
            && self.calibration_digest.is_none()
    }

    pub fn is_cachegen_archive(&self) -> bool {
        self.name == crate::cachegen::lmcache::CACHEGEN_CODEC_NAME
            && self.version == crate::cachegen::lmcache::CACHEGEN_CODEC_VERSION
            && self.class == CodecClass::Lossy
            && self
                .calibration_digest
                .as_deref()
                .is_some_and(|digest| !digest.is_empty())
    }

    /// Whether this build can assemble a segment encoded with this identity.
    /// Exact raw and native KV passthrough are implemented. Both assemble
    /// verbatim; anything else is unknown and must be refused before assembly.
    pub fn is_supported(&self) -> bool {
        let supported_representation = (self.name == CODEC_RAW
            && self.version == CODEC_RAW_VERSION)
            || self.is_native_kv_page()
            || self.is_cachegen_archive();
        supported_representation
            && (self.is_cachegen_archive()
                || (self.class == CodecClass::Exact && self.calibration_digest.is_none()))
    }

    /// Internal consistency the capability negotiation relies on: an exact
    /// identity must be raw-shaped (decoded length equals the stored bytes,
    /// no calibration), and every identity must agree that decoding yields
    /// what the manifest says it yields.
    pub fn is_self_consistent(&self, encoded_len: u64) -> bool {
        match self.class {
            CodecClass::Exact => {
                self.decoded_len == encoded_len && self.calibration_digest.is_none()
            }
            // A lossy identity must declare a calibration to decode against;
            // its decoded length is checked against the payload layout at
            // negotiation, not against the stored bytes.
            CodecClass::Lossy => self.calibration_digest.is_some(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HandoffSegmentRef {
    pub index: u32,
    pub offset: u64,
    pub bytes: u64,
    pub digest: String,
    /// Per-segment codec identity: what the stored bytes are, what decoding
    /// them promises, and (for lossy codecs) which calibration applies.
    /// Required on a current-version manifest; the `Option` exists only so a
    /// v4 manifest with the identity stripped is *detectable* and rejected —
    /// it never falls back to the payload codec or to raw.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codec_identity: Option<SegmentCodecIdentity>,
    /// Per-segment metadata for page-stream payloads (serialized
    /// `RuntimeKvPageDesc` plus token range), opaque to the store.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta_json: Option<String>,
}

/// How a manifest's key is derived.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KeyKind {
    /// `payload_digest` is BLAKE3 over the assembled payload bytes. Legacy
    /// entries and any manifest that does not declare otherwise.
    #[default]
    WireBlake3,
    /// `payload_digest` is a composition root over the ordered segment list,
    /// bound to `segmentation_id`. Commit and restore verify it without
    /// hashing payload bytes.
    CompositionRoot,
    /// `payload_digest` names another tier's entry. It is an in-memory index
    /// record (the L2 mirror), never a persisted L3 manifest, and nothing may
    /// claim a derivation for it.
    Alias,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffManifest {
    pub version: u32,
    /// Numerical model identity used by exact model-scoped lifecycle
    /// operations. This intentionally spans every split stage of one model.
    pub model_identity: String,
    /// `exact_state_identity` of the producing runtime — the numerical
    /// identity a loader must match before importing this state.
    pub state_identity: String,
    pub payload_kind: String,
    /// Codec identity of the segment payload bytes. Always present on a decoded
    /// manifest: a current-version manifest must carry it explicitly, and a
    /// [`LEGACY_MANIFEST_VERSION`] manifest is normalized to raw on decode. The
    /// field is optional only so a version-3 manifest with the codec stripped
    /// is detectable and rejected rather than silently defaulted to raw.
    #[serde(default)]
    pub codec: Option<PayloadCodec>,
    pub total_bytes: u64,
    /// BLAKE3 of the assembled payload; also the manifest's key.
    pub payload_digest: String,
    /// How `payload_digest` was derived. Absent in older manifests, which were
    /// all wire digests.
    #[serde(default)]
    pub key_kind: KeyKind,
    /// Identity of the cutting scheme that produced `segments`. Required for
    /// composition-root keys: the same bytes cut differently are different
    /// entries, and this is what makes that explicit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segmentation_id: Option<String>,
    pub segments: Vec<HandoffSegmentRef>,
    pub kv_bytes: u64,
    /// Decoded native KV length. Equal to `kv_bytes` for native pages; for a
    /// CacheGen envelope `kv_bytes` is the archive boundary.
    #[serde(default)]
    pub kv_decoded_bytes: u64,
    pub recurrent_bytes: u64,
    /// Serialized `RuntimeKvPageDesc` for kv-recurrent payloads; opaque to
    /// this crate so the store does not depend on the runtime.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_desc_json: Option<String>,
    pub token_count: u64,
    pub continuation_token: i32,
    /// Greedy continuation produced by the exporting session, when known —
    /// lets an offline restore self-verify determinism.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub expected_tokens: Vec<i32>,
}

impl HandoffManifest {
    pub fn new(state_identity: String, payload_kind: String) -> Self {
        Self::new_for_model(state_identity.clone(), state_identity, payload_kind)
    }

    pub fn new_for_model(
        model_identity: String,
        state_identity: String,
        payload_kind: String,
    ) -> Self {
        Self {
            version: MANIFEST_VERSION,
            model_identity,
            state_identity,
            payload_kind,
            codec: Some(PayloadCodec::raw()),
            total_bytes: 0,
            payload_digest: String::new(),
            key_kind: KeyKind::WireBlake3,
            segmentation_id: None,
            segments: Vec::new(),
            kv_bytes: 0,
            kv_decoded_bytes: 0,
            recurrent_bytes: 0,
            kv_desc_json: None,
            token_count: 0,
            continuation_token: 0,
            expected_tokens: Vec::new(),
        }
    }

    /// Whether this manifest contains runtime-native KV page segments.
    pub fn uses_native_kv_passthrough(&self) -> bool {
        self.segments.iter().any(|segment| {
            segment
                .codec_identity
                .as_ref()
                .is_some_and(SegmentCodecIdentity::is_native_kv_page)
        })
    }

    pub fn uses_cachegen_kv(&self) -> bool {
        self.segments.iter().any(|segment| {
            segment
                .codec_identity
                .as_ref()
                .is_some_and(SegmentCodecIdentity::is_cachegen_archive)
        })
    }
}

/// What one payload restore cost.
///
/// Restore latency tracks physical reads, packs touched and per-object work,
/// not bytes: a payload assembled from many small spill packs shows a high
/// `physical_reads` with a low `max_read_nanos`, and `max_read_nanos` close to
/// `read_nanos` means the reads were serialized.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AssembleStats {
    /// Logical manifest segments assembled.
    pub segments: u64,
    /// Distinct packs the manifest's segments live in.
    pub packs: u64,
    /// Pack files opened during this restore.
    pub packs_opened: u64,
    /// Physical read calls issued. One read may carry many logical segments.
    pub physical_reads: u64,
    /// Bytes read from disk.
    pub bytes_read: u64,
    /// Payload bytes assembled.
    pub bytes: u64,
    /// Wall time spent inside pack reads.
    pub read_nanos: u64,
    /// Longest single pack read.
    pub max_read_nanos: u64,
    /// Segments read from the uncompressed segment directory.
    pub loose_segment_reads: u64,
    /// CPU time hashing segment digests, summed across workers.
    pub verify_nanos: u64,
    /// CPU time hashing the assembled payload.
    pub hash_nanos: u64,
    /// Bytes fed to the payload hasher.
    pub hashed_bytes: u64,
    /// Whole-assemble wall time, including digest verification.
    pub total_nanos: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentPut {
    pub new: bool,
    pub bytes: u64,
}

/// What the store is allowed to occupy. Both bounds are hard: the budget caps
/// what the cache manages, the reserve caps what it may take from everything
/// else on the filesystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreLimits {
    /// Whole-root cap over segments, manifests, indexes and in-flight bytes.
    /// Zero disables the cap; the public configuration surface rejects zero
    /// rather than treating it as unlimited, so only internal callers and
    /// tests can reach that state.
    pub budget_bytes: u64,
    /// Free space the store preserves for everything else on the filesystem.
    pub minimum_free_bytes: u64,
}

impl StoreLimits {
    pub fn new(budget_bytes: u64, minimum_free_bytes: u64) -> Self {
        Self {
            budget_bytes,
            minimum_free_bytes,
        }
    }
}

/// Why a write was not admitted. Every refusal is a miss with a stable reason,
/// never a partial or best-effort write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteRefusal {
    /// The entry alone is larger than the whole budget. Storing it could only
    /// succeed by evicting everything else and still overflowing.
    SkippedOversize,
    /// The budget is full and eviction cannot free enough (everything else is
    /// pinned or the entry needs more than the whole cap).
    InsufficientSpace,
    /// The filesystem is at the minimum-free reserve. Reads still serve; the
    /// tier stops writing until space comes back.
    ReadOnlyLowSpace,
}

impl WriteRefusal {
    /// Stable reason code for the status surface.
    pub fn reason(self) -> &'static str {
        match self {
            Self::SkippedOversize => "skipped_oversize",
            Self::InsufficientSpace => "insufficient_space",
            Self::ReadOnlyLowSpace => "read_only_low_space",
        }
    }
}
