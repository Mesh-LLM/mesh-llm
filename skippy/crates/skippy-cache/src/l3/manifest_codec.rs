//! Versioned container for L3 manifests and packed indexes.
//!
//! The body is a protobuf message (`skippy.kv.v1`); the container prefixes it
//! with a four-byte magic so a reader can tell a binary entry from a legacy
//! JSON one without guessing at the first byte. Decoding is fail-closed:
//! protobuf bounds repeated fields by the input length, so a truncated file
//! cannot make the reader allocate from a declared count.
//!
//! A manifest written here is roughly half the size of the JSON it replaced
//! (the segment table dominates both) and decodes without tokenising.

use anyhow::{Context, Result, bail};
use prost::Message;
use skippy_protocol::proto::kv;

use super::{
    CodecClass, HandoffManifest, HandoffSegmentRef, KeyKind, PayloadCodec, SegmentCodecIdentity,
};

pub(super) const MAGIC: &[u8; 4] = b"SKV5";
pub(super) const KIND_MANIFEST: u8 = 1;
pub(super) const KIND_INDEX: u8 = 2;

/// Whether these bytes are one of our containers.
pub(super) fn looks_binary(bytes: &[u8]) -> bool {
    bytes.len() >= 5 && &bytes[..4] == MAGIC
}

pub(super) fn encode_manifest(manifest: &HandoffManifest) -> Result<Vec<u8>> {
    let message = kv::KvManifest {
        version: manifest.version,
        key_kind: match manifest.key_kind {
            KeyKind::WireBlake3 => kv::KeyKind::WireBlake3 as i32,
            KeyKind::CompositionRoot => kv::KeyKind::CompositionRoot as i32,
            KeyKind::Alias => kv::KeyKind::Alias as i32,
        },
        model_identity: manifest.model_identity.clone(),
        state_identity: manifest.state_identity.clone(),
        payload_kind: manifest.payload_kind.clone(),
        payload_digest: manifest.payload_digest.clone(),
        segmentation_id: manifest.segmentation_id.clone(),
        codec: manifest.codec.as_ref().map(|codec| kv::KvPayloadCodec {
            name: codec.name.clone(),
            version: codec.version,
        }),
        total_bytes: manifest.total_bytes,
        kv_bytes: manifest.kv_bytes,
        kv_decoded_bytes: manifest.kv_decoded_bytes,
        recurrent_bytes: manifest.recurrent_bytes,
        token_count: manifest.token_count,
        continuation_token: manifest.continuation_token,
        kv_desc_json: manifest.kv_desc_json.clone(),
        expected_tokens: manifest.expected_tokens.clone(),
        segments: manifest
            .segments
            .iter()
            .map(|segment| {
                Ok(kv::KvSegment {
                    index: segment.index,
                    offset: segment.offset,
                    bytes: segment.bytes,
                    digest: digest_to_bytes(&segment.digest)?.to_vec(),
                    codec_identity: segment.codec_identity.as_ref().map(|identity| {
                        kv::KvCodecIdentity {
                            name: identity.name.clone(),
                            version: identity.version,
                            class: match identity.class {
                                CodecClass::Exact => kv::CodecClass::Exact as i32,
                                CodecClass::Lossy => kv::CodecClass::Lossy as i32,
                            },
                            decoded_len: identity.decoded_len,
                            calibration_digest: identity.calibration_digest.clone(),
                        }
                    }),
                    meta_json: segment.meta_json.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?,
    };
    Ok(frame(KIND_MANIFEST, &message.encode_to_vec()))
}

pub(super) fn decode_manifest(bytes: &[u8]) -> Result<HandoffManifest> {
    let body = unframe(bytes, KIND_MANIFEST)?;
    let message = kv::KvManifest::decode(body).context("malformed manifest protobuf")?;
    if message.version == 0 {
        // A truncated frame can decode as a valid empty message; no real
        // manifest omits its version.
        bail!("manifest container has no version");
    }
    let segments = message
        .segments
        .into_iter()
        .map(|segment| {
            Ok(HandoffSegmentRef {
                index: segment.index,
                offset: segment.offset,
                bytes: segment.bytes,
                digest: digest_from_slice(&segment.digest)?,
                codec_identity: segment
                    .codec_identity
                    .map(|identity| -> Result<SegmentCodecIdentity> {
                        Ok(SegmentCodecIdentity {
                            name: identity.name,
                            version: identity.version,
                            class: match kv::CodecClass::try_from(identity.class) {
                                Ok(kv::CodecClass::Exact) => CodecClass::Exact,
                                Ok(kv::CodecClass::Lossy) => CodecClass::Lossy,
                                Err(_) => bail!("unknown codec class {}", identity.class),
                            },
                            decoded_len: identity.decoded_len,
                            calibration_digest: identity.calibration_digest,
                        })
                    })
                    .transpose()?,
                meta_json: segment.meta_json,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(HandoffManifest {
        version: message.version,
        model_identity: message.model_identity,
        state_identity: message.state_identity,
        payload_kind: message.payload_kind,
        codec: message.codec.map(|codec| PayloadCodec {
            name: codec.name,
            version: codec.version,
        }),
        total_bytes: message.total_bytes,
        payload_digest: message.payload_digest,
        key_kind: match kv::KeyKind::try_from(message.key_kind) {
            Ok(kv::KeyKind::WireBlake3) => KeyKind::WireBlake3,
            Ok(kv::KeyKind::CompositionRoot) => KeyKind::CompositionRoot,
            Ok(kv::KeyKind::Alias) => KeyKind::Alias,
            Err(_) => bail!("unknown key kind {}", message.key_kind),
        },
        segmentation_id: message.segmentation_id,
        segments,
        kv_bytes: message.kv_bytes,
        kv_decoded_bytes: message.kv_decoded_bytes,
        recurrent_bytes: message.recurrent_bytes,
        kv_desc_json: message.kv_desc_json,
        token_count: message.token_count,
        continuation_token: message.continuation_token,
        expected_tokens: message.expected_tokens,
    })
}

pub(super) fn encode_index(
    version: u32,
    payload_digest: &str,
    segments: &[Option<super::packed::PackedSegmentLocation>],
) -> Result<Vec<u8>> {
    let message = kv::KvPackedIndex {
        version,
        payload_digest: payload_digest.to_string(),
        entries: segments
            .iter()
            .map(|segment| -> Result<kv::KvPackedEntry> {
                Ok(kv::KvPackedEntry {
                    location: segment
                        .as_ref()
                        .map(|location| -> Result<kv::KvPackedLocation> {
                            Ok(kv::KvPackedLocation {
                                pack_digest: digest_to_bytes(&location.pack_digest)?.to_vec(),
                                offset: location.offset,
                                bytes: location.bytes,
                            })
                        })
                        .transpose()?,
                })
            })
            .collect::<Result<Vec<_>>>()?,
    };
    Ok(frame(KIND_INDEX, &message.encode_to_vec()))
}

pub(super) fn decode_index(
    bytes: &[u8],
) -> Result<(
    u32,
    String,
    Vec<Option<super::packed::PackedSegmentLocation>>,
)> {
    let body = unframe(bytes, KIND_INDEX)?;
    let message = kv::KvPackedIndex::decode(body).context("malformed packed index protobuf")?;
    let segments = message
        .entries
        .into_iter()
        .map(|entry| {
            entry
                .location
                .map(|location| {
                    Ok(super::packed::PackedSegmentLocation {
                        pack_digest: digest_from_slice(&location.pack_digest)?,
                        offset: location.offset,
                        bytes: location.bytes,
                    })
                })
                .transpose()
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((message.version, message.payload_digest, segments))
}

fn frame(kind: u8, body: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(5 + body.len());
    bytes.extend_from_slice(MAGIC);
    bytes.push(kind);
    bytes.extend_from_slice(body);
    bytes
}

fn unframe(bytes: &[u8], kind: u8) -> Result<&[u8]> {
    if !looks_binary(bytes) {
        bail!("not a skippy manifest container");
    }
    if bytes[4] != kind {
        bail!("unexpected container kind {}", bytes[4]);
    }
    Ok(&bytes[5..])
}

fn digest_to_bytes(digest: &str) -> Result<[u8; 32]> {
    let raw = digest.as_bytes();
    if raw.len() != 64 {
        bail!("segment digest is not 64 hex characters");
    }
    let mut out = [0u8; 32];
    for (index, chunk) in raw.as_chunks::<2>().0.iter().enumerate() {
        let high = (chunk[0] as char)
            .to_digit(16)
            .context("digest is not hex")?;
        let low = (chunk[1] as char)
            .to_digit(16)
            .context("digest is not hex")?;
        out[index] = ((high << 4) | low) as u8;
    }
    Ok(out)
}

fn digest_from_slice(bytes: &[u8]) -> Result<String> {
    if bytes.len() != 32 {
        bail!("segment digest must be 32 bytes, got {}", bytes.len());
    }
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Ok(out)
}
