//! Versioned binary container for L3 manifests and packed indexes.
//!
//! Layout: `MAGIC` (4 bytes) | format version (u16 LE) | kind (u8) | body.
//! Strings are length-prefixed UTF-8, digests are 32 raw bytes, and every
//! read is bounds-checked. The segment count is validated against the file
//! size before any allocation, so a truncated or hostile file fails closed
//! instead of allocating from a declared count.
//!
//! A manifest written here is an order of magnitude smaller than the JSON it
//! replaced (the segment table dominates both) and decodes as a bounds-checked
//! copy rather than a tokenising pass.

use anyhow::{Context, Result, bail};

use super::{
    CodecClass, HandoffManifest, HandoffSegmentRef, KeyKind, PayloadCodec, SegmentCodecIdentity,
};

pub(super) const MAGIC: &[u8; 4] = b"SKV5";
const FORMAT_VERSION: u16 = 1;
pub(super) const KIND_MANIFEST: u8 = 1;
pub(super) const KIND_INDEX: u8 = 2;

/// Smallest plausible encoded segment: index + offset + length + digest + a
/// codec-presence byte. Used to bound a declared count before allocating.
const MIN_SEGMENT_BYTES: u64 = 4 + 8 + 8 + 32 + 1;

/// Whether these bytes are one of our containers.
pub(super) fn looks_binary(bytes: &[u8]) -> bool {
    bytes.len() >= 7 && &bytes[..4] == MAGIC
}

pub(super) fn encode_manifest(manifest: &HandoffManifest) -> Result<Vec<u8>> {
    let mut writer = Writer::new(KIND_MANIFEST);
    writer.u32(manifest.version);
    writer.u8(match manifest.key_kind {
        KeyKind::WireBlake3 => 0,
        KeyKind::CompositionRoot => 1,
        KeyKind::Alias => 2,
    });
    writer.str(&manifest.model_identity);
    writer.str(&manifest.state_identity);
    writer.str(&manifest.payload_kind);
    writer.str(&manifest.payload_digest);
    writer.opt_str(manifest.segmentation_id.as_deref());
    match &manifest.codec {
        Some(codec) => {
            writer.u8(1);
            writer.str(&codec.name);
            writer.u32(codec.version);
        }
        None => writer.u8(0),
    }
    writer.u64(manifest.total_bytes);
    writer.u64(manifest.kv_bytes);
    writer.u64(manifest.kv_decoded_bytes);
    writer.u64(manifest.recurrent_bytes);
    writer.u64(manifest.token_count);
    writer.i32(manifest.continuation_token);
    writer.opt_str(manifest.kv_desc_json.as_deref());
    writer.u32(u32::try_from(manifest.expected_tokens.len()).context("too many expected tokens")?);
    for token in &manifest.expected_tokens {
        writer.i32(*token);
    }
    writer.u32(u32::try_from(manifest.segments.len()).context("too many segments")?);
    for segment in &manifest.segments {
        writer.u32(segment.index);
        writer.u64(segment.offset);
        writer.u64(segment.bytes);
        writer.digest(&segment.digest)?;
        match &segment.codec_identity {
            Some(identity) => {
                writer.u8(1);
                writer.str(&identity.name);
                writer.u32(identity.version);
                writer.u8(match identity.class {
                    CodecClass::Exact => 0,
                    CodecClass::Lossy => 1,
                });
                writer.u64(identity.decoded_len);
                writer.opt_str(identity.calibration_digest.as_deref());
            }
            None => writer.u8(0),
        }
        writer.opt_str(segment.meta_json.as_deref());
    }
    Ok(writer.finish())
}

pub(super) fn decode_manifest(bytes: &[u8]) -> Result<HandoffManifest> {
    let mut reader = Reader::open(bytes, KIND_MANIFEST)?;
    let version = reader.u32()?;
    let key_kind = match reader.u8()? {
        0 => KeyKind::WireBlake3,
        1 => KeyKind::CompositionRoot,
        2 => KeyKind::Alias,
        other => bail!("unknown key kind {other} in manifest"),
    };
    let model_identity = reader.str()?;
    let state_identity = reader.str()?;
    let payload_kind = reader.str()?;
    let payload_digest = reader.str()?;
    let segmentation_id = reader.opt_str()?;
    let codec = match reader.u8()? {
        0 => None,
        1 => Some(PayloadCodec {
            name: reader.str()?,
            version: reader.u32()?,
        }),
        other => bail!("unknown payload codec presence byte {other}"),
    };
    let total_bytes = reader.u64()?;
    let kv_bytes = reader.u64()?;
    let kv_decoded_bytes = reader.u64()?;
    let recurrent_bytes = reader.u64()?;
    let token_count = reader.u64()?;
    let continuation_token = reader.i32()?;
    let kv_desc_json = reader.opt_str()?;
    let expected_count = reader.count(MIN_SEGMENT_BYTES / 8)?;
    let mut expected_tokens = Vec::with_capacity(expected_count);
    for _ in 0..expected_count {
        expected_tokens.push(reader.i32()?);
    }
    let count = reader.count(MIN_SEGMENT_BYTES)?;
    let mut segments = Vec::with_capacity(count);
    for _ in 0..count {
        let index = reader.u32()?;
        let offset = reader.u64()?;
        let bytes = reader.u64()?;
        let digest = reader.digest()?;
        let codec_identity = match reader.u8()? {
            0 => None,
            1 => Some(SegmentCodecIdentity {
                name: reader.str()?,
                version: reader.u32()?,
                class: match reader.u8()? {
                    0 => CodecClass::Exact,
                    1 => CodecClass::Lossy,
                    other => bail!("unknown codec class {other}"),
                },
                decoded_len: reader.u64()?,
                calibration_digest: reader.opt_str()?,
            }),
            other => bail!("unknown segment codec presence byte {other}"),
        };
        let meta_json = reader.opt_str()?;
        segments.push(HandoffSegmentRef {
            index,
            offset,
            bytes,
            digest,
            codec_identity,
            meta_json,
        });
    }
    reader.finish()?;
    Ok(HandoffManifest {
        version,
        model_identity,
        state_identity,
        payload_kind,
        codec,
        total_bytes,
        payload_digest,
        key_kind,
        segmentation_id,
        segments,
        kv_bytes,
        kv_decoded_bytes,
        recurrent_bytes,
        kv_desc_json,
        token_count,
        continuation_token,
        expected_tokens,
    })
}

pub(super) fn encode_index(
    version: u32,
    payload_digest: &str,
    segments: &[Option<super::packed::PackedSegmentLocation>],
) -> Result<Vec<u8>> {
    let mut writer = Writer::new(KIND_INDEX);
    writer.u32(version);
    writer.str(payload_digest);
    writer.u32(u32::try_from(segments.len()).context("too many index entries")?);
    for segment in segments {
        match segment {
            Some(location) => {
                writer.u8(1);
                writer.digest(&location.pack_digest)?;
                writer.u64(location.offset);
                writer.u64(location.bytes);
            }
            None => writer.u8(0),
        }
    }
    Ok(writer.finish())
}

pub(super) fn decode_index(
    bytes: &[u8],
) -> Result<(
    u32,
    String,
    Vec<Option<super::packed::PackedSegmentLocation>>,
)> {
    let mut reader = Reader::open(bytes, KIND_INDEX)?;
    let version = reader.u32()?;
    let payload_digest = reader.str()?;
    let count = reader.count(1 + 32 + 8 + 8)?;
    let mut segments = Vec::with_capacity(count);
    for _ in 0..count {
        match reader.u8()? {
            0 => segments.push(None),
            1 => segments.push(Some(super::packed::PackedSegmentLocation {
                pack_digest: reader.digest()?,
                offset: reader.u64()?,
                bytes: reader.u64()?,
            })),
            other => bail!("unknown index entry presence byte {other}"),
        }
    }
    reader.finish()?;
    Ok((version, payload_digest, segments))
}

struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn new(kind: u8) -> Self {
        let mut bytes = Vec::with_capacity(1024);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.push(kind);
        Self { bytes }
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn i32(&mut self, value: i32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn str(&mut self, value: &str) {
        self.u32(value.len() as u32);
        self.bytes.extend_from_slice(value.as_bytes());
    }

    fn opt_str(&mut self, value: Option<&str>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.str(value);
            }
            None => self.u8(0),
        }
    }

    fn digest(&mut self, digest: &str) -> Result<()> {
        self.bytes.extend_from_slice(&digest_to_bytes(digest)?);
        Ok(())
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn open(bytes: &'a [u8], kind: u8) -> Result<Self> {
        if !looks_binary(bytes) {
            bail!("not a skippy manifest container");
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != FORMAT_VERSION {
            bail!("unsupported container version {version}");
        }
        if bytes[6] != kind {
            bail!("unexpected container kind {}", bytes[6]);
        }
        Ok(Self { bytes, position: 7 })
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(len)
            .context("container range overflows")?;
        let slice = self
            .bytes
            .get(self.position..end)
            .context("container is truncated")?;
        self.position = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        let mut raw = [0u8; 4];
        raw.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(raw))
    }

    fn u64(&mut self) -> Result<u64> {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(raw))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.u32()?.to_le_bytes()))
    }

    fn str(&mut self) -> Result<String> {
        let len = usize::try_from(self.u32()?).context("string length exceeds usize")?;
        let bytes = self.take(len)?;
        Ok(std::str::from_utf8(bytes)
            .context("container string is not UTF-8")?
            .to_string())
    }

    fn opt_str(&mut self) -> Result<Option<String>> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.str()?)),
            other => bail!("unknown optional-string presence byte {other}"),
        }
    }

    fn digest(&mut self) -> Result<String> {
        Ok(digest_from_bytes(self.take(32)?))
    }

    /// Read a count and bound it by the bytes remaining.
    fn count(&mut self, unit: u64) -> Result<usize> {
        let declared = u64::from(self.u32()?);
        let remaining = (self.bytes.len() - self.position) as u64;
        if declared.saturating_mul(unit.max(1)) > remaining {
            bail!("container declares {declared} entries but has {remaining} bytes left");
        }
        usize::try_from(declared).context("count exceeds usize")
    }

    fn finish(&self) -> Result<()> {
        if self.position != self.bytes.len() {
            bail!(
                "container has {} trailing bytes",
                self.bytes.len() - self.position
            );
        }
        Ok(())
    }
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

fn digest_from_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}
