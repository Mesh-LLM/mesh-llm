//! The CacheGen v1 segment container and its deterministic CPU encoder and
//! decoder.
//!
//! One container holds exactly one self-contained segment tile: affine
//! 4-bit quantization, token-axis delta decorrelation, static byte-rANS.
//! The container carries the calibration values and the 16-entry symbol
//! histogram (the CDF metadata), so decoding is self-contained and the
//! rANS table is bit-identical on both sides by construction.
//!
//! The calibration digest binds the calibration and tile shape into the
//! segment identity: a lossy lookup only matches entries calibrated
//! identically, and can never satisfy an exact lookup (the
//! `CodecClass::Lossy` contract on the v4 per-segment identity).
//!
//! Determinism contract: encoding is a pure function of the input bytes.
//! No wall clock, no map iteration, no float reassociation. The GPU work
//! in the later CubeCL slices must produce byte-identical containers.

use anyhow::{Result, anyhow, bail};
use skippy_protocol::binary::{f16_bits_to_f32, f32_to_f16_bits};

use super::rans::{RansDecoder, RansEncoder, SymbolTable};
use super::reference::{self, Calibration, TOKEN_COUNT};
use crate::l3::{CodecClass, SegmentCodecIdentity};

/// Segment codec name stamped into the per-segment identity.
pub const CACHEGEN_CODEC_NAME: &str = "cachegen";
/// Container format version.
pub const CACHEGEN_CODEC_VERSION: u32 = 1;
/// Container magic: CacheGen, version 1.
const MAGIC: [u8; 4] = *b"CGv1";
/// Fixed prefix: 4 magic + 2 dims + 4 rows + 4 min bits + 4 scale bits +
/// 4 stream length + 2 reserved.
const FIXED_HEADER_LEN: usize = 24;
/// The histogram travels as `TOKEN_COUNT` little-endian `u32` counts.
/// Wide counts keep the format honest at real tile sizes: a 4096x128 tile
/// is 524,288 symbols and would overflow a `u16` count, and CGv1 is one
/// format for every tile the codec can encode.
const HISTOGRAM_LEN: usize = TOKEN_COUNT * 4;
/// Total bytes before the rANS stream.
const HEADER_LEN: usize = FIXED_HEADER_LEN + HISTOGRAM_LEN;
/// Hard format ceiling on the tile one container may declare, in values.
/// Bounds `rows * dims` and the decoded allocation (`count * 2` bytes)
/// before any capacity is reserved. Far above any real KV segment tile
/// (the measured 4096x128 shape is 524,288 values), far below the
/// unbounded product a hostile `rows`/`dims` pair can claim; raising it
/// is a format-version decision, not a per-parse judgment call.
pub const MAX_DECODED_VALUES: usize = 1 << 30;

/// The per-segment identity a CacheGen segment carries: lossy, calibrated.
pub fn segment_identity(decoded_len: u64, calibration_digest: String) -> SegmentCodecIdentity {
    SegmentCodecIdentity {
        name: CACHEGEN_CODEC_NAME.to_string(),
        version: CACHEGEN_CODEC_VERSION,
        class: CodecClass::Lossy,
        decoded_len,
        calibration_digest: Some(calibration_digest),
    }
}

/// BLAKE3 digest binding the calibration parameters to the tile shape, in
/// f32 bit-exact form. Two entries decode against each other only if this
/// digest matches.
pub fn calibration_digest(calibration: &Calibration, dims: usize) -> Result<String> {
    let dims = u16::try_from(dims).map_err(|_| anyhow!("dims exceed container field width"))?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"skippy-cachegen-calibration-v1");
    hasher.update(&dims.to_le_bytes());
    hasher.update(&calibration.min_bits.to_le_bytes());
    hasher.update(&calibration.scale_bits.to_le_bytes());
    Ok(hasher.finalize().to_hex().to_string())
}

/// Encodes one segment of little-endian f16 KV values as a CacheGen v1
/// container. `dims` is the number of values per token along the token
/// axis (the delta stride); the value count must be a non-zero multiple
/// of it and each of its parts must fit the container's fixed fields.
pub fn encode_f16_segment(raw_segment: &[u8], dims: usize) -> Result<Vec<u8>> {
    if dims == 0 || raw_segment.is_empty() || !raw_segment.len().is_multiple_of(2) {
        bail!("segment must be a non-empty run of f16 pairs");
    }
    let values: Vec<f32> = raw_segment
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| f16_bits_to_f32(u16::from_le_bytes(*bytes)))
        .collect();
    if !values.len().is_multiple_of(dims) {
        bail!(
            "segment shape mismatch: {} values are not rows of {dims}",
            values.len()
        );
    }
    let rows = u32::try_from(values.len() / dims)
        .map_err(|_| anyhow!("segment row count exceeds container field width"))?;
    let dims16 = u16::try_from(dims).map_err(|_| anyhow!("dims exceed container field width"))?;

    let calibration = reference::calibrate(&values)?;
    let mut symbols = reference::quantize(&calibration, &values)?;
    reference::delta_encode(&mut symbols, dims)?;

    let mut histogram = vec![0u32; TOKEN_COUNT];
    for &symbol in &symbols {
        histogram[usize::from(symbol)] += 1;
    }
    let freqs = reference::histogram_to_freqs(&histogram, symbols.len())?;
    let table = SymbolTable::from_freqs(&freqs).expect("histogram_to_freqs yields a valid table");
    let mut encoder = RansEncoder::new();
    // rANS encodes in reverse stream order.
    for &symbol in symbols.iter().rev() {
        encoder.put(&table, usize::from(symbol));
    }
    let stream = encoder.finish();

    let mut out = Vec::with_capacity(HEADER_LEN + stream.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&dims16.to_le_bytes());
    out.extend_from_slice(&rows.to_le_bytes());
    out.extend_from_slice(&calibration.min_bits.to_le_bytes());
    out.extend_from_slice(&calibration.scale_bits.to_le_bytes());
    let stream_len = u32::try_from(stream.len())
        .map_err(|_| anyhow!("encoded stream exceeds container field width"))?;
    out.extend_from_slice(&stream_len.to_le_bytes());
    out.extend_from_slice(&[0u8; 2]);
    // Counts are `u32` throughout the encoder; the write is direct.
    for &count in &histogram {
        out.extend_from_slice(&count.to_le_bytes());
    }
    out.extend_from_slice(&stream);
    Ok(out)
}

/// Reconstructs the segment's little-endian f16 bytes from a container.
/// The output is within the quantization error of the original segment
/// and has exactly the declared decoded length.
pub fn decode_f16_segment(payload: &[u8]) -> Result<Vec<u8>> {
    let (header, histogram, stream) = parse_container(payload)?;
    let count = header.count;
    let freqs = reference::histogram_to_freqs(&histogram, count)?;
    let table = SymbolTable::from_freqs(&freqs).expect("histogram_to_freqs yields a valid table");

    let mut decoder = RansDecoder::new(stream)
        .ok_or_else(|| anyhow!("cachegen stream shorter than its initial state"))?;
    // The reservation uses the parse-validated count (checked product,
    // under the format ceiling), so a hostile header cannot inflate it;
    // the decoded byte size is computed with checked math as well.
    let mut symbols = Vec::with_capacity(count);
    for _ in 0..count {
        let symbol = decoder
            .get(&table)
            .ok_or_else(|| anyhow!("cachegen stream exhausted before tile completed"))?;
        symbols.push(symbol as u8);
    }
    reference::delta_decode(&mut symbols, header.dims)?;
    let values = reference::dequantize(&header.calibration, &symbols);
    let decoded_bytes = decoded_byte_len(count)?;
    let mut out = Vec::with_capacity(decoded_bytes);
    for value in values {
        out.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
    }
    Ok(out)
}

/// Checked byte length of a decoded tile: `values * 2` (little-endian
/// f16), computed without trusting the caller-provided count.
fn decoded_byte_len(count: usize) -> Result<usize> {
    count
        .checked_mul(2)
        .ok_or_else(|| anyhow!("cachegen decoded size exceeds the address space"))
}

/// Value count a container decodes to (`rows * dims`), for capability
/// negotiation against a segment's declared decoded length before any
/// decode work happens. The count is the parse-validated, bounded
/// product — never an unchecked multiply of header fields.
pub fn decoded_value_count(payload: &[u8]) -> Result<usize> {
    let (header, _, _) = parse_container(payload)?;
    Ok(header.count)
}

/// Calibration and tile shape from a container's header, without touching
/// the stream. Lets a caller digest the calibration into its lookup
/// identity before committing to a decode.
pub fn container_calibration(payload: &[u8]) -> Result<(Calibration, usize)> {
    let (header, _, _) = parse_container(payload)?;
    Ok((header.calibration, header.dims))
}

struct ContainerHeader {
    dims: usize,
    /// Parse-validated `rows * dims`: checked product, under
    /// [`MAX_DECODED_VALUES`], and equal to the histogram total.
    count: usize,
    calibration: Calibration,
}

fn parse_container(payload: &[u8]) -> Result<(ContainerHeader, Vec<u32>, &[u8])> {
    if payload.len() < HEADER_LEN {
        bail!("cachegen container shorter than its fixed header");
    }
    if payload[0..4] != MAGIC {
        bail!("not a cachegen container (bad magic)");
    }
    if payload[FIXED_HEADER_LEN - 2..FIXED_HEADER_LEN] != [0u8; 2] {
        bail!("cachegen container reserved bytes are not zero");
    }
    let dims = u16::from_le_bytes([payload[4], payload[5]]) as usize;
    let rows = u32::from_le_bytes([payload[6], payload[7], payload[8], payload[9]]) as usize;
    let min_bits = u32::from_le_bytes(payload[10..14].try_into().expect("4 bytes"));
    let scale_bits = u32::from_le_bytes(payload[14..18].try_into().expect("4 bytes"));
    let stream_len = u32::from_le_bytes(payload[18..22].try_into().expect("4 bytes")) as usize;
    if dims == 0 || rows == 0 {
        bail!("cachegen container declares an empty tile");
    }
    // The declared tile must be a real shape, not a hostile product: the
    // checked count bounds the decoded-size math below, and the format
    // ceiling bounds it further. Without these, a 92-byte payload could
    // claim billions of decoded values and drive the allocation at decode
    // time.
    let count = rows
        .checked_mul(dims)
        .ok_or_else(|| anyhow!("cachegen tile shape overflows the address space"))?;
    if count > MAX_DECODED_VALUES {
        bail!(
            "cachegen tile declares {count} values, above the format ceiling of {MAX_DECODED_VALUES}"
        );
    }
    // Calibration crosses the trust boundary as raw f32 bits; it must be
    // a usable affine map before anything derives from it. NaN/inf poison
    // every dequantized value, and a negative scale inverts the quantizer
    // ring. scale == 0 stays legal: it is the flat-tile encoding.
    let min = f32::from_bits(min_bits);
    let scale = f32::from_bits(scale_bits);
    if !min.is_finite() || !scale.is_finite() {
        bail!("cachegen calibration carries a non-finite value");
    }
    if scale < 0.0 {
        bail!("cachegen calibration scale is negative");
    }
    let histogram: Vec<u32> = payload[FIXED_HEADER_LEN..HEADER_LEN]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect();
    // The histogram must account for exactly the declared tile. A total
    // that disagrees with `rows * dims` is a corrupt (or hostile) header:
    // accepting it would build a CDF for a tile that never existed and
    // drive unbounded normalization repair at decode time.
    let histogram_total: u64 = histogram.iter().map(|&count| u64::from(count)).sum();
    let expected_total = (rows as u64) * (dims as u64);
    if histogram_total != expected_total {
        bail!(
            "cachegen histogram totals {histogram_total} but the tile declares {expected_total} symbols"
        );
    }
    if payload.len() != HEADER_LEN + stream_len {
        bail!(
            "cachegen container length {} disagrees with its declared stream length {stream_len}",
            payload.len()
        );
    }
    Ok((
        ContainerHeader {
            dims,
            count,
            calibration: Calibration {
                min_bits,
                scale_bits,
            },
        },
        histogram,
        &payload[HEADER_LEN..],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift* so fixtures never depend on an RNG crate.
    struct Xorshift(u64);

    impl Xorshift {
        fn next_unit(&mut self) -> f32 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / 16_777_216.0
        }
    }

    /// A smooth KV-like tile: slowly varying signal plus tiny jitter, the
    /// shape the CacheGen paper's gains come from.
    fn smooth_tile(rows: usize, dims: usize, seed: u64) -> Vec<u8> {
        let mut rng = Xorshift(seed);
        let mut out = Vec::with_capacity(rows * dims * 2);
        for row in 0..rows {
            for column in 0..dims {
                let phase = (row * dims + column) as f32;
                let value = (phase * 0.01).sin() * 0.4 + 0.5;
                let jitter = (rng.next_unit() - 0.5) * 0.01;
                out.extend_from_slice(&f32_to_f16_bits(value + jitter).to_le_bytes());
            }
        }
        out
    }

    #[test]
    fn smooth_segment_compresses_and_round_trips_within_quantization_error() {
        let dims = 16;
        let raw = smooth_tile(128, dims, 0xDECAFBAD);
        let encoded = encode_f16_segment(&raw, dims).expect("encode");
        assert!(
            encoded.len() < raw.len(),
            "smooth KV must compress: {} vs {} bytes",
            encoded.len(),
            raw.len()
        );
        let decoded = decode_f16_segment(&encoded).expect("decode");
        assert_eq!(decoded.len(), raw.len());
        // Per-value error is bounded by the quantization step plus the
        // final f16 rounding; smooth data at 4 bits stays well inside one
        // step.
        for (original, restored) in raw
            .as_chunks::<2>()
            .0
            .iter()
            .zip(decoded.as_chunks::<2>().0)
        {
            let original = f16_bits_to_f32(u16::from_le_bytes(*original));
            let restored = f16_bits_to_f32(u16::from_le_bytes(*restored));
            assert!(
                (original - restored).abs() < 0.05,
                "{original} rebuilt as {restored}"
            );
        }
    }

    #[test]
    fn encoding_is_deterministic() {
        let dims = 8;
        let raw = smooth_tile(64, dims, 42);
        let first = encode_f16_segment(&raw, dims).expect("encode");
        let second = encode_f16_segment(&raw, dims).expect("encode");
        assert_eq!(first, second);
    }

    #[test]
    fn calibration_digest_tracks_shape_and_values() {
        let raw = smooth_tile(32, 8, 7);
        let encoded = encode_f16_segment(&raw, 8).expect("encode");
        let (calibration, dims) = container_calibration(&encoded).expect("header");
        let digest = calibration_digest(&calibration, dims).expect("digest");
        assert_eq!(digest.len(), 64);
        // Different shape: same values, different digest.
        let wider = encode_f16_segment(&raw, 16).expect("encode");
        let (wider_calibration, wider_dims) = container_calibration(&wider).expect("header");
        assert_ne!(
            digest,
            calibration_digest(&wider_calibration, wider_dims).expect("digest")
        );
        // Different values: different calibration, different digest.
        let other = smooth_tile(32, 8, 8);
        let (other_calibration, other_dims) =
            container_calibration(&encode_f16_segment(&other, 8).expect("encode")).expect("header");
        assert_ne!(
            digest,
            calibration_digest(&other_calibration, other_dims).expect("digest")
        );
    }

    #[test]
    fn segment_identity_is_lossy_and_namespaced() {
        let identity = segment_identity(1024, "digest".to_string());
        assert_eq!(identity.name, CACHEGEN_CODEC_NAME);
        assert_eq!(identity.version, CACHEGEN_CODEC_VERSION);
        assert_eq!(identity.class, CodecClass::Lossy);
        assert_eq!(identity.decoded_len, 1024);
        assert!(!identity.is_supported());
        assert!(identity.is_self_consistent(2048));
        assert_ne!(
            identity.name,
            crate::l3::CODEC_RAW,
            "must stay out of the raw namespace"
        );
    }

    /// scama re-review blocker 1, reproduced exactly: a NaN calibration
    /// minimum, a negative scale, and a 92-byte container claiming
    /// billions of decoded values must all be refused at the boundary —
    /// `container_calibration` and `decoded_value_count` are capability
    /// probes, so they must reject, not just `decode_f16_segment`.
    #[test]
    fn hostile_calibration_and_shapes_are_refused_at_the_boundary() {
        let dims = 16usize;
        let raw = smooth_tile(4, dims, 5);
        let encoded = encode_f16_segment(&raw, dims).expect("encode");

        let mutate = |position: usize, bytes: [u8; 4]| {
            let mut payload = encoded.clone();
            payload[position..position + 4].copy_from_slice(&bytes);
            payload
        };
        // Header layout: min at 10, scale at 14.
        let nan_min = mutate(10, f32::NAN.to_bits().to_le_bytes());
        let inf_min = mutate(10, f32::INFINITY.to_bits().to_le_bytes());
        let nan_scale = mutate(14, f32::NAN.to_bits().to_le_bytes());
        let negative_scale = mutate(14, (-1.0f32).to_bits().to_le_bytes());
        for (name, hostile) in [
            ("NaN min", &nan_min),
            ("inf min", &inf_min),
            ("NaN scale", &nan_scale),
            ("negative scale", &negative_scale),
        ] {
            assert!(
                container_calibration(hostile).is_err(),
                "{name} must be refused by container_calibration"
            );
            assert!(
                decoded_value_count(hostile).is_err(),
                "{name} must be refused by decoded_value_count"
            );
            assert!(
                decode_f16_segment(hostile).is_err(),
                "{name} must be refused by decode_f16_segment"
            );
        }

        // dims=16, rows=u32::MAX, every histogram entry u32::MAX, a
        // 4-byte stream: the exact shape that used to decode as
        // 68,719,476,720 values.
        let mut bomb = vec![0u8; HEADER_LEN + 4];
        bomb[0..4].copy_from_slice(&MAGIC);
        bomb[4..6].copy_from_slice(&(dims as u16).to_le_bytes());
        bomb[6..10].copy_from_slice(&u32::MAX.to_le_bytes());
        bomb[18..22].copy_from_slice(&4u32.to_le_bytes());
        for entry in bomb[FIXED_HEADER_LEN..HEADER_LEN].chunks_exact_mut(4) {
            entry.copy_from_slice(&u32::MAX.to_le_bytes());
        }
        assert!(decoded_value_count(&bomb).is_err());
        assert!(decode_f16_segment(&bomb).is_err());
        // And a merely huge-but-well-formed total still exceeds the
        // format ceiling even when the histogram would have to lie to
        // support it.
        let mut huge = encoded.clone();
        huge[6..10].copy_from_slice(&(MAX_DECODED_VALUES as u32).to_le_bytes());
        assert!(decoded_value_count(&huge).is_err());
    }

    /// The flat tile (scale == 0) is a legal encoding and must stay one
    /// after calibration validation.
    #[test]
    fn flat_tile_calibration_remains_legal() {
        let values = vec![0.5f32; 4 * 8];
        let mut raw = Vec::new();
        for value in &values {
            raw.extend_from_slice(&f32_to_f16_bits(*value).to_le_bytes());
        }
        let encoded = encode_f16_segment(&raw, 8).expect("encode flat tile");
        let (calibration, _) = container_calibration(&encoded).expect("flat calibration");
        assert_eq!(calibration.scale_bits, 0.0f32.to_bits());
        let decoded = decode_f16_segment(&encoded).expect("decode flat tile");
        assert_eq!(decoded.len(), raw.len());
    }

    #[test]
    fn corrupt_containers_are_rejected_cleanly() {
        let dims = 8;
        let raw = smooth_tile(16, dims, 9);
        let encoded = encode_f16_segment(&raw, dims).expect("encode");

        let mut bad_magic = encoded.clone();
        bad_magic[0] = b'X';
        assert!(decode_f16_segment(&bad_magic).is_err());

        let mut reserved = encoded.clone();
        reserved[FIXED_HEADER_LEN - 1] = 1;
        assert!(decode_f16_segment(&reserved).is_err());

        assert!(decode_f16_segment(&encoded[..encoded.len() - 1]).is_err());
        assert!(decode_f16_segment(&encoded[..HEADER_LEN - 1]).is_err());
        assert!(decode_f16_segment(&[]).is_err());

        let mut length_lie = encoded.clone();
        length_lie[18] ^= 0xff;
        assert!(decode_f16_segment(&length_lie).is_err());

        // A corrupt histogram count must not silently decode: a zeroed
        // symbol frequency can leave symbols undecodable, and the derived
        // table must refuse to lie about totals.
        let mut bad_histogram = encoded.clone();
        bad_histogram[FIXED_HEADER_LEN] ^= 0xff;
        assert!(decode_f16_segment(&bad_histogram).is_err());

        // A histogram that plausibly formats but does not sum to the
        // declared tile is refused at parse, before any table is built:
        // this is the bounded-normalization contract (scama blocker 2).
        let mut total_lie = encoded.clone();
        let lie_position = FIXED_HEADER_LEN + 4 * 3;
        let current = u32::from_le_bytes(
            total_lie[lie_position..lie_position + 4]
                .try_into()
                .expect("4 bytes"),
        );
        total_lie[lie_position..lie_position + 4]
            .copy_from_slice(&current.wrapping_add(1).to_le_bytes());
        let error = decode_f16_segment(&total_lie).expect_err("histogram/shape disagreement");
        assert!(
            error.to_string().contains("histogram totals"),
            "rejection should name the histogram total: {error}"
        );
    }

    /// The exact shape scama's review measured and the spike reports:
    /// 4096 rows x 128 dims, end to end through the real CGv1 container,
    /// not a bypass. This is the regression for the u16 histogram cap.
    #[test]
    fn the_measured_4096x128_tile_round_trips_through_the_container() {
        let dims = 128;
        let rows = 4096;
        let raw = smooth_tile(rows, dims, 0xC0FFEE);
        let encoded = encode_f16_segment(&raw, dims).expect("encode a full-size tile");
        assert_eq!(
            decoded_value_count(&encoded).expect("count"),
            rows * dims,
            "container must declare the full measured tile"
        );
        let decoded = decode_f16_segment(&encoded).expect("decode a full-size tile");
        assert_eq!(decoded.len(), raw.len());
        let max_error: f32 = raw
            .as_chunks::<2>()
            .0
            .iter()
            .zip(decoded.as_chunks::<2>().0)
            .map(|(original, restored)| {
                (f16_bits_to_f32(u16::from_le_bytes(*original))
                    - f16_bits_to_f32(u16::from_le_bytes(*restored)))
                .abs()
            })
            .fold(0.0, f32::max);
        assert!(
            max_error < 0.05,
            "full-size tile decoded outside quantization error: {max_error}"
        );
    }

    #[test]
    fn shape_mismatch_is_refused_before_encoding() {
        let raw = smooth_tile(4, 8, 3);
        assert!(encode_f16_segment(&raw, 7).is_err());
        assert!(encode_f16_segment(&raw, 0).is_err());
        assert!(encode_f16_segment(&raw[..raw.len() - 1], 8).is_err());
    }

    #[test]
    fn decoded_value_count_reports_the_tile_size() {
        let dims = 8;
        let raw = smooth_tile(24, dims, 11);
        let encoded = encode_f16_segment(&raw, dims).expect("encode");
        assert_eq!(decoded_value_count(&encoded).expect("count"), 24 * dims);
    }

    #[test]
    fn random_noise_does_not_expand_but_still_round_trips() {
        // Incompressible input: the container must not blow up (bounded
        // overhead) and must still decode to quantization-accurate data.
        let dims = 8;
        let mut rng = Xorshift(0x5EED_5EED);
        let raw: Vec<u8> = (0..64 * dims)
            .flat_map(|_| f32_to_f16_bits(rng.next_unit()).to_le_bytes())
            .collect();
        let encoded = encode_f16_segment(&raw, dims).expect("encode");
        assert!(
            encoded.len() <= raw.len() + 256,
            "noise must not expand the payload materially: {} vs {}",
            encoded.len(),
            raw.len()
        );
        let decoded = decode_f16_segment(&encoded).expect("decode");
        assert_eq!(decoded.len(), raw.len());
    }
}

#[cfg(test)]
mod store_contract_tests {
    use super::*;
    use crate::l3::{HandoffManifest, HandoffSegmentRef, HandoffSegmentStore};

    /// The full #1652 contract, end to end: a real CacheGen-encoded segment
    /// carries a lossy identity the store refuses at commit. No lossy entry
    /// can enter the exact pipeline while raw is the only supported class.
    #[test]
    fn a_real_cachegen_segment_never_commits_to_the_store() {
        let dims = 8;
        let mut raw = Vec::new();
        for row in 0..16 {
            for column in 0..dims {
                let value = ((row * dims + column) as f32 * 0.05).sin() * 0.3 + 0.5;
                raw.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
            }
        }
        let encoded = encode_f16_segment(&raw, dims).expect("encode");
        let (calibration, container_dims) = container_calibration(&encoded).expect("header");
        let digest = calibration_digest(&calibration, container_dims).expect("digest");
        let identity = segment_identity(raw.len() as u64, digest);
        assert!(!identity.is_supported());

        let root = std::env::temp_dir()
            .join("skippy-cachegen-store-tests")
            .join(format!("reject-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let store = HandoffSegmentStore::open(&root, 0).expect("open store");
        let stored = store.put_segment(&encoded).expect("put segment");
        let mut manifest = HandoffManifest::new("blake3:cachegen".to_string(), "full-state".into());
        manifest.segments.push(HandoffSegmentRef {
            index: 0,
            offset: 0,
            bytes: encoded.len() as u64,
            digest: stored.digest.clone(),
            codec_identity: Some(identity),
            meta_json: None,
        });
        manifest.total_bytes = encoded.len() as u64;
        manifest.payload_digest = "blake3:cachegen-payload".to_string();
        let error = store
            .commit(&manifest)
            .expect_err("a lossy segment must not commit to the exact pipeline");
        assert!(
            error.to_string().contains("cachegen"),
            "commit error should name the codec: {error}"
        );
        std::fs::remove_dir_all(&root).ok();
    }
}
