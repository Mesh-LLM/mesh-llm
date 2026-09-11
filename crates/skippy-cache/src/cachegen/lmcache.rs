// SPDX-License-Identifier: Apache-2.0
//
// Derived from LMCache CacheGen at revision
// b5d109ea99a89b4d8a670ee4fc2e8cb76411ee5c:
// - lmcache/storage_backend/serde/cachegen_encoder.py
// - lmcache/storage_backend/serde/cachegen_decoder.py
// - csrc/cuda/ac_enc.cu
// - csrc/cuda/ac_dec.cu
// - csrc/cuda/cal_cdf.cu
//
// LMCache and Mesh-LLM are both licensed under Apache-2.0. This module keeps
// the reference arithmetic and tensor transforms explicit so compatibility
// can be checked without requiring Python, PyTorch, or a CUDA device at run
// time.

//! A portable Rust implementation of LMCache's CacheGen tensor codec.
//!
//! LMCache serializes PyTorch objects with pickle. Skippy uses a bounded,
//! language-neutral envelope, while preserving the codec inputs and outputs:
//! per-token maximum magnitude, model/layer-selected 16- or 32-bin symmetric
//! quantization, a 33-entry per-channel CDF, and one 32-bit arithmetic-coded
//! stream per channel. Each segment is one K or V layer with at most 256
//! token-major rows, matching LMCache's CUDA kernel limit.

use anyhow::{Result, anyhow, bail};
use skippy_protocol::binary::{f16_bits_to_f32, f32_to_f16_bits};

use crate::l3::{CodecClass, SegmentCodecIdentity};

pub const CACHEGEN_CODEC_NAME: &str = "lmcache-cachegen";
pub const CACHEGEN_CODEC_VERSION: u32 = 1;
pub const MAX_TOKENS_PER_CHUNK: usize = 256;

const MAGIC: [u8; 4] = *b"LCG1";
const HEADER_LEN: usize = 16;
const MAX_BINS: usize = 32;
const CDF_LEN: usize = MAX_BINS + 1;
const CDF_TOTAL: u32 = 1 << 16;
const QUARTER: u32 = 0x4000_0000;
const HALF: u32 = 0x8000_0000;
const THREE_QUARTERS: u32 = 0xc000_0000;
const MAX_DECODED_VALUES: usize = 1 << 24;

#[derive(Debug)]
struct Parsed<'a> {
    bins: u8,
    channels: usize,
    rows: usize,
    maxes: Vec<f32>,
    cdfs: Vec<[u16; CDF_LEN]>,
    lengths: Vec<usize>,
    streams: &'a [u8],
}

/// Geometry and bounded byte ranges validated from a portable CacheGen
/// segment. Device backends use this before accepting work from an archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentGeometry {
    pub bins: u8,
    pub rows: usize,
    pub channels: usize,
    pub metadata_bytes: usize,
    pub stream_bytes: usize,
}

/// LMCache's generic model-family bin schedule.
///
/// Models with fewer than ten layers use 32 bins everywhere. Larger models
/// use 32 bins for K layers 0..10 and V layers 0..2, then 16 bins.
pub fn bins_for_layer(layer_index: usize, layer_count: usize, is_key: bool) -> u8 {
    if layer_count < 10 || layer_index < if is_key { 10 } else { 2 } {
        32
    } else {
        16
    }
}

pub fn segment_identity(decoded_len: u64, codec_digest: String) -> SegmentCodecIdentity {
    SegmentCodecIdentity {
        name: CACHEGEN_CODEC_NAME.to_string(),
        version: CACHEGEN_CODEC_VERSION,
        class: CodecClass::Lossy,
        decoded_len,
        calibration_digest: Some(codec_digest),
    }
}

/// Encodes one token-major F16 K or V layer with LMCache CacheGen semantics.
pub fn encode_f16_segment(raw: &[u8], channels: usize, bins: u8) -> Result<Vec<u8>> {
    validate_input(raw, channels, bins)?;
    let rows = raw.len() / 2 / channels;
    if rows > MAX_TOKENS_PER_CHUNK {
        bail!("LMCache CacheGen chunks are limited to {MAX_TOKENS_PER_CHUNK} tokens, got {rows}");
    }

    let values: Vec<f32> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| f16_bits_to_f32(u16::from_le_bytes(*bytes)))
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        bail!("LMCache CacheGen input contains a non-finite F16 value");
    }

    let (symbols, maxes) = quantize(&values, rows, channels, bins);
    let cdfs = calculate_cdfs(&symbols, rows, channels)?;
    let mut lengths = Vec::with_capacity(channels);
    let mut streams = Vec::new();
    for (channel, cdf) in cdfs.iter().enumerate() {
        let stream = arithmetic_encode_channel(&symbols, rows, channels, channel, cdf);
        let length = u16::try_from(stream.len())
            .map_err(|_| anyhow!("LMCache channel stream exceeds u16 envelope field"))?;
        lengths.push(length);
        streams.extend_from_slice(&stream);
    }

    let channels_u32 = u32::try_from(channels).map_err(|_| anyhow!("channel count exceeds u32"))?;
    let rows_u16 = u16::try_from(rows).expect("row limit fits u16");
    let stream_len = u32::try_from(streams.len())
        .map_err(|_| anyhow!("LMCache segment stream exceeds u32 envelope field"))?;
    let metadata_len = maxes
        .len()
        .checked_mul(4)
        .and_then(|value| value.checked_add(channels.checked_mul(CDF_LEN * 2)?))
        .and_then(|value| value.checked_add(channels.checked_mul(2)?))
        .ok_or_else(|| anyhow!("LMCache segment metadata length overflow"))?;
    let mut out = Vec::with_capacity(HEADER_LEN + metadata_len + streams.len());
    out.extend_from_slice(&MAGIC);
    out.push(bins);
    out.push(0);
    out.extend_from_slice(&rows_u16.to_le_bytes());
    out.extend_from_slice(&channels_u32.to_le_bytes());
    out.extend_from_slice(&stream_len.to_le_bytes());
    for value in maxes {
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    for cdf in &cdfs {
        for value in cdf {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    for length in lengths {
        out.extend_from_slice(&length.to_le_bytes());
    }
    out.extend_from_slice(&streams);
    Ok(out)
}

/// Decodes a portable segment produced by [`encode_f16_segment`].
pub fn decode_f16_segment(payload: &[u8]) -> Result<Vec<u8>> {
    let parsed = parse(payload)?;
    let mut symbols = vec![0u8; parsed.rows * parsed.channels];
    let mut cursor = 0usize;
    for channel in 0..parsed.channels {
        let end = cursor
            .checked_add(parsed.lengths[channel])
            .ok_or_else(|| anyhow!("LMCache stream range overflow"))?;
        let stream = parsed
            .streams
            .get(cursor..end)
            .ok_or_else(|| anyhow!("LMCache channel stream exceeds payload"))?;
        arithmetic_decode_channel(
            stream,
            parsed.rows,
            parsed.channels,
            channel,
            &parsed.cdfs[channel],
            &mut symbols,
        )?;
        cursor = end;
    }
    if cursor != parsed.streams.len() {
        bail!("LMCache segment contains trailing stream bytes");
    }

    let center = f32::from(parsed.bins / 2 - 1);
    let max_symbol = parsed.bins - 2;
    let mut raw = Vec::with_capacity(symbols.len() * 2);
    for (row, max) in parsed.maxes.iter().copied().enumerate() {
        for channel in 0..parsed.channels {
            let symbol = symbols[row * parsed.channels + channel];
            if symbol > max_symbol {
                bail!(
                    "LMCache decoded symbol {symbol} exceeds the configured {}-bin range",
                    parsed.bins
                );
            }
            let centered = f32::from(symbol) - center;
            let normalized = centered / center;
            let value = normalized * max;
            raw.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
        }
    }
    Ok(raw)
}

/// Validates a portable segment without allocating its decoded F16 output.
pub fn validate_f16_segment(payload: &[u8]) -> Result<SegmentGeometry> {
    let parsed = parse(payload)?;
    Ok(SegmentGeometry {
        bins: parsed.bins,
        rows: parsed.rows,
        channels: parsed.channels,
        metadata_bytes: payload.len() - parsed.streams.len(),
        stream_bytes: parsed.streams.len(),
    })
}

fn validate_input(raw: &[u8], channels: usize, bins: u8) -> Result<()> {
    if !matches!(bins, 16 | 32) {
        bail!("LMCache CacheGen supports 16 or 32 bins, got {bins}");
    }
    if channels == 0 || raw.is_empty() || !raw.len().is_multiple_of(2) {
        bail!("LMCache CacheGen requires a non-empty whole-F16 segment");
    }
    let values = raw.len() / 2;
    if values > MAX_DECODED_VALUES {
        bail!("LMCache CacheGen segment exceeds the decoded-value ceiling");
    }
    if !values.is_multiple_of(channels) {
        bail!("LMCache CacheGen segment is not token-major rows of {channels} values");
    }
    Ok(())
}

fn quantize(values: &[f32], rows: usize, channels: usize, bins: u8) -> (Vec<u8>, Vec<f32>) {
    let center = f32::from(bins / 2 - 1);
    let mut symbols = Vec::with_capacity(values.len());
    let mut maxes = Vec::with_capacity(rows);
    for row in values.chunks_exact(channels) {
        let max = row.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        maxes.push(max);
        if max == 0.0 {
            symbols.extend(std::iter::repeat_n(center as u8, channels));
            continue;
        }
        let factor = center / max;
        symbols.extend(row.iter().map(|value| {
            let scaled = *value * factor;
            let shifted = scaled + center;
            shifted.round_ties_even().clamp(0.0, center * 2.0) as u8
        }));
    }
    (symbols, maxes)
}

fn calculate_cdfs(symbols: &[u8], rows: usize, channels: usize) -> Result<Vec<[u16; CDF_LEN]>> {
    let mut out = Vec::with_capacity(channels);
    for channel in 0..channels {
        let mut histogram = [0u32; CDF_LEN];
        for row in 0..rows {
            let symbol = usize::from(symbols[row * channels + channel]);
            let bucket = histogram
                .get_mut(symbol + 1)
                .ok_or_else(|| anyhow!("LMCache quantized symbol exceeds the CDF alphabet"))?;
            *bucket += 1;
        }
        let mut running = 0u32;
        for bucket in histogram.iter_mut().skip(1) {
            let count = *bucket;
            *bucket += running;
            running += count;
        }
        if running != rows as u32 {
            bail!("LMCache CDF histogram does not cover every token");
        }
        let mut cdf = [0u16; CDF_LEN];
        let normalization = u32::from(u16::MAX) - MAX_BINS as u32;
        for (index, count) in histogram.into_iter().enumerate() {
            let normalized = normalization * count / running + index as u32;
            cdf[index] = normalized as u16;
        }
        validate_cdf(&cdf)?;
        out.push(cdf);
    }
    Ok(out)
}

fn validate_cdf(cdf: &[u16; CDF_LEN]) -> Result<()> {
    if cdf[0] != 0 || cdf[CDF_LEN - 1] != u16::MAX {
        bail!("LMCache CDF endpoints are invalid");
    }
    if cdf.windows(2).any(|pair| pair[0] >= pair[1]) {
        bail!("LMCache CDF is not strictly increasing");
    }
    Ok(())
}

fn arithmetic_encode_channel(
    symbols: &[u8],
    rows: usize,
    channels: usize,
    channel: usize,
    cdf: &[u16; CDF_LEN],
) -> Vec<u8> {
    let mut low = 0u32;
    let mut high = u32::MAX;
    let mut pending = 0u64;
    let mut writer = BitWriter::default();
    for row in 0..rows {
        let symbol = usize::from(symbols[row * channels + channel]);
        let span = u64::from(high) - u64::from(low) + 1;
        let c_low = u64::from(cdf[symbol]);
        let c_high = if symbol == MAX_BINS - 1 {
            u64::from(CDF_TOTAL)
        } else {
            u64::from(cdf[symbol + 1])
        };
        high = low
            .wrapping_sub(1)
            .wrapping_add(((span * c_high) >> 16) as u32);
        low = low.wrapping_add(((span * c_low) >> 16) as u32);
        loop {
            if high < HALF {
                writer.append_with_pending(0, &mut pending);
            } else if low >= HALF {
                writer.append_with_pending(1, &mut pending);
            } else if low >= QUARTER && high < THREE_QUARTERS {
                pending += 1;
                low = (low << 1) & 0x7fff_ffff;
                high = (high << 1) | 0x8000_0001;
                continue;
            } else {
                break;
            }
            low <<= 1;
            high = (high << 1) | 1;
        }
    }
    pending += 1;
    writer.append_with_pending(u32::from(low >= QUARTER), &mut pending);
    writer.finish()
}

fn arithmetic_decode_channel(
    stream: &[u8],
    rows: usize,
    channels: usize,
    channel: usize,
    cdf: &[u16; CDF_LEN],
    output: &mut [u8],
) -> Result<()> {
    let mut reader = BitReader::new(stream);
    let mut low = 0u32;
    let mut high = u32::MAX;
    let mut value = reader.read_u32();
    for row in 0..rows {
        let span = u64::from(high) - u64::from(low) + 1;
        let count =
            (((u64::from(value) - u64::from(low) + 1) * u64::from(CDF_TOTAL) - 1) / span) as u16;
        let symbol = cdf
            .partition_point(|boundary| *boundary <= count)
            .saturating_sub(1);
        if symbol >= MAX_BINS {
            bail!("LMCache arithmetic stream decoded an out-of-range symbol");
        }
        output[row * channels + channel] = symbol as u8;
        if row + 1 == rows {
            break;
        }
        let c_low = u64::from(cdf[symbol]);
        let c_high = if symbol == MAX_BINS - 1 {
            u64::from(CDF_TOTAL)
        } else {
            u64::from(cdf[symbol + 1])
        };
        high = low
            .wrapping_sub(1)
            .wrapping_add(((span * c_high) >> 16) as u32);
        low = low.wrapping_add(((span * c_low) >> 16) as u32);
        loop {
            if low >= HALF || high < HALF {
                low <<= 1;
                high = (high << 1) | 1;
                value = (value << 1) | u32::from(reader.read_bit());
            } else if low >= QUARTER && high < THREE_QUARTERS {
                low = (low << 1) & 0x7fff_ffff;
                high = (high << 1) | 0x8000_0001;
                value = value.wrapping_sub(QUARTER);
                value = (value << 1) | u32::from(reader.read_bit());
            } else {
                break;
            }
        }
    }
    Ok(())
}

#[derive(Default)]
struct BitWriter {
    register: u32,
    bits: u32,
    bytes: Vec<u8>,
}

impl BitWriter {
    fn append_with_pending(&mut self, bit: u32, pending: &mut u64) {
        self.add_repeated(bit, 1);
        self.add_repeated(1 - bit, *pending);
        *pending = 0;
    }

    fn add_repeated(&mut self, bit: u32, mut count: u64) {
        while count > 0 {
            let take = count.min(u64::from(32 - self.bits)) as u32;
            self.register <<= take;
            if bit == 1 {
                self.register |= if take == 32 {
                    u32::MAX
                } else {
                    (1u32 << take) - 1
                };
            }
            self.bits += take;
            count -= u64::from(take);
            if self.bits == 32 {
                self.bytes.extend_from_slice(&self.register.to_be_bytes());
                self.register = 0;
                self.bits = 0;
            }
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bits > 0 {
            self.register <<= 32 - self.bits;
            let bytes = self.bits.div_ceil(8) as usize;
            self.bytes
                .extend_from_slice(&self.register.to_be_bytes()[..bytes]);
        }
        self.bytes
    }
}

struct BitReader<'a> {
    stream: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    fn new(stream: &'a [u8]) -> Self {
        Self { stream, bit: 0 }
    }

    fn read_bit(&mut self) -> u8 {
        let byte = self.stream.get(self.bit / 8).copied().unwrap_or(0);
        let value = (byte >> (7 - self.bit % 8)) & 1;
        self.bit += 1;
        value
    }

    fn read_u32(&mut self) -> u32 {
        let mut value = 0u32;
        for _ in 0..32 {
            value = (value << 1) | u32::from(self.read_bit());
        }
        value
    }
}

fn parse(payload: &[u8]) -> Result<Parsed<'_>> {
    if payload.len() < HEADER_LEN || payload[..4] != MAGIC {
        bail!("not an LMCache CacheGen portable segment");
    }
    let bins = payload[4];
    if payload[5] != 0 || !matches!(bins, 16 | 32) {
        bail!("invalid LMCache CacheGen segment header");
    }
    let rows = usize::from(u16::from_le_bytes([payload[6], payload[7]]));
    let channels = u32::from_le_bytes(payload[8..12].try_into().expect("four bytes")) as usize;
    let stream_len = u32::from_le_bytes(payload[12..16].try_into().expect("four bytes")) as usize;
    if rows == 0 || rows > MAX_TOKENS_PER_CHUNK || channels == 0 {
        bail!("invalid LMCache CacheGen segment geometry");
    }
    let values = rows
        .checked_mul(channels)
        .ok_or_else(|| anyhow!("LMCache CacheGen segment shape overflow"))?;
    if values > MAX_DECODED_VALUES {
        bail!("LMCache CacheGen segment exceeds the decoded-value ceiling");
    }
    let max_bytes = rows
        .checked_mul(4)
        .ok_or_else(|| anyhow!("max metadata overflow"))?;
    let cdf_bytes = channels
        .checked_mul(CDF_LEN * 2)
        .ok_or_else(|| anyhow!("CDF metadata overflow"))?;
    let length_bytes = channels
        .checked_mul(2)
        .ok_or_else(|| anyhow!("length metadata overflow"))?;
    let metadata_end = HEADER_LEN
        .checked_add(max_bytes)
        .and_then(|value| value.checked_add(cdf_bytes))
        .and_then(|value| value.checked_add(length_bytes))
        .ok_or_else(|| anyhow!("LMCache CacheGen metadata range overflow"))?;
    if payload.len()
        != metadata_end
            .checked_add(stream_len)
            .ok_or_else(|| anyhow!("payload overflow"))?
    {
        bail!("LMCache CacheGen payload length disagrees with its header");
    }
    let mut cursor = HEADER_LEN;
    let mut maxes = Vec::with_capacity(rows);
    for _ in 0..rows {
        let bits = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().expect("four bytes"));
        let value = f32::from_bits(bits);
        if !value.is_finite() || value < 0.0 {
            bail!("LMCache CacheGen segment contains an invalid row maximum");
        }
        maxes.push(value);
        cursor += 4;
    }
    let mut cdfs = Vec::with_capacity(channels);
    for _ in 0..channels {
        let mut cdf = [0u16; CDF_LEN];
        for value in &mut cdf {
            *value = u16::from_le_bytes(payload[cursor..cursor + 2].try_into().expect("two bytes"));
            cursor += 2;
        }
        validate_cdf(&cdf)?;
        cdfs.push(cdf);
    }
    let mut lengths = Vec::with_capacity(channels);
    let mut total = 0usize;
    for _ in 0..channels {
        let length = usize::from(u16::from_le_bytes(
            payload[cursor..cursor + 2].try_into().expect("two bytes"),
        ));
        if length == 0 || length > MAX_TOKENS_PER_CHUNK {
            bail!("LMCache CacheGen channel stream length is invalid");
        }
        total = total
            .checked_add(length)
            .ok_or_else(|| anyhow!("stream length overflow"))?;
        lengths.push(length);
        cursor += 2;
    }
    if total != stream_len {
        bail!("LMCache CacheGen channel lengths do not sum to the stream size");
    }
    Ok(Parsed {
        bins,
        channels,
        rows,
        maxes,
        cdfs,
        lengths,
        streams: &payload[cursor..],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_HEADER_LEN: usize = 16;

    fn input(rows: usize, channels: usize) -> Vec<u8> {
        let mut raw = Vec::with_capacity(rows * channels * 2);
        for row in 0..rows {
            for channel in 0..channels {
                let phase = (row * channels + channel) as f32 * 0.03125;
                let value = phase.sin() * 0.75 + (channel as f32 - 3.0) * 0.015625;
                raw.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
            }
        }
        raw
    }

    #[test]
    fn generic_bin_schedule_matches_lmcache() {
        assert_eq!(bins_for_layer(0, 8, true), 32);
        assert_eq!(bins_for_layer(7, 8, false), 32);
        assert_eq!(bins_for_layer(9, 28, true), 32);
        assert_eq!(bins_for_layer(10, 28, true), 16);
        assert_eq!(bins_for_layer(1, 28, false), 32);
        assert_eq!(bins_for_layer(2, 28, false), 16);
    }

    #[test]
    fn segment_round_trips_through_lmcache_quantization() {
        let raw = input(64, 8);
        for bins in [16, 32] {
            let encoded = encode_f16_segment(&raw, 8, bins).expect("encode");
            let decoded = decode_f16_segment(&encoded).expect("decode");
            assert_eq!(decoded.len(), raw.len());
            for (original, restored) in raw
                .as_chunks::<2>()
                .0
                .iter()
                .zip(decoded.as_chunks::<2>().0)
            {
                let original = f16_bits_to_f32(u16::from_le_bytes(*original));
                let restored = f16_bits_to_f32(u16::from_le_bytes(*restored));
                assert!(
                    (original - restored).abs() < 0.12,
                    "{original} rebuilt as {restored}"
                );
            }
        }
    }

    #[test]
    fn flat_rows_restore_exactly() {
        let mut raw = Vec::new();
        for value in [0.0f32, 0.5, -0.75] {
            for _ in 0..8 {
                raw.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
            }
        }
        let encoded = encode_f16_segment(&raw, 8, 32).expect("encode");
        assert_eq!(decode_f16_segment(&encoded).expect("decode"), raw);
    }

    #[test]
    fn malformed_envelopes_are_refused() {
        let raw = input(8, 4);
        let encoded = encode_f16_segment(&raw, 4, 32).expect("encode");
        assert!(decode_f16_segment(&encoded[..encoded.len() - 1]).is_err());
        let mut bad_cdf = encoded.clone();
        let cdf_start = HEADER_LEN + 8 * 4;
        bad_cdf[cdf_start + 2..cdf_start + 4].copy_from_slice(&0u16.to_le_bytes());
        assert!(decode_f16_segment(&bad_cdf).is_err());
    }

    fn assert_lmcache_fixture(fixture: &[u8], bins: u8) {
        assert_eq!(&fixture[..4], b"LCFX");
        let raw_len =
            u32::from_le_bytes(fixture[4..8].try_into().expect("fixture raw length")) as usize;
        let encoded_len =
            u32::from_le_bytes(fixture[8..12].try_into().expect("fixture encoded length")) as usize;
        let decoded_len =
            u32::from_le_bytes(fixture[12..16].try_into().expect("fixture decoded length"))
                as usize;
        assert_eq!(
            fixture.len(),
            FIXTURE_HEADER_LEN + raw_len + encoded_len + decoded_len
        );
        let raw = &fixture[FIXTURE_HEADER_LEN..FIXTURE_HEADER_LEN + raw_len];
        let encoded_start = FIXTURE_HEADER_LEN + raw_len;
        let expected = &fixture[encoded_start..encoded_start + encoded_len];
        let expected_decoded = &fixture[encoded_start + encoded_len..];
        let encoded = encode_f16_segment(raw, 8, bins).expect("encode fixture");
        assert_eq!(
            encoded, expected,
            "Rust encoder diverged from the pinned LMCache scalar reference"
        );
        assert_eq!(
            decode_f16_segment(expected).expect("decode fixture"),
            expected_decoded,
            "Rust decoder diverged from the pinned LMCache scalar reference"
        );
    }

    #[test]
    fn matches_lmcache_bins16_fixture() {
        assert_lmcache_fixture(include_bytes!("fixtures/lmcache_b5d109e_bins16.bin"), 16);
    }

    #[test]
    fn matches_lmcache_bins32_fixture() {
        assert_lmcache_fixture(include_bytes!("fixtures/lmcache_b5d109e_bins32.bin"), 32);
    }
}
