// SPDX-License-Identifier: Apache-2.0

//! Bounded CPU oracle for the CacheGen paper's ten-token anchor/delta codec.
//!
//! Profiles are trained on disjoint KV data and bound to each arithmetic-coded
//! tile by SHA-256. This module defines decoded values and the portable stream;
//! the Metal importer must produce the same values before serving qualification.

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 4] = b"CGP2";
const GROUP_ROWS: usize = 10;
const HEADER_BYTES: usize = 52;
const MAX_ROWS: usize = 250;
const CDF_TOTAL: u64 = 1 << 16;
const HALF: u32 = 1 << 31;
const QUARTER: u32 = 1 << 30;
const THREE_QUARTERS: u32 = 3 << 30;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(from = "ProfileWire", into = "ProfileWire")]
pub struct Profile {
    pub layer: usize,
    pub layers: usize,
    pub channels: usize,
    pub anchor_cdfs: Vec<Vec<u32>>,
    pub delta_cdfs: Vec<Vec<u32>>,
}

type ProfileWire = (usize, usize, usize, Vec<Vec<u32>>, Vec<Vec<u32>>);

impl From<ProfileWire> for Profile {
    fn from((layer, layers, channels, anchor_cdfs, delta_cdfs): ProfileWire) -> Self {
        Self {
            layer,
            layers,
            channels,
            anchor_cdfs,
            delta_cdfs,
        }
    }
}

impl From<Profile> for ProfileWire {
    fn from(profile: Profile) -> Self {
        (
            profile.layer,
            profile.layers,
            profile.channels,
            profile.anchor_cdfs,
            profile.delta_cdfs,
        )
    }
}

impl Profile {
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let profile: Self = serde_json::from_slice(bytes).context("parse CacheGen profile")?;
        profile.validate()?;
        Ok(profile)
    }

    pub fn to_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        serde_json::to_vec(&(
            self.layer,
            self.layers,
            self.channels,
            &self.anchor_cdfs,
            &self.delta_cdfs,
        ))
        .context("serialize CacheGen profile")
    }

    pub fn digest(&self) -> Result<[u8; 32]> {
        Ok(Sha256::digest(self.to_json()?).into())
    }

    fn validate(&self) -> Result<()> {
        let (_, center) = delta_quantizer(self.layer, self.layers)?;
        if self.channels == 0
            || self.channels > u16::MAX as usize
            || self.anchor_cdfs.len() != self.channels
            || self.delta_cdfs.len() != self.channels
        {
            bail!("invalid CacheGen profile geometry");
        }
        for cdf in &self.anchor_cdfs {
            validate_cdf(cdf, 255)?;
        }
        for cdf in &self.delta_cdfs {
            validate_cdf(cdf, center * 2 + 1)?;
        }
        Ok(())
    }
}

fn delta_quantizer(layer: usize, layers: usize) -> Result<(f32, usize)> {
    if layers < 3 || layer >= layers {
        bail!("invalid CacheGen layer geometry");
    }
    Ok(match (layer * 3 / layers).min(2) {
        0 => (0.5, 14),
        1 => (1.0, 7),
        _ => (1.5, 5),
    })
}

fn validate_cdf(cdf: &[u32], alphabet: usize) -> Result<()> {
    if cdf.len() != alphabet + 1
        || cdf.first() != Some(&0)
        || cdf.last() != Some(&(CDF_TOTAL as u32))
        || cdf.windows(2).any(|pair| pair[0] >= pair[1])
    {
        bail!("invalid CacheGen profile CDF");
    }
    Ok(())
}

fn cdf_from_symbols(symbols: impl Iterator<Item = u8>, alphabet: usize) -> Vec<u32> {
    let mut counts = vec![1u64; alphabet];
    for symbol in symbols {
        counts[usize::from(symbol)] += 1;
    }
    let total = counts.iter().sum::<u64>();
    let mut cumulative = 0u64;
    let mut cdf = Vec::with_capacity(alphabet + 1);
    cdf.push(0);
    for (index, count) in counts.into_iter().enumerate() {
        cumulative += count;
        cdf.push(((index + 1) as u64 + (CDF_TOTAL - alphabet as u64) * cumulative / total) as u32);
    }
    cdf
}

fn read_values(raw: &[u8], channels: usize) -> Result<Vec<f32>> {
    if channels == 0
        || channels > u16::MAX as usize
        || raw.is_empty()
        || !raw.len().is_multiple_of(channels * 4)
    {
        bail!("invalid CacheGen F32 row geometry");
    }
    let values = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        bail!("non-finite CacheGen KV value");
    }
    Ok(values)
}

fn row_symbols(
    values: &[f32],
    channels: usize,
    layer: usize,
    layers: usize,
) -> Result<(Vec<u8>, Vec<f32>)> {
    let (multiplier, center) = delta_quantizer(layer, layers)?;
    let rows = values.len() / channels;
    let mut symbols = Vec::with_capacity(values.len());
    let mut maxima = Vec::with_capacity(rows);
    for row in 0..rows {
        let current = &values[row * channels..(row + 1) * channels];
        let anchor = &values[(row / GROUP_ROWS * GROUP_ROWS) * channels
            ..(row / GROUP_ROWS * GROUP_ROWS + 1) * channels];
        let is_anchor = row.is_multiple_of(GROUP_ROWS);
        let differences = current
            .iter()
            .zip(anchor)
            .map(|(&value, &base)| if is_anchor { value } else { value - base })
            .collect::<Vec<_>>();
        let maximum = differences
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max);
        // The released PyTorch CPU scalar path scales an F32 reciprocal; the
        // vectorized delta path divides directly and offsets before rounding.
        let scale = if is_anchor { 127.0 } else { 7.0 / multiplier };
        let factor = if maximum == 0.0 {
            0.0
        } else if is_anchor {
            (1.0 / maximum) * scale
        } else {
            scale / maximum
        };
        for value in differences {
            let symbol = if is_anchor {
                ((value * factor).round_ties_even() as i32).clamp(-127, 127) + 127
            } else {
                ((value * factor + center as f32).round_ties_even() as i32)
                    .clamp(0, (center * 2) as i32)
            };
            symbols.push(symbol as u8);
        }
        maxima.push(maximum);
    }
    Ok((symbols, maxima))
}

pub fn train_profile(raw: &[u8], channels: usize, layer: usize, layers: usize) -> Result<Profile> {
    let values = read_values(raw, channels)?;
    let rows = values.len() / channels;
    let (symbols, _) = row_symbols(&values, channels, layer, layers)?;
    let (_, center) = delta_quantizer(layer, layers)?;
    let mut anchor_cdfs = Vec::with_capacity(channels);
    let mut delta_cdfs = Vec::with_capacity(channels);
    for channel in 0..channels {
        anchor_cdfs.push(cdf_from_symbols(
            (0..rows)
                .filter(|row| row.is_multiple_of(GROUP_ROWS))
                .map(|row| symbols[row * channels + channel]),
            255,
        ));
        delta_cdfs.push(cdf_from_symbols(
            (0..rows)
                .filter(|row| !row.is_multiple_of(GROUP_ROWS))
                .map(|row| symbols[row * channels + channel]),
            center * 2 + 1,
        ));
    }
    let profile = Profile {
        layer,
        layers,
        channels,
        anchor_cdfs,
        delta_cdfs,
    };
    profile.validate()?;
    Ok(profile)
}

pub fn encode(raw: &[u8], profile: &Profile) -> Result<Vec<u8>> {
    profile.validate()?;
    let values = read_values(raw, profile.channels)?;
    let rows = values.len() / profile.channels;
    if rows > MAX_ROWS || profile.layer > u16::MAX as usize || profile.layers > u16::MAX as usize {
        bail!("CacheGen tile exceeds bounded geometry");
    }
    let (symbols, maxima) = row_symbols(&values, profile.channels, profile.layer, profile.layers)?;
    let mut lengths = Vec::with_capacity(profile.channels * 2);
    let mut streams = Vec::new();
    for channel in 0..profile.channels {
        for (anchor, cdf) in [
            (true, &profile.anchor_cdfs[channel]),
            (false, &profile.delta_cdfs[channel]),
        ] {
            let selected = (0..rows)
                .filter(|row| row.is_multiple_of(GROUP_ROWS) == anchor)
                .map(|row| symbols[row * profile.channels + channel])
                .collect::<Vec<_>>();
            let encoded = arithmetic_encode(&selected, cdf);
            lengths.push(
                u32::try_from(encoded.len()).context("CacheGen channel stream is too large")?,
            );
            streams.extend_from_slice(&encoded);
        }
    }
    let stream_bytes = u32::try_from(streams.len()).context("CacheGen tile stream is too large")?;
    let mut output =
        Vec::with_capacity(HEADER_BYTES + rows * 4 + lengths.len() * 4 + streams.len());
    output.extend_from_slice(MAGIC);
    output.extend_from_slice(&(rows as u16).to_le_bytes());
    output.extend_from_slice(&(profile.channels as u16).to_le_bytes());
    output.extend_from_slice(&(profile.layer as u16).to_le_bytes());
    output.extend_from_slice(&(profile.layers as u16).to_le_bytes());
    output.push(GROUP_ROWS as u8);
    output.extend_from_slice(&[0; 3]);
    output.extend_from_slice(&profile.digest()?);
    output.extend_from_slice(&stream_bytes.to_le_bytes());
    for maximum in maxima {
        output.extend_from_slice(&maximum.to_le_bytes());
    }
    for length in lengths {
        output.extend_from_slice(&length.to_le_bytes());
    }
    output.extend_from_slice(&streams);
    Ok(output)
}

pub fn decode(payload: &[u8], profile: &Profile) -> Result<Vec<u8>> {
    profile.validate()?;
    let header = payload
        .get(..HEADER_BYTES)
        .ok_or_else(|| anyhow!("truncated CacheGen tile header"))?;
    if &header[..4] != MAGIC
        || header[12] != GROUP_ROWS as u8
        || header[13..16] != [0; 3]
        || header[16..48] != profile.digest()?
    {
        bail!("CacheGen tile profile or geometry mismatch");
    }
    let rows = usize::from(u16::from_le_bytes(header[4..6].try_into()?));
    let channels = usize::from(u16::from_le_bytes(header[6..8].try_into()?));
    let layer = usize::from(u16::from_le_bytes(header[8..10].try_into()?));
    let layers = usize::from(u16::from_le_bytes(header[10..12].try_into()?));
    if rows == 0
        || rows > MAX_ROWS
        || channels != profile.channels
        || layer != profile.layer
        || layers != profile.layers
    {
        bail!("CacheGen tile geometry disagrees with its profile");
    }
    let stream_bytes = usize::try_from(u32::from_le_bytes(header[48..52].try_into()?))?;
    let metadata_end = HEADER_BYTES
        .checked_add(rows * 4)
        .and_then(|value| value.checked_add(channels * 8))
        .context("CacheGen tile metadata overflow")?;
    if payload.len()
        != metadata_end
            .checked_add(stream_bytes)
            .context("CacheGen tile length overflow")?
    {
        bail!("CacheGen tile length mismatch");
    }
    let maxima = payload[HEADER_BYTES..HEADER_BYTES + rows * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    if maxima
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        bail!("invalid CacheGen tile maximum");
    }
    let lengths = payload[HEADER_BYTES + rows * 4..metadata_end]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes) as usize)
        .collect::<Vec<_>>();
    if lengths.iter().sum::<usize>() != stream_bytes {
        bail!("CacheGen channel streams do not cover the payload");
    }
    let mut symbols = vec![0u8; rows * channels];
    let mut cursor = metadata_end;
    for channel in 0..channels {
        for (anchor, cdf, count) in [
            (
                true,
                &profile.anchor_cdfs[channel],
                rows.div_ceil(GROUP_ROWS),
            ),
            (
                false,
                &profile.delta_cdfs[channel],
                rows - rows.div_ceil(GROUP_ROWS),
            ),
        ] {
            let length = lengths[channel * 2 + usize::from(!anchor)];
            let stream = &payload[cursor..cursor + length];
            let selected = arithmetic_decode(stream, count, cdf)?;
            cursor += length;
            for (row, symbol) in (0..rows)
                .filter(|row| row.is_multiple_of(GROUP_ROWS) == anchor)
                .zip(selected)
            {
                symbols[row * channels + channel] = symbol;
            }
        }
    }
    let (multiplier, center) = delta_quantizer(profile.layer, profile.layers)?;
    let mut output = Vec::with_capacity(rows * channels * 4);
    let mut anchor = vec![0.0f32; channels];
    for row in 0..rows {
        let maximum = maxima[row];
        let step = maximum / 7.0 * multiplier;
        for channel in 0..channels {
            let symbol = symbols[row * channels + channel];
            let value = if row.is_multiple_of(GROUP_ROWS) {
                if symbol > 254 {
                    bail!("CacheGen anchor symbol exceeds alphabet");
                }
                anchor[channel] = (f32::from(symbol) - 127.0) / 127.0 * maximum;
                anchor[channel]
            } else {
                if usize::from(symbol) > center * 2 {
                    bail!("CacheGen delta symbol exceeds alphabet");
                }
                anchor[channel] + (f32::from(symbol) - center as f32) * step
            };
            output.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(output)
}

fn arithmetic_encode(symbols: &[u8], cdf: &[u32]) -> Vec<u8> {
    let (mut low, mut high, mut pending) = (0u32, u32::MAX, 0usize);
    let mut writer = BitWriter::default();
    for &symbol in symbols {
        let span = u64::from(high) - u64::from(low) + 1;
        high = low
            .wrapping_sub(1)
            .wrapping_add(((span * u64::from(cdf[usize::from(symbol) + 1])) >> 16) as u32);
        low = low.wrapping_add(((span * u64::from(cdf[usize::from(symbol)])) >> 16) as u32);
        loop {
            if high < HALF {
                writer.emit(0, &mut pending);
            } else if low >= HALF {
                writer.emit(1, &mut pending);
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
    writer.emit(u8::from(low >= QUARTER), &mut pending);
    writer.finish()
}

fn arithmetic_decode(stream: &[u8], count: usize, cdf: &[u32]) -> Result<Vec<u8>> {
    let mut reader = BitReader { stream, bit: 0 };
    let (mut low, mut high) = (0u32, u32::MAX);
    let mut value = 0u32;
    for _ in 0..32 {
        value = (value << 1) | u32::from(reader.read());
    }
    let mut output = Vec::with_capacity(count);
    for position in 0..count {
        let span = u64::from(high) - u64::from(low) + 1;
        let target = ((u64::from(value.wrapping_sub(low)) + 1) * CDF_TOTAL - 1) / span;
        let symbol = cdf
            .partition_point(|boundary| u64::from(*boundary) <= target)
            .saturating_sub(1);
        if symbol >= cdf.len() - 1 {
            bail!("CacheGen arithmetic stream decoded an invalid symbol");
        }
        output.push(symbol as u8);
        if position + 1 == count {
            break;
        }
        high = low
            .wrapping_sub(1)
            .wrapping_add(((span * u64::from(cdf[symbol + 1])) >> 16) as u32);
        low = low.wrapping_add(((span * u64::from(cdf[symbol])) >> 16) as u32);
        loop {
            if low >= HALF || high < HALF {
                low <<= 1;
                high = (high << 1) | 1;
                value = (value << 1) | u32::from(reader.read());
            } else if low >= QUARTER && high < THREE_QUARTERS {
                low = (low << 1) & 0x7fff_ffff;
                high = (high << 1) | 0x8000_0001;
                value = (value.wrapping_sub(QUARTER) << 1) | u32::from(reader.read());
            } else {
                break;
            }
        }
    }
    Ok(output)
}

#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    bit: usize,
}

impl BitWriter {
    fn emit(&mut self, bit: u8, pending: &mut usize) {
        self.push(bit);
        for _ in 0..*pending {
            self.push(1 - bit);
        }
        *pending = 0;
    }

    fn push(&mut self, bit: u8) {
        if self.bit.is_multiple_of(8) {
            self.bytes.push(0);
        }
        self.bytes[self.bit / 8] |= bit << (7 - self.bit % 8);
        self.bit += 1;
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

struct BitReader<'a> {
    stream: &'a [u8],
    bit: usize,
}

impl BitReader<'_> {
    fn read(&mut self) -> u8 {
        let value = self.stream.get(self.bit / 8).copied().unwrap_or(0);
        let bit = (value >> (7 - self.bit % 8)) & 1;
        self.bit += 1;
        bit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type PythonFixture = (usize, &'static [u8], &'static [u8], &'static [u8]);

    const PYTHON_FIXTURES: [PythonFixture; 3] = [
        (
            0,
            include_bytes!("fixtures/paper_layer0_profile.json"),
            include_bytes!("fixtures/paper_layer0_tile.bin"),
            include_bytes!("fixtures/paper_layer0_decoded.bin"),
        ),
        (
            10,
            include_bytes!("fixtures/paper_layer10_profile.json"),
            include_bytes!("fixtures/paper_layer10_tile.bin"),
            include_bytes!("fixtures/paper_layer10_decoded.bin"),
        ),
        (
            20,
            include_bytes!("fixtures/paper_layer20_profile.json"),
            include_bytes!("fixtures/paper_layer20_tile.bin"),
            include_bytes!("fixtures/paper_layer20_decoded.bin"),
        ),
    ];

    #[test]
    fn matches_released_quantizer_rounding_order() {
        let mut rows = vec![0.0f32; 11 * 2];
        rows[2] = 1.366_129_5;
        rows[3] = 0.097_580_66;
        rows[4] = 2.620_091;
        rows[5] = -0.935_746_67;
        rows[20] = 87.252_525;
        rows[21] = -0.343_513_9;
        let (symbols, _) = row_symbols(&rows, 2, 10, 28).expect("symbols");
        assert_eq!(&symbols[2..4], &[14, 8]);
        assert_eq!(&symbols[4..6], &[14, 4]);
        assert_eq!(&symbols[20..22], &[254, 127]);
    }

    #[test]
    fn matches_disjoint_python_reference_fixtures() {
        let training = (0..37 * 8)
            .flat_map(|index| (((index % 37 - 18) as f32) * 0.125).to_le_bytes())
            .collect::<Vec<_>>();
        let held_out = (0..23)
            .flat_map(|row| {
                (0..8).map(move |channel| (((row * 8 + channel * 3) % 41 - 20) as f32) * 0.09375)
            })
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        for (layer, expected_profile, expected_tile, expected_decoded) in PYTHON_FIXTURES {
            let profile = train_profile(&training, 8, layer, 28).expect("train profile");
            assert_eq!(
                profile.to_json().expect("serialize profile"),
                expected_profile
            );
            let encoded = encode(&held_out, &profile).expect("encode tile");
            assert_eq!(encoded, expected_tile);
            let decoded = decode(&encoded, &profile).expect("decode tile");
            assert_eq!(decoded, expected_decoded);
            let mut wrong_profile = profile.clone();
            wrong_profile.layer = (layer + 1) % 28;
            assert!(decode(&encoded, &wrong_profile).is_err());
            let mut truncated = encoded.clone();
            truncated.pop();
            assert!(decode(&truncated, &profile).is_err());
        }
    }

    #[test]
    fn disjoint_profile_round_trip_all_layer_groups() {
        let make = |rows: usize, phase: f32| {
            (0..rows * 8)
                .flat_map(|index| ((index as f32 * phase).sin() * 3.0).to_le_bytes())
                .collect::<Vec<_>>()
        };
        let training = make(37, 0.13);
        let held_out = make(23, 0.29);
        for layer in [0, 10, 20] {
            let profile = train_profile(&training, 8, layer, 28).expect("profile");
            let json = profile.to_json().expect("serialize");
            let restored = Profile::from_json(&json).expect("deserialize");
            assert_eq!(
                profile.digest().expect("digest"),
                restored.digest().expect("digest")
            );
            let payload = encode(&held_out, &restored).expect("encode");
            let rebuilt = decode(&payload, &restored).expect("decode");
            assert_eq!(rebuilt.len(), held_out.len());
            assert_ne!(rebuilt, held_out);
            assert!(decode(&payload[..payload.len() - 1], &restored).is_err());
        }
    }
}
