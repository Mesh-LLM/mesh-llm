//! Ten-token anchor and delta codec for F32 KV tiles.
//!
//! This is a high-precision CacheGen candidate: the first row of each group
//! uses 8-bit symmetric quantization and the other rows encode differences
//! from that original anchor with 12 bits. Entropy coding can replace the
//! packed symbol stream without changing reconstructed values.

use anyhow::{Result, anyhow, bail};

pub const MAGIC: [u8; 4] = *b"CGP1";
pub const GROUP_ROWS: usize = 10;
pub const MAX_ROWS: usize = 250;
const HEADER_BYTES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    pub rows: usize,
    pub channels: usize,
}

pub fn encode(raw: &[u8], channels: usize) -> Result<Vec<u8>> {
    let row_bytes = channels
        .checked_mul(4)
        .ok_or_else(|| anyhow!("anchor-delta F32 row size overflow"))?;
    if row_bytes == 0 || raw.is_empty() || !raw.len().is_multiple_of(row_bytes) {
        bail!("anchor-delta input has invalid F32 row geometry");
    }
    let rows = raw.len() / row_bytes;
    if rows > MAX_ROWS || channels > u16::MAX as usize {
        bail!("anchor-delta tile exceeds bounded geometry");
    }
    let values = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        bail!("anchor-delta input contains a non-finite value");
    }
    let mut maxes = Vec::with_capacity(rows);
    let mut packed = Vec::new();
    let mut bit = 0usize;
    for row in 0..rows {
        let anchor_row = row / GROUP_ROWS * GROUP_ROWS;
        let anchor = &values[anchor_row * channels..(anchor_row + 1) * channels];
        let current = &values[row * channels..(row + 1) * channels];
        let center = if row == anchor_row { 127u16 } else { 2047u16 };
        let max = current
            .iter()
            .zip(anchor)
            .map(|(&value, &base)| {
                if row == anchor_row {
                    value.abs()
                } else {
                    (value - base).abs()
                }
            })
            .fold(0.0f32, f32::max);
        maxes.push(max);
        let factor = if max == 0.0 {
            0.0
        } else {
            f32::from(center) / max
        };
        for (&value, &base) in current.iter().zip(anchor) {
            let delta = if row == anchor_row {
                value
            } else {
                value - base
            };
            let signed = (delta * factor)
                .round_ties_even()
                .clamp(-f32::from(center), f32::from(center));
            let symbol = (signed as i32 + i32::from(center)) as u16;
            write_symbol(
                &mut packed,
                &mut bit,
                symbol,
                if row == anchor_row { 8 } else { 12 },
            );
        }
    }
    let stream_len =
        u32::try_from(packed.len()).map_err(|_| anyhow!("anchor-delta stream is too large"))?;
    let mut output = Vec::with_capacity(HEADER_BYTES + maxes.len() * 4 + packed.len());
    output.extend_from_slice(&MAGIC);
    output.extend_from_slice(&(rows as u16).to_le_bytes());
    output.extend_from_slice(&(channels as u16).to_le_bytes());
    output.extend_from_slice(&stream_len.to_le_bytes());
    output.push(GROUP_ROWS as u8);
    output.push(12);
    output.extend_from_slice(&[0, 0]);
    for max in maxes {
        output.extend_from_slice(&max.to_le_bytes());
    }
    output.extend_from_slice(&packed);
    Ok(output)
}

pub fn validate(payload: &[u8]) -> Result<Geometry> {
    let header = payload
        .get(..HEADER_BYTES)
        .ok_or_else(|| anyhow!("anchor-delta header is truncated"))?;
    if header[..4] != MAGIC
        || header[12] != GROUP_ROWS as u8
        || header[13] != 12
        || header[14..16] != [0, 0]
    {
        bail!("invalid anchor-delta segment header");
    }
    let rows = usize::from(u16::from_le_bytes([header[4], header[5]]));
    let channels = usize::from(u16::from_le_bytes([header[6], header[7]]));
    let stream_len = u32::from_le_bytes(header[8..12].try_into()?) as usize;
    if rows == 0 || rows > MAX_ROWS || channels == 0 {
        bail!("invalid anchor-delta tile geometry");
    }
    let anchors = rows.div_ceil(GROUP_ROWS);
    let bits = channels
        .checked_mul(anchors * 8 + (rows - anchors) * 12)
        .ok_or_else(|| anyhow!("anchor-delta bit length overflow"))?;
    if stream_len != bits.div_ceil(8) || payload.len() != HEADER_BYTES + rows * 4 + stream_len {
        bail!("anchor-delta stream length disagrees with geometry");
    }
    let maxes = &payload[HEADER_BYTES..HEADER_BYTES + rows * 4];
    if maxes.as_chunks::<4>().0.iter().any(|bytes| {
        let value = f32::from_le_bytes(*bytes);
        !value.is_finite() || value < 0.0
    }) {
        bail!("anchor-delta row maximum is invalid");
    }
    Ok(Geometry { rows, channels })
}

pub fn decode(payload: &[u8]) -> Result<Vec<u8>> {
    let geometry = validate(payload)?;
    let rows = geometry.rows;
    let channels = geometry.channels;
    let stream = &payload[HEADER_BYTES + rows * 4..];
    let mut output = Vec::with_capacity(rows * channels * 4);
    let mut anchor = vec![0.0f32; channels];
    let mut bit = 0usize;
    for row in 0..rows {
        let is_anchor = row.is_multiple_of(GROUP_ROWS);
        let center = if is_anchor { 127u16 } else { 2047u16 };
        let max_offset = HEADER_BYTES + row * 4;
        let max = f32::from_le_bytes(payload[max_offset..max_offset + 4].try_into()?);
        for base in &mut anchor {
            let symbol = read_symbol(stream, &mut bit, if is_anchor { 8 } else { 12 });
            let signed = f32::from(symbol) - f32::from(center);
            let value = signed / f32::from(center) * max;
            if is_anchor {
                *base = value;
                output.extend_from_slice(&value.to_le_bytes());
            } else {
                output.extend_from_slice(&(*base + value).to_le_bytes());
            }
        }
    }
    Ok(output)
}

fn write_symbol(stream: &mut Vec<u8>, bit: &mut usize, symbol: u16, width: usize) {
    let index = *bit / 8;
    let shift = *bit % 8;
    let bytes = (shift + width).div_ceil(8);
    stream.resize(stream.len().max(index + bytes), 0);
    let word = u32::from(symbol) << shift;
    stream[index] |= word as u8;
    if bytes > 1 {
        stream[index + 1] |= (word >> 8) as u8;
    }
    if bytes > 2 {
        stream[index + 2] |= (word >> 16) as u8;
    }
    *bit += width;
}

fn read_symbol(stream: &[u8], bit: &mut usize, width: usize) -> u16 {
    let index = *bit / 8;
    let shift = *bit % 8;
    let bytes = (shift + width).div_ceil(8);
    let mut word = u32::from(stream[index]);
    if bytes > 1 {
        word |= u32::from(stream[index + 1]) << 8;
    }
    if bytes > 2 {
        word |= u32::from(stream[index + 2]) << 16;
    }
    *bit += width;
    ((word >> shift) & ((1u32 << width) - 1)) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_and_delta_rows_round_trip_at_their_precision() {
        let raw = (0..23 * 8)
            .flat_map(|index| ((index as f32 / 17.0).sin() * 4.0).to_le_bytes())
            .collect::<Vec<_>>();
        let encoded = encode(&raw, 8).expect("encode");
        let geometry = validate(&encoded).expect("geometry");
        assert_eq!(
            geometry,
            Geometry {
                rows: 23,
                channels: 8
            }
        );
        let decoded = decode(&encoded).expect("decode");
        assert_eq!(decoded.len(), raw.len());
        assert!(
            raw.as_chunks::<4>()
                .0
                .iter()
                .zip(decoded.as_chunks::<4>().0)
                .all(|(a, b)| { (f32::from_le_bytes(*a) - f32::from_le_bytes(*b)).abs() < 0.1 })
        );
    }

    #[test]
    fn malformed_geometry_is_rejected() {
        let raw = [0.5f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat();
        let mut encoded = encode(&raw, 2).expect("encode");
        encoded[12] = 9;
        assert!(validate(&encoded).is_err());
    }
}
