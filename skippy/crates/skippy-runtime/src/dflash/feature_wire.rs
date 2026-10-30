//! How DFlash feature rows are encoded on the wire between split stages.
//!
//! Rows travel as Q8 blocks by default: 32 values share an F16 scale, as in
//! ggml's Q8_0, which is about half the bytes of BF16. A block keeps an
//! outlier channel from flattening the rest of its row. The draft only
//! proposes tokens the target then verifies, so feature precision can change
//! acceptance but never output. `SKIPPY_DFLASH_FEATURE_ENCODING=bf16` sends
//! BF16 rows instead; receivers read either.
//!
//! A Q8 part is an I32 part of packed rows, so activation codecs and the
//! native runtime carry it through untouched like any other opaque part.

use std::sync::OnceLock;

use anyhow::{Result, ensure};

use crate::ACTIVATION_IDENTITY_BYTES;

/// ggml's BF16 type id.
pub(super) const GGML_TYPE_BF16: u32 = 30;

/// Identity prefix of a BF16 DFlash feature part; the layer id follows in
/// decimal.
const FEATURE_IDENTITY_PREFIX: &[u8] = b"skippy.dflash.layer_inp.";

/// Identity prefix of a Q8-block DFlash feature part; `<layer>.<width>`
/// follows in decimal, since the packed rows do not show their width.
const Q8_FEATURE_IDENTITY_PREFIX: &[u8] = b"skippy.dflash.q8.";

/// Values that share one scale in a Q8-block row.
const Q8_BLOCK: usize = 32;

/// Environment override for how this stage encodes the rows it sends.
const FEATURE_ENCODING_ENV: &str = "SKIPPY_DFLASH_FEATURE_ENCODING";

/// How DFlash feature rows travel between stages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DFlashFeatureEncoding {
    /// Two bytes a value.
    Bf16,
    /// Blocks of 32 signed bytes with an F16 scale each, rows padded to four
    /// bytes, in an I32 part.
    #[default]
    Q8,
}

impl DFlashFeatureEncoding {
    /// The encoding this process sends: `SKIPPY_DFLASH_FEATURE_ENCODING`, or
    /// Q8.
    pub fn configured() -> Self {
        static CONFIGURED: OnceLock<DFlashFeatureEncoding> = OnceLock::new();
        *CONFIGURED.get_or_init(|| {
            let value = std::env::var(FEATURE_ENCODING_ENV).unwrap_or_default();
            match value.trim().to_ascii_lowercase().as_str() {
                "" | "q8" => Self::Q8,
                "bf16" => Self::Bf16,
                other => {
                    crate::write_native_log_note(format!(
                        "{FEATURE_ENCODING_ENV}={other} is not bf16 or q8; sending q8"
                    ));
                    Self::Q8
                }
            }
        })
    }

    /// Bytes one row of `n_embd` values takes.
    pub fn row_bytes(self, n_embd: usize) -> usize {
        match self {
            Self::Bf16 => n_embd * 2,
            Self::Q8 => (n_embd.div_ceil(Q8_BLOCK) * 2 + n_embd).next_multiple_of(4),
        }
    }
}

/// Activation part identity carrying the BF16 input rows of target layer
/// `layer`.
pub fn dflash_feature_identity(layer: u32) -> [u8; ACTIVATION_IDENTITY_BYTES] {
    let mut identity = [0_u8; ACTIVATION_IDENTITY_BYTES];
    let suffix = layer.to_string();
    identity[..FEATURE_IDENTITY_PREFIX.len()].copy_from_slice(FEATURE_IDENTITY_PREFIX);
    identity[FEATURE_IDENTITY_PREFIX.len()..][..suffix.len()].copy_from_slice(suffix.as_bytes());
    identity
}

/// Activation part identity carrying the Q8-block input rows, `n_embd` wide,
/// of target layer `layer`.
pub(super) fn q8_feature_identity(
    layer: u32,
    n_embd: usize,
) -> Result<[u8; ACTIVATION_IDENTITY_BYTES]> {
    let mut identity = [0_u8; ACTIVATION_IDENTITY_BYTES];
    let suffix = format!("{layer}.{n_embd}");
    let end = Q8_FEATURE_IDENTITY_PREFIX.len() + suffix.len();
    ensure!(
        end <= ACTIVATION_IDENTITY_BYTES,
        "Q8 feature identity for layer {layer} of width {n_embd} does not fit"
    );
    identity[..Q8_FEATURE_IDENTITY_PREFIX.len()].copy_from_slice(Q8_FEATURE_IDENTITY_PREFIX);
    identity[Q8_FEATURE_IDENTITY_PREFIX.len()..end].copy_from_slice(suffix.as_bytes());
    Ok(identity)
}

/// The decimal text of `identity` after `prefix`, up to its zero padding.
fn identity_suffix<'a>(
    identity: &'a [u8; ACTIVATION_IDENTITY_BYTES],
    prefix: &[u8],
) -> Option<&'a str> {
    let suffix = identity.strip_prefix(prefix)?;
    let end = suffix
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(suffix.len());
    if end == 0 || suffix[end..].iter().any(|&byte| byte != 0) {
        return None;
    }
    std::str::from_utf8(&suffix[..end]).ok()
}

/// What a DFlash feature part carries: its layer, its encoding, and for Q8
/// rows their width.
pub(super) fn dflash_feature_part(
    identity: &[u8; ACTIVATION_IDENTITY_BYTES],
) -> Option<(u32, DFlashFeatureEncoding, Option<usize>)> {
    if let Some(suffix) = identity_suffix(identity, FEATURE_IDENTITY_PREFIX) {
        return Some((suffix.parse().ok()?, DFlashFeatureEncoding::Bf16, None));
    }
    let suffix = identity_suffix(identity, Q8_FEATURE_IDENTITY_PREFIX)?;
    let (layer, width) = suffix.split_once('.')?;
    Some((
        layer.parse().ok()?,
        DFlashFeatureEncoding::Q8,
        Some(width.parse().ok().filter(|width| *width > 0)?),
    ))
}

/// The target layer a DFlash feature part carries, in any encoding, or
/// `None` for any other part.
pub fn dflash_feature_layer(identity: &[u8; ACTIVATION_IDENTITY_BYTES]) -> Option<u32> {
    dflash_feature_part(identity).map(|(layer, _, _)| layer)
}

pub(super) fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x3ff);
    let magnitude = match (exponent, mantissa) {
        (0, 0) => 0,
        // Subnormal: value is mantissa * 2^-24.
        (0, _) => {
            let value = mantissa as f32 * f32::powi(2.0, -24);
            return if sign == 0 { value } else { -value };
        }
        (0x1f, _) => 0x7f80_0000 | (mantissa << 13),
        _ => ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(sign | magnitude)
}

/// Rounds a finite, non-negative `value` to the nearest F16, saturating at
/// the largest finite F16.
fn f16_from_nonnegative(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = i32::try_from((bits >> 23) & 0xff).unwrap_or(0) - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exponent >= 0x1f {
        return 0x7bff;
    }
    if exponent <= 0 {
        // Subnormal in F16, or zero.
        if exponent < -10 {
            return 0;
        }
        let full = mantissa | 0x80_0000;
        let shift = u32::try_from(14 - exponent).unwrap_or(31);
        let rounded = (full + (1 << (shift - 1)) - 1 + ((full >> shift) & 1)) >> shift;
        return u16::try_from(rounded).unwrap_or(0x7bff);
    }
    let rounded = mantissa + 0x0fff + ((mantissa >> 13) & 1);
    let out = (u32::try_from(exponent).unwrap_or(0) << 10) + (rounded >> 13);
    u16::try_from(out.min(0x7bff)).unwrap_or(0x7bff)
}

pub(super) fn encode_bf16(values: &[f32], out: &mut Vec<u8>) {
    out.reserve(values.len() * 2);
    for value in values {
        let bits = value.to_bits();
        // Round to nearest even; NaN keeps a quiet payload.
        let rounded = if value.is_nan() {
            (bits >> 16) | 0x40
        } else {
            (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
        };
        out.extend_from_slice(&(rounded as u16).to_le_bytes());
    }
}

pub(super) fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16))
        .collect()
}

/// Encodes rows of `n_embd` values as Q8 blocks: per block of up to 32
/// values, an F16 scale and one signed byte a value, with each row padded to
/// four bytes. A value that is not finite encodes as zero.
pub(super) fn encode_q8_rows(values: &[f32], n_embd: usize, out: &mut Vec<u8>) {
    let row_bytes = DFlashFeatureEncoding::Q8.row_bytes(n_embd);
    out.reserve(values.len().div_ceil(n_embd.max(1)) * row_bytes);
    for row in values.chunks(n_embd.max(1)) {
        let row_start = out.len();
        for block in row.chunks(Q8_BLOCK) {
            let amax = block
                .iter()
                .filter(|value| value.is_finite())
                .fold(0.0_f32, |amax, value| amax.max(value.abs()));
            let scale_bits = f16_from_nonnegative(amax / 127.0);
            let scale = f16_to_f32(scale_bits);
            let inverse = if scale > 0.0 { scale.recip() } else { 0.0 };
            out.extend_from_slice(&scale_bits.to_le_bytes());
            out.extend(block.iter().map(|value| {
                let quantized = if value.is_finite() {
                    (value * inverse).round().clamp(-127.0, 127.0)
                } else {
                    0.0
                };
                (quantized as i8).to_le_bytes()[0]
            }));
        }
        out.resize(row_start + row_bytes, 0);
    }
}

/// Decodes rows `encode_q8_rows` wrote; the caller checked the length.
pub(super) fn decode_q8_rows(bytes: &[u8], n_embd: usize) -> Vec<f32> {
    let row_bytes = DFlashFeatureEncoding::Q8.row_bytes(n_embd);
    let mut values = Vec::with_capacity(bytes.len() / row_bytes.max(1) * n_embd);
    for row in bytes.chunks_exact(row_bytes.max(1)) {
        let mut offset = 0;
        let mut remaining = n_embd;
        while remaining > 0 {
            let len = remaining.min(Q8_BLOCK);
            let scale = f16_to_f32(u16::from_le_bytes([row[offset], row[offset + 1]]));
            values.extend(
                row[offset + 2..offset + 2 + len]
                    .iter()
                    .map(|&byte| f32::from(i8::from_le_bytes([byte])) * scale),
            );
            offset += 2 + len;
            remaining -= len;
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_identities_round_trip_and_ignore_other_parts() {
        for layer in [0, 6, 43, 4095] {
            assert_eq!(
                dflash_feature_layer(&dflash_feature_identity(layer)),
                Some(layer)
            );
        }
        assert_eq!(dflash_feature_layer(&[7; ACTIVATION_IDENTITY_BYTES]), None);
        let mut bare = [0; ACTIVATION_IDENTITY_BYTES];
        bare[..FEATURE_IDENTITY_PREFIX.len()].copy_from_slice(FEATURE_IDENTITY_PREFIX);
        assert_eq!(dflash_feature_layer(&bare), None);
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        let mut bytes = Vec::new();
        let values = [1.0_f32, 1.0 + f32::EPSILON, -3.140625, 65504.0, 1.0e30];
        encode_bf16(&values, &mut bytes);
        let decoded = decode_bf16(&bytes);
        assert_eq!(decoded[0], 1.0);
        assert_eq!(decoded[1], 1.0);
        assert_eq!(decoded[2], -3.140625);
        assert!((decoded[3] - 65504.0).abs() / 65504.0 < 0.01);
        assert!((decoded[4] - 1.0e30).abs() / 1.0e30 < 0.01);
    }

    #[test]
    fn q8_rows_keep_every_value_within_half_a_step_of_its_block() {
        for n_embd in [1, 31, 32, 33, 2560] {
            let rows = 3;
            let values = (0..rows * n_embd)
                .map(|index| {
                    let value = ((index * 7919) % 1000) as f32 / 100.0 - 5.0;
                    // An outlier channel now and then.
                    if index % 97 == 5 {
                        value * 400.0
                    } else {
                        value
                    }
                })
                .collect::<Vec<_>>();
            let mut bytes = Vec::new();
            encode_q8_rows(&values, n_embd, &mut bytes);
            assert_eq!(
                bytes.len(),
                rows * DFlashFeatureEncoding::Q8.row_bytes(n_embd)
            );
            let decoded = decode_q8_rows(&bytes, n_embd);
            assert_eq!(decoded.len(), values.len());
            for (got_row, want_row) in decoded.chunks(n_embd).zip(values.chunks(n_embd)) {
                for (got, want) in got_row.chunks(Q8_BLOCK).zip(want_row.chunks(Q8_BLOCK)) {
                    let amax = want
                        .iter()
                        .fold(0.0_f32, |amax, value| amax.max(value.abs()));
                    // Half a step, plus the F16 rounding of the block scale.
                    let bound = amax / 254.0 + amax * f32::powi(2.0, -10);
                    for (got, want) in got.iter().zip(want) {
                        assert!((got - want).abs() <= bound, "{n_embd}: {got} vs {want}");
                    }
                }
            }
        }
    }

    #[test]
    fn q8_rows_take_about_half_the_bytes_of_bf16() {
        assert_eq!(DFlashFeatureEncoding::Bf16.row_bytes(2560), 5120);
        assert_eq!(DFlashFeatureEncoding::Q8.row_bytes(2560), 2720);
        assert_eq!(DFlashFeatureEncoding::Q8.row_bytes(5120), 5440);
        // Two blocks of 32 and 1 values, padded to a word.
        assert_eq!(DFlashFeatureEncoding::Q8.row_bytes(33), 40);
    }

    #[test]
    fn q8_rows_encode_values_that_are_not_finite_as_zero() {
        let values = [f32::NAN, 1.0, f32::INFINITY, -2.0];
        let mut bytes = Vec::new();
        encode_q8_rows(&values, 4, &mut bytes);
        let decoded = decode_q8_rows(&bytes, 4);
        assert_eq!((decoded[0], decoded[2]), (0.0, 0.0));
        assert!((decoded[1] - 1.0).abs() < 1.0 / 64.0);
        assert!((decoded[3] + 2.0).abs() < 2.0 / 64.0);
    }

    #[test]
    fn q8_identities_name_their_layer_and_width() {
        let identity = q8_feature_identity(62, 5120).unwrap();
        assert_eq!(
            dflash_feature_part(&identity),
            Some((62, DFlashFeatureEncoding::Q8, Some(5120)))
        );
        assert_eq!(dflash_feature_layer(&identity), Some(62));
        assert!(q8_feature_identity(u32::MAX, usize::MAX).is_err());
        let mut no_width = [0; ACTIVATION_IDENTITY_BYTES];
        no_width[..Q8_FEATURE_IDENTITY_PREFIX.len()].copy_from_slice(Q8_FEATURE_IDENTITY_PREFIX);
        no_width[Q8_FEATURE_IDENTITY_PREFIX.len()] = b'7';
        assert_eq!(dflash_feature_layer(&no_width), None);
    }

    #[test]
    fn f16_decodes_normals_subnormals_and_specials() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), f32::powi(2.0, -24));
        assert_eq!(f16_to_f32(0x8001), -f32::powi(2.0, -24));
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn f16_scales_round_to_nearest() {
        for value in [0.0_f32, 1.0, 0.5, 2.0e-3, 65504.0, 6.0e-8, 1.0e-6, 3.3] {
            let back = f16_to_f32(f16_from_nonnegative(value));
            assert!(
                (back - value).abs() <= value * f32::powi(2.0, -11) + f32::powi(2.0, -25),
                "{value} -> {back}"
            );
        }
        assert_eq!(f16_from_nonnegative(1.0e9), 0x7bff);
        assert_eq!(f16_from_nonnegative(1.0e-12), 0);
    }
}
