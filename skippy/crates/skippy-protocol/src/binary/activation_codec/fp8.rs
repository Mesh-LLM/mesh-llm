//! FP8 activation payloads.
//!
//! Values use the OCP 8-bit floating-point formats: E4M3 in its "FN" variant
//! (no infinities, a single NaN code per sign, ±448 largest finite) and E5M2
//! (IEEE-style infinities and NaNs, ±57344 largest finite). Encoding rounds to
//! nearest, ties to even, and saturates to the largest finite value; it never
//! emits an infinity or NaN code, and the decoder rejects them.
//!
//! Two scale layouts keep activations with heavy-tailed outliers in range:
//!
//! - Row: each logical row is a little-endian finite positive F32 scale
//!   followed by `columns` FP8 codes, the same framing as S8 rows. The scale
//!   maps the row's largest magnitude onto the format's largest finite value.
//! - Block32: each logical row is `ceil(columns / 32)` E8M0 scale bytes
//!   followed by `columns` FP8 codes (the OCP MX layout). Scale byte `e`
//!   means `2^(e - 127)`; `0xff` is reserved. Each block takes the smallest
//!   power of two that keeps its largest magnitude within range, so dividing
//!   by the scale is exact and an outlier only costs precision in its own
//!   32 values.
//!
//! Both layouts keep every row self-contained, so concatenating independently
//! encoded row chunks is byte-identical to encoding the whole shape at once.

use std::io;
use std::sync::LazyLock;

use super::{ActivationShape, checked_wire_bytes, invalid_data, validate_payload_bytes};

/// Values sharing one E8M0 scale in the block layout.
pub(super) const BLOCK_VALUES: usize = 32;
const E8M0_BIAS: i32 = 127;
const E8M0_RESERVED: u8 = 0xff;

/// One OCP FP8 interchange format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Fp8Format {
    mantissa_bits: u32,
    bias: i32,
    /// Largest finite magnitude code (sign bit clear).
    max_code: u8,
    /// Smallest sign-clear code that is not finite (E4M3's NaN, E5M2's
    /// infinity); every code from it up to `0x7f` is reserved.
    first_non_finite_code: u8,
}

pub(super) const E4M3: Fp8Format = Fp8Format {
    mantissa_bits: 3,
    bias: 7,
    max_code: 0x7e,
    first_non_finite_code: 0x7f,
};

pub(super) const E5M2: Fp8Format = Fp8Format {
    mantissa_bits: 2,
    bias: 15,
    max_code: 0x7b,
    first_non_finite_code: 0x7c,
};

static E4M3_VALUES: LazyLock<[f32; 256]> = LazyLock::new(|| E4M3.value_table());
static E5M2_VALUES: LazyLock<[f32; 256]> = LazyLock::new(|| E5M2.value_table());

impl Fp8Format {
    fn values(self) -> &'static [f32; 256] {
        if self == E4M3 {
            &E4M3_VALUES
        } else {
            &E5M2_VALUES
        }
    }

    /// Exact magnitude of a sign-clear finite code.
    fn magnitude(self, code: u8) -> f64 {
        let mantissa_bits = self.mantissa_bits as i32;
        let exponent = i32::from(code >> self.mantissa_bits);
        let fraction = f64::from(code & ((1 << self.mantissa_bits) - 1));
        if exponent == 0 {
            fraction * 2_f64.powi(1 - self.bias - mantissa_bits)
        } else {
            (f64::from(1_u8 << self.mantissa_bits) + fraction)
                * 2_f64.powi(exponent - self.bias - mantissa_bits)
        }
    }

    pub(super) fn max_finite(self) -> f64 {
        self.magnitude(self.max_code)
    }

    /// Decoded value per code; NaN marks the codes the encoder never emits.
    fn value_table(self) -> [f32; 256] {
        let mut table = [f32::NAN; 256];
        for code in 0..self.first_non_finite_code {
            let magnitude = self.magnitude(code) as f32;
            table[usize::from(code)] = magnitude;
            table[usize::from(code | 0x80)] = -magnitude;
        }
        table
    }

    /// Round a finite value to nearest, ties to even, saturating to the
    /// largest finite code.
    pub(super) fn encode(self, value: f64) -> u8 {
        let sign = if value.is_sign_negative() { 0x80 } else { 0 };
        sign | self.encode_magnitude(value.abs())
    }

    fn encode_magnitude(self, magnitude: f64) -> u8 {
        let bits = magnitude.to_bits();
        let biased = ((bits >> 52) & 0x7ff) as i32;
        // Zero, or an f64 subnormal far below half the smallest FP8 subnormal.
        if biased == 0 {
            return 0;
        }
        let significand = (bits & ((1_u64 << 52) - 1)) | (1_u64 << 52);
        let exponent = biased - 1023 + self.bias;
        // Keep `mantissa_bits` fraction bits; subnormal targets drop more.
        let shift = 52 - self.mantissa_bits as i32 + (1 - exponent).max(0);
        if shift > 53 {
            // Below half the smallest subnormal: rounds to zero.
            return 0;
        }
        let shift = shift as u32;
        let mut quantized = significand >> shift;
        let remainder = significand & ((1_u64 << shift) - 1);
        let halfway = 1_u64 << (shift - 1);
        if remainder > halfway || (remainder == halfway && quantized & 1 == 1) {
            quantized += 1;
        }
        // A subnormal code is the quantized value itself (rounding up to
        // 2^mantissa_bits lands on the smallest normal). A normal code carries
        // a rounded-up mantissa into the exponent on its own.
        let code = if exponent >= 1 {
            ((exponent as u64 - 1) << self.mantissa_bits) + quantized
        } else {
            quantized
        };
        code.min(u64::from(self.max_code)) as u8
    }
}

pub(super) fn row_encoded_len(shape: ActivationShape) -> io::Result<usize> {
    checked_wire_bytes(shape.rows()?, row_bytes(shape)?)
}

pub(super) fn block_encoded_len(shape: ActivationShape) -> io::Result<usize> {
    checked_wire_bytes(shape.rows()?, block_row_bytes(shape)?)
}

fn row_bytes(shape: ActivationShape) -> io::Result<usize> {
    shape
        .columns
        .checked_add(4)
        .ok_or_else(|| invalid_data("FP8 row byte count overflow"))
}

fn block_row_bytes(shape: ActivationShape) -> io::Result<usize> {
    shape
        .columns
        .checked_add(shape.columns.div_ceil(BLOCK_VALUES))
        .ok_or_else(|| invalid_data("FP8 row byte count overflow"))
}

fn out_of_range() -> io::Error {
    invalid_data("FP8 activation value is out of range")
}

fn max_abs(values: &[f32]) -> f32 {
    values
        .iter()
        .fold(0.0_f32, |max_abs, value| max_abs.max(value.abs()))
}

/// Encode finite values with one F32 scale per logical row.
pub(super) fn encode_rows(
    format: Fp8Format,
    shape: ActivationShape,
    values: &[f32],
) -> io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(row_encoded_len(shape)?);
    let decoded = format.values();
    for row in values.chunks_exact(shape.columns) {
        let max_abs = max_abs(row);
        let scale = if max_abs == 0.0 {
            1.0
        } else {
            ((f64::from(max_abs) / format.max_finite()) as f32).max(f32::from_bits(1))
        };
        if !scale.is_finite() || scale <= 0.0 {
            return Err(invalid_data(
                "FP8 activation row scale is not finite and positive",
            ));
        }
        out.extend_from_slice(&scale.to_le_bytes());
        for value in row {
            let code = format.encode(f64::from(*value) / f64::from(scale));
            if !((f64::from(decoded[usize::from(code)]) * f64::from(scale)) as f32).is_finite() {
                return Err(out_of_range());
            }
            out.push(code);
        }
    }
    Ok(out)
}

pub(super) fn decode_rows(
    format: Fp8Format,
    shape: ActivationShape,
    payload: &[u8],
) -> io::Result<Vec<f32>> {
    let row_bytes = row_bytes(shape)?;
    validate_payload_bytes(payload.len(), row_encoded_len(shape)?)?;
    let decoded = format.values();
    let mut out = Vec::with_capacity(shape.elements()?);
    for row in payload.chunks_exact(row_bytes) {
        let scale = f32::from_le_bytes(row[..4].try_into().expect("FP8 scale prefix"));
        if !scale.is_finite() || scale <= 0.0 {
            return Err(invalid_data(
                "FP8 activation row scale is not finite and positive",
            ));
        }
        for code in &row[4..] {
            out.push(decode_code(decoded, *code, f64::from(scale))?);
        }
    }
    Ok(out)
}

/// The E8M0 exponent of the smallest power of two that keeps `max_abs`
/// within the format's finite range, clamped to E8M0's range.
fn block_exponent(format: Fp8Format, max_abs: f32) -> i32 {
    if max_abs == 0.0 {
        return 0;
    }
    let max_abs = f64::from(max_abs);
    // Every finite f32 is a normal f64, so this is floor(log2(max_abs)).
    let floor_log2 = ((max_abs.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    // max_finite lies in [2^(top - 1), 2^top), so the answer is one of two.
    let top = format.max_finite().log2().ceil() as i32;
    let exponent = if max_abs <= format.max_finite() * 2_f64.powi(floor_log2 - top + 1) {
        floor_log2 - top + 1
    } else {
        floor_log2 - top + 2
    };
    exponent.clamp(-E8M0_BIAS, E8M0_BIAS)
}

/// Encode finite values with one E8M0 scale per 32-value block of each row.
pub(super) fn encode_blocks(
    format: Fp8Format,
    shape: ActivationShape,
    values: &[f32],
) -> io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(block_encoded_len(shape)?);
    let decoded = format.values();
    for row in values.chunks_exact(shape.columns) {
        let exponents = row
            .chunks(BLOCK_VALUES)
            .map(|block| block_exponent(format, max_abs(block)))
            .collect::<Vec<_>>();
        out.extend(
            exponents
                .iter()
                .map(|exponent| (exponent + E8M0_BIAS) as u8),
        );
        for (block, exponent) in row.chunks(BLOCK_VALUES).zip(&exponents) {
            let scale = 2_f64.powi(*exponent);
            for value in block {
                let code = format.encode(f64::from(*value) / scale);
                if !((f64::from(decoded[usize::from(code)]) * scale) as f32).is_finite() {
                    return Err(out_of_range());
                }
                out.push(code);
            }
        }
    }
    Ok(out)
}

pub(super) fn decode_blocks(
    format: Fp8Format,
    shape: ActivationShape,
    payload: &[u8],
) -> io::Result<Vec<f32>> {
    let row_bytes = block_row_bytes(shape)?;
    let blocks = row_bytes - shape.columns;
    validate_payload_bytes(payload.len(), block_encoded_len(shape)?)?;
    let decoded = format.values();
    let mut out = Vec::with_capacity(shape.elements()?);
    for row in payload.chunks_exact(row_bytes) {
        let (scales, codes) = row.split_at(blocks);
        for (scale, block) in scales.iter().zip(codes.chunks(BLOCK_VALUES)) {
            if *scale == E8M0_RESERVED {
                return Err(invalid_data(
                    "FP8 activation payload contains reserved E8M0 scale",
                ));
            }
            let scale = 2_f64.powi(i32::from(*scale) - E8M0_BIAS);
            for code in block {
                out.push(decode_code(decoded, *code, scale)?);
            }
        }
    }
    Ok(out)
}

fn decode_code(decoded: &[f32; 256], code: u8, scale: f64) -> io::Result<f32> {
    let value = decoded[usize::from(code)];
    if value.is_nan() {
        return Err(invalid_data(
            "FP8 activation payload contains a non-finite code",
        ));
    }
    let value = (f64::from(value) * scale) as f32;
    if !value.is_finite() {
        return Err(invalid_data("activation payload contains non-finite value"));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_invalid(result: io::Result<impl std::fmt::Debug>, expected: &str) {
        let error = result.expect_err("operation should fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), expected);
    }

    /// Deterministic heavy-tailed activations: mostly small values with a few
    /// outliers two to three orders of magnitude larger.
    fn activations(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|index| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = ((state >> 40) as f32 / (1_u64 << 24) as f32) * 2.0 - 1.0;
                if index % 97 == 13 {
                    unit * 300.0
                } else {
                    unit * 0.75
                }
            })
            .collect()
    }

    #[test]
    fn largest_finite_values_match_the_ocp_formats() {
        assert_eq!(E4M3.max_finite(), 448.0);
        assert_eq!(E5M2.max_finite(), 57_344.0);
        assert_eq!(E4M3.magnitude(1), 2_f64.powi(-9));
        assert_eq!(E5M2.magnitude(1), 2_f64.powi(-16));
    }

    #[test]
    fn encode_rounds_to_nearest_even_including_mantissa_carry() {
        assert_eq!(E4M3.encode(1.0), 0x38);
        assert_eq!(
            E4M3.encode(1.0625),
            0x38,
            "tie between 1.0 and 1.125 keeps even"
        );
        assert_eq!(E4M3.encode(1.0626), 0x39);
        assert_eq!(
            E4M3.encode(1.1875),
            0x3a,
            "tie between 1.125 and 1.25 rounds to even"
        );
        assert_eq!(E4M3.encode(1.9375), 0x40, "carries into the exponent");
        assert_eq!(E4M3.encode(-2.5), 0xc2);
        assert_eq!(E5M2.encode(1.0), 0x3c);
        assert_eq!(
            E5M2.encode(1.125),
            0x3c,
            "tie between 1.0 and 1.25 keeps even"
        );
        assert_eq!(
            E5M2.encode(1.375),
            0x3e,
            "tie between 1.25 and 1.5 rounds to even"
        );
    }

    #[test]
    fn encode_rounds_subnormals_and_underflow_to_even() {
        let step = 2_f64.powi(-9);
        assert_eq!(E4M3.encode(step), 0x01);
        assert_eq!(
            E4M3.encode(step / 2.0),
            0x00,
            "half the smallest subnormal ties to zero"
        );
        assert_eq!(E4M3.encode(step * 0.51), 0x01);
        assert_eq!(E4M3.encode(step * 1.5), 0x02);
        assert_eq!(
            E4M3.encode(step * 7.5),
            0x08,
            "rounds up to the smallest normal"
        );
        assert_eq!(E4M3.encode(f64::from(f32::from_bits(1))), 0x00);
        assert_eq!(E4M3.encode(-f64::from(f32::from_bits(1))), 0x80);
        assert_eq!(E4M3.encode(0.0), 0x00);
        assert_eq!(E4M3.encode(-0.0), 0x80);
    }

    #[test]
    fn encode_saturates_instead_of_emitting_reserved_codes() {
        for value in [448.0, 463.9, 464.0, 480.0, 1.0e9, f64::from(f32::MAX)] {
            assert_eq!(E4M3.encode(value), 0x7e, "{value}");
            assert_eq!(E4M3.encode(-value), 0xfe, "-{value}");
        }
        for value in [57_344.0, 61_440.0, 1.0e12] {
            assert_eq!(E5M2.encode(value), 0x7b, "{value}");
            assert_eq!(E5M2.encode(-value), 0xfb, "-{value}");
        }
    }

    #[test]
    fn every_finite_code_round_trips_and_reserved_codes_decode_as_nan() {
        for format in [E4M3, E5M2] {
            let table = format.values();
            for code in 0_u8..=255 {
                let value = table[usize::from(code)];
                if code & 0x7f >= format.first_non_finite_code {
                    assert!(value.is_nan(), "{format:?} code {code:#04x}");
                } else {
                    assert_eq!(format.encode(f64::from(value)), code, "{format:?} {value}");
                }
            }
        }
    }

    #[test]
    fn row_layouts_have_deterministic_golden_bytes() {
        let shape = ActivationShape::new(2, 0, 4);
        let values = [0.0, 0.0, 0.0, 0.0, -448.0, 1.0, 0.5, 448.0];
        let payload = encode_rows(E4M3, shape, &values).unwrap();
        assert_eq!(
            payload,
            vec![
                0x00, 0x00, 0x80, 0x3f, 0, 0, 0, 0, // zero row keeps scale one
                0x00, 0x00, 0x80, 0x3f, 0xfe, 0x38, 0x30, 0x7e,
            ]
        );
        assert_eq!(decode_rows(E4M3, shape, &payload).unwrap(), values);

        let values = [0.0, 0.0, 0.0, 0.0, -57_344.0, 1.0, 0.5, 57_344.0];
        let payload = encode_rows(E5M2, shape, &values).unwrap();
        assert_eq!(
            &payload[8..],
            &[0x00, 0x00, 0x80, 0x3f, 0xfb, 0x3c, 0x38, 0x7b]
        );
        assert_eq!(decode_rows(E5M2, shape, &payload).unwrap(), values);
    }

    #[test]
    fn block_layout_has_golden_bytes_and_a_partial_last_block() {
        // 33 columns: one full block and a one-value block.
        let shape = ActivationShape::new(1, 0, 33);
        let mut values = vec![0.0_f32; 33];
        values[0] = 448.0;
        values[1] = -1.0;
        values[32] = 0.75;
        let payload = encode_blocks(E4M3, shape, &values).unwrap();
        assert_eq!(payload.len(), block_encoded_len(shape).unwrap());
        assert_eq!(payload.len(), 2 + 33);
        // 448 fits at 2^0; 0.75 needs 2^-9 so it lands on 384.
        assert_eq!(&payload[..2], &[127, 118]);
        assert_eq!(&payload[2..4], &[0x7e, 0xb8]);
        assert_eq!(payload[2 + 32], 0x7c);
        assert_eq!(decode_blocks(E4M3, shape, &payload).unwrap(), values);
    }

    #[test]
    fn block_exponent_is_the_smallest_power_of_two_that_fits() {
        assert_eq!(block_exponent(E4M3, 0.0), 0);
        assert_eq!(block_exponent(E4M3, 448.0), 0);
        assert_eq!(block_exponent(E4M3, 449.0), 1);
        assert_eq!(block_exponent(E4M3, 224.0), -1);
        assert_eq!(block_exponent(E4M3, 225.0), 0);
        assert_eq!(block_exponent(E4M3, f32::MAX), 120);
        assert_eq!(block_exponent(E4M3, f32::from_bits(1)), -127);
        assert_eq!(block_exponent(E5M2, 57_344.0), 0);
        assert_eq!(block_exponent(E5M2, 57_345.0), 1);
    }

    /// Round-to-nearest bound for one value: half a step of the format at the
    /// scaled magnitude, never finer than half the smallest subnormal step.
    fn error_bound(format: Fp8Format, value: f32, scale: f64) -> f64 {
        let half_relative = 2_f64.powi(-(format.mantissa_bits as i32) - 1);
        let half_subnormal = format.magnitude(1) / 2.0;
        let bound = (f64::from(value.abs()) * half_relative).max(half_subnormal * scale);
        // F32 scale and product roundings.
        bound * (1.0 + 1.0e-6) + f64::from(value.abs()) * 2.0e-7
    }

    #[test]
    fn row_round_trips_stay_within_the_rounding_bound() {
        let columns = 2560;
        let values = activations(columns * 3, 7);
        let shape = ActivationShape::new(3, 0, columns);
        for format in [E4M3, E5M2] {
            let decoded =
                decode_rows(format, shape, &encode_rows(format, shape, &values).unwrap()).unwrap();
            for (row, decoded_row) in values.chunks(columns).zip(decoded.chunks(columns)) {
                let scale = f64::from(max_abs(row)) / format.max_finite();
                for (value, decoded) in row.iter().zip(decoded_row) {
                    let error = f64::from((value - decoded).abs());
                    assert!(
                        error <= error_bound(format, *value, scale),
                        "{format:?}: {value} decoded as {decoded}"
                    );
                }
            }
        }
    }

    #[test]
    fn block_round_trips_stay_within_the_rounding_bound_per_block() {
        let columns = 2560;
        let values = activations(columns * 2, 11);
        let shape = ActivationShape::new(2, 0, columns);
        let decoded =
            decode_blocks(E4M3, shape, &encode_blocks(E4M3, shape, &values).unwrap()).unwrap();
        for (block, decoded_block) in values
            .chunks(BLOCK_VALUES)
            .zip(decoded.chunks(BLOCK_VALUES))
        {
            let scale = 2_f64.powi(block_exponent(E4M3, max_abs(block)));
            for (value, decoded) in block.iter().zip(decoded_block) {
                let error = f64::from((value - decoded).abs());
                assert!(
                    error <= error_bound(E4M3, *value, scale),
                    "{value} decoded as {decoded}"
                );
            }
        }
    }

    #[test]
    fn blocks_confine_an_outlier_to_its_own_block() {
        let shape = ActivationShape::new(1, 0, 64);
        let mut values = (0..64)
            .map(|index| 0.01 + index as f32 * 1.0e-4)
            .collect::<Vec<_>>();
        values[3] = 1000.0;
        let rows = decode_rows(E4M3, shape, &encode_rows(E4M3, shape, &values).unwrap()).unwrap();
        let blocks =
            decode_blocks(E4M3, shape, &encode_blocks(E4M3, shape, &values).unwrap()).unwrap();
        let worst = |decoded: &[f32]| {
            values[32..]
                .iter()
                .zip(&decoded[32..])
                .map(|(value, decoded)| ((value - decoded) / value).abs())
                .fold(0.0_f32, f32::max)
        };
        assert!(worst(&blocks) <= 0.0625, "{}", worst(&blocks));
        assert!(worst(&rows) > 0.1, "{}", worst(&rows));
    }

    #[test]
    fn rows_are_self_contained_chunks() {
        let columns = 40;
        let values = activations(columns * 3, 3);
        let shape = ActivationShape::new(2, 1, columns);
        let single = ActivationShape::new(1, 0, columns);
        for (all, chunked) in [
            (
                encode_rows(E4M3, shape, &values).unwrap(),
                values
                    .chunks(columns)
                    .flat_map(|row| encode_rows(E4M3, single, row).unwrap())
                    .collect::<Vec<_>>(),
            ),
            (
                encode_blocks(E4M3, shape, &values).unwrap(),
                values
                    .chunks(columns)
                    .flat_map(|row| encode_blocks(E4M3, single, row).unwrap())
                    .collect::<Vec<_>>(),
            ),
        ] {
            assert_eq!(all, chunked);
        }
    }

    #[test]
    fn tiny_rows_use_the_minimum_positive_subnormal_scale() {
        let shape = ActivationShape::new(1, 0, 2);
        let values = [f32::from_bits(1), -f32::from_bits(448)];
        let payload = encode_rows(E4M3, shape, &values).unwrap();
        assert_eq!(&payload[..4], &f32::from_bits(1).to_le_bytes());
        assert_eq!(decode_rows(E4M3, shape, &payload).unwrap(), values);
    }

    #[test]
    fn encoders_reject_values_whose_rounding_would_overflow_f32() {
        let shape = ActivationShape::new(1, 0, 1);
        assert_invalid(
            encode_blocks(E4M3, shape, &[f32::MAX]),
            "FP8 activation value is out of range",
        );
        assert_invalid(
            encode_blocks(E4M3, shape, &[-f32::MAX]),
            "FP8 activation value is out of range",
        );
        let large = encode_rows(E4M3, shape, &[f32::MAX]);
        if let Ok(payload) = large {
            assert!(decode_rows(E4M3, shape, &payload).unwrap()[0].is_finite());
        }
    }

    #[test]
    fn decoders_reject_reserved_codes_and_scales() {
        let shape = ActivationShape::new(1, 0, 1);
        let one = 1.0_f32.to_le_bytes();
        for (format, code) in [(E4M3, 0x7f), (E4M3, 0xff), (E5M2, 0x7c), (E5M2, 0xfd)] {
            let mut payload = one.to_vec();
            payload.push(code);
            assert_invalid(
                decode_rows(format, shape, &payload),
                "FP8 activation payload contains a non-finite code",
            );
        }
        assert_invalid(
            decode_blocks(E4M3, shape, &[127, 0x7f]),
            "FP8 activation payload contains a non-finite code",
        );
        assert_invalid(
            decode_blocks(E4M3, shape, &[E8M0_RESERVED, 0x38]),
            "FP8 activation payload contains reserved E8M0 scale",
        );
        assert_invalid(
            decode_blocks(E4M3, shape, &[254, 0x7e]),
            "activation payload contains non-finite value",
        );
        for scale in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
            let mut payload = scale.to_le_bytes().to_vec();
            payload.push(0x38);
            assert_invalid(
                decode_rows(E4M3, shape, &payload),
                "FP8 activation row scale is not finite and positive",
            );
        }
        let mut overflowing = f32::MAX.to_le_bytes().to_vec();
        overflowing.push(0x7e);
        assert_invalid(
            decode_rows(E4M3, shape, &overflowing),
            "activation payload contains non-finite value",
        );
        assert_invalid(
            decode_blocks(E4M3, shape, &[127]),
            "activation payload size mismatch",
        );
    }
}
