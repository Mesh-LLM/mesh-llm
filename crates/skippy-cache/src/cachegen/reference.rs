//! Deterministic quantization and token-axis delta reference for CacheGen.
//!
//! This is the CPU oracle the later GPU work must reproduce bit-for-bit.
//! Every operation is an IEEE-754 single-precision arithmetic op or an
//! integer op, so results are identical on any conformant platform and any
//! backend that performs the same operations in the same order — that
//! property is what makes this module a golden reference rather than an
//! implementation detail.
//!
//! Simplification versus the CacheGen paper, deliberate and documented:
//! calibration is per-segment min/max affine at 4 bits. The paper calibrates
//! once per model over a sample of requests and applies K/M-mixed
//! quantization across tensor dimensions; both are follow-up experiments
//! behind the same container, and neither changes the wire contract.

use anyhow::{Result, bail};

/// Bits per quantized symbol (CacheGen's default 4-bit KV quantization).
pub const QUANT_BITS: u32 = 4;
/// Quantized symbol alphabet size.
pub const TOKEN_COUNT: usize = 1 << QUANT_BITS;
/// Mask extracting a symbol from a byte; the delta ring is modulo this.
pub(crate) const ALPHABET_MASK: u8 = (TOKEN_COUNT - 1) as u8;

/// Per-tile affine calibration, carried bit-exactly in the container.
///
/// `min` and `scale` are the f32 bit patterns of the calibration values:
/// storing bits rather than decimal text is what makes the container
/// deterministic. `scale = (max - min) / (TOKEN_COUNT - 1)` over the tile's
/// f16-decoded values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Calibration {
    /// f32 bits of the calibration minimum.
    pub min_bits: u32,
    /// f32 bits of the quantization step.
    pub scale_bits: u32,
}

/// Derives the affine calibration for one tile of f16-decoded values.
/// Refuses empty or non-finite input: an NaN/inf element would poison the
/// calibration, and the crate's activation policy refuses non-finite
/// payloads rather than encoding them.
pub fn calibrate(values: &[f32]) -> Result<Calibration> {
    let Some(&first) = values.first() else {
        bail!("cannot calibrate an empty tile");
    };
    let mut min = first;
    let mut max = first;
    for &value in values {
        if !value.is_finite() {
            bail!("tile contains a non-finite value; refusing to calibrate");
        }
        min = min.min(value);
        max = max.max(value);
    }
    let scale = (max - min) / f32::from(TOKEN_COUNT as u16 - 1);
    Ok(Calibration {
        min_bits: min.to_bits(),
        scale_bits: scale.to_bits(),
    })
}

/// Quantizes one tile to symbols in `[0, TOKEN_COUNT)`. Midpoint rounding,
/// clamped at both ends. Must be preceded by [`calibrate`] over the same
/// values.
pub fn quantize(calibration: &Calibration, values: &[f32]) -> Result<Vec<u8>> {
    let min = f32::from_bits(calibration.min_bits);
    let scale = f32::from_bits(calibration.scale_bits);
    let max_symbol = (TOKEN_COUNT - 1) as f32;
    values
        .iter()
        .map(|&value| {
            if !value.is_finite() {
                bail!("tile contains a non-finite value; refusing to quantize");
            }
            if scale == 0.0 {
                // Flat tile: every value equals the calibration minimum.
                return Ok(0u8);
            }
            let scaled = ((value - min) / scale).round();
            let clamped = scaled.clamp(0.0, max_symbol);
            Ok(clamped as u8)
        })
        .collect()
}

/// Reconstructs f32 values from symbols. The inverse of quantization up to
/// the quantization error itself (at most half a step before the final
/// f16 round-trip).
pub fn dequantize(calibration: &Calibration, symbols: &[u8]) -> Vec<f32> {
    let min = f32::from_bits(calibration.min_bits);
    let scale = f32::from_bits(calibration.scale_bits);
    symbols
        .iter()
        .map(|&symbol| f32::from(symbol) * scale + min)
        .collect()
}

/// Token-axis delta transform, CacheGen's cheap decorrelation pass.
///
/// Symbols along the token axis (contiguous rows of `dims` values) become
/// the difference to the previous row's symbol, reduced modulo the 4-bit
/// alphabet: the mask keeps every symbol inside `[0, TOKEN_COUNT)` so the
/// entropy stage still sees a 16-symbol alphabet, and mod-`TOKEN_COUNT`
/// arithmetic is an exact ring, so the inverse is lossless.
///
/// `symbols.len()` must be a non-zero multiple of `dims`.
pub fn delta_encode(symbols: &mut [u8], dims: usize) -> Result<()> {
    validate_shape(symbols.len(), dims)?;
    for row in (dims..symbols.len()).rev() {
        symbols[row] = symbols[row].wrapping_sub(symbols[row - dims]) & ALPHABET_MASK;
    }
    Ok(())
}

/// Inverse of [`delta_encode`].
pub fn delta_decode(symbols: &mut [u8], dims: usize) -> Result<()> {
    validate_shape(symbols.len(), dims)?;
    for row in dims..symbols.len() {
        symbols[row] = symbols[row].wrapping_add(symbols[row - dims]) & ALPHABET_MASK;
    }
    Ok(())
}

/// Builds the static CDF from a symbol histogram: floor allocation with a
/// minimum frequency of one token per symbol, then deterministic
/// redistribution (subtract from the largest, add to the histogram-max with
/// lowest index) until the frequencies sum to the rANS scale. The
/// redistribution order is fixed, so the same histogram always yields the
/// same table on every backend.
///
/// `total` must equal `sum(histogram)` — the caller's contract, enforced
/// here. With it, both normalization loops are provably bounded: the
/// decrement loop removes at most one unit per min-clamped symbol (at most
/// `TOKEN_COUNT` iterations) and the increment loop adds at most
/// `SCALE - 1` missing units. Without it, a forged histogram/total pair
/// could drive unbounded repair work.
pub fn histogram_to_freqs(histogram: &[u32], total: usize) -> Result<Vec<u32>> {
    if histogram.len() != TOKEN_COUNT {
        bail!("histogram must cover every 4-bit symbol");
    }
    if total == 0 {
        bail!("cannot build a symbol table for an empty tile");
    }
    let histogram_total: u64 = histogram.iter().map(|&count| u64::from(count)).sum();
    if histogram_total != total as u64 {
        bail!("histogram totals {histogram_total} but the caller declares {total} symbols");
    }
    let mut freqs: Vec<u32> = histogram
        .iter()
        .map(|&count| (u64::from(count) * u64::from(super::rans::SCALE) / total as u64) as u32)
        .collect();
    // Every symbol must be decodable, even unused ones: a corrupt or
    // truncated stream must fail cleanly, not index an empty range.
    for freq in &mut freqs {
        *freq = (*freq).max(1);
    }
    let mut sum: u64 = freqs.iter().map(|&freq| u64::from(freq)).sum();
    while sum > u64::from(super::rans::SCALE) {
        let largest = freqs
            .iter()
            .enumerate()
            .max_by_key(|(index, freq)| (*freq, std::cmp::Reverse(*index)))
            .expect("non-empty")
            .0;
        freqs[largest] -= 1;
        sum -= 1;
    }
    let mut index = 0usize;
    while sum < u64::from(super::rans::SCALE) {
        // Unreachable with a validated histogram (every other symbol holds
        // at least one unit, so no target can sit at the full scale), but
        // the bound makes the loop's totality explicit rather than argued.
        if index > u64::from(super::rans::SCALE) as usize {
            bail!("histogram normalization failed to converge");
        }
        let candidate = histogram
            .iter()
            .enumerate()
            .max_by_key(|(position, count)| (*count, std::cmp::Reverse(*position)))
            .expect("non-empty")
            .0;
        let target = if index == 0 {
            candidate
        } else {
            (candidate + index) % TOKEN_COUNT
        };
        if freqs[target] < super::rans::SCALE {
            freqs[target] += 1;
            sum += 1;
        }
        index = index.wrapping_add(1);
    }
    Ok(freqs)
}

fn validate_shape(len: usize, dims: usize) -> Result<()> {
    if dims == 0 || !len.is_multiple_of(dims) {
        bail!("tile shape mismatch: {len} values are not rows of {dims}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantization_round_trips_within_half_a_step() {
        let values: Vec<f32> = (0..64).map(|index| index as f32 / 7.0).collect();
        let calibration = calibrate(&values).expect("calibrate");
        let symbols = quantize(&calibration, &values).expect("quantize");
        let rebuilt = dequantize(&calibration, &symbols);
        let scale = f32::from_bits(calibration.scale_bits);
        for (original, restored) in values.iter().zip(&rebuilt) {
            assert!(
                (original - restored).abs() <= scale,
                "{original} rebuilt as {restored} with step {scale}"
            );
        }
    }

    #[test]
    fn flat_tile_quantizes_to_zero_and_dequantizes_exactly() {
        let values = vec![0.25f32; 32];
        let calibration = calibrate(&values).expect("calibrate");
        assert_eq!(calibration.scale_bits, 0.0f32.to_bits());
        let symbols = quantize(&calibration, &values).expect("quantize");
        assert!(symbols.iter().all(|&symbol| symbol == 0));
        let rebuilt = dequantize(&calibration, &symbols);
        assert!(rebuilt.iter().all(|&value| value == 0.25));
    }

    #[test]
    fn delta_round_trips_across_the_wrap() {
        let dims = 3;
        let mut symbols = vec![15u8, 0, 7, 1, 14, 8, 2, 13, 9];
        let original = symbols.clone();
        delta_encode(&mut symbols, dims).expect("encode");
        delta_decode(&mut symbols, dims).expect("decode");
        assert_eq!(symbols, original);
    }

    #[test]
    fn delta_rejects_misshaped_tiles() {
        let mut symbols = vec![0u8; 7];
        assert!(delta_encode(&mut symbols, 3).is_err());
        assert!(delta_decode(&mut symbols, 0).is_err());
        assert!(delta_encode(&mut symbols, 7).is_ok());
    }

    #[test]
    fn non_finite_values_are_refused() {
        let mut values = vec![1.0f32, 2.0];
        values.push(f32::NAN);
        assert!(calibrate(&values).is_err());
        let calibration = calibrate(&[1.0, 2.0]).expect("calibrate");
        assert!(quantize(&calibration, &[f32::INFINITY]).is_err());
    }

    #[test]
    fn freqs_sum_to_scale_and_are_deterministic() {
        let mut histogram = vec![0u32; TOKEN_COUNT];
        histogram[0] = 700;
        histogram[3] = 200;
        histogram[9] = 100;
        let total: usize = histogram.iter().sum::<u32>() as usize;
        let first = histogram_to_freqs(&histogram, total).expect("freqs");
        let second = histogram_to_freqs(&histogram, total).expect("freqs");
        assert_eq!(first, second);
        let sum: u32 = first.iter().sum();
        assert_eq!(sum, super::super::rans::SCALE);
        assert!(first.iter().all(|&freq| freq >= 1));
    }

    #[test]
    fn a_total_that_disagrees_with_the_histogram_is_refused() {
        let mut histogram = vec![0u32; TOKEN_COUNT];
        histogram[0] = 700;
        histogram[3] = 200;
        histogram[9] = 100;
        // The forged-total cases: wildly low, wildly high, and off-by-one.
        // Any of them used to seed the CDF from a tile that never existed.
        for lie in [1usize, usize::MAX, 1001] {
            let error = histogram_to_freqs(&histogram, lie)
                .expect_err("a disagreeing total must be refused");
            assert!(
                error.to_string().contains("declares"),
                "error should name the disagreement: {error}"
            );
        }
    }

    #[test]
    fn unused_symbols_still_get_a_decodable_frequency() {
        let histogram = {
            let mut histogram = vec![0u32; TOKEN_COUNT];
            histogram[2] = 4096;
            histogram
        };
        let freqs = histogram_to_freqs(&histogram, 4096).expect("freqs");
        assert!(freqs.iter().all(|&freq| freq >= 1));
        assert_eq!(freqs.iter().sum::<u32>(), super::super::rans::SCALE);
    }
}
