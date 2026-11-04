//! Reference dequantization of GGUF weight rows, in f64, following ggml's
//! `dequantize_row_*` for each format the verifier checks.
//!
//! The values are exact: every format stores small integers and f16
//! scales, so f64 holds the products ggml computes in f32 without
//! rounding. The matmul check compares them with the node's output at a
//! tolerance that covers the backend's own rounding.

use anyhow::{Result, bail};
use half::f16;

use crate::weights::type_name;

/// ggml type ids of the weight formats the verifier can dequantize.
pub const GGML_F32: u32 = 0;
pub const GGML_F16: u32 = 1;
pub const GGML_Q4_0: u32 = 2;
pub const GGML_Q4_1: u32 = 3;
pub const GGML_Q5_0: u32 = 6;
pub const GGML_Q5_1: u32 = 7;
pub const GGML_Q8_0: u32 = 8;
pub const GGML_Q2_K: u32 = 10;
pub const GGML_Q3_K: u32 = 11;
pub const GGML_Q4_K: u32 = 12;
pub const GGML_Q5_K: u32 = 13;
pub const GGML_Q6_K: u32 = 14;
pub const GGML_IQ4_NL: u32 = 20;
pub const GGML_IQ4_XS: u32 = 23;
pub const GGML_BF16: u32 = 30;

/// The non-linear 4-bit levels of IQ4_NL.
const IQ4_NL_LEVELS: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// One IQ4_XS block of 256 values: an f16 scale, a 6-bit scale per 32
/// values (low 4 bits in `scales_l`, high 2 in `scales_h`), and IQ4_NL
/// levels (`dequantize_row_iq4_xs`).
fn iq4_xs(block: &[u8; 136], out: &mut Vec<f64>) {
    let d = half(&block[..2]);
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let scales_l = &block[4..8];
    for (ib, qs) in block[8..].as_chunks::<16>().0.iter().enumerate() {
        let low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF;
        let high = ((scales_h >> (2 * ib)) & 3) as u8;
        let dl = d * f64::from(i32::from(low | (high << 4)) - 32);
        let level = |q: u8| dl * f64::from(IQ4_NL_LEVELS[usize::from(q)]);
        out.extend(qs.iter().map(|&q| level(q & 0xF)));
        out.extend(qs.iter().map(|&q| level(q >> 4)));
    }
}

fn half(bytes: &[u8]) -> f64 {
    f64::from(f16::from_le_bytes([bytes[0], bytes[1]]).to_f32())
}

/// Whether [`dequantize_row`] decodes `ggml_type`.
pub fn supports(ggml_type: u32) -> bool {
    matches!(
        ggml_type,
        GGML_F32
            | GGML_F16
            | GGML_BF16
            | GGML_Q4_0
            | GGML_Q4_1
            | GGML_Q5_0
            | GGML_Q5_1
            | GGML_Q8_0
            | GGML_Q2_K
            | GGML_Q3_K
            | GGML_Q4_K
            | GGML_Q5_K
            | GGML_Q6_K
            | GGML_IQ4_NL
            | GGML_IQ4_XS
    )
}

/// Reference dequantization of one row of `n` values.
pub fn dequantize_row(ggml_type: u32, raw: &[u8], n: usize) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(n);
    match ggml_type {
        GGML_F32 => out.extend(
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f64::from(f32::from_le_bytes(*c))),
        ),
        GGML_F16 => out.extend(raw.as_chunks::<2>().0.iter().map(|c| half(c))),
        GGML_BF16 => out.extend(
            raw.as_chunks::<2>()
                .0
                .iter()
                .map(|c| f64::from(half::bf16::from_le_bytes(*c).to_f32())),
        ),
        GGML_Q4_0 => raw
            .as_chunks::<18>()
            .0
            .iter()
            .for_each(|b| q4_0(b, &mut out)),
        GGML_Q4_1 => raw
            .as_chunks::<20>()
            .0
            .iter()
            .for_each(|b| q4_1(b, &mut out)),
        GGML_Q5_0 => raw
            .as_chunks::<22>()
            .0
            .iter()
            .for_each(|b| q5_0(b, &mut out)),
        GGML_Q5_1 => raw
            .as_chunks::<24>()
            .0
            .iter()
            .for_each(|b| q5_1(b, &mut out)),
        GGML_Q8_0 => {
            for block in raw.as_chunks::<34>().0 {
                let d = half(block);
                out.extend(block[2..].iter().map(|&q| d * f64::from(q as i8)));
            }
        }
        GGML_Q2_K => raw
            .as_chunks::<84>()
            .0
            .iter()
            .for_each(|b| q2_k(b, &mut out)),
        GGML_Q3_K => raw
            .as_chunks::<110>()
            .0
            .iter()
            .for_each(|b| q3_k(b, &mut out)),
        GGML_Q4_K => raw
            .as_chunks::<144>()
            .0
            .iter()
            .for_each(|b| k_min_blocks(b, None, &b[16..], &mut |s| k_min_values(s, &mut out))),
        GGML_Q5_K => raw.as_chunks::<176>().0.iter().for_each(|b| {
            k_min_blocks(b, Some(&b[16..48]), &b[48..], &mut |s| {
                k_min_values(s, &mut out);
            });
        }),
        GGML_Q6_K => raw
            .as_chunks::<210>()
            .0
            .iter()
            .for_each(|b| q6_k(b, &mut out)),
        GGML_IQ4_NL => {
            for block in raw.as_chunks::<18>().0 {
                let d = half(block);
                let level = |q: u8| d * f64::from(IQ4_NL_LEVELS[usize::from(q)]);
                out.extend(block[2..].iter().map(|&q| level(q & 0xF)));
                out.extend(block[2..].iter().map(|&q| level(q >> 4)));
            }
        }
        GGML_IQ4_XS => raw
            .as_chunks::<136>()
            .0
            .iter()
            .for_each(|b| iq4_xs(b, &mut out)),
        other => bail!("no audit profile for weight type {} yet", type_name(other)),
    }
    if out.len() != n {
        bail!("row decoded to {} values, expected {n}", out.len());
    }
    Ok(out)
}

/// `d`, 16 bytes of nibbles: low nibbles are values 0..16, high 16..32.
fn q4_0(b: &[u8; 18], out: &mut Vec<f64>) {
    let d = half(b);
    out.extend(b[2..].iter().map(|&q| d * (f64::from(q & 0xF) - 8.0)));
    out.extend(b[2..].iter().map(|&q| d * (f64::from(q >> 4) - 8.0)));
}

fn q4_1(b: &[u8; 20], out: &mut Vec<f64>) {
    let (d, m) = (half(b), half(&b[2..]));
    out.extend(b[4..].iter().map(|&q| d * f64::from(q & 0xF) + m));
    out.extend(b[4..].iter().map(|&q| d * f64::from(q >> 4) + m));
}

/// The fifth bit of each value, from a 32-bit mask.
fn high_bits(qh: u32, j: usize) -> (u8, u8) {
    let low = (((qh >> j) << 4) & 0x10) as u8;
    let high = ((qh >> (j + 12)) & 0x10) as u8;
    (low, high)
}

fn q5_0(b: &[u8; 22], out: &mut Vec<f64>) {
    let d = half(b);
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    let qs = &b[6..];
    let mut hi = [0.0; 16];
    for (j, &q) in qs.iter().enumerate() {
        let (h0, h1) = high_bits(qh, j);
        out.push(d * (f64::from((q & 0xF) | h0) - 16.0));
        hi[j] = d * (f64::from((q >> 4) | h1) - 16.0);
    }
    out.extend(hi);
}

fn q5_1(b: &[u8; 24], out: &mut Vec<f64>) {
    let (d, m) = (half(b), half(&b[2..]));
    let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let qs = &b[8..];
    let mut hi = [0.0; 16];
    for (j, &q) in qs.iter().enumerate() {
        let (h0, h1) = high_bits(qh, j);
        out.push(d * f64::from((q & 0xF) | h0) + m);
        hi[j] = d * f64::from((q >> 4) | h1) + m;
    }
    out.extend(hi);
}

/// `scales[16]`, `qs[64]`, `d`, `dmin`: 2-bit values with a 4-bit scale
/// and minimum per 16.
fn q2_k(b: &[u8; 84], out: &mut Vec<f64>) {
    let (scales, qs) = (&b[..16], &b[16..80]);
    let (d, dmin) = (half(&b[80..]), half(&b[82..]));
    let mut is = 0;
    for q in qs.as_chunks::<32>().0 {
        for shift in (0..8).step_by(2) {
            for half_block in [&q[..16], &q[16..]] {
                let sc = scales[is];
                is += 1;
                let (dl, ml) = (d * f64::from(sc & 0xF), dmin * f64::from(sc >> 4));
                out.extend(
                    half_block
                        .iter()
                        .map(|&v| dl * f64::from((v >> shift) & 3) - ml),
                );
            }
        }
    }
}

/// `hmask[32]`, `qs[64]`, `scales[12]`, `d`: 3-bit values (two low bits
/// and a high bit that, when clear, subtracts 4) with a 6-bit scale per 16.
fn q3_k(b: &[u8; 110], out: &mut Vec<f64>) {
    let (hmask, qs) = (&b[..32], &b[32..96]);
    let d = half(&b[108..]);
    let word = |i: usize| u32::from_le_bytes([b[96 + i], b[97 + i], b[98 + i], b[99 + i]]);
    let (a0, a1, tmp) = (word(0), word(4), word(8));
    let (kmask1, kmask2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
    let aux = [
        (a0 & kmask2) | ((tmp & kmask1) << 4),
        (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4),
        ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4),
        ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4),
    ];
    let scales: Vec<i8> = aux
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .map(|v| v as i8)
        .collect();
    let mut is = 0;
    let mut m = 1u8;
    for q in qs.as_chunks::<32>().0 {
        for shift in (0..8).step_by(2) {
            for offset in [0, 16] {
                let dl = d * f64::from(i32::from(scales[is]) - 32);
                is += 1;
                out.extend((offset..offset + 16).map(|l| {
                    let low = i32::from((q[l] >> shift) & 3);
                    let high = if hmask[l] & m != 0 { 0 } else { 4 };
                    dl * f64::from(low - high)
                }));
            }
            m <<= 1;
        }
    }
}

/// The 6-bit scale and minimum of sub-block `j` of a Q4_K or Q5_K block.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// One sub-block of 32 values of a weight with a minimum per 32: value
/// `k` is `scale · quants[k] − min`.
#[derive(Clone, Debug, PartialEq)]
pub struct MinBlock {
    pub scale: f64,
    pub min: f64,
    pub quants: [u8; 32],
}

/// The sub-blocks of a row of a weight with a minimum per 32 values
/// (Q4_K, Q5_K), for the kernels that apply the minimum to a sum of the
/// activation rather than to each value.
pub fn min_blocks(ggml_type: u32, raw: &[u8]) -> Result<Vec<MinBlock>> {
    let mut out = Vec::new();
    let mut push = |block| out.push(block);
    match ggml_type {
        GGML_Q4_K => raw
            .as_chunks::<144>()
            .0
            .iter()
            .for_each(|b| k_min_blocks(b, None, &b[16..], &mut push)),
        GGML_Q5_K => raw
            .as_chunks::<176>()
            .0
            .iter()
            .for_each(|b| k_min_blocks(b, Some(&b[16..48]), &b[48..], &mut push)),
        other => bail!("{} weights have no minimum per 32 values", type_name(other)),
    }
    Ok(out)
}

/// The eight sub-blocks of a Q4_K block (`d`, `dmin`, `scales[12]`,
/// `qs[128]`: 4-bit values with a 6-bit scale and minimum per 32), or of a
/// Q5_K block with a fifth bit per value from `qh[32]`.
fn k_min_blocks(b: &[u8], qh: Option<&[u8]>, qs: &[u8], visit: &mut impl FnMut(MinBlock)) {
    let (d, dmin) = (half(b), half(&b[2..]));
    let scales = &b[4..16];
    for (i, q) in qs.as_chunks::<32>().0.iter().enumerate() {
        for high_nibble in [false, true] {
            let j = 2 * i + usize::from(high_nibble);
            let (s, m) = scale_min_k4(j, scales);
            let mut quants = [0u8; 32];
            for (k, (out, &v)) in quants.iter_mut().zip(q).enumerate() {
                let low = if high_nibble { v >> 4 } else { v & 0xF };
                let fifth = qh.map_or(0, |qh| ((qh[k] >> j) & 1) << 4);
                *out = low | fifth;
            }
            visit(MinBlock {
                scale: d * f64::from(s),
                min: dmin * f64::from(m),
                quants,
            });
        }
    }
}

fn k_min_values(block: MinBlock, out: &mut Vec<f64>) {
    let MinBlock { scale, min, quants } = block;
    out.extend(quants.iter().map(|&q| scale * f64::from(q) - min));
}

/// `ql[128]`, `qh[64]`, `scales[16]` (i8), `d`: 6-bit values less 32 with
/// an 8-bit scale per 16.
fn q6_k(b: &[u8; 210], out: &mut Vec<f64>) {
    let d = half(&b[208..]);
    for n in 0..2 {
        let ql = &b[n * 64..n * 64 + 64];
        let qh = &b[128 + n * 32..128 + n * 32 + 32];
        let sc = &b[192 + n * 8..192 + n * 8 + 8];
        let mut y = [0.0; 128];
        for l in 0..32 {
            let is = l / 16;
            let value = |low: u8, shift: u32| {
                f64::from(i32::from(low | (((qh[l] >> shift) & 3) << 4)) - 32)
            };
            let scale = |k: usize| d * f64::from(sc[is + k] as i8);
            y[l] = scale(0) * value(ql[l] & 0xF, 0);
            y[l + 32] = scale(2) * value(ql[l + 32] & 0xF, 2);
            y[l + 64] = scale(4) * value(ql[l] >> 4, 4);
            y[l + 96] = scale(6) * value(ql[l + 32] >> 4, 6);
        }
        out.extend(y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_bytes(v: f32) -> [u8; 2] {
        f16::from_f32(v).to_le_bytes()
    }

    #[test]
    fn q8_0_block_dequantizes_to_scale_times_quant() {
        let mut block = f16_bytes(0.5).to_vec();
        block.extend((0..32).map(|i| (i as i8 - 16) as u8));
        let row = dequantize_row(GGML_Q8_0, &block, 32).unwrap();
        assert_eq!(row[0], -8.0);
        assert_eq!(row[31], 7.5);
    }

    #[test]
    fn q4_0_and_q4_1_split_nibbles_into_halves() {
        let mut block = f16_bytes(2.0).to_vec();
        block.extend([0x9Fu8; 16]);
        let row = dequantize_row(GGML_Q4_0, &block, 32).unwrap();
        assert_eq!((row[0], row[16]), (14.0, 2.0));
        let mut block = f16_bytes(2.0).to_vec();
        block.extend(f16_bytes(-1.0));
        block.extend([0x9Fu8; 16]);
        let row = dequantize_row(GGML_Q4_1, &block, 32).unwrap();
        assert_eq!((row[0], row[16]), (29.0, 17.0));
    }

    #[test]
    fn q5_0_takes_the_fifth_bit_from_the_mask() {
        let mut block = f16_bytes(1.0).to_vec();
        // fifth bit set for value 0 and for value 16 + 3
        block.extend((1u32 | (1 << 19)).to_le_bytes());
        block.extend([0u8; 16]);
        let row = dequantize_row(GGML_Q5_0, &block, 32).unwrap();
        assert_eq!(row[0], 0.0);
        assert_eq!(row[1], -16.0);
        assert_eq!(row[19], 0.0);
        assert_eq!(row[20], -16.0);
    }

    #[test]
    fn q4_k_applies_scales_and_minimums_per_32() {
        let mut block = f16_bytes(1.0).to_vec();
        block.extend(f16_bytes(0.5));
        // sub-blocks 0..4: scale j + 1, minimum 2; sub-blocks 4..8: scale 1, minimum 0
        let mut scales = [0u8; 12];
        for j in 0..4 {
            scales[j] = j as u8 + 1;
            scales[j + 4] = 2;
            scales[j + 8] = 1;
        }
        block.extend(scales);
        block.extend([0x21u8; 128]);
        let row = dequantize_row(GGML_Q4_K, &block, 256).unwrap();
        // sub-block 0 reads low nibbles (1), sub-block 1 high nibbles (2)
        assert_eq!(row[0], 1.0 * 1.0 - 1.0);
        assert_eq!(row[32], 2.0 * 2.0 - 1.0);
        assert_eq!(row[64], 3.0 * 1.0 - 1.0);
        assert_eq!(row[255], 1.0 * 2.0);
    }

    #[test]
    fn q6_k_subtracts_32_and_scales_per_16() {
        let mut block = vec![0u8; 210];
        block[..128].fill(0x11);
        block[128..192].fill(0b1010_1010);
        for (i, s) in block[192..208].iter_mut().enumerate() {
            *s = (i as i8 - 4) as u8;
        }
        block[208..].copy_from_slice(&f16_bytes(0.25));
        let row = dequantize_row(GGML_Q6_K, &block, 256).unwrap();
        // value (1 | 2 << 4) - 32 = 1 in every position
        assert_eq!(row[0], 0.25 * -4.0);
        assert_eq!(row[16], 0.25 * -3.0);
        assert_eq!(row[32], 0.25 * -2.0);
        assert_eq!(row[128], 0.25 * 4.0);
    }

    #[test]
    fn every_k_quant_decodes_a_whole_row() {
        for (t, bytes) in [(GGML_Q2_K, 84), (GGML_Q3_K, 110), (GGML_Q5_K, 176)] {
            let row = dequantize_row(t, &vec![0u8; bytes * 2], 512).unwrap();
            assert_eq!(row.len(), 512);
            assert!(row.iter().all(|v| v.is_finite()));
        }
        let mut q3 = vec![0u8; 110];
        q3[108..].copy_from_slice(&f16_bytes(1.0));
        // every scale is 0 - 32 and every value 0 - 4: 128 everywhere
        let row = dequantize_row(GGML_Q3_K, &q3, 256).unwrap();
        assert!(row.iter().all(|&v| v == 128.0), "{:?}", &row[..4]);
    }

    #[test]
    fn iq4_xs_joins_its_six_bit_scales_and_maps_levels() {
        let mut block = vec![0u8; 136];
        block[..2].copy_from_slice(&f16_bytes(1.0));
        // sub-block 0: scale 1 | 2 << 4 = 33, so 33 - 32 = 1; sub-block 1: 0 - 32
        block[2] = 0b10;
        block[4] = 0x01;
        block[8] = 0x8F;
        let row = dequantize_row(GGML_IQ4_XS, &block, 256).unwrap();
        assert_eq!(row[0], 113.0);
        assert_eq!(row[16], 1.0);
        assert_eq!(row[32], -32.0 * -127.0);
        assert_eq!(row.len(), 256);
    }

    #[test]
    fn unknown_types_are_refused() {
        assert!(dequantize_row(19, &[0u8; 50], 256).is_err());
    }
}
