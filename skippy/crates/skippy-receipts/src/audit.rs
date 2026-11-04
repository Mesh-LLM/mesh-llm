//! Client-side row checks of an opening: weight rows of the claimed GGUF
//! against the opened matmul columns, random rows for most matmuls and
//! every row for a router. The Merkle paths are checked by
//! [`crate::binding`].
//!
//! The rows come from the VRF output of the opening's hash under the
//! client's key, drawn per weight, so the node cannot know them when it
//! opens and the client cannot pick them.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};

use crate::draw::DrawStream;
use crate::weights::{TensorInfo, WeightFile};
use crate::wire::{LayerOpening, MatrixOpening};

mod block_sums;
pub mod pairs;
pub mod rounding_flips;

/// A column check fails above this relative error (see RECEIPTS.md).
pub const TOLERANCE: f64 = 1e-3;
pub const ROWS_PER_MATRIX: usize = 16;

/// Failing columns a matrix may have and still pass. Honest backends fail
/// at most 0.34% of column checks, so an honest matrix of 16 columns has
/// three failures about once in 45,000 audits. A majority rule would let a
/// node cheat on half the positions, for example on every decode token of a
/// long prompt.
pub fn allowed_failing_columns(columns: usize) -> usize {
    2 + columns / 32
}

/// How a backend rounds the f32 activation before it meets the weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Raw,
    F16,
    /// The activation and the dequantized weights both rounded to f16, as
    /// Vulkan's cooperative-matrix kernels compute with f32 accumulation.
    F16Weights,
    /// Rounded to bf16, as CPU bf16 dot products and cuBLAS bf16 do.
    Bf16,
    /// CUDA `quantize_q8_1` (MMVQ): d = amax/127, q = round(x/d), d stored as f16.
    Q8DivHalf,
    /// CUDA `quantize_mmq_q8_1` (MMQ): q = round(x·127/amax), d stored as f16
    /// (the layout with sums, for Q4_0).
    Q8MulHalf,
    /// MMQ for Q4_K and Q5_K: [`Self::Q8MulHalf`], with the minimum applied
    /// to the f16 sum of the unquantized activation per 32 values, and the
    /// weight's scale and minimum rounded to f16 (the MMA kernels).
    Q8MulHalfSums,
    /// [`Self::Q8MulHalfSums`] with exact weight scales (the dp4a kernels).
    Q8MulHalfSumsDp4a,
    /// MMQ with d stored as f32 (Q5_0, Q3_K, Q6_K, IQ4_NL, IQ4_XS).
    Q8MulF32,
    /// MMQ with one f32 d per 64 values (Q2_K).
    Q8Mul64F32,
    /// CPU `quantize_row_q8_0`: d = amax/127, q = round(x·(1/d)), d stored as f16.
    CpuQ8,
    /// CPU `quantize_row_q8_K` for the K-quants: per 256 values, with `max`
    /// the value of largest magnitude, q = min(127, round_even(x·-127/max)),
    /// d = -max/127 stored as f32.
    CpuQ8K,
}

/// The backends with measured rounding profiles. Metal has none: through
/// `skippy serve`, its prefill outputs differ from every profile by up to
/// 8e-4, which would fail honest nodes (RECEIPTS.md, "Detection").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendFamily {
    Cpu,
    Cuda,
    Vulkan,
}

impl BackendFamily {
    /// The family of a receipt's backend name, such as `CPU` or `CUDA0`.
    pub fn of(backend: &str) -> Option<Self> {
        if backend.eq_ignore_ascii_case("cpu") {
            Some(Self::Cpu)
        } else if backend.starts_with("CUDA") {
            Some(Self::Cuda)
        } else if backend.starts_with("Vulkan") {
            Some(Self::Vulkan)
        } else {
            None
        }
    }

    /// [`Self::of`], failing for a backend without rounding profiles.
    pub fn checked(backend: &str) -> Result<Self> {
        Self::of(backend).with_context(|| format!("no rounding profiles for backend {backend} yet"))
    }
}

/// The profiles a backend may use for a weight type. A pair without an
/// entry cannot be checked, which makes the audit unusable rather than
/// letting the most lenient profile decide.
///
/// CUDA picks its kernel by batch size: MMVQ (`Q8DivHalf`) for decode,
/// MMQ (`Q8MulHalf`) for prefill, and a dequantize-and-cuBLAS path that
/// rounds the activation to f16. f16 and bf16 weights meet f32 activations
/// in the decode kernels and rounded ones in cuBLAS.
///
/// Vulkan decodes with f32 activations, or q8_1 ones where the device has
/// integer dot products (`CpuQ8`). Its prefill kernels meet f16 weights with
/// f16 activations, accumulated in f32 because receipts force that precision
/// on a Vulkan stage (`F16Weights`); with f16 accumulation, as Vulkan picks by
/// default, honest outputs are 1e-3 to 4e-3 off.
///
/// Q4_1 and Q5_1 have none: their kernels round the product of the weight
/// and activation scales, and the activation block sum, which a profile of
/// the activation alone cannot reproduce (RECEIPTS.md, "Detection").
pub fn profiles_for(family: BackendFamily, weight_type: u32) -> Option<&'static [Profile]> {
    use crate::dequant::{
        GGML_BF16, GGML_F16, GGML_F32, GGML_IQ4_NL, GGML_IQ4_XS, GGML_Q2_K, GGML_Q3_K, GGML_Q4_0,
        GGML_Q4_K, GGML_Q5_0, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0,
    };
    use Profile::{
        Bf16, CpuQ8, CpuQ8K, F16, F16Weights, Q8DivHalf, Q8Mul64F32, Q8MulF32, Q8MulHalf,
        Q8MulHalfSums, Q8MulHalfSumsDp4a, Raw,
    };
    let profiles: &'static [Profile] = match (family, weight_type) {
        (_, GGML_F32) => &[Raw],
        (BackendFamily::Cpu, GGML_F16) => &[F16],
        (BackendFamily::Cpu, GGML_BF16) => &[Bf16],
        (BackendFamily::Cpu, GGML_Q8_0 | GGML_Q4_0 | GGML_Q5_0 | GGML_IQ4_NL) => &[CpuQ8],
        (
            BackendFamily::Cpu,
            GGML_Q2_K | GGML_Q3_K | GGML_Q4_K | GGML_Q5_K | GGML_Q6_K | GGML_IQ4_XS,
        ) => &[CpuQ8K],
        (BackendFamily::Cuda, GGML_F16) => &[Raw, F16],
        (BackendFamily::Cuda, GGML_BF16) => &[Raw, Bf16],
        (BackendFamily::Cuda, GGML_Q8_0) => &[Q8DivHalf, Q8MulHalf, F16],
        (BackendFamily::Cuda, GGML_Q4_0) => &[Q8DivHalf, Q8MulHalf, F16],
        (BackendFamily::Cuda, GGML_Q4_K | GGML_Q5_K) => {
            &[Q8DivHalf, Q8MulHalfSums, Q8MulHalfSumsDp4a, F16]
        }
        (BackendFamily::Cuda, GGML_Q5_0 | GGML_Q3_K | GGML_Q6_K | GGML_IQ4_NL | GGML_IQ4_XS) => {
            &[Q8DivHalf, Q8MulF32, F16]
        }
        (BackendFamily::Cuda, GGML_Q2_K) => &[Q8DivHalf, Q8Mul64F32, F16],
        (BackendFamily::Vulkan, GGML_BF16) => &[Raw, Bf16],
        (
            BackendFamily::Vulkan,
            GGML_F16 | GGML_Q8_0 | GGML_Q4_0 | GGML_Q5_0 | GGML_Q2_K | GGML_Q3_K | GGML_Q4_K
            | GGML_Q5_K | GGML_Q6_K | GGML_IQ4_NL | GGML_IQ4_XS,
        ) => &[Raw, F16Weights, CpuQ8],
        _ => return None,
    };
    Some(profiles)
}

/// [`profiles_for`], failing for a pair without profiles.
pub fn checked_profiles(family: BackendFamily, weight_type: u32) -> Result<&'static [Profile]> {
    profiles_for(family, weight_type).with_context(|| {
        format!(
            "no rounding profile for {} weights on {family:?}",
            crate::weights::type_name(weight_type)
        )
    })
}

impl Profile {
    pub fn name(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::F16 => "f16",
            Self::F16Weights => "f16_weights",
            Self::Bf16 => "bf16",
            Self::Q8DivHalf => "q8_32_div_dh",
            Self::Q8MulHalf => "q8_32_mul_dh",
            Self::Q8MulHalfSums => "q8_32_mul_dh_sums",
            Self::Q8MulHalfSumsDp4a => "q8_32_mul_dh_sums_dp4a",
            Self::Q8MulF32 => "q8_32_mul_d",
            Self::Q8Mul64F32 => "q8_64_mul_d",
            Self::CpuQ8 => "cpu_q8_0",
            Self::CpuQ8K => "cpu_q8_K",
        }
    }

    /// Whether the weights meet the activation rounded to f16 too.
    fn rounds_weights(self) -> bool {
        self == Self::F16Weights
    }

    /// For the profiles that apply a weight's minimum to block sums of the
    /// activation: whether the weight's scales are rounded to f16.
    fn block_sums(self) -> Option<bool> {
        match self {
            Self::Q8MulHalfSums => Some(true),
            Self::Q8MulHalfSumsDp4a => Some(false),
            _ => None,
        }
    }

    pub fn apply(self, x: &[f32]) -> Vec<f64> {
        match self {
            Self::Raw => x.iter().map(|&v| f64::from(v)).collect(),
            Self::F16 | Self::F16Weights => x
                .iter()
                .map(|&v| f64::from(half::f16::from_f32(v).to_f32()))
                .collect(),
            Self::Bf16 => x
                .iter()
                .map(|&v| f64::from(half::bf16::from_f32(v).to_f32()))
                .collect(),
            Self::Q8DivHalf | Self::Q8MulHalf | Self::CpuQ8 => quantize_q8_blocks(self, x),
            Self::Q8MulHalfSums | Self::Q8MulHalfSumsDp4a => block_sums::quantized(x),
            Self::Q8MulF32 => quantize_mul_blocks(x, 32),
            Self::Q8Mul64F32 => quantize_mul_blocks(x, 64),
            Self::CpuQ8K => quantize_q8_k(x),
        }
    }
}

/// Dequantized weights as an f16 kernel holds them.
fn round_to_f16(w: &[f64]) -> Vec<f64> {
    w.iter()
        .map(|&v| f64::from(half::f16::from_f64(v).to_f32()))
        .collect()
}

fn quantize_q8_blocks(profile: Profile, x: &[f32]) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for block in x.chunks(32) {
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if amax == 0.0 {
            out.extend(std::iter::repeat_n(0.0, block.len()));
            continue;
        }
        let (d, quant): (f32, Box<dyn Fn(f32) -> f32>) = match profile {
            Profile::Q8DivHalf => {
                let d = amax / 127.0;
                (d, Box::new(move |v| (v / d).round()))
            }
            Profile::Q8MulHalf => {
                let d_inv = 127.0 / amax;
                (1.0 / d_inv, Box::new(move |v| (v * d_inv).round()))
            }
            _ => {
                let d = amax / 127.0;
                let id = 1.0 / d;
                (d, Box::new(move |v| (v * id).round()))
            }
        };
        let dd = f64::from(half::f16::from_f32(d).to_f32());
        out.extend(block.iter().map(|&v| dd * f64::from(quant(v))));
    }
    out
}

/// CUDA MMQ blocks of `block` values with an f32 scale.
fn quantize_mul_blocks(x: &[f32], block: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for values in x.chunks(block) {
        let amax = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if amax == 0.0 {
            out.extend(std::iter::repeat_n(0.0, values.len()));
            continue;
        }
        let d_inv = 127.0 / amax;
        let d = f64::from(1.0 / d_inv);
        out.extend(values.iter().map(|&v| d * f64::from((v * d_inv).round())));
    }
    out
}

/// CPU `quantize_row_q8_K_ref`, blocks of 256.
fn quantize_q8_k(x: &[f32]) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for values in x.chunks(256) {
        let max = values
            .iter()
            .fold(0.0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
        if max == 0.0 {
            out.extend(std::iter::repeat_n(0.0, values.len()));
            continue;
        }
        let iscale = -127.0 / max;
        let d = f64::from(1.0 / iscale);
        out.extend(
            values
                .iter()
                .map(|&v| d * f64::from((iscale * v).round_ties_even().min(127.0))),
        );
    }
    out
}

/// Relative error of the checked rows (RECEIPTS.md, "Fail rule"):
/// `||y[R] - y_ref[R]|| / (||y|| · sqrt(|R| / M))`.
pub fn column_error(rows: &[(usize, &[f64])], x: &[f64], y: &[f32]) -> f64 {
    let references: Vec<(usize, f64)> = rows
        .iter()
        .map(|(m, w)| (*m, w.iter().zip(x).map(|(a, b)| a * b).sum()))
        .collect();
    reference_error(&references, y)
}

/// [`column_error`] of the reference outputs of the checked rows.
fn reference_error(references: &[(usize, f64)], y: &[f32]) -> f64 {
    let norm_y = y
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt();
    let diff2: f64 = references
        .iter()
        .map(|&(m, reference)| (f64::from(y[m]) - reference).powi(2))
        .sum();
    let scale = norm_y * (references.len() as f64 / y.len() as f64).sqrt();
    if scale == 0.0 {
        return if diff2 == 0.0 { 0.0 } else { f64::INFINITY };
    }
    diff2.sqrt() / scale
}

#[derive(Clone, Debug)]
pub struct MatrixVerdict {
    pub weight: String,
    pub weight_type: String,
    /// A router: every row was checked.
    pub router: bool,
    /// Columns checked: one per position, or one per expert used.
    pub tokens: usize,
    pub failing_tokens: usize,
    pub max_error: f64,
    pub best_profile: &'static str,
    pub pass: bool,
}

#[derive(Clone, Debug)]
pub struct AuditReport {
    pub layer: u32,
    pub tokens: usize,
    pub matrices: Vec<MatrixVerdict>,
    pub pass: bool,
}

/// One opened matmul column split into the slices one weight row meets: a
/// `mul_mat_id` has one per expert used, a batched matmul one per batch.
struct MatmulSlices {
    /// Values per weight row (K) and rows per weight matrix (M).
    k: usize,
    m: usize,
    /// Per output slice: the first weight row and the input slice.
    slices: Vec<(u64, usize)>,
}

fn matmul_slices(
    tensor: &TensorInfo,
    m: &MatrixOpening,
    x: usize,
    y: usize,
) -> Result<MatmulSlices> {
    let k = usize::try_from(tensor.row_len())?;
    let rows = usize::try_from(tensor.rows())?;
    let matrices = tensor.dims.get(2).copied().unwrap_or(1);
    let shape = || {
        format!(
            "{}: opened input of {x} and output of {y} values do not fit the claimed weight {:?}",
            m.weight, tensor.dims
        )
    };
    if k == 0 || rows == 0 || !x.is_multiple_of(k) || !y.is_multiple_of(rows) {
        bail!(shape());
    }
    let (groups_in, groups_out) = (x / k, y / rows);
    let slices = match &m.experts {
        Some(experts) => {
            if experts.len() != groups_out || (groups_in != 1 && groups_in != groups_out) {
                bail!(shape());
            }
            experts
                .iter()
                .enumerate()
                .map(|(g, &e)| {
                    let e = u64::try_from(e)
                        .ok()
                        .filter(|&e| e < matrices)
                        .with_context(|| {
                            format!("{}: expert {e} outside {matrices} experts", m.weight)
                        })?;
                    Ok((e * rows as u64, if groups_in == 1 { 0 } else { g }))
                })
                .collect::<Result<_>>()?
        }
        None => {
            if groups_in != groups_out {
                bail!(shape());
            }
            match matrices {
                1 => (0..groups_out).map(|g| (0, g)).collect(),
                n if n == groups_out as u64 => (0..groups_out)
                    .map(|g| (g as u64 * rows as u64, g))
                    .collect(),
                _ => bail!(shape()),
            }
        }
    };
    Ok(MatmulSlices { k, m: rows, slices })
}

/// Fetches every row the checks of `opening` read in one round, so a
/// sparse copy of the claimed GGUF makes its requests at once and not one
/// per row.
fn prefetch_rows(
    weights: &WeightFile,
    cache: &mut RowCache,
    opening: &LayerOpening,
    seed: &[u8],
) -> Result<()> {
    let mut ranges = Vec::new();
    let leaves = opening.leaves.iter();
    for (position, m) in leaves.flat_map(|leaf| leaf.matrices.iter().map(|m| (leaf.position, m))) {
        let tensor = weights.tensor(&m.weight)?;
        let (inputs, outputs) = m.value_counts()?;
        let shape = matmul_slices(tensor, m, inputs, outputs)?;
        let picked = cache.picked(&m.weight, shape.m, m.router, seed);
        for g in cache.checked_slices(m, position, shape.slices.len()) {
            let (base, _) = shape.slices[g];
            for &r in &picked {
                ranges.push(weights.row_range(&m.weight, base + r as u64)?);
            }
        }
    }
    weights.prefetch(&ranges)
}

/// Weight rows already read, by tensor and row.
#[derive(Default)]
struct RowCache {
    rows: BTreeMap<(String, u64), Vec<f64>>,
    /// The same rows as sub-blocks, for the profiles with block sums.
    blocks: BTreeMap<(String, u64), Vec<crate::dequant::MinBlock>>,
    /// Row indices checked per weight; every row for a router.
    picked: BTreeMap<String, Vec<usize>>,
    /// The (position, expert) pairs whose expert matmuls are checked, for
    /// a MoE layer.
    pairs: Option<std::collections::BTreeSet<(u32, i32)>>,
}

impl RowCache {
    /// The output slices of `m` at `position` to check: every one, but for
    /// an expert matmul only the slices of the picked pairs.
    fn checked_slices(&self, m: &MatrixOpening, position: u32, slices: usize) -> Vec<usize> {
        match (&m.experts, &self.pairs) {
            (Some(experts), Some(pairs)) => (0..slices)
                .filter(|&g| {
                    experts
                        .get(g)
                        .is_some_and(|&e| pairs.contains(&(position, e)))
                })
                .collect(),
            _ => (0..slices).collect(),
        }
    }

    fn picked(&mut self, weight: &str, m: usize, all: bool, seed: &[u8]) -> Vec<usize> {
        self.picked
            .entry(weight.to_owned())
            .or_insert_with(|| {
                if all || m <= ROWS_PER_MATRIX {
                    return (0..m).collect();
                }
                let mut stream = DrawStream::new(seed, &format!("rows/{weight}"));
                let mut picked = std::collections::BTreeSet::new();
                while picked.len() < ROWS_PER_MATRIX {
                    picked.insert(stream.below(m as u64) as usize);
                }
                picked.into_iter().collect()
            })
            .clone()
    }

    fn load_blocks(&mut self, weights: &WeightFile, weight: &str, row: u64) -> Result<()> {
        if let std::collections::btree_map::Entry::Vacant(entry) =
            self.blocks.entry((weight.to_owned(), row))
        {
            entry.insert(weights.min_blocks(weight, row)?);
        }
        Ok(())
    }

    fn load(&mut self, weights: &WeightFile, weight: &str, row: u64) -> Result<()> {
        if let std::collections::btree_map::Entry::Vacant(entry) =
            self.rows.entry((weight.to_owned(), row))
        {
            entry.insert(weights.row(weight, row)?);
        }
        Ok(())
    }
}

/// Per checked column: its error, and the profile that gave it.
type ColumnErrors = Vec<(f64, &'static str)>;

/// Errors of every slice of one opened matmul column, with the best profile.
fn matmul_errors(
    weights: &WeightFile,
    cache: &mut RowCache,
    position: u32,
    m: &MatrixOpening,
    seed: &[u8],
    family: BackendFamily,
) -> Result<ColumnErrors> {
    let x = m.input_values()?;
    let y = m.output_values()?;
    let tensor = weights.tensor(&m.weight)?;
    let profiles = checked_profiles(family, tensor.ggml_type)?;
    let shape = matmul_slices(tensor, m, x.len(), y.len())?;
    let picked = cache.picked(&m.weight, shape.m, m.router, seed);
    let mut errors = Vec::with_capacity(shape.slices.len());
    for g in cache.checked_slices(m, position, shape.slices.len()) {
        let (base, input) = shape.slices[g];
        let sums = profiles.iter().any(|p| p.block_sums().is_some());
        for &r in &picked {
            cache.load(weights, &m.weight, base + r as u64)?;
            if sums {
                cache.load_blocks(weights, &m.weight, base + r as u64)?;
            }
        }
        let rows: Vec<(usize, &[f64])> = picked
            .iter()
            .map(|&r| {
                (
                    r,
                    cache.rows[&(m.weight.clone(), base + r as u64)].as_slice(),
                )
            })
            .collect();
        let xs = &x[input * shape.k..(input + 1) * shape.k];
        let ys = &y[g * shape.m..(g + 1) * shape.m];
        let error = |p: Profile| match p.block_sums() {
            Some(rounded) => {
                let xq = p.apply(xs);
                let references: Vec<(usize, f64)> = picked
                    .iter()
                    .map(|&r| {
                        let blocks = &cache.blocks[&(m.weight.clone(), base + r as u64)];
                        (r, block_sums::reference(blocks, xs, &xq, rounded))
                    })
                    .collect();
                reference_error(&references, ys)
            }
            None if p.rounds_weights() => {
                let rounded: Vec<(usize, Vec<f64>)> =
                    rows.iter().map(|&(r, w)| (r, round_to_f16(w))).collect();
                let rounded: Vec<(usize, &[f64])> =
                    rounded.iter().map(|(r, w)| (*r, w.as_slice())).collect();
                column_error(&rounded, &p.apply(xs), ys)
            }
            None => column_error(&rows, &p.apply(xs), ys),
        };
        let best = profiles
            .iter()
            .map(|&p| (error(p), p.name()))
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .expect("profiles");
        errors.push(best);
    }
    Ok(errors)
}

/// Row checks of every opened matrix against the claimed weights, on rows
/// drawn from `seed`, with the profiles of `backend`, the backend the
/// receipt names for the layer's stage.
pub fn check_rows(
    weights: &WeightFile,
    opening: &LayerOpening,
    seed: &[u8],
    backend: &str,
) -> Result<AuditReport> {
    let family = BackendFamily::checked(backend)?;
    // weight name -> per-column (error, profile), and whether it is a router
    let mut per_matrix: BTreeMap<String, (bool, ColumnErrors)> = BTreeMap::new();
    let mut cache = RowCache {
        pairs: pairs::picked_pairs(opening, seed),
        ..RowCache::default()
    };
    prefetch_rows(weights, &mut cache, opening, seed)?;
    for leaf in &opening.leaves {
        for m in &leaf.matrices {
            let errors = matmul_errors(weights, &mut cache, leaf.position, m, seed, family)?;
            let entry = per_matrix.entry(m.weight.clone()).or_default();
            entry.0 |= m.router;
            entry.1.extend(errors);
        }
    }
    let matrices: Vec<MatrixVerdict> = per_matrix
        .into_iter()
        .map(|(weight, (router, errors))| verdict(weights, weight, router, &errors))
        .collect::<Result<_>>()?;
    if matrices.is_empty() {
        bail!("the opening has no matrices to check");
    }
    let pass = matrices.iter().all(|m| m.pass);
    Ok(AuditReport {
        layer: opening.layer,
        tokens: opening.leaves.len(),
        matrices,
        pass,
    })
}

fn verdict(
    weights: &WeightFile,
    weight: String,
    router: bool,
    errors: &[(f64, &'static str)],
) -> Result<MatrixVerdict> {
    let failing = errors.iter().filter(|(e, _)| *e > TOLERANCE).count();
    let (max_error, _) = errors
        .iter()
        .copied()
        .fold((0.0, ""), |acc, e| if e.0 > acc.0 { e } else { acc });
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for (_, p) in errors {
        *counts.entry(p).or_default() += 1;
    }
    let best_profile = counts
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map(|(p, _)| p)
        .unwrap_or("raw");
    let weight_type = crate::weights::type_name(weights.tensor(&weight)?.ggml_type);
    Ok(MatrixVerdict {
        router,
        tokens: errors.len(),
        failing_tokens: failing,
        max_error,
        best_profile,
        // a router steers the selection, so no column of it may fail
        pass: if router {
            failing == 0
        } else {
            failing <= allowed_failing_columns(errors.len())
        },
        weight,
        weight_type,
    })
}

/// One weight role (`attn_q.weight`, `ffn_down_exps.weight`, ...) across
/// every opened layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoleVerdict {
    pub role: String,
    pub layers: Vec<u32>,
    pub router: bool,
    pub tokens: usize,
    pub failing_tokens: usize,
    pub pass: bool,
}

/// The fail rule over a whole challenge. A challenge opens a few positions
/// in each of several layers, too few per matrix for
/// [`allowed_failing_columns`] to bite: with four columns, two failures
/// would pass. So each weight role is judged on its columns in every opened
/// layer together, which gives the rule as many columns as one layer of 16
/// positions had.
pub fn pool_roles(reports: &[AuditReport]) -> Vec<RoleVerdict> {
    let mut roles: BTreeMap<String, RoleVerdict> = BTreeMap::new();
    for report in reports {
        for m in &report.matrices {
            let role = weight_role(&m.weight).to_owned();
            let entry = roles.entry(role.clone()).or_insert_with(|| RoleVerdict {
                role,
                layers: Vec::new(),
                router: false,
                tokens: 0,
                failing_tokens: 0,
                pass: true,
            });
            entry.layers.push(report.layer);
            entry.router |= m.router;
            entry.tokens += m.tokens;
            entry.failing_tokens += m.failing_tokens;
        }
    }
    roles
        .into_values()
        .map(|mut role| {
            role.pass = if role.router {
                role.failing_tokens == 0
            } else {
                role.failing_tokens <= allowed_failing_columns(role.tokens)
            };
            role
        })
        .collect()
}

/// A weight's name without its `blk.N.` prefix.
fn weight_role(weight: &str) -> &str {
    weight
        .strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
        .filter(|(layer, _)| layer.bytes().all(|b| b.is_ascii_digit()))
        .map_or(weight, |(_, role)| role)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verdict_of(weight: &str, tokens: usize, failing: usize) -> MatrixVerdict {
        MatrixVerdict {
            weight: weight.into(),
            weight_type: "q8_0".into(),
            router: weight.contains("gate_inp"),
            tokens,
            failing_tokens: failing,
            max_error: 0.0,
            best_profile: "raw",
            pass: failing <= allowed_failing_columns(tokens),
        }
    }

    fn report(layer: u32, matrices: Vec<MatrixVerdict>) -> AuditReport {
        AuditReport {
            layer,
            tokens: 4,
            pass: matrices.iter().all(|m| m.pass),
            matrices,
        }
    }

    #[test]
    fn a_role_is_judged_on_every_opened_layer() {
        // cheating on half the positions: two of four columns fail in each
        // layer, which each layer alone would let pass
        let reports: Vec<AuditReport> = (0..4)
            .map(|l| {
                report(
                    l,
                    vec![
                        verdict_of(&format!("blk.{l}.attn_q.weight"), 4, 2),
                        verdict_of(&format!("blk.{l}.ffn_up.weight"), 4, 0),
                    ],
                )
            })
            .collect();
        assert!(reports.iter().all(|r| r.pass));
        let roles = pool_roles(&reports);
        assert_eq!(roles.len(), 2);
        assert_eq!(roles[0].role, "attn_q.weight");
        assert_eq!((roles[0].tokens, roles[0].failing_tokens), (16, 8));
        assert!(!roles[0].pass);
        assert_eq!(roles[0].layers, vec![0, 1, 2, 3]);
        assert!(roles[1].pass);
    }

    #[test]
    fn honest_outliers_spread_over_layers_still_pass() {
        let reports = [
            report(3, vec![verdict_of("blk.3.attn_k.weight", 4, 1)]),
            report(9, vec![verdict_of("blk.9.attn_k.weight", 4, 1)]),
            report(20, vec![verdict_of("blk.20.ffn_gate_inp.weight", 4, 1)]),
        ];
        let roles = pool_roles(&reports);
        assert!(roles[0].pass, "{roles:?}");
        // a router may not fail a single column
        assert!(roles[1].router && !roles[1].pass);
    }

    #[test]
    fn roles_drop_only_a_layer_prefix() {
        assert_eq!(
            weight_role("blk.12.ffn_down_exps.weight"),
            "ffn_down_exps.weight"
        );
        assert_eq!(weight_role("output.weight"), "output.weight");
        assert_eq!(weight_role("blk.x.attn_q.weight"), "blk.x.attn_q.weight");
    }

    #[test]
    fn a_matrix_fails_on_a_few_bad_columns() {
        // decode tokens are often a small share of a turn's positions; a
        // node that cheats on four of 16 must fail
        assert!(4 > allowed_failing_columns(16));
        assert_eq!(allowed_failing_columns(16), 2);
        assert_eq!(allowed_failing_columns(128), 6);
    }

    #[test]
    fn exact_columns_have_zero_error() {
        let (a, b) = ([1.0, 2.0], [-1.0, 0.5]);
        let rows = vec![(0, &a[..]), (1, &b[..])];
        let x = [0.5f32, 0.25];
        let y = [1.0f32, -0.375];
        assert_eq!(column_error(&rows, &Profile::Raw.apply(&x), &y), 0.0);
    }

    #[test]
    fn a_wrong_output_shows_up() {
        let a = [1.0, 2.0];
        let rows = vec![(0, &a[..])];
        let y = [1.01f32, 5.0];
        assert!(column_error(&rows, &Profile::Raw.apply(&[0.5, 0.25]), &y) > TOLERANCE);
    }

    #[test]
    fn f16_weights_rounds_the_weights_too() {
        // values f16 cannot hold, as a q8_0 scale times a quant gives them
        let w: Vec<f64> = (0..64).map(|i| 0.0123 * f64::from(i - 31) + 1e-4).collect();
        let x: Vec<f32> = (0..64).map(|i| 0.37 - 0.011 * i as f32).collect();
        let xh = Profile::F16Weights.apply(&x);
        let kernel = round_to_f16(&w)
            .iter()
            .zip(&xh)
            .map(|(a, b)| a * b)
            .sum::<f64>() as f32;
        let rows = [(0, w.as_slice())];
        let rounded = round_to_f16(&w);
        let rounded = [(0, rounded.as_slice())];
        assert!(column_error(&rounded, &xh, &[kernel]) < 1e-7);
        let e = column_error(&rows, &Profile::F16.apply(&x), &[kernel]);
        assert!(e > 1e-6, "{e}");
    }

    #[test]
    fn each_backend_names_its_own_profiles() {
        use crate::dequant::{GGML_F16, GGML_F32, GGML_Q8_0};
        assert_eq!(BackendFamily::of("CPU"), Some(BackendFamily::Cpu));
        assert_eq!(BackendFamily::of("CUDA0"), Some(BackendFamily::Cuda));
        assert_eq!(BackendFamily::of("Vulkan0"), Some(BackendFamily::Vulkan));
        assert_eq!(BackendFamily::of("MTL0"), None);
        assert_eq!(
            profiles_for(BackendFamily::Vulkan, GGML_Q8_0),
            Some(&[Profile::Raw, Profile::F16Weights, Profile::CpuQ8][..])
        );
        assert_eq!(
            profiles_for(BackendFamily::Cpu, GGML_Q8_0),
            Some(&[Profile::CpuQ8][..])
        );
        // CUDA's q8 kernels, never the CPU one
        let cuda = profiles_for(BackendFamily::Cuda, GGML_Q8_0).unwrap();
        assert!(!cuda.contains(&Profile::CpuQ8) && !cuda.contains(&Profile::Raw));
        assert_eq!(
            profiles_for(BackendFamily::Cpu, GGML_F32),
            Some(&[Profile::Raw][..])
        );
        assert_eq!(
            profiles_for(BackendFamily::Cpu, GGML_F16),
            Some(&[Profile::F16][..])
        );
        // the CPU quantizes K-quant activations per 256, CUDA per 32 or 64
        assert_eq!(
            profiles_for(BackendFamily::Cpu, crate::dequant::GGML_Q4_K),
            Some(&[Profile::CpuQ8K][..])
        );
        assert!(
            profiles_for(BackendFamily::Cuda, crate::dequant::GGML_Q2_K)
                .unwrap()
                .contains(&Profile::Q8Mul64F32)
        );
        // types without a reference dequantization or a known profile have none
        assert_eq!(profiles_for(BackendFamily::Cpu, 19), None);
        assert_eq!(
            profiles_for(BackendFamily::Cuda, crate::dequant::GGML_Q4_1),
            None
        );
    }

    #[test]
    fn the_cpu_k_quant_profile_scales_by_the_largest_value() {
        let mut x = vec![0.5f32; 256];
        x[7] = -2.0;
        let y = Profile::CpuQ8K.apply(&x);
        // d = 2/127; the largest value is exact and 0.5 is 31.75 steps
        assert!((y[7] + 2.0).abs() < 1e-6);
        assert!((y[0] - 32.0 * 2.0 / 127.0).abs() < 1e-6);
        assert_eq!(Profile::CpuQ8K.apply(&[0.0; 256]), vec![0.0; 256]);
        // MMQ blocks of 64 share one scale
        let mut x = vec![1.0f32; 128];
        x[0] = 4.0;
        let y = Profile::Q8Mul64F32.apply(&x);
        assert!((y[63] - 32.0 * 4.0 / 127.0).abs() < 1e-6);
        assert!((y[64] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn q8_profiles_round_to_the_block_grid() {
        let x: Vec<f32> = (0..32).map(|i| i as f32 / 31.0).collect();
        for p in [Profile::Q8DivHalf, Profile::Q8MulHalf, Profile::CpuQ8] {
            let q = p.apply(&x);
            assert!((q[31] - 1.0).abs() < 1e-3, "{:?}", p);
            assert!(
                q.iter()
                    .zip(&x)
                    .all(|(a, b)| (a - f64::from(*b)).abs() < 5e-3)
            );
        }
    }

    fn tensor(dims: Vec<u64>) -> TensorInfo {
        TensorInfo {
            name: "w".into(),
            dims,
            ggml_type: 0,
            offset: 0,
        }
    }

    fn matrix(experts: Option<Vec<i32>>) -> MatrixOpening {
        MatrixOpening {
            index: 0,
            weight: "w".into(),
            router: false,
            experts,
            input: String::new(),
            output: String::new(),
        }
    }

    #[test]
    fn expert_slices_start_at_their_expert() {
        let w = tensor(vec![4, 3, 8]);
        let s = matmul_slices(&w, &matrix(Some(vec![5, 2])), 4, 6).unwrap();
        assert_eq!(s.slices, vec![(15, 0), (6, 0)]);
        let s = matmul_slices(&w, &matrix(Some(vec![5, 2])), 8, 6).unwrap();
        assert_eq!(s.slices, vec![(15, 0), (6, 1)]);
        assert!(matmul_slices(&w, &matrix(Some(vec![9, 2])), 4, 6).is_err());
        assert!(matmul_slices(&w, &matrix(Some(vec![1])), 4, 6).is_err());
    }

    #[test]
    fn a_shared_weight_meets_every_batch() {
        let s = matmul_slices(&tensor(vec![4, 3]), &matrix(None), 8, 6).unwrap();
        assert_eq!(s.slices, vec![(0, 0), (0, 1)]);
        let s = matmul_slices(&tensor(vec![4, 3, 2]), &matrix(None), 8, 6).unwrap();
        assert_eq!(s.slices, vec![(0, 0), (3, 1)]);
        assert!(matmul_slices(&tensor(vec![4, 3]), &matrix(None), 5, 3).is_err());
    }
}
