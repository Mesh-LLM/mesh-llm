//! Client-side row checks of an opening: weight rows of the claimed GGUF
//! against the opened matmul columns, random rows for most matmuls and
//! every row for a router. The Merkle paths are checked by
//! [`crate::binding`].
//!
//! The rows come from the VRF output of the opening's hash under the
//! client's key, drawn per weight, so the node cannot know them when it
//! opens and the client cannot pick them.

use std::collections::BTreeMap;

use anyhow::{Context, Result};

use crate::activation_rounding::Profile;
use crate::draw::DrawStream;
use crate::error_bound::{all_finite, exceeds, worst_of};
use crate::glue::PlannedMatmul;
use crate::node_fault::{BlameNode, NodeFault, ensure_node, node_fault};
use crate::weights::{TensorInfo, WeightFile};
use crate::wire::{LayerOpening, MatrixOpening};
use column_check::{CheckedRows, ColumnCheck};

mod block_sums;
mod column_check;
mod near_ties;
pub(crate) mod pairs;
pub(crate) mod rounding_flips;

/// A column check fails above this relative error (see RECEIPTS.md).
pub(crate) const TOLERANCE: f64 = 1e-3;
pub(crate) const ROWS_PER_MATRIX: usize = 16;

/// Failing columns a matrix may have and still pass, for honest outliers
/// that the near-tie repair does not explain. Before the repair, honest
/// backends failed at most 1 in 300 column checks, so an honest matrix of
/// 16 columns had three failures about 1 in 47,000 audits; with it, at
/// most 1 in 8,350 failed, and three failures come about 1 in a billion
/// audits. A majority rule
/// would let a node cheat on half the positions, for example on every
/// decode token of a long prompt.
pub(crate) fn allowed_failing_columns(columns: usize) -> usize {
    2 + columns / 32
}

/// The backends with measured rounding profiles. Metal has none: through
/// `skippy serve`, its prefill outputs differ from every profile by up to
/// 8e-4, which would fail honest nodes (RECEIPTS_MEASUREMENTS.md, "Detection").
///
/// Vulkan has none either. Its integer-dot kernels round K-quant weights
/// in a way no profile models, so an honest node with Q4_K_M weights on
/// an integrated GPU passed 2 of 16 audits, and Q8_0 weights there passed
/// with 1% of margin (RECEIPTS_MEASUREMENTS.md, "Vulkan on the final
/// build"). A stage on a
/// backend without a family issues no receipts, and a receipt that names
/// one is not audited ([`crate::coverage`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendFamily {
    Cpu,
    Cuda,
}

impl BackendFamily {
    /// The family of a receipt's backend name, such as `CPU` or `CUDA0`.
    pub fn of(backend: &str) -> Option<Self> {
        if backend.eq_ignore_ascii_case("cpu") {
            Some(Self::Cpu)
        } else if backend.starts_with("CUDA") {
            Some(Self::Cuda)
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
/// Q4_1 and Q5_1 have none: their kernels round the product of the weight
/// and activation scales, and the activation block sum, which a profile of
/// the activation alone cannot reproduce (RECEIPTS_MEASUREMENTS.md, "Detection").
pub(crate) fn profiles_for(family: BackendFamily, weight_type: u32) -> Option<&'static [Profile]> {
    use crate::ggml_types::{
        GGML_BF16, GGML_F16, GGML_F32, GGML_IQ4_NL, GGML_IQ4_XS, GGML_Q2_K, GGML_Q3_K, GGML_Q4_0,
        GGML_Q4_K, GGML_Q5_0, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0,
    };
    use Profile::{
        Bf16, CpuQ8, CpuQ8K, F16, Q8DivHalf, Q8Mul64F32, Q8MulF32, Q8MulHalf, Q8MulHalfSums,
        Q8MulHalfSumsDp4a, Raw,
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
        _ => return None,
    };
    Some(profiles)
}

/// [`profiles_for`], failing for a pair without profiles.
pub(crate) fn checked_profiles(
    family: BackendFamily,
    weight_type: u32,
) -> Result<&'static [Profile]> {
    profiles_for(family, weight_type).with_context(|| {
        format!(
            "no rounding profile for {} weights on {family:?}",
            crate::ggml_types::type_name(weight_type)
        )
    })
}

/// Relative error of the checked rows (RECEIPTS.md, "Fail rule"):
/// `||y[R] - y_ref[R]|| / (||y|| · sqrt(|R| / M))`.
pub(crate) fn column_error(rows: &[(usize, &[f64])], x: &[f64], y: &[f32]) -> f64 {
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
    /// Columns that pass only once a repair rounds near-ties of their
    /// activation the other way ([`near_ties`]).
    pub repaired_tokens: usize,
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

/// Checks that an opened matmul of `x` input and `y` output values fits
/// the claimed weight `tensor`: whole rows and columns of it, one slice per
/// expert named, and every expert one the weight holds.
///
/// An error is the node's fault: the weight's shape comes from the claimed
/// GGUF's header.
pub(crate) fn check_matmul_shape(
    tensor: &TensorInfo,
    m: &MatrixOpening,
    x: usize,
    y: usize,
) -> Result<()> {
    matmul_slices(tensor, m, x, y).map(|_| ())
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
        NodeFault::error(format!(
            "{}: opened input of {x} and output of {y} values do not fit the claimed weight {:?}",
            m.weight, tensor.dims
        ))
    };
    // an empty column would pass every comparison
    if k == 0 || rows == 0 || x == 0 || y == 0 || !x.is_multiple_of(k) || !y.is_multiple_of(rows) {
        return Err(shape());
    }
    let (groups_in, groups_out) = (x / k, y / rows);
    let slices = match &m.experts {
        Some(experts) => {
            if experts.len() != groups_out || (groups_in != 1 && groups_in != groups_out) {
                return Err(shape());
            }
            experts
                .iter()
                .enumerate()
                .map(|(g, &e)| {
                    let Some(e) = u64::try_from(e).ok().filter(|&e| e < matrices) else {
                        node_fault!("{}: expert {e} outside {matrices} experts", m.weight);
                    };
                    Ok((e * rows as u64, if groups_in == 1 { 0 } else { g }))
                })
                .collect::<Result<_>>()?
        }
        None => {
            if groups_in != groups_out {
                return Err(shape());
            }
            match matrices {
                1 => (0..groups_out).map(|g| (0, g)).collect(),
                n if n == groups_out as u64 => (0..groups_out)
                    .map(|g| (g as u64 * rows as u64, g))
                    .collect(),
                _ => return Err(shape()),
            }
        }
    };
    Ok(MatmulSlices { k, m: rows, slices })
}

/// What the row check knows of a layer's matmuls.
#[derive(Clone, Copy, Debug)]
pub enum Routers<'a> {
    /// The layer as the auditor's own evaluation has it: which weight each
    /// matmul reads, and which are routers.
    Planned(&'a MatmulPlan),
    /// The auditor could not evaluate the layer. No matmul is taken for a
    /// router, so each is checked on a sample of its rows and none on all
    /// of them: the opening may not decide what the client reads. A column
    /// that fails is still a failure; a layer that passes this way was not
    /// held to the weights its matmuls read.
    Unknown,
}

impl Routers<'_> {
    /// Whether the opened matmul `m` is a router. With a plan, an opening
    /// that names another weight for the index, or an index the layer does
    /// not have, is the node's fault. The router flag the node sent is
    /// never read.
    fn is_router(self, position: u32, m: &MatrixOpening) -> Result<bool> {
        let Self::Planned(plan) = self else {
            return Ok(false);
        };
        let Some(planned) = plan.get(&m.index) else {
            node_fault!(
                "position {position}: the layer has no matmul {} ({})",
                m.index,
                m.weight
            );
        };
        ensure_node!(
            planned.weight == m.weight,
            "position {position}: matmul {} reads {}, opened as {}",
            m.index,
            planned.weight,
            m.weight
        );
        Ok(planned.router)
    }

    /// Whether a layer with these matmuls may open none.
    fn allows_none(self) -> bool {
        match self {
            Self::Planned(plan) => plan.is_empty(),
            Self::Unknown => true,
        }
    }
}

/// The claimed weight an opened matmul names. A name the GGUF does not
/// have is the node's fault.
fn named_weight<'a>(weights: &'a WeightFile, m: &MatrixOpening) -> Result<&'a TensorInfo> {
    let Some(tensor) = weights.tensors.get(&m.weight) else {
        node_fault!(
            "matmul {} names {:?}, a tensor the claimed GGUF does not have",
            m.index,
            m.weight
        );
    };
    Ok(tensor)
}

/// Fetches every row the checks of `opening` read in one round, so a
/// sparse copy of the claimed GGUF makes its requests at once and not one
/// per row.
fn prefetch_rows(
    weights: &WeightFile,
    cache: &mut RowCache,
    opening: &LayerOpening,
    routers: Routers<'_>,
    seed: &[u8],
) -> Result<()> {
    let mut ranges = Vec::new();
    let leaves = opening.leaves.iter();
    for (position, m) in leaves.flat_map(|leaf| leaf.matrices.iter().map(|m| (leaf.position, m))) {
        let router = routers.is_router(position, m)?;
        let tensor = named_weight(weights, m)?;
        let (inputs, outputs) = m.value_counts().or_node_fault()?;
        let shape = matmul_slices(tensor, m, inputs, outputs)?;
        let picked = cache.picked(&m.weight, shape.m, router, seed);
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

/// Column checks of every slice of one opened matmul column, each with
/// its best profile.
fn matmul_errors(
    weights: &WeightFile,
    cache: &mut RowCache,
    (position, m, router): (u32, &MatrixOpening, bool),
    seed: &[u8],
    family: BackendFamily,
) -> Result<Vec<ColumnCheck>> {
    let x = m.input_values().or_node_fault()?;
    let y = m.output_values().or_node_fault()?;
    ensure_node!(
        all_finite(&x) && all_finite(&y),
        "position {position}: {} is opened with a value that is not a number",
        m.weight
    );
    let tensor = named_weight(weights, m)?;
    let profiles = checked_profiles(family, tensor.ggml_type)?;
    let shape = matmul_slices(tensor, m, x.len(), y.len())?;
    let picked = cache.picked(&m.weight, shape.m, router, seed);
    let sums = profiles.iter().any(|p| p.block_sums().is_some());
    let mut checks = Vec::with_capacity(shape.slices.len());
    for g in cache.checked_slices(m, position, shape.slices.len()) {
        let (base, input) = shape.slices[g];
        for &r in &picked {
            cache.load(weights, &m.weight, base + r as u64)?;
            if sums {
                cache.load_blocks(weights, &m.weight, base + r as u64)?;
            }
        }
        let key = |r: usize| (m.weight.clone(), base + r as u64);
        let rows = CheckedRows {
            rows: picked
                .iter()
                .map(|&r| (r, cache.rows[&key(r)].as_slice()))
                .collect(),
            blocks: match sums {
                true => picked
                    .iter()
                    .map(|&r| cache.blocks[&key(r)].as_slice())
                    .collect(),
                false => Vec::new(),
            },
        };
        let xs = &x[input * shape.k..(input + 1) * shape.k];
        let ys = &y[g * shape.m..(g + 1) * shape.m];
        let best = profiles
            .iter()
            .map(|&p| rows.check(p, xs, ys))
            .min_by(|a, b| a.error.total_cmp(&b.error))
            .expect("profiles");
        checks.push(best);
    }
    Ok(checks)
}

/// A layer's weight matmuls by index, as the auditor's own graph has them
/// ([`crate::glue::GlueReport::matmuls`]).
pub(crate) type MatmulPlan = BTreeMap<u32, PlannedMatmul>;

/// Row checks of every opened matrix against the claimed weights, on rows
/// drawn from `seed`, with the profiles of `backend`, the backend the
/// receipt names for the layer's stage.
///
/// `routers` is the layer as the auditor built it, when it could: which
/// weight each matmul reads and which are routers. The opening's own
/// router flags are not read, since a router costs a read of every row.
/// Each matmul may be opened once per position, so a column counts once
/// toward [`allowed_failing_columns`].
///
/// An error that blames the node ([`crate::node_fault::NodeFault`]) is a
/// defect of the opening. Any other error is the auditor's: it could not
/// read the claimed weights.
pub fn check_rows(
    weights: &WeightFile,
    opening: &LayerOpening,
    routers: Routers<'_>,
    seed: &[u8],
    backend: &str,
) -> Result<AuditReport> {
    let family = BackendFamily::checked(backend)?;
    for leaf in &opening.leaves {
        let mut seen = std::collections::BTreeSet::new();
        if let Some(twice) = leaf.matrices.iter().find(|m| !seen.insert(m.index)) {
            node_fault!(
                "position {}: matmul {} ({}) is opened more than once",
                leaf.position,
                twice.index,
                twice.weight
            );
        }
    }
    // weight name -> per-column (error, profile), and whether it is a router
    let mut per_matrix: BTreeMap<String, (bool, Vec<ColumnCheck>)> = BTreeMap::new();
    let mut cache = RowCache {
        pairs: pairs::picked_pairs(opening, seed),
        ..RowCache::default()
    };
    prefetch_rows(weights, &mut cache, opening, routers, seed)?;
    for leaf in &opening.leaves {
        for m in &leaf.matrices {
            let router = routers.is_router(leaf.position, m)?;
            let column = (leaf.position, m, router);
            let checks = matmul_errors(weights, &mut cache, column, seed, family)?;
            let entry = per_matrix.entry(m.weight.clone()).or_default();
            entry.0 |= router;
            entry.1.extend(checks);
        }
    }
    let matrices: Vec<MatrixVerdict> = per_matrix
        .into_iter()
        .map(|(weight, (router, checks))| verdict(weights, weight, router, &checks))
        .collect::<Result<_>>()?;
    ensure_node!(
        !matrices.is_empty() || routers.allows_none(),
        "the opening has no matrices to check"
    );
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
    checks: &[ColumnCheck],
) -> Result<MatrixVerdict> {
    // an error that is not a number fails its column
    let failing = checks
        .iter()
        .filter(|c| exceeds(c.error, TOLERANCE))
        .count();
    let repaired = checks.iter().filter(|c| c.flips > 0).count();
    let max_error = worst_of(checks.iter().map(|c| c.error));
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for c in checks {
        *counts.entry(c.profile).or_default() += 1;
    }
    let best_profile = counts
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .map(|(p, _)| p)
        .unwrap_or("raw");
    let weight_type = crate::ggml_types::type_name(weights.tensor(&weight)?.ggml_type);
    Ok(MatrixVerdict {
        router,
        tokens: checks.len(),
        failing_tokens: failing,
        repaired_tokens: repaired,
        max_error,
        best_profile,
        // a router steers the selection, so no column of it may fail
        pass: if router {
            failing == 0
        } else {
            failing <= allowed_failing_columns(checks.len())
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
mod tests;
