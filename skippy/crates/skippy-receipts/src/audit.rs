//! Client-side checks of an opening: Merkle paths against the receipt, then
//! weight rows against the opened matmul columns: random rows for most
//! matmuls, every row for a router.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use rand::RngExt;

use crate::commitment::{
    leaf_index, mixing_leaf, mixing_slot, residual_leaf, residual_slot, root_from_path,
    slots_per_position,
};
use crate::layer_tensors::f32_bytes;
use crate::weights::{TensorInfo, WeightFile};
use crate::wire::{
    Challenge, MatrixOpening, MixingKind, Opening, Receipt, mixing_bytes, mixing_values,
};

/// A column check fails above this relative error (see RECEIPTS.md).
pub const TOLERANCE: f64 = 1e-3;
pub const ROWS_PER_MATRIX: usize = 16;

/// How a backend rounds the f32 activation before it meets the weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Raw,
    F16,
    /// CUDA `quantize_q8_1` (MMVQ): d = amax/127, q = round(x/d), d stored as f16.
    Q8DivHalf,
    /// CUDA `quantize_mmq_q8_1` (MMQ): q = round(x·127/amax), d stored as f16.
    Q8MulHalf,
    /// CPU `quantize_row_q8_0`: d = amax/127, q = round(x·(1/d)), d stored as f16.
    CpuQ8,
}

pub const PROFILES: [Profile; 5] = [
    Profile::Raw,
    Profile::F16,
    Profile::Q8DivHalf,
    Profile::Q8MulHalf,
    Profile::CpuQ8,
];

impl Profile {
    pub fn name(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::F16 => "f16",
            Self::Q8DivHalf => "q8_32_div_dh",
            Self::Q8MulHalf => "q8_32_mul_dh",
            Self::CpuQ8 => "cpu_q8_0",
        }
    }

    pub fn apply(self, x: &[f32]) -> Vec<f64> {
        match self {
            Self::Raw => x.iter().map(|&v| f64::from(v)).collect(),
            Self::F16 => x
                .iter()
                .map(|&v| f64::from(half::f16::from_f32(v).to_f32()))
                .collect(),
            Self::Q8DivHalf | Self::Q8MulHalf | Self::CpuQ8 => quantize_q8_blocks(self, x),
        }
    }
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

/// Relative error of the checked rows (RECEIPTS.md, "Fail rule"):
/// `||y[R] - y_ref[R]|| / (||y|| · sqrt(|R| / M))`.
pub fn column_error(rows: &[(usize, &[f64])], x: &[f64], y: &[f32]) -> f64 {
    let norm_y = y
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt();
    let diff2: f64 = rows
        .iter()
        .map(|(m, w)| {
            let reference: f64 = w.iter().zip(x).map(|(a, b)| a * b).sum();
            (f64::from(y[*m]) - reference).powi(2)
        })
        .sum();
    let scale = norm_y * (rows.len() as f64 / y.len() as f64).sqrt();
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
    pub stage_index: u32,
    pub tokens: usize,
    pub matrices: Vec<MatrixVerdict>,
    pub pass: bool,
}

/// Checks that every opened leaf hashes to the stage root in the receipt:
/// the residual stream entering and leaving the layer, and the mixing leaf.
/// Stored rows must be in a cache type the receipt names.
pub fn verify_paths(receipt: &Receipt, challenge: &Challenge, opening: &Opening) -> Result<()> {
    let layer = challenge.layer;
    let stage = receipt
        .stage_for_layer(layer)
        .context("receipt has no stage for the challenged layer")?;
    if opening.layer != layer || opening.stage_index != stage.stage_index {
        bail!("opening answers a different layer or stage");
    }
    let opened: Vec<u32> = opening.leaves.iter().map(|l| l.position).collect();
    if opened != challenge.positions {
        bail!(
            "opening covers positions {opened:?}, challenge asked {:?}",
            challenge.positions
        );
    }
    let root: [u8; 32] = hex::decode(&stage.root)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("stage root is not 32 bytes"))?;
    let slots = slots_per_position(stage.layer_start, stage.layer_end);
    for leaf in &opening.leaves {
        let position = leaf.position;
        if position < receipt.position_start || position >= receipt.position_end {
            bail!("position {position} is outside the receipt");
        }
        for stored in leaf.mixing.iter().filter(|t| t.kind == MixingKind::Stored) {
            let name = crate::weights::type_name(stored.ggml_type as u32);
            if !stage.cache_types.contains(&name) {
                bail!(
                    "position {position}: a stored row is {name}, which the receipt does not name"
                );
            }
        }
        let residual_in = f32_bytes(&leaf.residual_in_values()?);
        let residual_out = leaf.residual_out_values()?.map(|v| f32_bytes(&v));
        let decoded = mixing_bytes(&leaf.mixing)?;
        let checks = [
            (
                "residual-in",
                residual_slot(layer, stage.layer_start),
                &leaf.residual_in_path,
                residual_leaf(position, layer, Some(&residual_in)),
            ),
            (
                "mixing",
                mixing_slot(layer, stage.layer_start),
                &leaf.mixing_path,
                mixing_leaf(position, layer, &mixing_values(&decoded)),
            ),
            (
                "residual-out",
                residual_slot(layer + 1, stage.layer_start),
                &leaf.residual_out_path,
                residual_leaf(position, layer + 1, residual_out.as_deref()),
            ),
        ];
        for (name, slot, path, hash) in checks {
            let index = leaf_index(position, slot, receipt.position_start, slots);
            if root_from_path(hash, index, path)? != root {
                bail!(
                    "position {position}: the opened {name} values do not match the committed root"
                );
            }
        }
    }
    Ok(())
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

/// Weight rows already read, by tensor and row.
#[derive(Default)]
struct RowCache {
    rows: BTreeMap<(String, u64), Vec<f64>>,
    /// Row indices checked per weight; every row for a router.
    picked: BTreeMap<String, Vec<usize>>,
}

impl RowCache {
    fn picked(
        &mut self,
        weight: &str,
        m: usize,
        all: bool,
        rng: &mut impl rand::Rng,
    ) -> Vec<usize> {
        self.picked
            .entry(weight.to_owned())
            .or_insert_with(|| {
                if all || m <= ROWS_PER_MATRIX {
                    return (0..m).collect();
                }
                let mut picked = std::collections::BTreeSet::new();
                while picked.len() < ROWS_PER_MATRIX {
                    picked.insert(rng.random_range(0..m));
                }
                picked.into_iter().collect()
            })
            .clone()
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
    m: &MatrixOpening,
    rng: &mut impl rand::Rng,
) -> Result<ColumnErrors> {
    let x = m.input_values()?;
    let y = m.output_values()?;
    let tensor = weights.tensor(&m.weight)?;
    let shape = matmul_slices(tensor, m, x.len(), y.len())?;
    let picked = cache.picked(&m.weight, shape.m, m.router, rng);
    let mut errors = Vec::with_capacity(shape.slices.len());
    for (g, &(base, input)) in shape.slices.iter().enumerate() {
        for &r in &picked {
            cache.load(weights, &m.weight, base + r as u64)?;
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
        let best = PROFILES
            .iter()
            .map(|p| (column_error(&rows, &p.apply(xs), ys), p.name()))
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .expect("profiles");
        errors.push(best);
    }
    Ok(errors)
}

/// Row checks of every opened matrix against the claimed weights.
pub fn check_rows(
    weights: &WeightFile,
    opening: &Opening,
    rng: &mut impl rand::Rng,
) -> Result<AuditReport> {
    // weight name -> per-column (error, profile), and whether it is a router
    let mut per_matrix: BTreeMap<String, (bool, ColumnErrors)> = BTreeMap::new();
    let mut cache = RowCache::default();
    for leaf in &opening.leaves {
        for m in &leaf.matrices {
            let errors = matmul_errors(weights, &mut cache, m, rng)?;
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
        stage_index: opening.stage_index,
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
            failing * 2 <= errors.len()
        },
        weight,
        weight_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
