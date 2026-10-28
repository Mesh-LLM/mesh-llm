//! Client-side checks of an opening: Merkle paths against the receipt, then
//! random weight rows against the opened matmul columns.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use rand::RngExt;

use crate::commitment::{
    attention_leaf, attention_slot, leaf_index, residual_leaf, residual_slot, root_from_path,
    slots_per_position,
};
use crate::weights::WeightFile;
use crate::wire::{Challenge, Opening, Receipt};

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
pub fn column_error(rows: &[(usize, Vec<f64>)], x: &[f64], y: &[f32]) -> f64 {
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
/// the residual stream entering and leaving the layer, and the attention
/// leaf (the attn_q, attn_k and attn_v outputs and the attn_output input).
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
        let output = |role: &str| -> Result<Vec<f32>> {
            leaf.matrix(layer, role)
                .with_context(|| format!("position {position}: no {role} matmul opened"))?
                .output_values()
        };
        let attn_out = leaf
            .matrix(layer, "attn_output")
            .with_context(|| format!("position {position}: no attn_output matmul opened"))?
            .input_values()?;
        let residual_in = leaf.residual_in_values()?;
        let residual_out = leaf.residual_out_values()?;
        let checks = [
            (
                "residual-in",
                residual_slot(layer, stage.layer_start),
                &leaf.residual_in_path,
                residual_leaf(position, layer, Some(&residual_in)),
            ),
            (
                "attention",
                attention_slot(layer, stage.layer_start),
                &leaf.attention_path,
                attention_leaf(
                    position,
                    layer,
                    &output("attn_q")?,
                    &output("attn_k")?,
                    &output("attn_v")?,
                    &attn_out,
                ),
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

/// Row checks of every opened matrix against the claimed weights.
pub fn check_rows(
    weights: &WeightFile,
    opening: &Opening,
    rng: &mut impl rand::Rng,
) -> Result<AuditReport> {
    // weight name -> per-token (error, profile)
    let mut per_matrix: BTreeMap<String, Vec<(f64, &'static str)>> = BTreeMap::new();
    let mut row_cache: BTreeMap<String, Vec<(usize, Vec<f64>)>> = BTreeMap::new();
    for leaf in &opening.leaves {
        for m in &leaf.matrices {
            let x = m.input_values()?;
            let y = m.output_values()?;
            let tensor = weights.tensor(&m.weight)?;
            if tensor.row_len() as usize != x.len() || tensor.rows() as usize != y.len() {
                bail!(
                    "{}: opened shape {}x{} does not match the claimed weight {}x{}",
                    m.weight,
                    y.len(),
                    x.len(),
                    tensor.rows(),
                    tensor.row_len()
                );
            }
            if !row_cache.contains_key(&m.weight) {
                let rows = sample_rows(weights, &m.weight, y.len(), rng)?;
                row_cache.insert(m.weight.clone(), rows);
            }
            let rows = &row_cache[&m.weight];
            let best = PROFILES
                .iter()
                .map(|p| (column_error(rows, &p.apply(&x), &y), p.name()))
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .expect("profiles");
            per_matrix.entry(m.weight.clone()).or_default().push(best);
        }
    }
    let matrices: Vec<MatrixVerdict> = per_matrix
        .into_iter()
        .map(|(weight, errors)| verdict(weights, weight, &errors))
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

fn sample_rows(
    weights: &WeightFile,
    name: &str,
    m: usize,
    rng: &mut impl rand::Rng,
) -> Result<Vec<(usize, Vec<f64>)>> {
    let mut picked = std::collections::BTreeSet::new();
    while picked.len() < ROWS_PER_MATRIX.min(m) {
        picked.insert(rng.random_range(0..m));
    }
    picked
        .into_iter()
        .map(|r| Ok((r, weights.row(name, r as u64)?)))
        .collect()
}

fn verdict(
    weights: &WeightFile,
    weight: String,
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
        tokens: errors.len(),
        failing_tokens: failing,
        max_error,
        best_profile,
        pass: failing * 2 <= errors.len(),
        weight,
        weight_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_columns_have_zero_error() {
        let rows = vec![(0, vec![1.0, 2.0]), (1, vec![-1.0, 0.5])];
        let x = [0.5f32, 0.25];
        let y = [1.0f32, -0.375];
        assert_eq!(column_error(&rows, &Profile::Raw.apply(&x), &y), 0.0);
    }

    #[test]
    fn a_wrong_output_shows_up() {
        let rows = vec![(0, vec![1.0, 2.0])];
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
}
