//! Client-side checks that bind an opening's matmuls to the committed
//! residual stream.
//!
//! A replay recomputes the challenged layer, so its matmul columns pass the
//! row checks no matter which weights served the turn. These checks tie the
//! opened columns to the committed residual stream instead. For a pre-norm
//! transformer layer with an RMS norm and a SwiGLU FFN:
//!
//! ```text
//! attn_in      = rms_norm(residual_in) · attn_norm     (input of attn_q, attn_k, attn_v)
//! mid          = residual_in + attn_output(attn_out)
//! ffn_in       = rms_norm(mid) · ffn_norm              (input of ffn_gate, ffn_up)
//! down_in      = silu(ffn_gate) · ffn_up               (input of ffn_down)
//! residual_out = mid + ffn_down
//! ```
//!
//! The attention leaf commits to the attn_q, attn_k and attn_v outputs and to
//! the attn_output input as served, so the row checks of those matmuls test
//! the served values. Attention itself is not checked.

use anyhow::{Context, Result, bail};

use crate::weights::WeightFile;
use crate::wire::{LeafOpening, Opening};

/// Architectures whose layers follow the glue equations above.
const ARCHITECTURES: [&str; 3] = ["llama", "qwen2", "qwen3"];
/// A glue check fails above this error, relative to the largest value of
/// the reference vector.
pub const GLUE_TOLERANCE: f64 = 1e-4;

#[derive(Clone, Debug)]
pub struct GlueVerdict {
    pub check: &'static str,
    pub tokens: usize,
    pub failing_tokens: usize,
    pub max_error: f64,
    pub pass: bool,
}

struct LayerNorms {
    attn: Vec<f64>,
    ffn: Vec<f64>,
    eps: f64,
}

impl LayerNorms {
    fn load(weights: &WeightFile, layer: u32) -> Result<Self> {
        let arch = weights.metadata_str("general.architecture")?;
        if !ARCHITECTURES.contains(&arch) {
            bail!("no glue checks for architecture {arch} yet");
        }
        Ok(Self {
            attn: weights.vector(&format!("blk.{layer}.attn_norm.weight"))?,
            ffn: weights.vector(&format!("blk.{layer}.ffn_norm.weight"))?,
            eps: weights.metadata_f64(&format!("{arch}.attention.layer_norm_rms_epsilon"))?,
        })
    }
}

fn rms_norm(x: &[f32], weight: &[f64], eps: f64) -> Result<Vec<f64>> {
    if x.len() != weight.len() {
        bail!(
            "a row of {} values meets a norm of {}",
            x.len(),
            weight.len()
        );
    }
    let mean = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len() as f64;
    let scale = 1.0 / (mean + eps).sqrt();
    Ok(x.iter()
        .zip(weight)
        .map(|(&v, w)| f64::from(v) * scale * w)
        .collect())
}

/// Largest difference, relative to the largest reference value.
fn relative_error(actual: &[f32], reference: &[f64]) -> f64 {
    if actual.len() != reference.len() {
        return f64::INFINITY;
    }
    let scale = reference.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let diff = actual
        .iter()
        .zip(reference)
        .fold(0.0f64, |m, (&a, r)| m.max((f64::from(a) - r).abs()));
    if scale == 0.0 {
        return if diff == 0.0 { 0.0 } else { f64::INFINITY };
    }
    diff / scale
}

fn add_f32(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        bail!("adding rows of {} and {} values", a.len(), b.len());
    }
    Ok(a.iter().zip(b).map(|(x, y)| x + y).collect())
}

fn widen(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
}

/// Glue errors of one position, by check name. FFN checks are skipped
/// where the layer kept no row for the position.
fn position_errors(
    leaf: &LeafOpening,
    layer: u32,
    norms: &LayerNorms,
) -> Result<Vec<(&'static str, f64)>> {
    let matrix = |role: &str| {
        leaf.matrix(layer, role)
            .with_context(|| format!("position {}: no {role} matmul opened", leaf.position))
    };
    let residual_in = leaf.residual_in_values()?;
    let attn_in = matrix("attn_q")?.input_values()?;
    let mut errors = Vec::new();
    let shared = matrix("attn_k")?.input_values()? == attn_in
        && matrix("attn_v")?.input_values()? == attn_in;
    let norm_error = relative_error(&attn_in, &rms_norm(&residual_in, &norms.attn, norms.eps)?);
    errors.push((
        "attn_in = rms_norm(residual_in)",
        if shared { norm_error } else { f64::INFINITY },
    ));

    let Some(residual_out) = leaf.residual_out_values()? else {
        return Ok(errors);
    };
    let mid = add_f32(&residual_in, &matrix("attn_output")?.output_values()?)?;
    let ffn_in = matrix("ffn_gate")?.input_values()?;
    let shared = matrix("ffn_up")?.input_values()? == ffn_in;
    let norm_error = relative_error(&ffn_in, &rms_norm(&mid, &norms.ffn, norms.eps)?);
    errors.push((
        "ffn_in = rms_norm(mid)",
        if shared { norm_error } else { f64::INFINITY },
    ));

    let gate = matrix("ffn_gate")?.output_values()?;
    let up = matrix("ffn_up")?.output_values()?;
    if gate.len() != up.len() {
        bail!(
            "position {}: ffn_gate and ffn_up disagree in length",
            leaf.position
        );
    }
    let swiglu: Vec<f64> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| {
            let g = f64::from(g);
            g / (1.0 + (-g).exp()) * f64::from(u)
        })
        .collect();
    let down = matrix("ffn_down")?;
    errors.push((
        "down_in = silu(gate) · up",
        relative_error(&down.input_values()?, &swiglu),
    ));
    let expected = add_f32(&mid, &down.output_values()?)?;
    errors.push((
        "residual_out = mid + down",
        relative_error(&residual_out, &widen(&expected)),
    ));
    Ok(errors)
}

/// Checks every opened position against the glue equations.
pub fn check_glue(weights: &WeightFile, opening: &Opening) -> Result<Vec<GlueVerdict>> {
    let norms = LayerNorms::load(weights, opening.layer)?;
    let mut verdicts: Vec<GlueVerdict> = Vec::new();
    for leaf in &opening.leaves {
        for (check, error) in position_errors(leaf, opening.layer, &norms)? {
            let verdict = match verdicts.iter_mut().find(|v| v.check == check) {
                Some(verdict) => verdict,
                None => {
                    verdicts.push(GlueVerdict {
                        check,
                        tokens: 0,
                        failing_tokens: 0,
                        max_error: 0.0,
                        pass: true,
                    });
                    verdicts.last_mut().expect("pushed")
                }
            };
            verdict.tokens += 1;
            verdict.max_error = verdict.max_error.max(error);
            if error > GLUE_TOLERANCE {
                verdict.failing_tokens += 1;
                verdict.pass = false;
            }
        }
    }
    if verdicts.is_empty() {
        bail!("the opening has no positions to check");
    }
    Ok(verdicts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_scales_to_unit_mean_square() {
        let out = rms_norm(&[3.0, 4.0], &[1.0, 2.0], 0.0).unwrap();
        let rms = (12.5f64).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-12);
        assert!((out[1] - 8.0 / rms).abs() < 1e-12);
    }

    #[test]
    fn relative_error_is_scaled_by_the_largest_reference() {
        assert_eq!(relative_error(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        assert!((relative_error(&[1.0, 2.5], &[1.0, 2.0]) - 0.25).abs() < 1e-12);
        assert!(relative_error(&[1.0], &[1.0, 2.0]).is_infinite());
    }
}
