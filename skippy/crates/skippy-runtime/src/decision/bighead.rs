//! BigHead: the trained pointer-head decision backend ("System One" readout).
//!
//! Scores each offered option from its own final hidden state, against a query
//! projected from the hidden state at the `<answer>` position — the Strands
//! Decider pointer-head design, ported for serving in skippy. Unlike the causal
//! backend (logit readout, `relative_candidate_probability_not_calibrated_truth`),
//! a trained BigHead is a true probability measure: this is the backend that
//! fixes probability-measure coherence.
//!
//! Reference math (strands-labs/strands-decider, src/strands_decider/modeling.py,
//! PointerHead.forward):
//!
//! ```text
//! logits[k] = ( k_proj(LayerNorm(h_option[k])) . q_proj(LayerNorm(h_answer)) ) * dim^-0.5
//! probs = softmax(logits)
//! ```
//!
//! The head trains in fp32 even on a bf16 torso ("a low-precision classifier is a
//! needless source of calibration error"), so the readout computes in f32.
//!
//! ## Integration seam (what this module does NOT yet do)
//!
//! - Hidden states: `llama_get_embeddings_ith` is already bound in skippy-ffi, so
//!   per-token final hidden states are reachable from a prefill pass. The serving
//!   backend must render the BigHead prompt layout (state, question, options with
//!   the `<answer>` marker), prefill, and collect the option-last-token and
//!   `<answer>` hidden states.
//! - The LoRA adapter (~67 MB) that makes the torso decision-friendly is not yet
//!   applied at serve time: llama.cpp's lora-adapter API is not bound in
//!   skippy-ffi. Until it is, BigHead serving runs against the un-adapted torso
//!   (weaker — the A2 skew gate measures exactly this class of train/serve gap).
//! - Artifact layout: `head.safetensors` (norm weight/bias, q/k projections) plus
//!   the config below, discovered beside the model like `calibration.json`.
//!   Safetensors parsing is intentionally not in this module yet; the loader
//!   takes parsed tensors so both tests and the future integration stay honest.

use serde::{Deserialize, Serialize};

/// BigHead artifact schema (v1), discovered as `<model>.bighead.json` or
/// `bighead.json` beside the model, exactly like `calibration.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BigHeadConfig {
    #[serde(default)]
    pub schema_version: Option<String>,
    /// Torso hidden size the head was trained against.
    pub hidden_size: usize,
    /// Pointer projection dimension (q/k width).
    pub pointer_dim: usize,
    /// Per-question-kind temperatures, composed with `calibration.json` semantics:
    /// softmax(scores / T).
    #[serde(default)]
    pub kinds: std::collections::BTreeMap<String, f64>,
}

pub const BIGHEAD_SCHEMA_VERSION: &str = "1";

impl BigHeadConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if let Some(v) = &self.schema_version {
            anyhow::ensure!(
                v == BIGHEAD_SCHEMA_VERSION,
                "unsupported bighead schema {v:?}"
            );
        }
        anyhow::ensure!(self.hidden_size > 0, "hidden_size must be positive");
        anyhow::ensure!(self.pointer_dim > 0, "pointer_dim must be positive");
        Ok(())
    }

    /// Temperature for a question kind; 1.0 when absent.
    pub fn temperature(&self, kind: &str) -> f64 {
        self.kinds.get(kind).copied().unwrap_or(1.0)
    }
}

/// Parsed fp32 head weights: LayerNorm affine, q/k projections (row-major
/// `[out][in]`, torch Linear layout).
#[derive(Debug, Clone)]
pub struct BigHeadWeights {
    pub norm_weight: Vec<f32>,
    pub norm_bias: Vec<f32>,
    /// `[pointer_dim][hidden_size]`
    pub q: Vec<f32>,
    /// `[pointer_dim][hidden_size]`
    pub k: Vec<f32>,
}

impl BigHeadWeights {
    pub fn validate(&self, cfg: &BigHeadConfig) -> anyhow::Result<()> {
        let h = cfg.hidden_size;
        let d = cfg.pointer_dim;
        anyhow::ensure!(
            self.norm_weight.len() == h && self.norm_bias.len() == h,
            "norm params must have hidden_size {h} elements"
        );
        anyhow::ensure!(
            self.q.len() == d * h && self.k.len() == d * h,
            "q/k projections must be pointer_dim*hidden_size = {} elements",
            d * h
        );
        Ok(())
    }
}

fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / x.len() as f32;
    let std = (var + 1e-5).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| (v - mean) / std * weight[i] + bias[i])
        .collect()
}

fn matvec(linear: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(linear.len(), rows * cols);
    (0..rows)
        .map(|r| {
            linear[r * cols..(r + 1) * cols]
                .iter()
                .zip(x)
                .map(|(w, v)| w * v)
                .sum()
        })
        .collect()
}

/// Pointer logits for one question: `h_answer` is the hidden state at the
/// `<answer>` position; `h_options[k]` is the hidden state at the last token of
/// option k's line. Returns one logit per option.
pub fn pointer_logits(
    cfg: &BigHeadConfig,
    w: &BigHeadWeights,
    h_answer: &[f32],
    h_options: &[&[f32]],
) -> Vec<f32> {
    let h = cfg.hidden_size;
    debug_assert_eq!(h_answer.len(), h);
    let a = layer_norm(h_answer, &w.norm_weight, &w.norm_bias);
    let q = matvec(&w.q, cfg.pointer_dim, h, &a);
    let scale = (cfg.pointer_dim as f32).powf(-0.5);
    h_options
        .iter()
        .map(|opt| {
            debug_assert_eq!(opt.len(), h);
            let o = layer_norm(opt, &w.norm_weight, &w.norm_bias);
            let k = matvec(&w.k, cfg.pointer_dim, h, &o);
            k.iter().zip(&q).map(|(kv, qv)| kv * qv).sum::<f32>() * scale
        })
        .collect()
}

/// Softmax over pointer logits; empty input is an error upstream.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
    let z: f32 = exps.iter().sum();
    exps.iter().map(|e| e / z).collect()
}

/// Calibrated probabilities for a question kind: softmax(logits / T).
pub fn kind_probabilities(cfg: &BigHeadConfig, kind: &str, logits: &[f32]) -> Vec<f32> {
    let t = cfg.temperature(kind);
    let scaled: Vec<f64> = logits.iter().map(|l| *l as f64 / t).collect();
    let m = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|l| (l - m).exp()).collect();
    let z: f64 = exps.iter().sum();
    exps.iter().map(|e| (e / z) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(h: usize, d: usize) -> BigHeadConfig {
        BigHeadConfig {
            schema_version: Some("1".into()),
            hidden_size: h,
            pointer_dim: d,
            kinds: [("noul".to_string(), 0.9)].into_iter().collect(),
        }
    }

    fn identity_weights(h: usize, d: usize) -> BigHeadWeights {
        // Norm = identity; q = k = first-`d` coordinate selectors, so the score is the
        // dot product of the option's and answer's leading components, times scale.
        let mut q = vec![0.0; d * h];
        let mut k = vec![0.0; d * h];
        for i in 0..d {
            q[i * h + i] = 1.0;
            k[i * h + i] = 1.0;
        }
        BigHeadWeights {
            norm_weight: vec![1.0; h],
            norm_bias: vec![0.0; h],
            q,
            k,
        }
    }

    #[test]
    fn logits_match_reference_math() {
        // h=4, d=2: score = (o1*a1 + o2*a2) * 2^-0.5 after LayerNorm. Build states whose
        // normed leading components are known: LayerNorm of [x,0,0,0] is [1,..]*affine.
        let c = cfg(4, 2);
        let w = identity_weights(4, 2);
        let answer = &[1.0f32, 0.0, 0.0, 0.0];
        let opt_a = &[1.0f32, 0.0, 0.0, 0.0];
        let opt_b = &[0.0f32, 1.0, 0.0, 0.0];
        let logits = pointer_logits(&c, &w, answer, &[opt_a, opt_b]);
        // LayerNorm (population, eps inside sqrt) of one-hot [1,0,0,0] is
        // [sqrt(3), -1/sqrt(3), -1/sqrt(3), -1/sqrt(3)]: mean 1/4, var 3/16.
        // q = k = first-two selectors, so:
        //   opt_a dot = sqrt(3)^2 + (1/sqrt(3))^2 = 10/3
        //   opt_b dot = sqrt(3)(-1/sqrt(3)) + (-1/sqrt(3))sqrt(3) = -2
        let expect_a = (10.0f32 / 3.0) * 0.5f32.sqrt();
        let expect_b = -2.0 * 0.5f32.sqrt();
        assert!(
            (logits[0] - expect_a).abs() < 1e-3,
            "opt_a: {} vs {expect_a}",
            logits[0]
        );
        assert!(
            (logits[1] - expect_b).abs() < 1e-3,
            "opt_b: {} vs {expect_b}",
            logits[1]
        );
        let p = softmax(&logits);
        assert!((p[0] + p[1] - 1.0).abs() < 1e-6);
        assert!(p[0] > 0.97, "option a should dominate: {p:?}");
    }

    #[test]
    fn temperature_flattens() {
        let c = cfg(4, 2);
        let logits = [2.0f32, 0.0];
        let sharp = kind_probabilities(&c, "noul", &logits); // T=0.9 sharpens slightly
        let flat = kind_probabilities(&c, "score", &logits); // T=1 default
        assert!(sharp[0] > flat[0]);
        assert!((flat[0] + flat[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn weights_validation_catches_shape_mismatch() {
        let c = cfg(4, 2);
        let bad = BigHeadWeights {
            norm_weight: vec![1.0; 3],
            norm_bias: vec![0.0; 4],
            q: vec![0.0; 8],
            k: vec![0.0; 8],
        };
        assert!(bad.validate(&c).is_err());
        assert!(identity_weights(4, 2).validate(&c).is_ok());
    }
}
