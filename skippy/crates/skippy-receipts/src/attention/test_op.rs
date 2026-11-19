//! A small attention op with a context and a query, and the exact attention
//! of a head over that context, for tests.

use super::{AttentionOp, ContextRow, Window, head_slice};

pub(crate) fn op() -> AttentionOp {
    AttentionOp {
        n_head: 4,
        n_head_kv: 2,
        head_dim_k: 3,
        head_dim_v: 2,
        scale: 0.5,
        softcap: 0.0,
        sinks: None,
        window: Window::Causal,
        output_rotation: None,
        q_rounding: None,
    }
}

/// A deterministic value in [-1, 1).
pub(crate) fn value(seed: u64) -> f32 {
    let x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(17);
    (x % 2000) as f32 / 1000.0 - 1.0
}

pub(crate) fn context(op: &AttentionOp, n: u32) -> Vec<ContextRow> {
    (0..n)
        .map(|position| ContextRow {
            position,
            key: (0..op.n_head_kv * op.head_dim_k)
                .map(|i| f64::from(value(u64::from(position) * 1000 + i as u64)))
                .collect(),
            value: (0..op.n_head_kv * op.head_dim_v)
                .map(|i| f64::from(value(u64::from(position) * 1000 + 500 + i as u64)))
                .collect(),
        })
        .collect()
}

pub(crate) fn query(op: &AttentionOp) -> Vec<f32> {
    (0..op.n_head * op.head_dim_k)
        .map(|i| value(77 + i as u64) * 3.0)
        .collect()
}

/// Plain softmax attention for one head over every row.
pub(crate) fn exact(op: &AttentionOp, head: usize, q: &[f32], rows: &[ContextRow]) -> Vec<f64> {
    let g = op.kv_head(head);
    let q = op.rounded_q(q, head).unwrap();
    let logits: Vec<f64> = rows
        .iter()
        .map(|r| op.logit(&q, head_slice(&r.key, g, op.head_dim_k).unwrap()))
        .collect();
    let sink = op
        .sinks
        .as_ref()
        .map_or(f64::NEG_INFINITY, |s| f64::from(s[head]));
    let max = logits.iter().copied().fold(sink, f64::max);
    let weights: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = weights.iter().sum::<f64>() + (sink - max).exp();
    (0..op.head_dim_v)
        .map(|d| {
            rows.iter()
                .zip(&weights)
                .map(|(r, w)| w * head_slice(&r.value, g, op.head_dim_v).unwrap()[d])
                .sum::<f64>()
                / sum
        })
        .collect()
}
