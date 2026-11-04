//! The loose attention check (`mesh/docs/design/RECEIPTS.md`,
//! Verification, step 5).
//!
//! An opened attention output is checked against the committed context
//! without the client reading all of it. The positions a query attends to
//! are cut into blocks of [`BLOCK_POSITIONS`]. For a head, a block's
//! flash-attention partial is its largest logit, the sum of the
//! exponentials of its logits below that, and their sum of values weighted
//! by them. The node commits every head's partials in its opening. The
//! client then draws a head and a few blocks, recomputes those blocks'
//! partials from the committed q, K and V, and checks that all the head's
//! partials combine to the opened output.
//!
//! A node that cut the context computed its output from fewer blocks. Its
//! partials either combine to that output, and then a block it dropped
//! gives a partial the client recomputes differently, or they are the
//! honest ones and do not combine to its output.
//!
//! Partials are a function of the committed values alone, so the node
//! computes them on its CPU, in f64, and the client's recomputation of a
//! block matches them closely. Only the combination is compared loosely,
//! because the node's backend computed the output with its own kernel.

use anyhow::{Context, Result, bail, ensure};

use crate::audit::Profile;
use crate::wire::{MixingKind, TensorOpening};

pub mod opening;
pub mod verify;

/// Attended positions per block.
pub const BLOCK_POSITIONS: usize = 64;
/// A recomputed block's partial passes within this error, relative to the
/// partial's own scale. Both sides compute it in f64 from the same values.
pub const PARTIAL_TOLERANCE: f64 = 1e-6;
/// The partials' combination passes within this error of the opened
/// output, relative to the output's largest value. Flash attention on a
/// GPU computes in f16 and f32; honest prefill was measured at up to 9e-3
/// at 16k positions (`RECEIPTS.md`, Client checks).
pub const COMBINE_TOLERANCE: f64 = 2e-2;

/// Which positions a query reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    /// Every earlier position, and its own.
    Causal,
    /// The last `n` positions up to its own.
    Sliding(u32),
    /// Earlier positions in its own chunk of `n`.
    Chunked(u32),
}

impl Window {
    pub fn attends(self, query: u32, key: u32) -> bool {
        if key > query {
            return false;
        }
        match self {
            Self::Causal => true,
            Self::Sliding(n) => query - key < n,
            Self::Chunked(n) => n > 0 && key / n == query / n,
        }
    }
}

/// Blocks of a picked head the client recomputes.
pub const BLOCKS_PER_PICK: usize = 2;

/// One attention op of a layer as the graph computes it, by the indices
/// of its values among the layer's committed tensors.
#[derive(Clone, Debug, PartialEq)]
pub struct AttentionLayout {
    pub layer: u32,
    pub op: AttentionOp,
    /// q, among the layer's mixing inputs.
    pub q_input: u32,
    /// K and V rows, among the layer's stored rows.
    pub k_store: u32,
    pub v_store: u32,
    /// The attention output, among the layer's mixing outputs.
    pub output: u32,
    /// A gate among the layer's mixing inputs that multiplies the
    /// attention output, value by value, before it leaves the token-mixing
    /// region; the committed output is the gated one.
    pub output_gate: Option<u32>,
}

/// An attention op of a layer, or why the check cannot follow it.
pub type AttentionDescriptor = std::result::Result<AttentionLayout, String>;

/// The attention op the check follows in a layer described by
/// `descriptors`: `None` for a layer without attention, an error for one
/// the check cannot follow.
pub fn layer_layout(descriptors: &[AttentionDescriptor]) -> Result<Option<&AttentionLayout>> {
    match descriptors {
        [] => Ok(None),
        [Ok(layout)] => Ok(Some(layout)),
        [Err(reason)] => bail!("{reason}"),
        _ => bail!("a layer with {} attention ops", descriptors.len()),
    }
}

/// The positions a query at `position` attends to, in order, when the
/// sequence holds every position from 0.
pub fn context_positions(op: &AttentionOp, position: u32) -> Vec<u32> {
    (0..=position)
        .filter(|&key| op.window.attends(position, key))
        .collect()
}

impl AttentionLayout {
    /// The layer's q at one position, from its opened mixing values.
    pub fn q_values(&self, mixing: &[TensorOpening]) -> Result<Vec<f32>> {
        opened_f32(mixing, MixingKind::Input, self.q_input, "q")
    }

    /// The layer's attention output at one position.
    pub fn output_values(&self, mixing: &[TensorOpening]) -> Result<Vec<f32>> {
        opened_f32(
            mixing,
            MixingKind::Output,
            self.output,
            "the attention output",
        )
    }

    /// The gate the attention output is multiplied by at one position, or
    /// `None` for an ungated output.
    pub fn gate_values(&self, mixing: &[TensorOpening]) -> Result<Option<Vec<f32>>> {
        self.output_gate
            .map(|gate| opened_f32(mixing, MixingKind::Input, gate, "the attention output gate"))
            .transpose()
    }

    /// The K and V rows of one position from the layer's stored rows, by
    /// index, in their cache types.
    pub fn context_row(&self, position: u32, stored: &[(i32, &[u8])]) -> Result<ContextRow> {
        let op = &self.op;
        let read = |index: u32, len: usize| -> Result<Vec<f64>> {
            let (ggml_type, bytes) = stored
                .get(index as usize)
                .with_context(|| format!("stored row {index} is missing"))?;
            let ggml_type = u32::try_from(*ggml_type)?;
            ensure!(
                crate::dequant::supports(ggml_type),
                "a stored row of ggml type {ggml_type}"
            );
            crate::dequant::dequantize_row(ggml_type, bytes, len)
        };
        Ok(ContextRow {
            position,
            key: read(self.k_store, op.n_head_kv * op.head_dim_k)?,
            value: read(self.v_store, op.n_head_kv * op.head_dim_v)?,
        })
    }
}

fn opened_f32(
    mixing: &[TensorOpening],
    kind: MixingKind,
    index: u32,
    what: &str,
) -> Result<Vec<f32>> {
    mixing
        .iter()
        .find(|m| m.kind == kind && m.index == index)
        .with_context(|| format!("{what} is not opened"))?
        .f32_values()?
        .with_context(|| format!("{what} is absent or not f32"))
}

/// One layer's attention, as the client's own build of the claimed model
/// computes it.
#[derive(Clone, Debug, PartialEq)]
pub struct AttentionOp {
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim_k: usize,
    pub head_dim_v: usize,
    /// Multiplies q·k.
    pub scale: f64,
    /// `softcap · tanh(logit / softcap)` when nonzero.
    pub softcap: f64,
    /// A per-head logit that only adds to the softmax's denominator.
    pub sinks: Option<Vec<f32>>,
    pub window: Window,
    /// A rotation of the output before it leaves the token-mixing region.
    pub output_rotation: Option<OutputRotation>,
    /// How the kernel rounds each head of q before its dot with K, or
    /// `None`.
    pub q_rounding: Option<Profile>,
}

/// Values in one block of a q rounding.
const Q_ROUNDING_BLOCK: usize = 32;

/// How a kernel that rounds q to `ggml_type` rounds it. The CPU kernel
/// rounds to q8_0 as [`Profile::CpuQ8`] does. The CUDA kernel rounds the
/// scaled q to q8_1 by division, which differs from that by an ulp at most
/// for a value, well inside the combine tolerance.
pub fn q_rounding(ggml_type: u32) -> Result<Profile> {
    match ggml_type {
        crate::dequant::GGML_Q8_0 => Ok(Profile::CpuQ8),
        other => bail!("attention that rounds q to ggml type {other}"),
    }
}

/// A rotation of each head's attention output in chunks of `n` values:
/// value `i` of a chunk becomes `sum_k matrix[i * n + k] * value k`.
/// llama.cpp rotates V before a quantized KV cache stores it, and rotates
/// the output back.
#[derive(Clone, Debug, PartialEq)]
pub struct OutputRotation {
    pub n: usize,
    pub matrix: Vec<f32>,
}

impl OutputRotation {
    /// Rotates `values`, a whole number of chunks, in place.
    pub fn apply(&self, values: &mut [f64]) -> Result<()> {
        let n = self.n;
        ensure!(
            n > 0 && self.matrix.len() == n * n && values.len().is_multiple_of(n),
            "a rotation of {} values in chunks of {n}",
            values.len()
        );
        for chunk in values.chunks_mut(n) {
            let input = chunk.to_vec();
            for (value, row) in chunk.iter_mut().zip(self.matrix.chunks(n)) {
                *value = row
                    .iter()
                    .zip(&input)
                    .map(|(&m, &x)| f64::from(m) * x)
                    .sum();
            }
        }
        Ok(())
    }
}

impl AttentionOp {
    pub fn check_shape(&self) -> Result<()> {
        ensure!(
            self.n_head > 0
                && self.n_head_kv > 0
                && self.n_head.is_multiple_of(self.n_head_kv)
                && self.head_dim_k > 0
                && self.head_dim_v > 0,
            "attention shape {self:?} is not one the check follows"
        );
        if let Some(sinks) = &self.sinks {
            ensure!(sinks.len() == self.n_head, "one sink per head");
        }
        if let Some(rotation) = &self.output_rotation {
            ensure!(
                rotation.n > 0 && self.head_dim_v.is_multiple_of(rotation.n),
                "an output rotation in chunks of {} across heads of {}",
                rotation.n,
                self.head_dim_v
            );
        }
        if self.q_rounding.is_some() {
            ensure!(
                self.head_dim_k.is_multiple_of(Q_ROUNDING_BLOCK),
                "q rounded in blocks of {Q_ROUNDING_BLOCK} across heads of {}",
                self.head_dim_k
            );
        }
        Ok(())
    }

    /// Head `head` of q as the kernel takes it into its dots with K.
    fn rounded_q(&self, q: &[f32], head: usize) -> Result<Vec<f64>> {
        let q = head_slice(q, head, self.head_dim_k).context("q")?;
        Ok(match self.q_rounding {
            Some(profile) => profile.apply(q),
            None => q.iter().map(|&x| f64::from(x)).collect(),
        })
    }

    /// The K and V head that query head `head` reads.
    pub fn kv_head(&self, head: usize) -> usize {
        head / (self.n_head / self.n_head_kv)
    }

    /// Values in one head's partials: per block, the largest logit, the
    /// sum and the weighted values.
    pub fn partial_len(&self) -> usize {
        self.head_dim_v + 2
    }

    fn logit(&self, q: &[f64], k: &[f64]) -> f64 {
        let dot: f64 = q.iter().zip(k).map(|(&a, &b)| a * b).sum();
        let logit = dot * self.scale;
        if self.softcap == 0.0 {
            logit
        } else {
            self.softcap * (logit / self.softcap).tanh()
        }
    }
}

/// One attended position's K and V rows, all heads, dequantized from the
/// cache type.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextRow {
    pub position: u32,
    pub key: Vec<f64>,
    pub value: Vec<f64>,
}

/// A block's partial for one head: the largest logit, the sum of
/// `exp(logit - max)`, and the values weighted by those exponentials.
pub fn block_partial(
    op: &AttentionOp,
    head: usize,
    q: &[f32],
    rows: &[ContextRow],
) -> Result<Vec<f64>> {
    let g = op.kv_head(head);
    let q = op.rounded_q(q, head)?;
    let mut logits = Vec::with_capacity(rows.len());
    for row in rows {
        let k = head_slice(&row.key, g, op.head_dim_k).context("a K row")?;
        logits.push(op.logit(&q, k));
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut partial = vec![0.0; op.partial_len()];
    partial[0] = max;
    for (row, logit) in rows.iter().zip(logits) {
        let weight = (logit - max).exp();
        partial[1] += weight;
        let v = head_slice(&row.value, g, op.head_dim_v).context("a V row")?;
        for (out, &x) in partial[2..].iter_mut().zip(v) {
            *out += weight * x;
        }
    }
    Ok(partial)
}

/// Every block's partial for one head, block after block, over `context`
/// (the attended positions, in order).
pub fn head_partials(
    op: &AttentionOp,
    head: usize,
    q: &[f32],
    context: &[ContextRow],
) -> Result<Vec<f32>> {
    let mut partials = Vec::with_capacity(blocks(context.len()) * op.partial_len());
    for block in context.chunks(BLOCK_POSITIONS) {
        partials.extend(block_partial(op, head, q, block)?.iter().map(|&x| x as f32));
    }
    Ok(partials)
}

/// Blocks over `positions` attended positions.
pub fn blocks(positions: usize) -> usize {
    positions.div_ceil(BLOCK_POSITIONS)
}

/// The head's normalized output from its partials.
pub fn combine(op: &AttentionOp, head: usize, partials: &[f32]) -> Result<Vec<f64>> {
    let len = op.partial_len();
    ensure!(
        !partials.is_empty() && partials.len().is_multiple_of(len),
        "partials of {} values are not whole blocks of {len}",
        partials.len()
    );
    let sink = op
        .sinks
        .as_ref()
        .map(|s| f64::from(s[head]))
        .unwrap_or(f64::NEG_INFINITY);
    let max = partials
        .chunks(len)
        .map(|p| f64::from(p[0]))
        .fold(sink, f64::max);
    ensure!(max.is_finite(), "no partial has a finite logit");
    let mut sum = (sink - max).exp();
    let mut out = vec![0.0; op.head_dim_v];
    for p in partials.chunks(len) {
        let scale = (f64::from(p[0]) - max).exp();
        sum += scale * f64::from(p[1]);
        for (o, &x) in out.iter_mut().zip(&p[2..]) {
            *o += scale * f64::from(x);
        }
    }
    ensure!(sum > 0.0, "the partials sum to nothing");
    for o in &mut out {
        *o /= sum;
    }
    Ok(out)
}

/// How far an opened block partial is from the client's recomputation:
/// the largest of the logit's error (relative to the logit's size, at
/// least 1) and the sum's and values' errors, each relative to the
/// largest of them, which the node rounds to f32.
pub fn partial_error(opened: &[f32], recomputed: &[f64]) -> f64 {
    if opened.len() != recomputed.len() || recomputed.len() < 2 {
        return f64::INFINITY;
    }
    let max_err = (f64::from(opened[0]) - recomputed[0]).abs() / recomputed[0].abs().max(1.0);
    let scale = recomputed[1..]
        .iter()
        .fold(f64::MIN_POSITIVE, |m, x| m.max(x.abs()));
    let rest = opened[1..]
        .iter()
        .zip(&recomputed[1..])
        .map(|(&a, &b)| (f64::from(a) - b).abs() / scale)
        .fold(0.0, f64::max);
    if max_err.is_nan() || rest.is_nan() {
        return f64::INFINITY;
    }
    max_err.max(rest)
}

/// How far the combination is from the opened output's head, relative to
/// the larger of their largest values.
pub fn combine_error(combined: &[f64], opened: &[f32]) -> f64 {
    if combined.len() != opened.len() {
        return f64::INFINITY;
    }
    let scale = combined
        .iter()
        .map(|x| x.abs())
        .chain(opened.iter().map(|&x| f64::from(x).abs()))
        .fold(0.0, f64::max)
        .max(f64::MIN_POSITIVE);
    let err = combined
        .iter()
        .zip(opened)
        .map(|(&a, &b)| (a - f64::from(b)).abs())
        .fold(0.0, f64::max);
    if err.is_nan() {
        f64::INFINITY
    } else {
        err / scale
    }
}

/// The values of head `head` in a column of heads of `dim` values each.
pub fn head_slice<T>(column: &[T], head: usize, dim: usize) -> Result<&[T]> {
    let start = head * dim;
    match column.get(start..start + dim) {
        Some(slice) => Ok(slice),
        None => bail!(
            "a column of {} values has no head {head} of {dim}",
            column.len()
        ),
    }
}

#[cfg(test)]
#[path = "attention/tests.rs"]
mod tests;
