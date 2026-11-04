//! The client's side of the loose attention check ([`crate::attention`]):
//! the combination of a head's partials against the opened output, the
//! blocks it then picks, the proofs of the rows the node opens for them,
//! and the recomputation of those blocks' partials.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail, ensure};

use crate::attention::opening::{
    AttentionPartials, AttentionPick, BlockChallenge, BlockPick, RowOpening, RowProof, RowsOpening,
    parse_hash, stored_digests,
};
use crate::attention::{
    AttentionLayout, AttentionOp, BLOCK_POSITIONS, BLOCKS_PER_PICK, ContextRow, block_partial,
    block_positions, blocks, context_len, context_positions, head_slice,
};
use crate::commitment::{
    leaf_index, mixing_leaf_from, mixing_slot, prefix_leaf, root_from_path, slots_per_position,
    stored_leaf,
};
use crate::draw::DrawStream;
use crate::error_bound::{all_finite, within, worst, worst_of};
use crate::node_fault::BlameNode;
use crate::wire::{LeafOpening, Receipt, StageCommitment};

/// A recomputed block's partial passes within this error, relative to the
/// partial's own scale. Both sides compute it in f64 from the same values.
pub const PARTIAL_TOLERANCE: f64 = 1e-6;
/// The least error within which the partials' combination passes against
/// the opened output, relative to the output's largest value: the tolerance
/// of a short context ([`combine_tolerance`]).
pub(crate) const COMBINE_TOLERANCE: f64 = 2e-2;
/// How the tolerance grows with the square root of the attended positions.
/// A flash-attention kernel that sums f16 values in f16 (CUDA prefill)
/// rounds each partial sum to 11 bits, and the rounding adds up like a
/// random walk over the context. Honest CUDA prefill was measured at up to
/// `0.54 * 2^-11 * sqrt(positions)` over 163 picks from 3,000 to 31,000
/// positions (`RECEIPTS_MEASUREMENTS.md`); this is `2 * 2^-11`, about 3.7
/// times that.
const COMBINE_TOLERANCE_PER_SQRT_POSITION: f64 = 1.0 / 1024.0;

/// The error within which the partials of a head that attends to `context`
/// positions pass against the opened output: [`COMBINE_TOLERANCE`] up to
/// about 420 positions, then growing with their square root (6.3e-2 at
/// 4,096 positions, 1.8e-1 at 32,768). A fixed tolerance fails honest long
/// turns: at 2e-2 a CUDA node was at 97% of it at 7,000 positions and past
/// it at 19,000.
pub(crate) fn combine_tolerance(context: usize) -> f64 {
    COMBINE_TOLERANCE.max(COMBINE_TOLERANCE_PER_SQRT_POSITION * (context as f64).sqrt())
}

/// One picked head after the node opened its partials.
#[derive(Clone, Debug)]
pub struct OpenedHead {
    pub pick: AttentionPick,
    pub head: u32,
    pub partials: Vec<f32>,
    /// The positions the query attends to, in order.
    pub context: Vec<u32>,
    /// How far the partials combine from the opened output.
    pub combine_error: f64,
}

impl OpenedHead {
    pub fn passed(&self) -> bool {
        within(self.combine_error, self.tolerance())
    }

    /// The combine error this head passes within, for its context.
    pub fn tolerance(&self) -> f64 {
        combine_tolerance(self.context.len())
    }

    pub fn blocks(&self) -> usize {
        blocks(self.context.len())
    }
}

/// Checks that the partials the node opened for `pick` combine to the
/// output it opened at that position, `leaf`.
///
/// An error that blames the node ([`crate::node_fault::NodeFault`]) is a
/// defect of what it opened. Any other is an attention shape the check
/// does not follow, which says nothing about the node.
pub fn check_partials(
    layout: &AttentionLayout,
    pick: &AttentionPick,
    leaf: &LeafOpening,
    opened: &AttentionPartials,
) -> Result<OpenedHead> {
    let op = &layout.op;
    op.check_shape()?;
    let head = pick.head_draw % u32::try_from(op.n_head)?;
    opened_head(layout, pick, leaf, opened, head).or_node_fault()
}

/// [`check_partials`] for the model's head `head`, once the auditor's own
/// layout is known to be one the check follows.
fn opened_head(
    layout: &AttentionLayout,
    pick: &AttentionPick,
    leaf: &LeafOpening,
    opened: &AttentionPartials,
    head: u32,
) -> Result<OpenedHead> {
    let op = &layout.op;
    ensure!(
        opened.head == head,
        "the node opened head {} for draw {}, the model's head is {head}",
        opened.head,
        pick.head_draw
    );
    // the length first: the context is as long as the position is large,
    // and only the receipt says how large that is
    let attended = context_len(op, pick.position);
    let partials = opened.values()?;
    let expected = attended.div_ceil(BLOCK_POSITIONS as u64) * op.partial_len() as u64;
    ensure!(
        partials.len() as u64 == expected,
        "{} partial values for {attended} attended positions",
        partials.len()
    );
    ensure!(all_finite(&partials), "a partial is not a number");
    let context = context_positions(op, pick.position);
    let output = layout.output_values(&leaf.mixing)?;
    let head_index = usize::try_from(head)?;
    let output = head_slice(&output, head_index, op.head_dim_v)?;
    let mut combined = combine(op, head_index, &partials)?;
    if let Some(rotation) = &op.output_rotation {
        rotation.apply(&mut combined)?;
    }
    if let Some(gate) = layout.gate_values(&leaf.mixing)? {
        let gate = head_slice(&gate, head_index, op.head_dim_v)?;
        for (value, &g) in combined.iter_mut().zip(gate) {
            *value *= f64::from(g);
        }
    }
    Ok(OpenedHead {
        pick: pick.clone(),
        head,
        combine_error: combine_error(&combined, output),
        partials,
        context,
    })
}

/// The blocks the client recomputes for each opened head, drawn from
/// `seed`, which the node cannot know before it opened the partials.
pub fn block_challenge(receipt_id: &str, heads: &[OpenedHead], seed: &[u8]) -> BlockChallenge {
    let picks = heads
        .iter()
        .map(|h| BlockPick {
            position: h.pick.position,
            layer: h.pick.layer,
            blocks: draw_blocks(seed, &h.pick, h.blocks()),
        })
        .collect();
    BlockChallenge {
        receipt_id: receipt_id.to_owned(),
        picks,
        proof: None,
    }
}

/// [`BLOCKS_PER_PICK`] distinct blocks of `n_blocks`, sorted; every block
/// when there are no more.
pub(crate) fn draw_blocks(seed: &[u8], pick: &AttentionPick, n_blocks: usize) -> Vec<u32> {
    let n = u32::try_from(n_blocks).unwrap_or(u32::MAX);
    if n_blocks <= BLOCKS_PER_PICK {
        return (0..n).collect();
    }
    let label = format!("attention/{}/{}", pick.position, pick.layer);
    let mut stream = DrawStream::new(seed, &label);
    let mut picked = BTreeSet::new();
    while picked.len() < BLOCKS_PER_PICK {
        picked.insert(stream.in_range(0..n));
    }
    picked.into_iter().collect()
}

/// Checks that `opening` holds exactly the rows `blocks` asks of `heads`
/// (one head per pick of `blocks`, in order): one part per stage of the
/// receipt, in order, and every position of every picked block, once. A
/// node that left a row out, sent one twice or added one fails here, so
/// the recomputation reads the rows it asked for and no others.
///
/// An error is the node's fault.
pub fn verify_answer(
    receipt: &Receipt,
    blocks: &BlockChallenge,
    heads: &[OpenedHead],
    opening: &RowsOpening,
) -> Result<()> {
    answered_rows(receipt, blocks, heads, opening).or_node_fault()
}

fn answered_rows(
    receipt: &Receipt,
    blocks: &BlockChallenge,
    heads: &[OpenedHead],
    opening: &RowsOpening,
) -> Result<()> {
    ensure!(
        opening.receipt_id == receipt.receipt_id,
        "the rows name another receipt"
    );
    let stages: Vec<u32> = receipt.stages.iter().map(|s| s.stage_index).collect();
    let opened: Vec<u32> = opening.stages.iter().map(|s| s.stage_index).collect();
    ensure!(
        opened == stages,
        "the rows come from stages {opened:?}; the receipt's stages are {stages:?}"
    );
    let mut wanted = BTreeSet::new();
    for (head, pick) in heads.iter().zip(&blocks.picks) {
        for &block in &pick.blocks {
            wanted.extend(block_positions(&head.context, block).map(|p| (pick.layer, p)));
        }
    }
    let mut seen = BTreeSet::new();
    for row in opening.stages.iter().flat_map(|s| &s.rows) {
        let key = (row.layer, row.position);
        ensure!(
            wanted.contains(&key) && seen.insert(key),
            "the row at position {} of layer {} is not one the blocks ask for, once",
            row.position,
            row.layer
        );
    }
    if let Some((layer, position)) = wanted.difference(&seen).next() {
        bail!("the row at position {position} of layer {layer} is not opened");
    }
    Ok(())
}

/// The K and V rows of every opened position, by (layer, position), each
/// checked against the receipt.
///
/// An error is the node's fault.
pub fn verify_rows(
    receipt: &Receipt,
    layouts: &BTreeMap<u32, AttentionLayout>,
    opening: &RowsOpening,
) -> Result<BTreeMap<(u32, u32), ContextRow>> {
    committed_rows(receipt, layouts, opening).or_node_fault()
}

fn committed_rows(
    receipt: &Receipt,
    layouts: &BTreeMap<u32, AttentionLayout>,
    opening: &RowsOpening,
) -> Result<BTreeMap<(u32, u32), ContextRow>> {
    ensure!(
        opening.receipt_id == receipt.receipt_id,
        "the rows name another receipt"
    );
    let mut rows = BTreeMap::new();
    for stage_rows in &opening.stages {
        let stage = receipt
            .stages
            .iter()
            .find(|s| s.stage_index == stage_rows.stage_index)
            .with_context(|| format!("the receipt has no stage {}", stage_rows.stage_index))?;
        for row in &stage_rows.rows {
            ensure!(
                (stage.layer_start..stage.layer_end).contains(&row.layer),
                "stage {} opened a row of layer {}",
                stage.stage_index,
                row.layer
            );
            verify_row(receipt, stage, row)
                .with_context(|| format!("position {}, layer {}", row.position, row.layer))?;
            let layout = layouts
                .get(&row.layer)
                .with_context(|| format!("layer {} has no attention to check", row.layer))?;
            let stored = row
                .stored
                .iter()
                .map(|s| Ok((s.ggml_type, s.bytes()?.context("a stored row is absent")?)))
                .collect::<Result<Vec<_>>>()?;
            let stored: Vec<(i32, &[u8])> =
                stored.iter().map(|(t, b)| (*t, b.as_slice())).collect();
            rows.insert(
                (row.layer, row.position),
                layout.context_row(row.position, &stored)?,
            );
        }
    }
    Ok(rows)
}

fn verify_row(receipt: &Receipt, stage: &StageCommitment, row: &RowOpening) -> Result<()> {
    let stored = stored_leaf(row.position, row.layer, &stored_digests(&row.stored)?);
    match &row.proof {
        RowProof::Turn { rest, path } => {
            ensure!(
                (receipt.position_start..receipt.position_end).contains(&row.position),
                "the turn did not compute this position"
            );
            let leaf = mixing_leaf_from(row.position, row.layer, &parse_hash(rest)?, &stored);
            let index = leaf_index(
                row.position,
                mixing_slot(row.layer, stage.layer_start),
                receipt.position_start,
                slots_per_position(stage.layer_start, stage.layer_end),
            );
            let root = root_from_path(leaf, index, stage.leaf_count, path)?;
            ensure!(
                hex::encode(root) == stage.root,
                "the mixing leaf is not in the stage tree"
            );
        }
        RowProof::Prefix {
            cell,
            cell_path,
            leaves,
            prefix_path,
        } => {
            let prefix = stage
                .prefix
                .as_ref()
                .context("the stage committed no prefix")?;
            ensure!(
                prefix.positions == receipt.position_start && row.position < prefix.positions,
                "the prefix does not hold one cell per position below the turn"
            );
            let digest = root_from_path(
                stored,
                u64::from(row.layer - stage.layer_start),
                u64::from(stage.layer_end - stage.layer_start),
                cell_path,
            )?;
            let root = root_from_path(
                // the leaf binds the cell's index alone, not its stream
                prefix_leaf(
                    i32::try_from(row.position)?,
                    *cell,
                    &digest,
                    &parse_hash(leaves)?,
                ),
                u64::from(row.position),
                u64::from(prefix.positions),
                prefix_path,
            )?;
            ensure!(
                root == parse_hash(&prefix.root)?,
                "the cell is not in the prefix tree"
            );
        }
    }
    Ok(())
}

/// The largest error of the picked blocks' partials, recomputed from the
/// committed q and the opened rows, against the partials the node opened.
///
/// An error is the node's fault: a row or a partial the recomputation
/// needs is missing from what it opened.
pub fn check_blocks(
    layout: &AttentionLayout,
    head: &OpenedHead,
    pick: &BlockPick,
    q: &[f32],
    rows: &BTreeMap<(u32, u32), ContextRow>,
) -> Result<f64> {
    recomputed_blocks(layout, head, pick, q, rows).or_node_fault()
}

fn recomputed_blocks(
    layout: &AttentionLayout,
    head: &OpenedHead,
    pick: &BlockPick,
    q: &[f32],
    rows: &BTreeMap<(u32, u32), ContextRow>,
) -> Result<f64> {
    let op = &layout.op;
    let len = op.partial_len();
    let mut worst: f64 = 0.0;
    for &block in &pick.blocks {
        let context = block_positions(&head.context, block)
            .map(|position| {
                rows.get(&(pick.layer, position))
                    .cloned()
                    .with_context(|| format!("the row at position {position} is not opened"))
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(!context.is_empty(), "block {block} is past the context");
        let recomputed = block_partial(op, usize::try_from(head.head)?, q, &context)?;
        let start = block as usize * len;
        let opened = head
            .partials
            .get(start..start + len)
            .context("the partials end before the block")?;
        worst = crate::error_bound::worst(worst, partial_error(opened, &recomputed));
    }
    Ok(worst)
}

/// Whether a block's recomputed partial is within the tolerance of the
/// node's.
pub fn block_passed(error: f64) -> bool {
    within(error, PARTIAL_TOLERANCE)
}
/// The head's normalized output from its partials.
pub fn combine(op: &AttentionOp, head: usize, partials: &[f32]) -> Result<Vec<f64>> {
    let len = op.partial_len();
    ensure!(
        !partials.is_empty() && partials.len().is_multiple_of(len),
        "partials of {} values are not whole blocks of {len}",
        partials.len()
    );
    // `f64::max` below would drop a partial that is not a number
    ensure!(
        crate::error_bound::all_finite(partials),
        "a partial is not a number"
    );
    let sink = op
        .sinks
        .as_ref()
        .and_then(|s| s.get(head))
        .map_or(f64::NEG_INFINITY, |&s| f64::from(s));
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
pub(crate) fn partial_error(opened: &[f32], recomputed: &[f64]) -> f64 {
    if opened.len() != recomputed.len() || recomputed.len() < 2 {
        return f64::INFINITY;
    }
    let max_err = (f64::from(opened[0]) - recomputed[0]).abs() / recomputed[0].abs().max(1.0);
    let scale = worst(
        f64::MIN_POSITIVE,
        worst_of(recomputed[1..].iter().map(|x| x.abs())),
    );
    let rest = worst_of(
        opened[1..]
            .iter()
            .zip(&recomputed[1..])
            .map(|(&a, &b)| (f64::from(a) - b).abs() / scale),
    );
    worst(max_err, rest)
}

/// How far the combination is from the opened output's head, relative to
/// the larger of their largest values.
pub fn combine_error(combined: &[f64], opened: &[f32]) -> f64 {
    if combined.len() != opened.len() {
        return f64::INFINITY;
    }
    let values = combined
        .iter()
        .map(|x| x.abs())
        .chain(opened.iter().map(|&x| f64::from(x).abs()));
    let scale = worst(worst_of(values), f64::MIN_POSITIVE);
    let err = worst_of(
        combined
            .iter()
            .zip(opened)
            .map(|(&a, &b)| (a - f64::from(b)).abs()),
    );
    // an infinite error over an infinite scale is not a number
    worst(err / scale, 0.0)
}

#[cfg(test)]
mod tests;
