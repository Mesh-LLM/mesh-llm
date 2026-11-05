//! The client's side of the attention check: the combination of a head's
//! partials against the opened output, the blocks it then picks, the
//! proofs of the rows the node opens for them, and the recomputation of
//! those blocks' partials.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, ensure};

use super::opening::{
    AttentionPartials, AttentionPick, BlockChallenge, BlockPick, RowOpening, RowProof, RowsOpening,
    parse_hash,
};
use super::{
    AttentionLayout, BLOCK_POSITIONS, BLOCKS_PER_PICK, COMBINE_TOLERANCE, ContextRow,
    PARTIAL_TOLERANCE, block_partial, blocks, combine, combine_error, context_positions,
    head_slice, partial_error,
};
use crate::commitment::{
    MixingDigest, ValueDigest, leaf_index, mixing_leaf_from, mixing_slot, root_from_path,
    slots_per_position, stored_leaf,
};
use crate::draw::DrawStream;
use crate::layer_tensors::TensorKind;
use crate::recorder::{PrefixCell, prefix_leaf};
use crate::wire::{LeafOpening, MixingKind, Receipt, StageCommitment, TensorOpening};

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
        self.combine_error <= COMBINE_TOLERANCE
    }

    pub fn blocks(&self) -> usize {
        blocks(self.context.len())
    }
}

/// Checks that the partials the node opened for `pick` combine to the
/// output it opened at that position, `leaf`.
pub fn check_partials(
    layout: &AttentionLayout,
    pick: &AttentionPick,
    leaf: &LeafOpening,
    opened: &AttentionPartials,
) -> Result<OpenedHead> {
    let op = &layout.op;
    op.check_shape()?;
    let head = pick.head_draw % u32::try_from(op.n_head)?;
    ensure!(
        opened.head == head,
        "the node opened head {} for draw {}, the model's head is {head}",
        opened.head,
        pick.head_draw
    );
    let context = context_positions(op, pick.position);
    let partials = opened.values()?;
    ensure!(
        partials.len() == blocks(context.len()) * op.partial_len(),
        "{} partial values for {} attended positions",
        partials.len(),
        context.len()
    );
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
pub fn draw_blocks(seed: &[u8], pick: &AttentionPick, n_blocks: usize) -> Vec<u32> {
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

/// The positions of block `block` of a context.
pub fn block_positions(context: &[u32], block: u32) -> impl Iterator<Item = u32> + '_ {
    context
        .chunks(BLOCK_POSITIONS)
        .nth(block as usize)
        .into_iter()
        .flatten()
        .copied()
}

/// The K and V rows of every opened position, by (layer, position), each
/// checked against the receipt.
pub fn verify_rows(
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
            let cell = PrefixCell {
                pos: i32::try_from(row.position)?,
                // the leaf binds the index alone
                stream: 0,
                cell: *cell,
                digest,
            };
            let root = root_from_path(
                prefix_leaf(&cell),
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

/// The digests of a layer's stored rows, which must be opened in index
/// order.
pub(crate) fn stored_digests(stored: &[TensorOpening]) -> Result<Vec<MixingDigest>> {
    stored
        .iter()
        .enumerate()
        .map(|(i, row)| {
            ensure!(
                row.kind == MixingKind::Stored && row.index as usize == i,
                "stored rows must be opened in index order"
            );
            Ok(MixingDigest {
                kind: TensorKind::MixingStored,
                index: row.index,
                value: row
                    .bytes()?
                    .map(|bytes| (row.ggml_type, ValueDigest::of(&bytes))),
            })
        })
        .collect()
}

/// The largest error of the picked blocks' partials, recomputed from the
/// committed q and the opened rows, against the partials the node opened.
pub fn check_blocks(
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
        worst = worst.max(partial_error(opened, &recomputed));
    }
    Ok(worst)
}

/// Whether a block's recomputed partial is within the tolerance of the
/// node's.
pub fn block_passed(error: f64) -> bool {
    error <= PARTIAL_TOLERANCE
}
