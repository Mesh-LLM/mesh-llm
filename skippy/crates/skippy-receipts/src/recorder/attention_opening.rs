//! A stage's side of the attention check ([`crate::attention`]).
//!
//! With the challenge, the stage opens the partials of each attention pick
//! in its layers: it reads the K and V rows of every position the query
//! attends to from its KV cache, and computes the picked head's partials
//! from them and the committed q, on its CPU. After the client picks the
//! blocks it recomputes, the stage opens their rows with proofs: through
//! the mixing leaf for a position the turn computed, and through the cell's
//! digest and the prefix root for one it read from cache.
//!
//! The rows are read from the live cache, so they are there only while the
//! cells still hold what the turn read or wrote; the stage checks that
//! before it reads. A freed cell keeps its rows until a ubatch writes it,
//! and the stage pins the cells of a turn a client may still challenge
//! ([`super::pins`]), so a later request writes other cells while it fits
//! there.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};

use super::{
    CommittedTurn, Event, PrefixCell, Recorder, ReplayedLayer, Replayer, StoredRows, opening_turn,
    turn_mut,
};
use crate::attention::opening::{
    AttentionPartials, AttentionPick, BlockChallenge, RowOpening, RowProof, StageRows,
};
use crate::attention::verify::{block_positions, stored_digests};
use crate::attention::{
    AttentionLayout, BLOCKS_PER_PICK, ContextRow, blocks, context_positions, head_partials,
    layer_layout,
};
use crate::commitment::{
    Hash, MerkleTree, encode_path, leaf_index, mixing_leaf_from, mixing_slot, slots_per_position,
    stored_leaf,
};
use crate::wire::{LayerOpening, MixingKind, TensorOpening};

use super::cells::prefix_leaf;

/// An attention pick a stage opened partials for: the positions its query
/// attends to, whose blocks the client may then ask the rows of.
#[derive(Clone, Debug)]
pub(super) struct OpenedAttention {
    position: u32,
    layer: u32,
    context: Vec<u32>,
}

/// Replayed values by (position, layer).
type Replayed = BTreeMap<(u32, u32), ReplayedLayer>;

impl Recorder {
    /// The partials of each attention pick of `picks` in this stage's
    /// layers that has attention, from the opened q and the rows in the
    /// stage's cache.
    pub(super) fn open_attention(
        &self,
        receipt_id: &str,
        picks: &[AttentionPick],
        layers: &[LayerOpening],
        replayed: &Replayed,
        replayer: &dyn Replayer,
    ) -> Result<Vec<AttentionPartials>> {
        let mut by_layer: BTreeMap<u32, Vec<(&AttentionPick, AttentionLayout)>> = BTreeMap::new();
        for pick in picks.iter().filter(|p| self.layers.contains(&p.layer)) {
            let values = replayed
                .get(&(pick.position, pick.layer))
                .with_context(|| format!("position {} was not replayed", pick.position))?;
            // the client decides from its own build what a layer without a
            // layout the check follows means
            if let Ok(Some(layout)) = layer_layout(&values.attention) {
                by_layer
                    .entry(pick.layer)
                    .or_default()
                    .push((pick, layout.clone()));
            }
        }
        let mut partials = Vec::new();
        let mut opened = Vec::new();
        for (layer, picks) in by_layer {
            let last = picks.iter().map(|(p, _)| p.position).max().unwrap_or(0);
            let cells = self.context_cells(receipt_id, 0..last + 1)?;
            self.check_cells(&cells)?;
            let rows = read_context(replayer, layer, &cells, &picks[0].1)?;
            for (pick, layout) in picks {
                let leaf = layers
                    .iter()
                    .find(|l| l.layer == layer)
                    .and_then(|l| l.leaves.iter().find(|leaf| leaf.position == pick.position))
                    .context("an attention pick outside the opened positions")?;
                let q = layout.q_values(&leaf.mixing)?;
                let head = pick.head_draw % u32::try_from(layout.op.n_head)?;
                let context = context_positions(&layout.op, pick.position);
                let values =
                    head_partials(&layout.op, head as usize, &q, &attended(&rows, &context))?;
                partials.push(AttentionPartials::new(pick, head, &values));
                opened.push(OpenedAttention {
                    position: pick.position,
                    layer,
                    context,
                });
            }
        }
        let mut state = self.lock();
        if let Some(turn) = turn_mut(&mut state.committed, receipt_id) {
            turn.pin.opened(Instant::now(), !opened.is_empty());
            turn.attention.extend(opened);
        }
        drop(state);
        Ok(partials)
    }

    /// The cell that held each position in `positions` for the turn, with
    /// its digest: a prefix cell below the turn, else the one the turn
    /// wrote last.
    fn context_cells(
        &self,
        receipt_id: &str,
        positions: impl IntoIterator<Item = u32>,
    ) -> Result<Vec<PrefixCell>> {
        let state = self.lock();
        let turn = opening_turn(&state.committed, receipt_id)?;
        let written = written_cells(turn);
        positions
            .into_iter()
            .map(|position| {
                if let Some(cell) = turn.prefix.get(position as usize) {
                    return Ok(*cell);
                }
                let index = position
                    .checked_sub(turn.position_start)
                    .context("a position between the prefix and the turn")?;
                let digest = turn
                    .cells
                    .get(index as usize)
                    .with_context(|| format!("position {position} is after the turn"))?
                    .digest;
                let cell = *written
                    .get(&position)
                    .with_context(|| format!("the turn wrote no cell for position {position}"))?;
                Ok(PrefixCell {
                    pos: i32::try_from(position)?,
                    stream: state.cells.streams().of(turn.seq),
                    cell,
                    digest,
                })
            })
            .collect()
    }

    /// Opens the rows of the blocks the client picked in this stage's
    /// layers. Each pick must be one this stage opened partials for, and
    /// name at most [`BLOCKS_PER_PICK`] of its blocks.
    pub fn open_rows(
        &self,
        challenge: &BlockChallenge,
        replayer: &dyn Replayer,
    ) -> Result<StageRows> {
        let wanted = self.requested_rows(challenge)?;
        let (prefix_len, prefix_tree) = {
            let state = self.lock();
            let turn = opening_turn(&state.committed, &challenge.receipt_id)?;
            let tree = if turn.prefix.is_empty() {
                None
            } else {
                Some(MerkleTree::new(
                    turn.prefix.iter().map(prefix_leaf).collect(),
                )?)
            };
            (turn.prefix.len() as u32, tree)
        };
        // a prefix cell's digest is over every layer of the stage, so its
        // rows are read once for all the layers the challenge names
        let prefix_positions: BTreeSet<u32> = wanted
            .values()
            .flatten()
            .copied()
            .filter(|&p| p < prefix_len)
            .collect();
        let prefix = self.context_cells(&challenge.receipt_id, prefix_positions)?;
        self.check_cells(&prefix)?;
        let prefix_trees = self.prefix_trees(replayer, &prefix)?;
        let mut rows = Vec::new();
        for (layer, positions) in wanted {
            let cells = self.context_cells(&challenge.receipt_id, positions)?;
            self.check_cells(&cells)?;
            let stored = replayer.read_rows(layer, &cells)?;
            let index = u64::from(layer - self.layers.start);
            for (i, cell) in cells.iter().enumerate() {
                let position = u32::try_from(cell.pos)?;
                let opened: Vec<TensorOpening> = stored
                    .iter()
                    .enumerate()
                    .map(|(index, rows)| {
                        TensorOpening::new(
                            MixingKind::Stored,
                            index as u32,
                            Some((rows.ggml_type, rows.row(i))),
                        )
                    })
                    .collect();
                let proof = match (&prefix_tree, prefix_trees.get(&position)) {
                    (Some(tree), Some(cell_tree)) => RowProof::Prefix {
                        cell: cell.cell,
                        cell_path: encode_path(&cell_tree.path(index)?),
                        prefix_path: encode_path(&tree.path(u64::from(position))?),
                    },
                    _ => self.turn_proof(&challenge.receipt_id, position, layer, &opened)?,
                };
                rows.push(RowOpening {
                    position,
                    layer,
                    stored: opened,
                    proof,
                });
            }
        }
        self.release_pin(&challenge.receipt_id);
        Ok(StageRows {
            stage_index: self.stage_index,
            rows,
        })
    }

    /// The positions whose rows the challenge asks for, by layer.
    fn requested_rows(&self, challenge: &BlockChallenge) -> Result<BTreeMap<u32, Vec<u32>>> {
        let state = self.lock();
        let turn = opening_turn(&state.committed, &challenge.receipt_id)?;
        let mut wanted: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
        for pick in challenge
            .picks
            .iter()
            .filter(|p| self.layers.contains(&p.layer))
        {
            let opened = turn
                .attention
                .iter()
                .find(|a| a.position == pick.position && a.layer == pick.layer)
                .with_context(|| {
                    format!(
                        "no partials were opened at position {}, layer {}",
                        pick.position, pick.layer
                    )
                })?;
            let distinct: BTreeSet<u32> = pick.blocks.iter().copied().collect();
            ensure!(
                distinct.len() == pick.blocks.len() && distinct.len() <= BLOCKS_PER_PICK,
                "at most {BLOCKS_PER_PICK} distinct blocks per pick"
            );
            let n_blocks = blocks(opened.context.len());
            for &block in &distinct {
                ensure!(
                    (block as usize) < n_blocks,
                    "block {block} is past the context"
                );
                wanted
                    .entry(pick.layer)
                    .or_default()
                    .extend(block_positions(&opened.context, block));
            }
        }
        Ok(wanted
            .into_iter()
            .map(|(layer, positions)| (layer, positions.into_iter().collect()))
            .collect())
    }

    /// The tree over the stored leaf of every layer of the stage in each
    /// prefix cell, by position, checked against the digest the turn read.
    fn prefix_trees(
        &self,
        replayer: &dyn Replayer,
        prefix: &[PrefixCell],
    ) -> Result<BTreeMap<u32, MerkleTree>> {
        if prefix.is_empty() {
            return Ok(BTreeMap::new());
        }
        let mut leaves: Vec<Vec<Hash>> = vec![Vec::new(); prefix.len()];
        for l in self.layers.clone() {
            let stored = replayer.read_rows(l, prefix)?;
            for (i, cell) in prefix.iter().enumerate() {
                let digests = stored_digests(&cell_openings(&stored, i))?;
                leaves[i].push(stored_leaf(u32::try_from(cell.pos)?, l, &digests));
            }
        }
        prefix
            .iter()
            .zip(leaves)
            .map(|(cell, leaves)| {
                // a cell's digest is the root of this tree
                let tree = MerkleTree::new(leaves)?;
                if tree.root() != cell.digest {
                    bail!(
                        "the cell at position {} no longer holds the rows the turn read",
                        cell.pos
                    );
                }
                Ok((u32::try_from(cell.pos)?, tree))
            })
            .collect()
    }

    /// The proof of a stored row the turn computed: the rest of its mixing
    /// leaf and the leaf's path.
    fn turn_proof(
        &self,
        receipt_id: &str,
        position: u32,
        layer: u32,
        stored: &[TensorOpening],
    ) -> Result<RowProof> {
        let state = self.lock();
        let turn = opening_turn(&state.committed, receipt_id)?;
        let rest = turn
            .cells
            .get((position - turn.position_start) as usize)
            .and_then(|c| c.rests.get((layer - self.layers.start) as usize))
            .context("no mixing leaf at the position")?;
        let slots = slots_per_position(self.layers.start, self.layers.end);
        let index = leaf_index(
            position,
            mixing_slot(layer, self.layers.start),
            turn.position_start,
            slots,
        );
        let leaf = mixing_leaf_from(
            position,
            layer,
            rest,
            &stored_leaf(position, layer, &stored_digests(stored)?),
        );
        if turn.tree.leaf(index) != Some(leaf) {
            bail!("the cell of position {position} no longer holds the rows the turn wrote");
        }
        Ok(RowProof::Turn {
            rest: hex::encode(rest),
            path: encode_path(&turn.tree.path(index)?),
        })
    }
}

/// The K and V rows of the cells, by position from 0.
fn read_context(
    replayer: &dyn Replayer,
    layer: u32,
    cells: &[PrefixCell],
    layout: &AttentionLayout,
) -> Result<Vec<ContextRow>> {
    let stored = replayer.read_rows(layer, cells)?;
    cells
        .iter()
        .enumerate()
        .map(|(i, cell)| {
            let rows: Vec<(i32, &[u8])> = stored.iter().map(|s| (s.ggml_type, s.row(i))).collect();
            layout.context_row(u32::try_from(cell.pos)?, &rows)
        })
        .collect()
}

/// The stored rows of cell `i`, as openings.
fn cell_openings(stored: &[StoredRows], i: usize) -> Vec<TensorOpening> {
    stored
        .iter()
        .enumerate()
        .map(|(index, rows)| {
            TensorOpening::new(
                MixingKind::Stored,
                index as u32,
                Some((rows.ggml_type, rows.row(i))),
            )
        })
        .collect()
}

/// The rows of the attended positions: a slice of `rows` when the
/// positions are consecutive, as every attention window keeps them.
fn attended<'a>(rows: &'a [ContextRow], context: &[u32]) -> Cow<'a, [ContextRow]> {
    match (context.first(), context.last()) {
        (Some(&lo), Some(&hi)) if (hi - lo) as usize + 1 == context.len() => {
            Cow::Borrowed(&rows[lo as usize..=hi as usize])
        }
        _ => Cow::Owned(context.iter().map(|&p| rows[p as usize].clone()).collect()),
    }
}

/// The cell each position of the turn's sequence was last written to.
pub(super) fn written_cells(turn: &CommittedTurn) -> BTreeMap<u32, u32> {
    let mut cells = BTreeMap::new();
    for event in &turn.events {
        match event {
            Event::Ubatch(record) => {
                for ((&seq, &pos), &cell) in record.seq.iter().zip(&record.pos).zip(&record.cell) {
                    if seq == turn.seq
                        && let Ok(pos) = u32::try_from(pos)
                    {
                        cells.insert(pos, cell);
                    }
                }
            }
            Event::SeqRm { p0, p1 } => {
                let start = u32::try_from((*p0).max(0)).unwrap_or(0);
                let end = u32::try_from(*p1).unwrap_or(u32::MAX);
                cells.retain(|pos, _| !(start..end).contains(pos));
            }
        }
    }
    cells
}
