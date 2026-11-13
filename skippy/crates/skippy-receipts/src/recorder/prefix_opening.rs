//! Opens the picked positions of a turn's cached prefix.
//!
//! A turn did not compute its prefix, so a replay of the turn does not
//! recompute it. Each cell of the stage's cache keeps the ubatch that wrote
//! it ([`super::cells`]). For a picked position of the prefix the stage
//! replays that one ubatch, from a copy of the prefix cells below it, which
//! are the cells the ubatch ran on. The replay is checked against the
//! leaves the stage committed when the ubatch ran, like a replay of the
//! turn against the turn's tree.
//!
//! Another sequence may have computed the prefix: a cache sequence, or an
//! earlier request whose cells the turn's sequence shares. The replay runs
//! in the turn's sequence all the same, so the ubatch's columns of the
//! writer are renamed to it ([`lineage_columns`]).
//!
//! The leaves of a prefix position are in no tree of the turn. The stage
//! opens them all, by hash, with the leaves of the rows the cell stores and
//! the position's path in the prefix tree the receipt commits
//! ([`PrefixOpening`]).

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, Result};

use super::cells::{CellMirror, Lineage, TurnPrefix, prefix_leaves};
use super::stage_opening::{Replay, replay_columns, seed_for};
use super::{CommittedTurn, PrefixCell, Recorder, ReplayStep, Replayer, UbatchRecord};
use crate::commitment::{Hash, MerkleTree, encode_path};
use crate::wire::PrefixOpening;

/// The picked positions of a turn's cached prefix: the turn's prefix, and
/// by position the leaves the stage committed when it computed it.
pub(super) struct PrefixPlan {
    prefix: TurnPrefix,
    picked: BTreeMap<u32, Arc<[Hash]>>,
}

impl PrefixPlan {
    /// The plan for the positions `lineage` holds, or `None` for none.
    pub(super) fn of(turn: &CommittedTurn, lineage: &BTreeMap<u32, Lineage>) -> Option<Self> {
        (!lineage.is_empty()).then(|| Self {
            prefix: turn.prefix.clone(),
            picked: lineage
                .iter()
                .map(|(&position, found)| (position, Arc::clone(&found.leaves)))
                .collect(),
        })
    }
}

impl Recorder {
    /// How each of `positions` below the turn was computed, by position. A
    /// cell that no longer holds the position, or whose ubatch the stage
    /// dropped, is a [`crate::wire::CannotOpen`].
    pub(super) fn lineage(
        &self,
        turn: &CommittedTurn,
        cells: &CellMirror,
        positions: &[u32],
    ) -> Result<BTreeMap<u32, Lineage>> {
        positions
            .iter()
            .filter(|&&position| position < turn.position_start)
            .map(|&position| {
                let at = position as usize;
                let (cell, digest) = turn
                    .prefix
                    .cells
                    .get(at)
                    .zip(turn.prefix.positions.get(at))
                    .with_context(|| format!("position {position} is outside the prefix"))?;
                let found = cells
                    .lineage(cell, digest)
                    .map_err(|record| self.lost_cell(cell, record))?;
                Ok((position, found))
            })
            .collect()
    }

    /// What the stage committed at each picked position of the prefix: its
    /// leaves, the leaves of the rows its cell stores, read from the cache,
    /// and its path in the prefix tree.
    pub(super) fn open_prefix(
        &self,
        plan: Option<&PrefixPlan>,
        replayer: &dyn Replayer,
    ) -> Result<Vec<PrefixOpening>> {
        let Some(PrefixPlan { prefix, picked }) = plan else {
            return Ok(Vec::new());
        };
        let cells: Vec<PrefixCell> = picked
            .keys()
            .map(|&position| prefix.cells[position as usize])
            .collect();
        self.check_cells(&cells)?;
        let stored = self.prefix_trees(replayer, &cells)?;
        let tree = MerkleTree::new(prefix_leaves(prefix))?;
        picked
            .iter()
            .zip(&cells)
            .map(|((&position, leaves), cell)| {
                let stored = stored
                    .get(&position)
                    .context("the rows of a picked prefix cell were not read")?;
                Ok(PrefixOpening {
                    position,
                    cell: cell.cell,
                    leaves: leaves.iter().map(hex::encode).collect(),
                    stored: (0..stored.leaf_count())
                        .filter_map(|layer| stored.leaf(layer))
                        .map(hex::encode)
                        .collect(),
                    path: encode_path(&tree.path(u64::from(position))?),
                })
            })
            .collect()
    }
}

/// The replays of the picked positions of `turn`'s prefix: for each ubatch
/// that wrote one, that ubatch alone, from the prefix cells below it.
/// Positions one ubatch wrote share a replay.
pub(super) fn lineage_replays(
    turn: &CommittedTurn,
    lineage: &BTreeMap<u32, Lineage>,
) -> Vec<Replay> {
    let mut by_ubatch: Vec<(&Lineage, Vec<u32>)> = Vec::new();
    for (&position, found) in lineage {
        let same = |other: &Lineage| {
            Arc::ptr_eq(&other.record, &found.record) && other.writer() == found.writer()
        };
        match by_ubatch.iter_mut().find(|(other, _)| same(other)) {
            Some((_, positions)) => positions.push(position),
            None => by_ubatch.push((found, vec![position])),
        }
    }
    by_ubatch
        .into_iter()
        .map(|(found, capture)| {
            let steps = vec![ReplayStep::Decode {
                record: lineage_columns(&found.record, found.writer(), turn.seq),
                capture,
                mixing: Vec::new(),
            }];
            let below = usize::try_from(found.first).unwrap_or(0);
            let cells = turn.prefix.cells[..below.min(turn.prefix.len())].to_vec();
            let seed = (!cells.is_empty()).then(|| seed_for(cells, &steps, None));
            Replay { seed, steps }
        })
        .collect()
}

/// A ubatch that computed positions for `writer`, as `seq` replays it: the
/// columns of `writer` become columns of `seq`, at their recorded
/// positions and cells, and every other column a dummy. The prefix of
/// `seq` shares the cells `writer` wrote, so a replay of `seq` from the
/// cells below the ubatch computes what `writer` did.
pub(super) fn lineage_columns(
    record: &Arc<UbatchRecord>,
    writer: i32,
    seq: i32,
) -> Arc<UbatchRecord> {
    if writer == seq {
        return replay_columns(record, seq);
    }
    let mut renamed = UbatchRecord::clone(record);
    for s in &mut renamed.seq {
        // a column the ubatch computed for `seq` itself is another
        // request's, and must not join the replayed sequence
        *s = match *s {
            s if s == writer => seq,
            s if s == seq => writer,
            s => s,
        };
    }
    replay_columns(&Arc::new(renamed), seq)
}
