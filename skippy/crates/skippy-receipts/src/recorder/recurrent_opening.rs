//! A stage's side of the recurrent check: the ubatch that computed an
//! attention pick in a layer whose token mixing reads a recurrent state
//! (the gated delta net, Mamba's scan, a short convolution).
//!
//! Such a layer has no KV rows to open: each ubatch continues one state per
//! sequence, which it reads before its tokens and writes after them. The
//! stage commits the state each ubatch read in the mixing leaf of its first
//! position. For the first such pick of a challenge, it opens the mixing
//! leaves of the pick's whole ubatch, and of the position after it, whose
//! state is the one the ubatch left. The replay that opens the picked
//! layers reports them, mixing values alone.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use anyhow::{Context, Result, ensure};

use super::CommittedTurn;
use super::event_log::{Event, Written};
use super::stage_opening::{Divergence, ReplayPlan, Replayed, computing_step};
use crate::attention::opening::AttentionPick;
use crate::commitment::{mixing_leaf, mixing_slot};
use crate::wire::{MixingLeafOpening, RecurrentOpening, mixing_bytes, mixing_values};

/// The ubatch an attention pick was computed in, on a layer that reads a
/// recurrent state.
pub(super) struct RegionPlan {
    pub position: u32,
    pub layer: u32,
    /// The ubatch's positions of the turn's sequence.
    pub start: u32,
    pub end: u32,
    /// The turn goes on past the ubatch, so the leaf at `end` holds the
    /// state the ubatch left.
    pub continues: bool,
    /// By replay step, the positions whose mixing values the opening shows.
    pub mixing: BTreeMap<usize, Vec<u32>>,
}

impl RegionPlan {
    /// Every position the opening shows a mixing leaf of.
    pub fn positions(&self) -> Range<u32> {
        self.start..self.end + u32::from(self.continues)
    }
}

/// The region of the first attention pick of `picks` in `layers` on a layer
/// of `state_layers`, if there is one.
pub(super) fn region_plan(
    turn: &CommittedTurn,
    written: &BTreeMap<u32, Written>,
    picks: &[AttentionPick],
    layers: &Range<u32>,
    state_layers: &BTreeSet<u32>,
) -> Result<Option<RegionPlan>> {
    let Some(pick) = picks
        .iter()
        .find(|p| layers.contains(&p.layer) && state_layers.contains(&p.layer))
    else {
        return Ok(None);
    };
    let step = computing_step(written, pick.position)?;
    let Event::Ubatch(record) = &turn.events[step] else {
        unreachable!("a computing step is a ubatch");
    };
    let positions = record
        .seq
        .iter()
        .zip(&record.pos)
        .filter(|(seq, _)| **seq == turn.seq)
        .map(|(_, &pos)| u32::try_from(pos))
        .collect::<Result<BTreeSet<u32>, _>>()?;
    let (start, end) = match (positions.first(), positions.last()) {
        (Some(&first), Some(&last)) => (first, last + 1),
        _ => unreachable!("the ubatch computed the pick"),
    };
    ensure!(
        (end - start) as usize == positions.len(),
        "the ubatch that computed position {} skips positions",
        pick.position
    );
    ensure!(
        start >= turn.position_start,
        "the ubatch that computed position {} began before the turn",
        pick.position
    );
    let mut mixing = BTreeMap::new();
    mixing.insert(step, (start..end).collect::<Vec<u32>>());
    let continues = end < turn.position_end;
    if continues {
        mixing
            .entry(computing_step(written, end)?)
            .or_insert_with(Vec::new)
            .push(end);
    }
    Ok(Some(RegionPlan {
        position: pick.position,
        layer: pick.layer,
        start,
        end,
        continues,
        mixing,
    }))
}

/// The opening of the plan's region, checked against the committed leaves.
pub(super) fn open_region(
    plan: &ReplayPlan,
    replayed: &Replayed,
    layer_start: u32,
) -> Result<Vec<RecurrentOpening>> {
    let Some(region) = &plan.region else {
        return Ok(Vec::new());
    };
    let layer = region.layer;
    let leaves = region
        .positions()
        .map(|position| {
            let values = replayed.get(&(position, layer)).with_context(|| {
                format!("replay did not report the mixing of position {position}, layer {layer}")
            })?;
            let mixing = values.mixing_openings();
            let decoded = mixing_bytes(&mixing)?;
            let committed = plan.committed(position, mixing_slot(layer, layer_start))?;
            if mixing_leaf(position, layer, &mixing_values(&decoded)) != committed.hash {
                return Err(Divergence(format!(
                    "replay diverged from the committed mixing leaf at position {position}, \
                     layer {layer}"
                ))
                .into());
            }
            Ok(MixingLeafOpening {
                position,
                mixing,
                path: committed.path.clone(),
            })
        })
        .collect::<Result<_>>()?;
    Ok(vec![RecurrentOpening {
        position: region.position,
        layer,
        start: region.start,
        end: region.end,
        leaves,
    }])
}
