//! A replay that starts from the turn's own cells, not only its prefix.
//!
//! After a turn, the stage's KV cache still holds most of the cells the turn
//! wrote; serving may drop the last ones, past the prefix cache's stride. A
//! replay can copy the cells still held along with the prefix and compute
//! only the ubatches that hold a captured position before the first dropped
//! cell, and every ubatch from that one on. An opening then costs a few
//! ubatches, not a prefill of the whole turn, so a long turn opens within
//! the challenge window.
//!
//! The node runs its own replay, so this is no weaker for the client: the
//! client checks the opened values against leaves committed before the
//! challenge, however the node recomputed them. The copied cells are checked
//! against the turn's committed leaves, so an honest replay reproduces them.
//!
//! The copy equals what the skipped ubatches would write only when the turn
//! wrote its positions in increasing order, removed only ranges of them (a
//! rejected draft removes the tail of its verify window), and nothing the
//! replay computes overwrites a copied cell first. Any other turn replays
//! every ubatch, as before.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result};

use super::CommittedTurn;
use super::cells::{CellMirror, PrefixCell};
use super::event_log::{Event, removed};
use crate::commitment::Hash;

/// One replay of consecutive events of the turn, from the turn's cells
/// that the events before them wrote.
///
/// A replay cannot copy the cells of positions after the ones it computes:
/// llama.cpp requires a sequence's new positions to follow the last one in
/// its cache. So events that are not consecutive replay separately, each
/// from the cells before it.
pub(super) struct Segment {
    /// The turn's cells to copy, besides the prefix.
    pub cells: Vec<PrefixCell>,
    /// The events to replay, consecutive and in order.
    pub events: Vec<usize>,
}

/// The replays that compute the `computed` events from the turn's own cells,
/// the last of which is `last`. `None` when the turn must be replayed in
/// full from its prefix.
pub(super) fn segments(
    turn: &CommittedTurn,
    computed: &BTreeMap<usize, Vec<u32>>,
    last: usize,
    mirror: &CellMirror,
) -> Result<Option<Vec<Segment>>> {
    let Some(written) = held_cells(turn) else {
        return Ok(None);
    };
    let mut cells = Vec::with_capacity(written.len());
    let mut first_lost = last + 1;
    for (&pos, &(cell, event)) in &written {
        let cell = PrefixCell {
            pos,
            stream: mirror.streams().of(turn.seq),
            cell,
            digest: committed_digest(turn, pos)?,
        };
        if mirror.holds(&cell) {
            cells.push((cell, event));
        } else {
            first_lost = first_lost.min(event);
        }
    }
    if first_lost == 0 {
        return Ok(None);
    }
    let events: Vec<usize> = computed
        .keys()
        .copied()
        .filter(|&event| event < first_lost)
        .chain(first_lost..=last)
        .collect();
    let mut segments: Vec<Segment> = Vec::new();
    for event in events {
        match segments.last_mut() {
            Some(segment) if segment.events.last() == Some(&(event - 1)) => {
                segment.events.push(event);
            }
            _ => segments.push(Segment {
                // the cells written before the segment, all still held
                cells: cells
                    .iter()
                    .filter(|&&(_, written)| written < event)
                    .map(|&(cell, _)| cell)
                    .collect(),
                events: vec![event],
            }),
        }
    }
    if segments.iter().any(|s| overwrites_a_seeded_cell(turn, s)) {
        return Ok(None);
    }
    Ok(Some(segments))
}

/// The digest of the cell the turn wrote `pos` to.
fn committed_digest(turn: &CommittedTurn, pos: i32) -> Result<Hash> {
    let index = u32::try_from(pos)?
        .checked_sub(turn.position_start)
        .context("a position before the turn")?;
    turn.cells
        .get(index as usize)
        .map(|cell| cell.digest)
        .context("a position after the turn")
}

/// By position, the cell that holds it after the turn and the event that
/// wrote it last, when every ubatch of the turn wrote positions after the
/// last one held, each to a cell no held position used. A removal drops a
/// range of positions: a rejected draft drops the tail of its verify
/// window, which the next window writes again.
///
/// A copy of the cells written before an event is then what that event
/// read: a later removal below the event's first position either leaves a
/// position of the turn uncomputed, so the turn is not committed, or makes
/// the turn write the event's positions again, so they are not its own.
fn held_cells(turn: &CommittedTurn) -> Option<BTreeMap<i32, (u32, usize)>> {
    let mut held: BTreeMap<i32, (u32, usize)> = BTreeMap::new();
    let mut cells = BTreeSet::new();
    for (index, event) in turn.events.iter().enumerate() {
        let record = match event {
            Event::Ubatch(record) => record,
            Event::SeqRm { p0, p1 } => {
                let dropped: Vec<i32> = held.range(removed(*p0, *p1)).map(|(&p, _)| p).collect();
                for pos in dropped {
                    if let Some((cell, _)) = held.remove(&pos) {
                        cells.remove(&cell);
                    }
                }
                continue;
            }
        };
        let last = held.keys().next_back().copied();
        for col in (0..record.seq.len()).filter(|&col| record.seq[col] == turn.seq) {
            let (pos, cell) = (record.pos[col], record.cell[col]);
            if last.is_some_and(|last| pos <= last)
                || !u32::try_from(pos).is_ok_and(|p| p >= turn.position_start)
                || held.insert(pos, (cell, index)).is_some()
                || !cells.insert(cell)
            {
                return None;
            }
        }
    }
    // drafts past the turn's end, which the turn rejected as it stopped,
    // were written last, so no replayed ubatch read them
    let end = i32::try_from(turn.position_end).ok()?;
    held.split_off(&end);
    Some(held)
}

/// Whether a replayed ubatch of `segment` places another sequence's column,
/// which a replay fills with a dummy, in a cell the segment copies.
fn overwrites_a_seeded_cell(turn: &CommittedTurn, segment: &Segment) -> bool {
    let copied: BTreeSet<u32> = segment.cells.iter().map(|c| c.cell).collect();
    segment
        .events
        .iter()
        .filter_map(|&index| turn.events[index].ubatch())
        .any(|record| {
            (0..record.seq.len())
                .any(|col| record.seq[col] != turn.seq && copied.contains(&record.cell[col]))
        })
}
