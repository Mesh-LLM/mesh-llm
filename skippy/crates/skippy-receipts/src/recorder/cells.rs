//! A mirror of the stage's KV cells: which positions of which sequences each
//! cell holds, what the stage committed when it wrote the cell, and the
//! ubatch that wrote it.
//!
//! A turn attends to cells written before it: by the sequence's earlier turns,
//! or by another request whose cached prefix the sequence shares (`seq_cp`
//! shares cells without copying them). A replay of the turn starts from a
//! copy of those cells instead of replaying how they were written, and the
//! receipt binds them by their digests ([`prefix_root`]). The mirror follows
//! the native cache from the ubatches that write cells and the memory events
//! that move them, so it knows which cells a sequence holds, and whether they
//! still hold what the turn attended to when it is opened.
//!
//! A client may challenge a position of that prefix. The stage then replays
//! the one ubatch that wrote the position, from a copy of the cells below
//! it. So every cell keeps its lineage: the leaves the stage committed for
//! its position, and the recorded ubatch ([`Lineage`]). The copy equals what
//! the ubatch read only when the cells below the ubatch are still the ones
//! it ran on. The mirror tracks that with a stamp per sequence
//! ([`lineage`](super::lineage)): a ubatch that appends to a sequence notes
//! the stamp it found and leaves a new one, and [`CellMirror::prefix`]
//! follows the stamps from position 0. A sequence the mirror cannot follow
//! (a shift, a state load, a write that is not an append, cells taken from
//! two sources) has no prefix until it is emptied.
//!
//! A cache that is not unified keeps a stream of cells per sequence, each
//! numbering its own cells from 0. There `seq_cp` copies the source's whole
//! stream into the destination's, rows and all.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};

use super::UbatchRecord;
use super::event_log::{removed, removes_all};
use super::lineage::{EMPTY, Runs, Shape, Stamp};
use crate::commitment::{self, Hash, MerkleTree};

/// A KV-cache operation the stage reports (`skippy_receipt_memory_op`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryOp {
    Clear,
    SeqRm,
    SeqCp,
    SeqKeep,
    /// `seq_add` or `seq_div`.
    SeqShift,
    StateRead,
    Compact,
}

impl MemoryOp {
    pub fn from_code(code: i32) -> Option<Self> {
        Some(match code {
            0 => Self::Clear,
            1 => Self::SeqRm,
            2 => Self::SeqCp,
            3 => Self::SeqKeep,
            4 => Self::SeqShift,
            5 => Self::StateRead,
            6 => Self::Compact,
            _ => return None,
        })
    }
}

/// How the stage's KV cache lays out its cells.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvStreams {
    /// One stream that every sequence shares.
    #[default]
    Shared,
    /// A stream per sequence, numbered as the sequence.
    PerSequence,
}

impl KvStreams {
    /// The stream that holds the cells of `seq`.
    pub fn of(self, seq: i32) -> u32 {
        match self {
            Self::Shared => 0,
            Self::PerSequence => u32::try_from(seq).unwrap_or(0),
        }
    }
}

/// A cell of the stage's KV cache: its stream and its index there.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct KvCell {
    pub stream: u32,
    pub cell: u32,
}

/// One cell a sequence holds below a turn's first position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixCell {
    pub pos: i32,
    /// The stream that holds the cell, which a turn's cells share, and the
    /// cell's index in it. The receipt binds the index alone.
    pub stream: u32,
    pub cell: u32,
    /// Digest of the leaves the stage committed when it computed the
    /// position, which bind the rows it stored in the cell
    /// ([`crate::commitment::cell_digest`]).
    pub digest: Hash,
}

/// What a stage committed in one column of a ubatch, kept with the cell
/// the column wrote.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ColumnLeaves {
    /// The digest of the rows the stage stored in the cell.
    pub cell: Hash,
    /// Every leaf the stage committed at the position, in slot order.
    pub leaves: Arc<[Hash]>,
    /// The digest of the leaves ([`commitment::position_digest`]).
    pub position: Hash,
}

/// The cells a turn attended to below its first position, from position 0:
/// what its replay starts from and what its receipt commits.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TurnPrefix {
    pub cells: Vec<PrefixCell>,
    /// By position, the digest of the leaves the stage committed there.
    pub positions: Vec<Hash>,
    /// By position, the token the stage computed there. Meaningful on the
    /// stage that holds layer 0.
    pub tokens: Vec<i32>,
}

impl TurnPrefix {
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Bytes the prefix holds.
    pub fn bytes(&self) -> usize {
        self.cells.len() * (size_of::<PrefixCell>() + size_of::<Hash>() + size_of::<i32>())
    }
}

/// How a prefix position was computed: the recorded ubatch that wrote its
/// cell, and the leaves the stage committed for it then.
#[derive(Clone, Debug)]
pub(crate) struct Lineage {
    pub record: Arc<UbatchRecord>,
    /// The column of the ubatch that computed the position.
    pub col: usize,
    /// The first position the ubatch computed for the column's sequence.
    pub first: i32,
    pub leaves: Arc<[Hash]>,
}

impl Lineage {
    /// The sequence the ubatch computed the position for.
    pub fn writer(&self) -> i32 {
        self.record.seq[self.col]
    }
}

#[derive(Clone, Debug)]
struct Cell {
    pos: i32,
    /// Empty once every sequence freed the cell: it keeps its rows until a
    /// ubatch writes it again, so a turn that used it can still open them.
    seqs: BTreeSet<i32>,
    /// `None` when the recorder could not hash what the stage computed.
    committed: Option<ColumnLeaves>,
    /// The run that wrote the cell ([`Runs`]), and its column there.
    run: u64,
    col: u32,
}

impl Cell {
    fn digest(&self) -> Option<Hash> {
        self.committed.as_ref().map(|committed| committed.cell)
    }
}

/// A cell: its stream and its index there.
type CellKey = (u32, u32);

#[derive(Default)]
pub(crate) struct CellMirror {
    streams: KvStreams,
    cells: HashMap<CellKey, Cell>,
    /// By sequence, its cells (in its stream) by position.
    by_seq: HashMap<i32, BTreeMap<i32, u32>>,
    /// The ubatches that wrote the cells.
    runs: Runs,
    /// What the mirror knows of each sequence's cells as a whole. A
    /// sequence without an entry holds no cells, unless `unknown`.
    shapes: HashMap<i32, Shape>,
    /// A sequence without an entry may hold cells the mirror does not know.
    unknown: bool,
}

/// What the mirror knows of a cell a turn read or wrote.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CellRecord {
    /// It still holds what the turn read or wrote.
    Held,
    /// A later ubatch wrote it.
    Overwritten,
    /// The mirror no longer knows what it holds.
    Unknown,
    /// It holds what the turn read, but the stage dropped the ubatch that
    /// computed it, so it cannot be replayed.
    Dropped,
}

impl CellMirror {
    pub fn streams(&self) -> KvStreams {
        self.streams
    }

    /// Sets how the cache lays out its cells; the mirror forgets every cell.
    pub fn set_streams(&mut self, streams: KvStreams) {
        self.forget_all(false);
        self.streams = streams;
    }

    /// Sets the bytes of later-stage input rows the mirror keeps for the
    /// lineage of its cells.
    pub fn set_input_budget(&mut self, bytes: usize) {
        self.runs.set_input_budget(bytes);
    }

    fn shape(&self, seq: i32) -> Shape {
        self.shapes.get(&seq).copied().unwrap_or(Shape {
            stamp: EMPTY,
            replayable: !self.unknown,
        })
    }

    fn held(&self, seq: i32) -> usize {
        self.by_seq.get(&seq).map_or(0, BTreeMap::len)
    }

    /// One past the last position `seq` holds.
    fn end(&self, seq: i32) -> i32 {
        self.by_seq
            .get(&seq)
            .and_then(|positions| positions.keys().next_back())
            .map_or(0, |last| last + 1)
    }

    /// The stamp of the cells `seq` holds below `end`, when a replay can
    /// follow the sequence and it holds them all.
    fn stamp_below(&self, seq: i32, end: i32) -> Option<Stamp> {
        if !self.shape(seq).replayable {
            return None;
        }
        if end <= 0 {
            return Some(EMPTY);
        }
        let cell = *self.by_seq.get(&seq)?.get(&(end - 1))?;
        let cell = self.cells.get(&(self.streams.of(seq), cell))?;
        self.runs.stamp_through(cell.run, end - 1)
    }

    /// A ubatch wrote each column's position into its cell; whatever the
    /// cells held before is gone. `committed` holds what the stage committed
    /// in each column, `None` where the recorder could not hash it.
    pub fn write_ubatch(
        &mut self,
        record: &Arc<UbatchRecord>,
        committed: Vec<Option<ColumnLeaves>>,
    ) {
        let mut columns: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
        for (col, &seq) in record.seq.iter().enumerate() {
            self.free((self.streams.of(seq), record.cell[col]));
            columns.entry(seq).or_default().push(col);
        }
        let mut committed = committed.into_iter().map(Some).collect::<Vec<_>>();
        for (seq, cols) in columns {
            let first = record.pos[cols[0]];
            let consecutive = cols
                .iter()
                .zip(first..)
                .all(|(&col, pos)| record.pos[col] == pos);
            let shape = self.shape(seq);
            // an append continues what the sequence holds; any other write
            // leaves cells a replay of one ubatch cannot start from
            let appends =
                shape.replayable && consecutive && usize::try_from(first) == Ok(self.held(seq));
            let last = first + i32::try_from(cols.len()).unwrap_or(i32::MAX) - 1;
            let (run, after) = self.runs.begin(
                record,
                cols.len(),
                first..last + 1,
                appends.then_some(shape.stamp),
            );
            for col in cols {
                let cell = Cell {
                    pos: record.pos[col],
                    seqs: BTreeSet::from([seq]),
                    committed: committed.get_mut(col).and_then(Option::take).flatten(),
                    run,
                    col: u32::try_from(col).unwrap_or(u32::MAX),
                };
                self.by_seq
                    .entry(seq)
                    .or_default()
                    .insert(cell.pos, record.cell[col]);
                self.runs.hold(run);
                self.cells
                    .insert((self.streams.of(seq), record.cell[col]), cell);
            }
            self.shapes.insert(
                seq,
                Shape {
                    stamp: after,
                    replayable: appends,
                },
            );
        }
        self.runs.enforce_budget();
    }

    /// Forgets the cell at `key`. A sequence that still held it loses a
    /// position a replay cannot account for.
    fn free(&mut self, key: CellKey) {
        if let Some(old) = self.cells.remove(&key) {
            for seq in old.seqs {
                self.unlink(seq, old.pos, key.1);
                self.shapes.insert(seq, Shape::UNREPLAYABLE);
            }
            self.runs.release(old.run);
        }
    }

    fn unlink(&mut self, seq: i32, pos: i32, cell: u32) {
        if let Some(positions) = self.by_seq.get_mut(&seq) {
            if positions.get(&pos) == Some(&cell) {
                positions.remove(&pos);
            }
            if positions.is_empty() {
                self.by_seq.remove(&seq);
            }
        }
    }

    /// Forgets every cell. Afterwards a sequence holds nothing, or, with
    /// `unknown`, cells the mirror does not know.
    fn forget_all(&mut self, unknown: bool) {
        self.cells.clear();
        self.by_seq.clear();
        self.runs.clear();
        self.shapes.clear();
        self.unknown = unknown;
    }

    /// `seq` (every sequence if negative) lost its cells in `[p0, p1)`. The
    /// cache frees a cell without changing its rows.
    pub fn seq_rm(&mut self, seq: i32, p0: i32, p1: i32) {
        let range = removed(p0, p1);
        let seqs: Vec<i32> = if seq < 0 {
            self.by_seq.keys().copied().collect()
        } else {
            vec![seq]
        };
        if seq < 0 && removes_all(p0, p1) {
            // every sequence is empty, also one the mirror did not know
            self.shapes.clear();
            self.unknown = false;
        }
        for s in seqs {
            let end = self.end(s);
            let known = self.shape(s).replayable;
            let shape = if removes_all(p0, p1) || (known && range.start <= 0 && range.end >= end) {
                Some(Shape {
                    stamp: EMPTY,
                    replayable: true,
                })
            } else if range.start >= end {
                None
            } else if range.end >= end {
                // the tail went: what is left is what the sequence held
                // when it reached the position below the cut
                Some(
                    self.stamp_below(s, range.start)
                        .map_or(Shape::UNREPLAYABLE, |stamp| Shape {
                            stamp,
                            replayable: true,
                        }),
                )
            } else {
                Some(Shape::UNREPLAYABLE)
            };
            if let Some(shape) = shape {
                self.shapes.insert(s, shape);
            }
            let Some(positions) = self.by_seq.get(&s) else {
                continue;
            };
            let removed: Vec<(i32, u32)> = positions
                .range(range.clone())
                .map(|(&p, &c)| (p, c))
                .collect();
            let stream = self.streams.of(s);
            for (pos, cell) in removed {
                self.unlink(s, pos, cell);
                if let Some(entry) = self.cells.get_mut(&(stream, cell)) {
                    entry.seqs.remove(&s);
                }
            }
        }
    }

    /// The cells of `src` in `[p0, p1)` are now also cells of `dst`. Across
    /// streams, the destination's stream becomes a copy of the source's.
    pub fn seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) {
        if src == dst {
            return;
        }
        let (from, to) = (self.streams.of(src), self.streams.of(dst));
        if from != to {
            self.copy_stream(src, dst);
            return;
        }
        let range = removed(p0, p1);
        // an empty destination that takes the source's cells from 0 holds
        // what the source held when it reached the last of them
        let fresh = self.held(dst) == 0 && self.shape(dst).replayable && range.start == 0;
        let stamp = fresh
            .then(|| self.stamp_below(src, range.end.min(self.end(src))))
            .flatten();
        let shared: Vec<(i32, u32)> = self
            .by_seq
            .get(&src)
            .map(|positions| positions.range(range).map(|(&p, &c)| (p, c)).collect())
            .unwrap_or_default();
        for (pos, cell) in shared {
            if let Some(entry) = self.cells.get_mut(&(from, cell)) {
                entry.seqs.insert(dst);
            }
            self.by_seq.entry(dst).or_default().insert(pos, cell);
        }
        self.shapes.insert(
            dst,
            stamp.map_or(Shape::UNREPLAYABLE, |stamp| Shape {
                stamp,
                replayable: true,
            }),
        );
    }

    /// The cache copies every row of the source's stream into the
    /// destination's, which then holds `dst` where the source held `src`.
    fn copy_stream(&mut self, src: i32, dst: i32) {
        let (from, to) = (self.streams.of(src), self.streams.of(dst));
        let shape = self.shape(src);
        let replaced: Vec<CellKey> = self.cells.keys().copied().filter(|k| k.0 == to).collect();
        for key in replaced {
            self.free(key);
        }
        let copied: Vec<(u32, Cell)> = self
            .cells
            .iter()
            .filter(|(k, _)| k.0 == from)
            .map(|(k, c)| {
                let seqs = if c.seqs.contains(&src) {
                    BTreeSet::from([dst])
                } else {
                    BTreeSet::new()
                };
                (k.1, Cell { seqs, ..c.clone() })
            })
            .collect();
        for (cell, entry) in copied {
            if entry.seqs.contains(&dst) {
                self.by_seq.entry(dst).or_default().insert(entry.pos, cell);
            }
            self.runs.hold(entry.run);
            self.cells.insert((to, cell), entry);
        }
        self.shapes.insert(dst, shape);
    }

    /// Every sequence but `seq` lost its cells.
    pub fn seq_keep(&mut self, seq: i32) {
        let others: Vec<i32> = self.by_seq.keys().copied().filter(|&s| s != seq).collect();
        for other in others {
            self.seq_rm(other, -1, -1);
        }
    }

    /// The positions of the cells `seq` holds in `[p0, p1)` changed, and the
    /// cache rewrites their rows to match: every sequence that shares one
    /// of those cells loses it. No other cell changes.
    fn forget_shifted(&mut self, seq: i32, p0: i32, p1: i32) {
        self.shapes.insert(seq, Shape::UNREPLAYABLE);
        let Some(positions) = self.by_seq.get(&seq) else {
            return;
        };
        let stream = self.streams.of(seq);
        let shifted: Vec<u32> = positions.range(removed(p0, p1)).map(|(_, &c)| c).collect();
        for cell in shifted {
            self.free((stream, cell));
        }
    }

    /// Something the recorder did not see wrote cells. The cache places a
    /// write in cells no sequence holds, so those are forgotten, and a cell
    /// some sequence still holds keeps its record. (Under a sliding
    /// attention window the cache may also take a cell whose position its
    /// sequence no longer attends to; no opening reads such a cell.)
    ///
    /// The write gave some sequence positions the mirror has no cell for.
    /// A prefix over them fails for the missing cell, and the next ubatch
    /// the recorder sees on that sequence is not an append.
    pub fn unseen_write(&mut self) {
        let freed: Vec<CellKey> = self
            .cells
            .iter()
            .filter(|(_, cell)| cell.seqs.is_empty())
            .map(|(&key, _)| key)
            .collect();
        for key in freed {
            self.free(key);
        }
    }

    /// Follows a KV-cache operation (`skippy_receipt_memory_op`), forgetting
    /// only the cells the operation can have changed:
    ///
    /// - a shift of one sequence changes the cells that sequence holds in
    ///   the shifted span;
    /// - a state load into one sequence replaces that sequence's cells with
    ///   rows written to cells no sequence held;
    /// - a clear may zero every row, compaction moves cells, and a shift or
    ///   state load that names no sequence touches them all, so each of
    ///   these does invalidate the whole cache.
    pub fn memory_op(&mut self, op: MemoryOp, seq: i32, seq_dst: i32, p0: i32, p1: i32) {
        match op {
            MemoryOp::SeqRm => self.seq_rm(seq, p0, p1),
            MemoryOp::SeqCp => self.seq_cp(seq, seq_dst, p0, p1),
            MemoryOp::SeqKeep => self.seq_keep(seq),
            MemoryOp::SeqShift if seq >= 0 => self.forget_shifted(seq, p0, p1),
            MemoryOp::StateRead if seq >= 0 => {
                self.seq_rm(seq, -1, -1);
                self.unseen_write();
                // the sequence holds the loaded rows, which nothing recorded
                self.shapes.insert(seq, Shape::UNREPLAYABLE);
            }
            MemoryOp::Clear => self.forget_all(false),
            MemoryOp::SeqShift | MemoryOp::StateRead | MemoryOp::Compact => self.forget_all(true),
        }
    }

    /// The cells `seq` holds below `end`, which must be every position from
    /// 0, each with what the stage committed for it and the ubatch that
    /// wrote it, and each written on top of the cells below it.
    pub fn prefix(&self, seq: i32, end: u32) -> Result<TurnPrefix> {
        let end = i32::try_from(end)?;
        let mut prefix = TurnPrefix::default();
        if end == 0 {
            return Ok(prefix);
        }
        ensure!(
            self.shape(seq).replayable,
            "the sequence's cells changed in a way a replay of one ubatch cannot follow"
        );
        let stream = self.streams.of(seq);
        let positions = self.by_seq.get(&seq);
        // the run and position of the cell below the one in hand
        let mut below: Option<(u64, i32)> = None;
        for pos in 0..end {
            let Some(&cell) = positions.and_then(|p| p.get(&pos)) else {
                bail!("the sequence holds no recorded cell at position {pos}");
            };
            let held = self.cells.get(&(stream, cell));
            let Some((held, committed)) = held.and_then(|c| Some((c, c.committed.as_ref()?)))
            else {
                bail!("the rows stored at position {pos} were not recorded");
            };
            let run = self
                .runs
                .get(held.run)
                .with_context(|| format!("nothing recorded the ubatch of position {pos}"))?;
            let Some(record) = &run.record else {
                bail!("the ubatch that wrote position {pos} is no longer held");
            };
            let found = match below {
                None => Some(EMPTY),
                Some((run, at)) => self.runs.stamp_through(run, at),
            };
            let follows = if pos == run.positions.start {
                run.before.is_some() && run.before == found
            } else {
                below.is_some_and(|(run, _)| run == held.run)
            };
            ensure!(
                follows && record.pos.get(held.col as usize) == Some(&pos),
                "position {pos} was not computed on top of the cells below it"
            );
            below = Some((held.run, pos));
            prefix.cells.push(PrefixCell {
                pos,
                stream,
                cell,
                digest: committed.cell,
            });
            prefix.positions.push(committed.position);
            prefix.tokens.push(record.token[held.col as usize]);
        }
        Ok(prefix)
    }

    /// Whether `cell` still holds what it held when the prefix was taken.
    pub fn holds(&self, cell: &PrefixCell) -> bool {
        self.record(cell) == CellRecord::Held
    }

    /// What the mirror knows of `cell` since a turn read or wrote it.
    pub fn record(&self, cell: &PrefixCell) -> CellRecord {
        match self.cells.get(&(cell.stream, cell.cell)) {
            Some(c) if c.pos == cell.pos && c.digest() == Some(cell.digest) => CellRecord::Held,
            Some(_) => CellRecord::Overwritten,
            None => CellRecord::Unknown,
        }
    }

    /// Fails unless every cell of `prefix` still holds what it held when
    /// the prefix was taken, with the ubatch that computed it.
    pub fn check_lineage(&self, prefix: &TurnPrefix) -> Result<()> {
        for (cell, position) in prefix.cells.iter().zip(&prefix.positions) {
            let run = match self.cells.get(&(cell.stream, cell.cell)) {
                Some(held)
                    if held.pos == cell.pos
                        && held.committed.as_ref().is_some_and(|committed| {
                            committed.cell == cell.digest && &committed.position == position
                        }) =>
                {
                    self.runs.get(held.run)
                }
                _ => bail!(
                    "the cell of position {} of the cached prefix was written again",
                    cell.pos
                ),
            };
            ensure!(
                run.is_some_and(|run| run.record.is_some()),
                "the ubatch that wrote position {} of the cached prefix is no longer held",
                cell.pos
            );
        }
        Ok(())
    }

    /// How `cell`, a prefix cell whose leaves hash to `position`, was
    /// computed. The error says what the mirror knows of a cell that no
    /// longer holds it, or whose ubatch the stage dropped.
    pub fn lineage(
        &self,
        cell: &PrefixCell,
        position: &Hash,
    ) -> std::result::Result<Lineage, CellRecord> {
        let held = match self.record(cell) {
            CellRecord::Held => &self.cells[&(cell.stream, cell.cell)],
            record => return Err(record),
        };
        let committed = held
            .committed
            .as_ref()
            .filter(|committed| &committed.position == position)
            .ok_or(CellRecord::Overwritten)?;
        let run = self.runs.get(held.run).ok_or(CellRecord::Unknown)?;
        Ok(Lineage {
            record: run.record.clone().ok_or(CellRecord::Dropped)?,
            col: held.col as usize,
            first: run.positions.start,
            leaves: committed.leaves.clone(),
        })
    }
}

/// The leaves of a turn's prefix, in position order.
pub(super) fn prefix_leaves(prefix: &TurnPrefix) -> Vec<Hash> {
    prefix
        .cells
        .iter()
        .zip(&prefix.positions)
        .map(|(cell, position)| prefix_leaf(cell, position))
        .collect()
}

/// The Merkle root over a turn's prefix cells, in position order, or `None`
/// for a turn that starts at position 0.
pub(crate) fn prefix_root(prefix: &TurnPrefix) -> Result<Option<Hash>> {
    if prefix.is_empty() {
        return Ok(None);
    }
    Ok(Some(MerkleTree::new(prefix_leaves(prefix))?.root()))
}

/// The leaf of one prefix cell ([`commitment::prefix_leaf`]). It binds the
/// cell's position, index and digest and the digest of the position's
/// leaves, and not the cell's stream.
pub(super) fn prefix_leaf(cell: &PrefixCell, position: &Hash) -> Hash {
    commitment::prefix_leaf(cell.pos, cell.cell, &cell.digest, position)
}

#[cfg(test)]
#[path = "cells_tests.rs"]
mod tests;
