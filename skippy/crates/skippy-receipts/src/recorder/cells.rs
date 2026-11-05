//! A mirror of the stage's KV cells: which positions of which sequences each
//! cell holds, and a digest of the rows the stage stored in it.
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
//! A cache that is not unified keeps a stream of cells per sequence, each
//! numbering its own cells from 0. There `seq_cp` copies the source's whole
//! stream into the destination's, rows and all.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;

use anyhow::{Result, bail};

use crate::commitment::{Hash, MerkleTree};

const PREFIX_DOMAIN: &[u8] = b"skippy-receipt-prefix-v1";

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
    /// ([`super::leaves::cell_digest`]).
    pub digest: Hash,
}

#[derive(Clone, Debug)]
struct Cell {
    pos: i32,
    /// Empty once every sequence freed the cell: it keeps its rows until a
    /// ubatch writes it again, so a turn that used it can still open them.
    seqs: BTreeSet<i32>,
    /// `None` when the recorder could not hash what the stage computed.
    digest: Option<Hash>,
}

/// A cell: its stream and its index there.
type CellKey = (u32, u32);

#[derive(Default)]
pub(crate) struct CellMirror {
    streams: KvStreams,
    cells: HashMap<CellKey, Cell>,
    /// By sequence, its cells (in its stream) by position.
    by_seq: HashMap<i32, BTreeMap<i32, u32>>,
}

/// Positions `[p0, p1)` with the native cache's conventions: `p0 < 0` starts
/// at 0 and `p1 < 0` has no end.
fn span(p0: i32, p1: i32) -> Range<i32> {
    p0.max(0)..if p1 < 0 { i32::MAX } else { p1 }
}

impl CellMirror {
    pub fn streams(&self) -> KvStreams {
        self.streams
    }

    /// Sets how the cache lays out its cells; the mirror forgets every cell.
    pub fn set_streams(&mut self, streams: KvStreams) {
        self.clear();
        self.streams = streams;
    }

    /// A ubatch wrote position `pos` of `seq` into `cell`; whatever the cell
    /// held before is gone.
    pub fn write(&mut self, cell: u32, pos: i32, seq: i32, digest: Option<Hash>) {
        let key = (self.streams.of(seq), cell);
        self.free(key);
        self.cells.insert(
            key,
            Cell {
                pos,
                seqs: BTreeSet::from([seq]),
                digest,
            },
        );
        self.by_seq.entry(seq).or_default().insert(pos, cell);
    }

    fn free(&mut self, key: CellKey) {
        if let Some(old) = self.cells.remove(&key) {
            for seq in old.seqs {
                self.unlink(seq, old.pos, key.1);
            }
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

    pub fn clear(&mut self) {
        self.cells.clear();
        self.by_seq.clear();
    }

    /// `seq` (every sequence if negative) lost its cells in `[p0, p1)`. The
    /// cache frees a cell without changing its rows.
    pub fn seq_rm(&mut self, seq: i32, p0: i32, p1: i32) {
        let range = span(p0, p1);
        let seqs: Vec<i32> = if seq < 0 {
            self.by_seq.keys().copied().collect()
        } else {
            vec![seq]
        };
        for s in seqs {
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
        let Some(positions) = self.by_seq.get(&src) else {
            return;
        };
        let shared: Vec<(i32, u32)> = positions
            .range(span(p0, p1))
            .map(|(&p, &c)| (p, c))
            .collect();
        for (pos, cell) in shared {
            if let Some(entry) = self.cells.get_mut(&(from, cell)) {
                entry.seqs.insert(dst);
            }
            self.by_seq.entry(dst).or_default().insert(pos, cell);
        }
    }

    /// The cache copies every row of the source's stream into the
    /// destination's, which then holds `dst` where the source held `src`.
    fn copy_stream(&mut self, src: i32, dst: i32) {
        let (from, to) = (self.streams.of(src), self.streams.of(dst));
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
            self.cells.insert((to, cell), entry);
        }
    }

    /// Every sequence but `seq` lost its cells.
    pub fn seq_keep(&mut self, seq: i32) {
        let others: Vec<i32> = self.by_seq.keys().copied().filter(|&s| s != seq).collect();
        for other in others {
            self.seq_rm(other, -1, -1);
        }
    }

    /// Follows a KV-cache operation (`skippy_receipt_memory_op`). After a
    /// position shift, a state load or compaction, cells hold what the
    /// recorder did not see, so it forgets every cell.
    pub fn memory_op(&mut self, op: MemoryOp, seq: i32, seq_dst: i32, p0: i32, p1: i32) {
        match op {
            MemoryOp::SeqRm => self.seq_rm(seq, p0, p1),
            MemoryOp::SeqCp => self.seq_cp(seq, seq_dst, p0, p1),
            MemoryOp::SeqKeep => self.seq_keep(seq),
            MemoryOp::Clear | MemoryOp::SeqShift | MemoryOp::StateRead | MemoryOp::Compact => {
                self.clear();
            }
        }
    }

    /// The cells `seq` holds below `end`, which must be every position from
    /// 0, each with a known digest.
    pub fn prefix(&self, seq: i32, end: u32) -> Result<Vec<PrefixCell>> {
        let end = i32::try_from(end)?;
        let stream = self.streams.of(seq);
        let positions = self.by_seq.get(&seq);
        let mut out = Vec::with_capacity(usize::try_from(end).unwrap_or(0));
        for pos in 0..end {
            let Some(&cell) = positions.and_then(|p| p.get(&pos)) else {
                bail!("the sequence holds no recorded cell at position {pos}");
            };
            let Some(digest) = self.cells.get(&(stream, cell)).and_then(|c| c.digest) else {
                bail!("the rows stored at position {pos} were not recorded");
            };
            out.push(PrefixCell {
                pos,
                stream,
                cell,
                digest,
            });
        }
        Ok(out)
    }

    /// Whether `cell` still holds what it held when the prefix was taken.
    pub fn holds(&self, cell: &PrefixCell) -> bool {
        self.cells
            .get(&(cell.stream, cell.cell))
            .is_some_and(|c| c.pos == cell.pos && c.digest == Some(cell.digest))
    }
}

/// The Merkle root over a turn's prefix cells, in position order, or `None`
/// for a turn that starts at position 0.
pub(crate) fn prefix_root(prefix: &[PrefixCell]) -> Result<Option<Hash>> {
    if prefix.is_empty() {
        return Ok(None);
    }
    let leaves = prefix.iter().map(prefix_leaf).collect();
    Ok(Some(MerkleTree::new(leaves)?.root()))
}

/// The leaf of one prefix cell: its position, its index in the cache's
/// stream, and the digest of what it stores.
pub(crate) fn prefix_leaf(cell: &PrefixCell) -> Hash {
    let mut h = blake3::Hasher::new();
    h.update(PREFIX_DOMAIN);
    h.update(&cell.pos.to_le_bytes());
    h.update(&cell.cell.to_le_bytes());
    h.update(&cell.digest);
    *h.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(n: u8) -> Option<Hash> {
        Some([n; 32])
    }

    #[test]
    fn a_shared_prefix_is_the_sources_cells() {
        let mut m = CellMirror::default();
        for pos in 0..4 {
            m.write(10 + pos as u32, pos, 7, digest(pos as u8));
        }
        m.seq_cp(7, 1, 0, 3);
        let prefix = m.prefix(1, 3).unwrap();
        assert_eq!(
            prefix.iter().map(|c| c.cell).collect::<Vec<_>>(),
            vec![10, 11, 12]
        );
        assert!(m.prefix(1, 4).is_err(), "position 3 was not shared");
        // the source losing its cells leaves the shared ones to the copy
        m.seq_rm(7, -1, -1);
        assert_eq!(m.prefix(1, 3).unwrap(), prefix);
        assert!(prefix.iter().all(|c| m.holds(c)));
        m.seq_rm(1, 2, -1);
        assert!(m.prefix(1, 3).is_err(), "the sequence lost position 2");
        assert!(m.holds(&prefix[2]), "a freed cell keeps its rows");
        m.write(12, 0, 4, digest(8));
        assert!(!m.holds(&prefix[2]), "until it is written again");
        m.write(11, 1, 3, digest(9));
        assert!(!m.holds(&prefix[1]), "a rewritten cell holds new rows");
    }

    #[test]
    fn a_rewritten_cell_leaves_every_sequence_that_shared_it() {
        let mut m = CellMirror::default();
        m.write(5, 0, 7, digest(1));
        m.seq_cp(7, 1, -1, -1);
        m.seq_rm(7, -1, -1);
        m.seq_rm(1, -1, -1);
        m.write(5, 0, 2, digest(2));
        assert!(m.prefix(1, 1).is_err());
        assert_eq!(m.prefix(2, 1).unwrap()[0].digest, [2; 32]);
    }

    #[test]
    fn unrecorded_rows_or_gaps_have_no_prefix() {
        let mut m = CellMirror::default();
        m.write(0, 0, 1, digest(1));
        m.write(1, 1, 1, None);
        m.write(3, 3, 1, digest(3));
        assert!(m.prefix(1, 1).is_ok());
        assert!(m.prefix(1, 2).is_err(), "unknown rows");
        m.write(1, 1, 1, digest(2));
        assert!(m.prefix(1, 4).is_err(), "position 2 is missing");
        m.seq_keep(2);
        assert!(m.prefix(1, 1).is_err());
    }

    #[test]
    fn the_prefix_root_binds_positions_cells_and_rows() {
        let cell = PrefixCell {
            pos: 0,
            stream: 0,
            cell: 4,
            digest: [1; 32],
        };
        let root = prefix_root(&[cell]).unwrap().unwrap();
        for other in [
            PrefixCell { pos: 1, ..cell },
            PrefixCell { cell: 5, ..cell },
            PrefixCell {
                digest: [2; 32],
                ..cell
            },
        ] {
            assert_ne!(prefix_root(&[other]).unwrap().unwrap(), root);
        }
        assert_eq!(prefix_root(&[]).unwrap(), None);
    }

    #[test]
    fn a_cache_per_sequence_numbers_cells_per_stream() {
        let mut m = CellMirror::default();
        m.set_streams(KvStreams::PerSequence);
        m.write(0, 0, 1, digest(1));
        m.write(0, 0, 2, digest(2));
        let one = m.prefix(1, 1).unwrap()[0];
        let two = m.prefix(2, 1).unwrap()[0];
        assert_eq!((one.stream, one.cell, two.stream, two.cell), (1, 0, 2, 0));
        assert!(m.holds(&one) && m.holds(&two), "the streams share no cell");
        // a freed cell of sequence 2 keeps its rows, which a copy takes along
        m.write(1, 1, 2, digest(3));
        m.seq_rm(2, 1, -1);
        m.seq_cp(2, 3, -1, -1);
        let copied = m.prefix(3, 1).unwrap()[0];
        assert_eq!((copied.stream, copied.cell, copied.digest), (3, 0, [2; 32]));
        assert!(
            m.prefix(3, 2).is_err(),
            "position 1 was freed before the copy"
        );
        let freed = PrefixCell {
            pos: 1,
            stream: 3,
            cell: 1,
            digest: [3; 32],
        };
        assert!(m.holds(&freed));
        // a copy replaces what the destination held
        m.seq_cp(1, 3, -1, -1);
        assert_eq!(m.prefix(3, 1).unwrap()[0].digest, [1; 32]);
        assert!(!m.holds(&copied) && !m.holds(&freed));
        assert!(m.holds(&two), "the source keeps its cells");
    }
}
