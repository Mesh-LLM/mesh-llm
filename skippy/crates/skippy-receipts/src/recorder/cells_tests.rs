use super::*;

fn hash(n: u8) -> Hash {
    [n; 32]
}

/// What a ubatch committed in a column whose cell digest is `n`.
fn committed(n: u8) -> Option<ColumnLeaves> {
    let leaves: Arc<[Hash]> = Arc::from(vec![hash(n), hash(n.wrapping_add(100))]);
    Some(ColumnLeaves {
        cell: hash(n),
        position: commitment::position_digest(&leaves),
        leaves,
    })
}

/// A ubatch of `columns`, each a sequence, a position and a cell; the token
/// at a position is 1000 more than the position.
fn record(columns: &[(i32, i32, u32)]) -> Arc<UbatchRecord> {
    Arc::new(UbatchRecord {
        token: columns.iter().map(|(_, pos, _)| 1000 + pos).collect(),
        seq: columns.iter().map(|(seq, _, _)| *seq).collect(),
        pos: columns.iter().map(|(_, pos, _)| *pos).collect(),
        output: vec![0; columns.len()],
        cell: columns.iter().map(|(_, _, cell)| *cell).collect(),
        n_kv: 64,
        input: None,
    })
}

/// Writes one ubatch whose column `i` committed digest `digests[i]`.
fn write(m: &mut CellMirror, columns: &[(i32, i32, u32)], digests: &[u8]) -> Arc<UbatchRecord> {
    let record = record(columns);
    m.write_ubatch(&record, digests.iter().map(|&n| committed(n)).collect());
    record
}

/// Writes positions `positions` of `seq` in one ubatch, to cells
/// `first_cell..`, each with its position as digest.
fn append(m: &mut CellMirror, seq: i32, positions: std::ops::Range<i32>, first_cell: u32) {
    let columns: Vec<(i32, i32, u32)> = positions
        .clone()
        .zip(first_cell..)
        .map(|(pos, cell)| (seq, pos, cell))
        .collect();
    let digests: Vec<u8> = positions.map(|pos| pos as u8).collect();
    write(m, &columns, &digests);
}

fn cells_of(prefix: &TurnPrefix) -> Vec<u32> {
    prefix.cells.iter().map(|c| c.cell).collect()
}

#[test]
fn a_shared_prefix_is_the_sources_cells() {
    let mut m = CellMirror::default();
    append(&mut m, 7, 0..4, 10);
    m.seq_cp(7, 1, 0, 3);
    let prefix = m.prefix(1, 3).unwrap();
    assert_eq!(cells_of(&prefix), vec![10, 11, 12]);
    assert_eq!(prefix.tokens, vec![1000, 1001, 1002]);
    assert!(m.prefix(1, 4).is_err(), "position 3 was not shared");
    // the source losing its cells leaves the shared ones to the copy
    m.seq_rm(7, -1, -1);
    assert_eq!(m.prefix(1, 3).unwrap(), prefix);
    assert!(prefix.cells.iter().all(|c| m.holds(c)));
    m.seq_rm(1, 2, -1);
    assert!(m.prefix(1, 3).is_err(), "the sequence lost position 2");
    assert!(m.holds(&prefix.cells[2]), "a freed cell keeps its rows");
    write(&mut m, &[(4, 0, 12)], &[8]);
    assert!(!m.holds(&prefix.cells[2]), "until it is written again");
    write(&mut m, &[(3, 0, 11)], &[9]);
    assert!(
        !m.holds(&prefix.cells[1]),
        "a rewritten cell holds new rows"
    );
}

#[test]
fn a_rewritten_cell_leaves_every_sequence_that_shared_it() {
    let mut m = CellMirror::default();
    write(&mut m, &[(7, 0, 5)], &[1]);
    m.seq_cp(7, 1, -1, -1);
    m.seq_rm(7, -1, -1);
    m.seq_rm(1, -1, -1);
    write(&mut m, &[(2, 0, 5)], &[2]);
    assert!(m.prefix(1, 1).is_err());
    assert_eq!(m.prefix(2, 1).unwrap().cells[0].digest, hash(2));
}

#[test]
fn unrecorded_rows_or_gaps_have_no_prefix() {
    let mut m = CellMirror::default();
    write(&mut m, &[(1, 0, 0)], &[1]);
    let unhashed = record(&[(1, 1, 1)]);
    m.write_ubatch(&unhashed, vec![None]);
    assert!(m.prefix(1, 1).is_ok());
    assert!(m.prefix(1, 2).is_err(), "unknown rows");
    // a position written past a gap is no append
    write(&mut m, &[(1, 3, 3)], &[3]);
    assert!(m.prefix(1, 1).is_err(), "the sequence has a gap");
    m.seq_keep(2);
    assert!(m.prefix(1, 1).is_err());
}

// A prefix position can be replayed from the cells below its ubatch only
// when those are the cells the ubatch ran on.
#[test]
fn a_prefix_follows_the_cells_each_ubatch_ran_on() {
    // two ubatches, the second on top of the first
    let mut m = CellMirror::default();
    append(&mut m, 1, 0..3, 0);
    append(&mut m, 1, 3..5, 3);
    let prefix = m.prefix(1, 5).unwrap();
    assert_eq!(cells_of(&prefix), vec![0, 1, 2, 3, 4]);
    let lineage = |m: &CellMirror, at: usize| m.lineage(&prefix.cells[at], &prefix.positions[at]);
    let (low, high) = (lineage(&m, 1).unwrap(), lineage(&m, 4).unwrap());
    assert_eq!((low.first, low.col, low.writer()), (0, 1, 1));
    assert_eq!((high.first, high.col), (3, 1));
    assert_eq!(high.record.pos, vec![3, 4]);

    // a tail cut inside a ubatch, written again: the prefix holds half of
    // the first ubatch, and the second ran on that half
    m.seq_rm(1, 2, -1);
    append(&mut m, 1, 2..4, 10);
    let cut = m.prefix(1, 4).unwrap();
    assert_eq!(cells_of(&cut), vec![0, 1, 10, 11]);
    assert_eq!(
        m.lineage(&cut.cells[3], &cut.positions[3]).unwrap().first,
        2
    );
    // the cells the cut freed keep their rows, but hold no position of it
    assert!(m.holds(&prefix.cells[2]));

    // a copy from 0 holds what the source held through the copied position
    m.seq_cp(1, 2, 0, 3);
    assert_eq!(cells_of(&m.prefix(2, 3).unwrap()), vec![0, 1, 10]);
    // and a ubatch appended to the copy ran on the copy's cells
    append(&mut m, 2, 3..4, 20);
    assert_eq!(cells_of(&m.prefix(2, 4).unwrap()), vec![0, 1, 10, 20]);
    // the source goes on by itself
    assert_eq!(cells_of(&m.prefix(1, 4).unwrap()), vec![0, 1, 10, 11]);
}

#[test]
fn cells_from_two_sources_have_no_prefix() {
    let mut m = CellMirror::default();
    append(&mut m, 1, 0..2, 0);
    append(&mut m, 2, 0..4, 10);
    // sequence 3 takes position 0 and 1 of one request, 2 and 3 of another
    m.seq_cp(1, 3, 0, 2);
    assert!(m.prefix(3, 2).is_ok());
    m.seq_cp(2, 3, 2, 4);
    let error = m.prefix(3, 4).unwrap_err();
    assert!(format!("{error:#}").contains("cannot follow"), "{error:#}");
    // until it is emptied and filled from one
    m.seq_rm(3, -1, -1);
    m.seq_cp(2, 3, 0, 4);
    assert_eq!(cells_of(&m.prefix(3, 4).unwrap()), vec![10, 11, 12, 13]);
}

#[test]
fn a_write_that_is_no_append_has_no_prefix() {
    // a lower position written after a higher one: the higher position's
    // ubatch did not run on the new cell
    let mut m = CellMirror::default();
    append(&mut m, 1, 0..3, 0);
    write(&mut m, &[(1, 1, 9)], &[9]);
    assert!(m.prefix(1, 3).is_err());
    // a removal from the middle leaves a gap
    let mut m = CellMirror::default();
    append(&mut m, 1, 0..4, 0);
    m.seq_rm(1, 1, 2);
    assert!(m.prefix(1, 1).is_err());
    // a ubatch whose positions skip one
    let mut m = CellMirror::default();
    write(&mut m, &[(1, 0, 0), (1, 2, 2)], &[0, 2]);
    assert!(m.prefix(1, 1).is_err());
    // each is forgotten when the sequence is emptied
    m.seq_rm(1, -1, -1);
    append(&mut m, 1, 0..2, 4);
    assert!(m.prefix(1, 2).is_ok());
}

#[test]
fn a_ubatch_of_several_sequences_is_a_run_of_each() {
    let mut m = CellMirror::default();
    append(&mut m, 1, 0..2, 0);
    // a decode step of two lanes in one ubatch
    let shared = write(&mut m, &[(2, 0, 10), (1, 2, 2), (2, 1, 11)], &[50, 2, 51]);
    let one = m.prefix(1, 3).unwrap();
    let two = m.prefix(2, 2).unwrap();
    assert_eq!(cells_of(&one), vec![0, 1, 2]);
    assert_eq!(cells_of(&two), vec![10, 11]);
    let lineage = m.lineage(&two.cells[1], &two.positions[1]).unwrap();
    assert!(Arc::ptr_eq(&lineage.record, &shared));
    assert_eq!((lineage.col, lineage.first, lineage.writer()), (2, 0, 2));
    let lineage = m.lineage(&one.cells[2], &one.positions[2]).unwrap();
    assert_eq!((lineage.col, lineage.first, lineage.writer()), (1, 2, 1));
}

#[test]
fn the_lineage_of_a_cell_is_gone_with_what_it_held() {
    let mut m = CellMirror::default();
    append(&mut m, 1, 0..2, 0);
    let prefix = m.prefix(1, 2).unwrap();
    let cell = prefix.cells[1];
    assert!(m.lineage(&cell, &prefix.positions[1]).is_ok());
    // other leaves than the turn read
    assert_eq!(
        m.lineage(&cell, &hash(77)).unwrap_err(),
        CellRecord::Overwritten
    );
    m.seq_rm(1, -1, -1);
    assert!(m.lineage(&cell, &prefix.positions[1]).is_ok(), "freed");
    write(&mut m, &[(5, 0, 1)], &[9]);
    assert_eq!(
        m.lineage(&cell, &prefix.positions[1]).unwrap_err(),
        CellRecord::Overwritten
    );
    m.memory_op(MemoryOp::Compact, -1, -1, -1, -1);
    assert_eq!(
        m.lineage(&cell, &prefix.positions[1]).unwrap_err(),
        CellRecord::Unknown
    );
}

// A stage after the first keeps a ubatch's input rows for its cells, within
// a budget. Past it the oldest ubatch goes, and its cells have no prefix.
#[test]
fn a_ubatch_past_the_input_budget_has_no_prefix() {
    let with_input = |columns: &[(i32, i32, u32)]| {
        let mut record = UbatchRecord::clone(&record(columns));
        record.input = Some(crate::recorder::StageInput {
            rows: vec![0; columns.len() * 8],
            row_bytes: 8,
            ggml_type: 0,
        });
        Arc::new(record)
    };
    let mut m = CellMirror::default();
    m.set_input_budget(40);
    for (seq, cell) in [(1, 0), (2, 10), (3, 20)] {
        let record = with_input(&[(seq, 0, cell), (seq, 1, cell + 1)]);
        m.write_ubatch(&record, vec![committed(1), committed(2)]);
    }
    let error = m.prefix(1, 2).unwrap_err();
    assert!(format!("{error:#}").contains("no longer held"), "{error:#}");
    assert!(m.prefix(2, 2).is_ok() && m.prefix(3, 2).is_ok());
    // the cell still holds its rows: only its ubatch is gone
    let cell = PrefixCell {
        pos: 0,
        stream: 0,
        cell: 0,
        digest: hash(1),
    };
    assert!(m.holds(&cell));
    let leaves = committed(1).unwrap().position;
    assert_eq!(m.lineage(&cell, &leaves).unwrap_err(), CellRecord::Dropped);
}

// What the recorder could not follow costs only the cells it can have
// changed.
#[test]
fn an_event_forgets_only_the_cells_it_can_have_changed() {
    let filled = || {
        let mut m = CellMirror::default();
        append(&mut m, 1, 0..3, 0);
        append(&mut m, 2, 0..3, 10);
        // sequence 3 finished: its cells are free and keep their rows
        write(&mut m, &[(3, 0, 20)], &[20]);
        m.seq_rm(3, -1, -1);
        m
    };
    let cell = |pos: i32, cell: u32, n: u8| PrefixCell {
        pos,
        stream: 0,
        cell,
        digest: hash(n),
    };
    let held = |m: &CellMirror, seq: i32| m.prefix(seq, 3).is_ok();
    let freed = cell(0, 20, 20);

    let mut m = filled();
    m.memory_op(MemoryOp::SeqShift, 1, -1, 1, -1);
    assert!(m.holds(&cell(0, 0, 0)), "position 0 was not shifted");
    assert_eq!(m.record(&cell(1, 1, 1)), CellRecord::Unknown);
    assert!(held(&m, 2) && m.holds(&freed));
    assert!(m.prefix(1, 1).is_err(), "a shifted sequence has no prefix");

    let mut m = filled();
    m.memory_op(MemoryOp::StateRead, 1, -1, -1, -1);
    assert!(!held(&m, 1), "the sequence's cells were replaced");
    assert!(held(&m, 2), "another sequence's cells were not");
    assert_eq!(
        m.record(&freed),
        CellRecord::Unknown,
        "a free cell may hold the loaded rows"
    );
    // a ubatch on top of loaded rows did not run on recorded cells
    append(&mut m, 1, 0..1, 30);
    assert!(m.prefix(1, 1).is_err());

    let mut m = filled();
    m.unseen_write();
    assert!(held(&m, 1) && held(&m, 2));
    assert_eq!(m.record(&freed), CellRecord::Unknown);

    for op in [MemoryOp::Clear, MemoryOp::Compact] {
        let mut m = filled();
        m.memory_op(op, -1, -1, -1, -1);
        assert!(!held(&m, 1) && !held(&m, 2) && !m.holds(&freed));
        // a cleared cache is empty; a compacted one holds what it held
        append(&mut m, 1, 0..3, 0);
        assert_eq!(held(&m, 1), op == MemoryOp::Clear, "{op:?}");
    }
    let mut m = filled();
    m.memory_op(MemoryOp::StateRead, -1, -1, -1, -1);
    assert!(!held(&m, 1) && !held(&m, 2));

    // a cell a later ubatch wrote is told from one the mirror lost
    let mut m = filled();
    write(&mut m, &[(2, 3, 20)], &[99]);
    assert_eq!(m.record(&freed), CellRecord::Overwritten);
}

#[test]
fn the_prefix_root_binds_positions_cells_rows_and_leaves() {
    let prefix = |pos: i32, cell: u32, digest: u8, leaves: u8| TurnPrefix {
        cells: vec![PrefixCell {
            pos,
            stream: 0,
            cell,
            digest: hash(digest),
        }],
        positions: vec![hash(leaves)],
        tokens: vec![7],
    };
    let root = prefix_root(&prefix(0, 4, 1, 9)).unwrap().unwrap();
    for other in [
        prefix(1, 4, 1, 9),
        prefix(0, 5, 1, 9),
        prefix(0, 4, 2, 9),
        prefix(0, 4, 1, 8),
    ] {
        assert_ne!(prefix_root(&other).unwrap().unwrap(), root);
    }
    assert_eq!(prefix_root(&TurnPrefix::default()).unwrap(), None);
}

#[test]
fn a_cache_per_sequence_numbers_cells_per_stream() {
    let mut m = CellMirror::default();
    m.set_streams(KvStreams::PerSequence);
    write(&mut m, &[(1, 0, 0)], &[1]);
    write(&mut m, &[(2, 0, 0)], &[2]);
    let one = m.prefix(1, 1).unwrap().cells[0];
    let two = m.prefix(2, 1).unwrap().cells[0];
    assert_eq!((one.stream, one.cell, two.stream, two.cell), (1, 0, 2, 0));
    assert!(m.holds(&one) && m.holds(&two), "the streams share no cell");
    // a freed cell of sequence 2 keeps its rows, which a copy takes along
    write(&mut m, &[(2, 1, 1)], &[3]);
    m.seq_rm(2, 1, -1);
    m.seq_cp(2, 3, -1, -1);
    let copied = m.prefix(3, 1).unwrap().cells[0];
    assert_eq!((copied.stream, copied.cell, copied.digest), (3, 0, hash(2)));
    assert!(
        m.prefix(3, 2).is_err(),
        "position 1 was freed before the copy"
    );
    let freed = PrefixCell {
        pos: 1,
        stream: 3,
        cell: 1,
        digest: hash(3),
    };
    assert!(m.holds(&freed));
    // the copy's lineage is the source's ubatch
    let prefix = m.prefix(3, 1).unwrap();
    let lineage = m.lineage(&prefix.cells[0], &prefix.positions[0]).unwrap();
    assert_eq!(lineage.writer(), 2);
    // a copy replaces what the destination held
    m.seq_cp(1, 3, -1, -1);
    assert_eq!(m.prefix(3, 1).unwrap().cells[0].digest, hash(1));
    assert!(!m.holds(&copied) && !m.holds(&freed));
    assert!(m.holds(&two), "the source keeps its cells");
}
