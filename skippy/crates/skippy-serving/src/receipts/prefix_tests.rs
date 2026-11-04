//! A turn that continues a cached prefix, on the fixture chain of
//! `tests.rs`: its turn at positions 4..10 continues four positions that an
//! earlier turn of the session computed.

use skippy_receipts::binding::{verify_paths, verify_shape};

use super::{Chain, POSITIONS, SESSION, chain, challenge, session_turn, turn};
use crate::receipts::TurnTokens;

// The fixture's turn continues a prefix of four positions. A challenge
// picks two of them, and every stage opens them by a replay of the ubatch
// that wrote them.
#[test]
fn opens_positions_of_the_cached_prefix_on_every_stage() {
    let Chain {
        stage0,
        probe0,
        probe1,
        ..
    } = chain();
    let receipt = stage0
        .finish_turn("r1", SESSION, session_turn(), turn(), None)
        .expect("receipt");
    verify_shape(&receipt, 4).expect("shape");
    let prefix = |stage: usize| receipt.stages[stage].prefix.clone().expect("a prefix");
    assert_eq!((prefix(0).positions, prefix(1).positions), (4, 4));
    assert!(prefix(0).tokens.is_some() && prefix(1).tokens.is_none());

    let mut challenge = challenge("r1", &[1, 3]);
    for pick in &mut challenge.picks {
        pick.positions = vec![1, 3, 7, 9];
    }
    skippy_receipts::challenge::check_shape(&challenge, &receipt).expect("a draw's shape");
    let opening = stage0.open(&challenge, None).expect("opening");
    verify_paths(&receipt, &challenge, &opening).expect("paths verify");
    assert_eq!(opening.stages[0].prefix_tokens, vec![100, 101, 102, 103]);
    assert!(opening.stages[1].prefix_tokens.is_empty());
    for stage in &opening.stages {
        let opened: Vec<u32> = stage.prefix.iter().map(|p| p.position).collect();
        assert_eq!(opened, [1, 3]);
    }
    // the turn, and the one ubatch that wrote both prefix positions
    assert_eq!((probe0.replays(), probe1.replays()), (2, 2));

    // three positions of the prefix are more than a draw picks
    for pick in &mut challenge.picks {
        pick.positions = vec![0, 1, 3, 7];
    }
    let error = stage0.open(&challenge, None).expect_err("no draw's shape");
    assert!(format!("{error:#}").contains("of the prefix"), "{error:#}");
}

// Receipts forbid the reuse of generated tokens, so a prefix holds only
// what a request fed. A turn whose prefix is longer than its prompt would
// hold sampled tokens that no head check covers: it gets no receipt.
#[test]
fn a_prefix_longer_than_the_prompt_gets_no_receipt() {
    let Chain { stage0, .. } = chain();
    let tokens = TurnTokens {
        generation_start: POSITIONS.start - 1,
        ..turn()
    };
    let error = stage0
        .finish_turn("r1", SESSION, session_turn(), tokens, None)
        .expect_err("a prefix past the prompt");
    assert!(
        format!("{error:#}").contains("longer than its prompt"),
        "{error:#}"
    );
}
