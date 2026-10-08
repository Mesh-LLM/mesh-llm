//! Converts a decoded DFlash noise block into proposal tokens.
//!
//! The draft decodes `[anchor, <mask> * n]` in one pass. Block position 0 is
//! the anchor (the last committed target token); positions `1..=n` predict the
//! draft tokens. DFlash reads each position's logits greedily. DFlash2 packs a
//! candidate lattice into the nextn output instead: per position, `top_k`
//! candidate token ids followed by a `top_k x top_k` transition score matrix
//! indexed by the chosen predecessor candidate.

/// Greedy token for one logits row. Ties resolve to the lowest token id.
///
/// A block reads one full-vocabulary row per draft position, so this runs over
/// millions of floats per proposal. Lane-wise maxima keep the scan
/// vectorizable; a second pass finds the first index holding the maximum.
pub(crate) fn argmax_token(logits: &[f32]) -> Option<i32> {
    const LANES: usize = 16;
    if logits.is_empty() {
        return None;
    }
    let (chunks, remainder) = logits.as_chunks::<LANES>();
    let mut lane_max = [f32::NEG_INFINITY; LANES];
    for chunk in chunks {
        for (max, &value) in lane_max.iter_mut().zip(chunk) {
            if value > *max {
                *max = value;
            }
        }
    }
    let max = lane_max.into_iter().chain(remainder.iter().copied()).fold(
        f32::NEG_INFINITY,
        |max, value| if value > max { value } else { max },
    );
    // A row of NaNs has no maximum; keep the first token like a scalar scan.
    let index = logits.iter().position(|&value| value == max).unwrap_or(0);
    i32::try_from(index).ok()
}

/// Walks a DFlash2 selector lattice and returns one token per draft position.
///
/// `rows` holds the lattice rows for block positions `1..` in order, each at
/// least `top_k + top_k * top_k` floats wide. The first draft position scores
/// against the anchor, so every predecessor row is identical there and the
/// walk starts from predecessor 0. The walk stops at a candidate outside the
/// `n_vocab` vocabulary, which only a malformed draft produces.
pub(crate) fn walk_dflash2_lattice<'a>(
    rows: impl IntoIterator<Item = &'a [f32]>,
    top_k: usize,
    n_vocab: usize,
) -> Vec<i32> {
    let row_used = top_k + top_k * top_k;
    let mut tokens = Vec::new();
    let mut predecessor = 0usize;
    for row in rows {
        if top_k == 0 || row.len() < row_used {
            break;
        }
        let scores_start = top_k + predecessor * top_k;
        let scores = &row[scores_start..scores_start + top_k];
        predecessor = first_max_index(scores);
        let candidate = row[predecessor];
        let Some(token) = lattice_token(candidate, n_vocab) else {
            break;
        };
        tokens.push(token);
    }
    tokens
}

/// The token id a lattice candidate names, truncated as upstream reads it, if
/// it lies in the vocabulary.
fn lattice_token(candidate: f32, n_vocab: usize) -> Option<i32> {
    if !(candidate.is_finite() && candidate >= 0.0) {
        return None;
    }
    let token = i32::try_from(candidate as i64).ok()?;
    (usize::try_from(token).ok()? < n_vocab).then_some(token)
}

fn first_max_index(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (index, &value) in values.iter().enumerate().skip(1) {
        if value > values[best] {
            best = index;
        }
    }
    best
}

/// Number of draft tokens one block can produce for a requested maximum.
///
/// The anchor occupies block position 0, so a block of `block_size` tokens
/// yields at most `block_size - 1` draft tokens.
pub(crate) fn block_draft_capacity(block_size: usize, requested: usize) -> usize {
    requested.min(block_size.saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_prefers_the_lowest_id_on_ties() {
        assert_eq!(argmax_token(&[0.5, 2.0, 2.0, -1.0]), Some(1));
        assert_eq!(argmax_token(&[]), None);
    }

    #[test]
    fn argmax_scans_whole_lanes_and_the_remainder() {
        let mut row = vec![0.0f32; 1_000];
        row[37] = 3.0;
        row[901] = 3.0;
        assert_eq!(argmax_token(&row), Some(37));

        row[999] = 4.0;
        assert_eq!(argmax_token(&row), Some(999));

        assert_eq!(argmax_token(&[f32::NAN, f32::NAN]), Some(0));
        assert_eq!(argmax_token(&[f32::NEG_INFINITY; 20]), Some(0));
    }

    #[test]
    fn block_capacity_reserves_the_anchor_position() {
        assert_eq!(block_draft_capacity(16, 32), 15);
        assert_eq!(block_draft_capacity(16, 4), 4);
        assert_eq!(block_draft_capacity(1, 4), 0);
        assert_eq!(block_draft_capacity(0, 4), 0);
    }

    fn lattice_row(candidates: &[f32], scores: &[[f32; 2]; 2]) -> Vec<f32> {
        let mut row = candidates.to_vec();
        for predecessor in scores {
            row.extend_from_slice(predecessor);
        }
        // Rows are padded to the draft hidden size.
        row.extend_from_slice(&[0.0; 3]);
        row
    }

    #[test]
    fn dflash2_walk_follows_the_chosen_predecessor() {
        // Position 1 scores against the anchor: both predecessor rows match.
        let first = lattice_row(&[100.0, 200.0], &[[0.1, 0.9], [0.1, 0.9]]);
        // Position 2 must read predecessor 1 (candidate 200 was chosen).
        let second = lattice_row(&[300.0, 400.0], &[[5.0, 0.0], [0.0, 5.0]]);
        // Position 3 must read predecessor 1 again.
        let third = lattice_row(&[500.0, 600.0], &[[9.0, 0.0], [1.0, 0.0]]);

        let tokens = walk_dflash2_lattice(
            [first.as_slice(), second.as_slice(), third.as_slice()],
            2,
            1_000,
        );

        assert_eq!(tokens, vec![200, 400, 500]);
    }

    #[test]
    fn dflash2_walk_stops_at_rows_too_narrow_for_the_lattice() {
        let first = lattice_row(&[7.0, 8.0], &[[1.0, 0.0], [1.0, 0.0]]);
        let narrow = vec![1.0, 2.0, 3.0];

        assert_eq!(
            walk_dflash2_lattice([first.as_slice(), narrow.as_slice()], 2, 1_000),
            vec![7]
        );
        assert!(walk_dflash2_lattice([first.as_slice()], 0, 1_000).is_empty());
    }

    #[test]
    fn dflash2_walk_rejects_invalid_candidate_ids() {
        let scores = [[1.0, 0.0], [1.0, 0.0]];
        for candidate in [-1.0, 1_000.0, f32::NAN, f32::INFINITY, 1e12] {
            let row = lattice_row(&[candidate, 3.0], &scores);
            assert!(
                walk_dflash2_lattice([row.as_slice()], 2, 1_000).is_empty(),
                "{candidate}"
            );
        }
        let last = lattice_row(&[999.0, 3.0], &scores);
        assert_eq!(walk_dflash2_lattice([last.as_slice()], 2, 1_000), vec![999]);
    }
}
