//! How large a peer's answer to a challenge may be.
//!
//! This node reads no more of an opening than its own challenge can
//! produce for the claimed model: the leaves of the picked positions, each
//! with the columns of its layer's matmuls, the head residuals and logits,
//! the attention partials, and for a turn that continued a cached prefix
//! the tokens of the prefix and the leaf hashes of its picked positions.
//! The sizes come from the claimed GGUF's header, which this node read
//! itself, and the receipt's stages. The bound is twice the sum, and never
//! above [`MAX_ANSWER_BYTES`], so a peer cannot make this node hold
//! hundreds of megabytes by answering with them. An honest opening of a 4B
//! model is about 10 MB.
//!
//! A stage that keeps a recurrent state also opens states, whose size the
//! header does not give. An opening of such a turn is bounded by
//! [`MAX_ANSWER_BYTES`] alone.

use std::collections::BTreeMap;

use anyhow::Result;
use skippy_receipts::attention::opening::BlockChallenge;
use skippy_receipts::attention::{BLOCK_POSITIONS, blocks};
use skippy_receipts::opened_values::ModelShapes;
use skippy_receipts::weights::WeightFile;
use skippy_receipts::wire::{Challenge, Receipt, StageCommitment};

/// The largest answer to a challenge this node reads, whatever the model:
/// an opening, or attention rows.
pub(super) const MAX_ANSWER_BYTES: usize = 128 << 20;
/// The most this node reads of an answer that is not an opening: an error
/// status and its body.
pub(super) const MAX_REFUSAL_BYTES: usize = 64 << 10;
/// A hex hash as JSON, quoted, with its comma.
const HASH_BYTES: usize = 67;
/// A Merkle path as JSON: at most 64 hex hashes, quoted.
const PATH_BYTES: usize = 64 * HASH_BYTES;
/// A token id as JSON, with its comma.
const TOKEN_BYTES: usize = 12;
/// Fields and punctuation around the values of one JSON object.
const OBJECT_BYTES: usize = 1024;
/// Rows of the residual width a leaf's token-mixing values may hold,
/// besides its two residuals.
const MIXING_ROWS: usize = 32;
/// Stored rows of a layer at one position, each at most one width.
const STORED_ROWS: usize = 8;

fn base64_len(bytes: usize) -> usize {
    bytes.div_ceil(3).saturating_mul(4)
}

/// Base64 of `values` f32 values.
fn values_len(values: usize) -> usize {
    base64_len(values.saturating_mul(size_of::<f32>()))
}

/// The bounds of the answers to challenges of one turn.
pub(super) struct AnswerLimits {
    hidden: usize,
    vocabulary: usize,
    /// Per layer: the values of the matmul columns one opened leaf holds,
    /// and its matmuls.
    matmuls: BTreeMap<u32, (usize, usize)>,
    stages: usize,
    /// A stage keeps a recurrent state, which its openings hold.
    recurrent: bool,
    /// Positions of the sequence the turn attends over.
    positions: usize,
    /// Positions of the cached prefix the turn continued.
    prefix: u32,
    /// Leaf hashes every stage together opens for one picked position of
    /// the prefix: its leaves, and a stored leaf per layer.
    prefix_hashes: usize,
}

impl AnswerLimits {
    /// The bounds for a turn `receipt` records of the model `weights`
    /// holds. Fails when the header does not name the model's shapes,
    /// which is this node's file and says nothing about the peer.
    pub(super) fn of(weights: &WeightFile, receipt: &Receipt) -> Result<Self> {
        let shapes = ModelShapes::of(weights)?;
        let size = |n: u64| usize::try_from(n).unwrap_or(usize::MAX);
        let mut matmuls: BTreeMap<u32, (usize, usize)> = BTreeMap::new();
        for tensor in weights.tensors.values().filter(|t| t.is_matrix()) {
            let Some(layer) = tensor.layer() else {
                continue;
            };
            // a matmul opens its input, one row long, and its output, a
            // value per row: once, or for every expert at most
            let experts = tensor
                .dims
                .iter()
                .skip(2)
                .copied()
                .fold(1u64, u64::saturating_mul);
            let column = tensor.row_len().saturating_add(tensor.rows());
            let values = size(column.saturating_mul(experts));
            let (sum, count) = matmuls.entry(layer).or_default();
            *sum = sum.saturating_add(values);
            *count += 1;
        }
        let recurrent = |stage: &StageCommitment| stage.recurrent_state.is_some();
        let prefix_hashes = receipt
            .stages
            .iter()
            .map(|stage| {
                let layers = u64::from(stage.layer_end.saturating_sub(stage.layer_start));
                size(layers.saturating_mul(3).saturating_add(1))
            })
            .fold(0usize, usize::saturating_add);
        Ok(Self {
            hidden: shapes.hidden,
            vocabulary: size(shapes.vocabulary),
            matmuls,
            stages: receipt.stages.len(),
            recurrent: receipt.stages.iter().any(recurrent),
            positions: receipt.position_end as usize,
            prefix: receipt.position_start,
            prefix_hashes,
        })
    }

    fn row(&self) -> usize {
        values_len(self.hidden)
    }

    /// One opened leaf of `layer`: its residuals and mixing values, the
    /// columns of its matmuls, and its three paths.
    fn leaf(&self, layer: u32) -> usize {
        let (values, count) = self.matmuls.get(&layer).copied().unwrap_or_default();
        let rows = (2 + MIXING_ROWS).saturating_mul(self.row());
        // each matmul's input and output are encoded apart
        let columns = values_len(values).saturating_add(count.saturating_mul(8));
        rows.saturating_add(columns)
            .saturating_add(3 * PATH_BYTES)
            .saturating_add((8 + count).saturating_mul(OBJECT_BYTES))
    }

    /// The most bytes an opening of `challenge` may take.
    pub(super) fn opening(&self, challenge: &Challenge) -> usize {
        if self.recurrent {
            return MAX_ANSWER_BYTES;
        }
        let positions = challenge.positions().len();
        let boundary = PATH_BYTES + OBJECT_BYTES;
        let mut total = OBJECT_BYTES + self.stages * OBJECT_BYTES;
        // every stage: the leaves entering and leaving it at each position
        total = total.saturating_add(self.stages * positions * 2 * boundary);
        // the input of layer 0 at each position
        total = total.saturating_add(positions * (self.row() + boundary));
        for pick in &challenge.picks {
            let leaves = pick.positions.len().saturating_mul(self.leaf(pick.layer));
            total = total.saturating_add(leaves);
        }
        let head = self.row() + values_len(self.vocabulary) + boundary;
        total = total.saturating_add(challenge.head.len().saturating_mul(head));
        // one head's partials: per block of attended positions, a logit, a
        // sum and a value row
        let partials = blocks(self.positions).saturating_mul(values_len(2 + self.hidden));
        let attention = challenge.attention.len();
        total = total.saturating_add(attention.saturating_mul(partials + OBJECT_BYTES));
        // the tokens of the prefix, and what every stage committed at each
        // of its picked positions
        let picked = challenge.prefix_positions(self.prefix).len();
        let position = self
            .prefix_hashes
            .saturating_mul(HASH_BYTES)
            .saturating_add(self.stages * (PATH_BYTES + OBJECT_BYTES));
        total = total
            .saturating_add((self.prefix as usize).saturating_mul(TOKEN_BYTES))
            .saturating_add(picked.saturating_mul(position));
        total.saturating_mul(2).min(MAX_ANSWER_BYTES)
    }

    /// The most bytes the rows of `blocks` may take: the stored rows of
    /// every position of every picked block, each with its proof.
    pub(super) fn rows(&self, blocks: &BlockChallenge) -> usize {
        let row = STORED_ROWS * self.row() + 2 * PATH_BYTES + OBJECT_BYTES;
        let positions: usize = blocks
            .picks
            .iter()
            .map(|pick| pick.blocks.len() * BLOCK_POSITIONS)
            .sum();
        let total = (1 + self.stages) * OBJECT_BYTES + positions.saturating_mul(row);
        total.saturating_mul(2).min(MAX_ANSWER_BYTES)
    }
}

#[cfg(test)]
mod tests {
    use skippy_receipts::attention::opening::{AttentionPick, BlockPick};
    use skippy_receipts::toy_turn::{self, Turn};

    use super::*;

    fn limits() -> (Turn, AnswerLimits) {
        let turn = Turn::new();
        let limits = AnswerLimits::of(&toy_turn::weights(), &turn.receipt).expect("toy shapes");
        (turn, limits)
    }

    #[test]
    fn an_honest_opening_fits_and_the_bound_follows_the_challenge() {
        let (turn, limits) = limits();
        let small = turn.challenge(&[(0, &[3])], &[]);
        let large = turn.challenge(&[(0, &[3, 4]), (1, &[3, 4])], &[4]);
        for challenge in [&small, &large] {
            let honest = serde_json::to_vec(&turn.open(challenge)).unwrap().len();
            let bound = limits.opening(challenge);
            assert!(honest <= bound, "{honest} > {bound}");
        }
        // a peer may not answer a small challenge with a large opening
        assert!(limits.opening(&small) < limits.opening(&large));
        // and the toy model's bound is nowhere near the cap
        assert!(
            limits.opening(&large) < 1 << 20,
            "{}",
            limits.opening(&large)
        );
    }

    #[test]
    fn an_honest_opening_of_a_cached_turn_fits() {
        // the opening holds the 40 tokens of the prefix, and what both
        // stages committed at its two picked positions
        let turn = Turn::starting_at(40);
        let limits = AnswerLimits::of(&toy_turn::weights(), &turn.receipt).expect("toy shapes");
        let with_prefix = turn.challenge(&[(0, &[7, 31, 41]), (1, &[7, 31, 43])], &[44]);
        let honest = serde_json::to_vec(&turn.open(&with_prefix)).unwrap().len();
        let bound = limits.opening(&with_prefix);
        assert!(honest <= bound, "{honest} > {bound}");
        let turn_alone = turn.challenge(&[(0, &[41]), (1, &[43])], &[44]);
        assert!(limits.opening(&turn_alone) < bound);
        // a receipt that names a prefix past any sequence allows the cap
        let mut far = turn.receipt.clone();
        far.position_start = u32::MAX - 5;
        far.position_end = u32::MAX;
        let limits = AnswerLimits::of(&toy_turn::weights(), &far).expect("toy shapes");
        assert_eq!(limits.opening(&turn_alone), MAX_ANSWER_BYTES);
    }

    #[test]
    fn no_challenge_allows_more_than_the_cap() {
        let (turn, limits) = limits();
        assert_eq!(MAX_ANSWER_BYTES, 128 << 20);
        // a challenge no draw makes: every position, a million times
        let positions: Vec<u32> = vec![3; 1 << 20];
        let mut huge = turn.challenge(&[(0, &positions), (1, &positions)], &positions);
        huge.attention = (0..1 << 16)
            .map(|position| AttentionPick {
                position,
                layer: 0,
                head_draw: 0,
            })
            .collect();
        assert_eq!(limits.opening(&huge), MAX_ANSWER_BYTES);
        let blocks = BlockChallenge {
            receipt_id: "r".into(),
            picks: vec![
                BlockPick {
                    position: 3,
                    layer: 0,
                    blocks: (0..1 << 16).collect(),
                };
                64
            ],
            proof: None,
        };
        assert_eq!(limits.rows(&blocks), MAX_ANSWER_BYTES);
    }

    #[test]
    fn the_rows_bound_follows_the_picked_blocks() {
        let (_, limits) = limits();
        let pick = |blocks: Vec<u32>| BlockChallenge {
            receipt_id: "r".into(),
            picks: vec![BlockPick {
                position: 3,
                layer: 0,
                blocks,
            }],
            proof: None,
        };
        let one = limits.rows(&pick(vec![0]));
        let two = limits.rows(&pick(vec![0, 1]));
        assert!(one < two && two < MAX_ANSWER_BYTES, "{one} {two}");
    }

    #[test]
    fn a_turn_with_a_recurrent_state_is_bounded_by_the_cap_alone() {
        // its openings hold states whose size the header does not give
        let mut turn = Turn::new();
        turn.receipt.stages[1].recurrent_state = Some("ab".repeat(32));
        let limits = AnswerLimits::of(&toy_turn::weights(), &turn.receipt).unwrap();
        let small = turn.challenge(&[(0, &[3])], &[]);
        assert_eq!(limits.opening(&small), MAX_ANSWER_BYTES);
    }

    #[test]
    fn a_header_without_shapes_gives_no_bound() {
        let turn = Turn::new();
        let mut weights = toy_turn::weights();
        weights.metadata.clear();
        assert!(AnswerLimits::of(&weights, &turn.receipt).is_err());
    }
}
