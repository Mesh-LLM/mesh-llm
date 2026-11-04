//! Bounds on what the stages after this one may answer, and the checks a
//! stage runs on their commitments before it keeps them.
//!
//! A stage reads no more of a later stage's reply than its own request can
//! produce. A commit produces one commitment per later stage. An opening
//! produces, per later stage, the boundary leaves of the picked positions,
//! the leaves of the picks in that stage's layers, the head residuals on
//! the last stage, the partials of its attention picks, the leaf hashes of
//! the picked positions of a cached prefix, and on a stage with hybrid
//! memory one recurrent region; rows produce the stored rows of
//! the picked blocks. The widths are the model's residual width, which
//! every stage of a chain shares, times a generous count of values, since
//! a stage does not know a later stage's matmul shapes. The bound is
//! twice the sum, and never above the protocol's own.

use std::ops::Range;

use anyhow::{Result, bail};
use skippy_protocol::binary::MAX_STAGE_RECEIPTS_REPLY_BYTES;
use skippy_receipts::attention::opening::BlockChallenge;
use skippy_receipts::attention::{BLOCK_POSITIONS, blocks};
use skippy_receipts::challenge::PREFIX_POSITIONS;
use skippy_receipts::commitment::slots_per_position;
use skippy_receipts::wire::{Challenge, StageCommitment};

use super::kept_turns::LaterStages;

/// Stages a chain may have after any one stage.
pub(super) const LATER_STAGES: usize = 64;
/// A residual width for a stage that could not read its own.
const DEFAULT_WIDTH: usize = 16_384;
/// A hex hash as JSON, quoted, with its comma.
const HASH_BYTES: usize = 67;
/// A Merkle path as JSON: at most 64 hex hashes, quoted.
const PATH_BYTES: usize = 64 * HASH_BYTES;
/// Rows of the residual width an opened leaf may hold: both residuals, its
/// token-mixing values, and the inputs and outputs of its weight matmuls,
/// which may be several widths wide.
const LEAF_ROWS: usize = 2 + 32 + 64 * 16;
/// Rows of the residual width a mixing leaf of a recurrent region may hold.
const MIXING_LEAF_ROWS: usize = 32;
/// Stored rows of a layer at one position, each at most one width.
const STORED_ROWS: usize = 8;
/// The logits of one head position, base64: a vocabulary of 780,000.
const LOGITS_BYTES: usize = 4 << 20;
/// The recurrent states a region opens: the one its ubatch read and the
/// one it left.
const REGION_STATES_BYTES: usize = 64 << 20;
/// Fields and punctuation around the values of one JSON object.
const OBJECT_BYTES: usize = 1024;

#[derive(Clone, Copy, Debug)]
pub(super) struct ReplyLimits {
    /// Bytes of one residual row, base64.
    row: usize,
    /// The residual width, in values.
    width: usize,
}

fn base64_len(bytes: usize) -> usize {
    bytes.div_ceil(3) * 4
}

impl ReplyLimits {
    /// Limits for a model whose residual stream is `width` values wide;
    /// `None` when the stage could not read it.
    pub(super) fn for_width(width: Option<usize>) -> Self {
        let width = width.filter(|&w| w > 0).unwrap_or(DEFAULT_WIDTH);
        Self {
            row: base64_len(width * size_of::<f32>()),
            width,
        }
    }

    /// Bytes the commitments of the later stages may take for a turn of
    /// `positions` positions: per stage its root, backend, cache types and
    /// prefix, and the first position of each of its ubatches.
    pub(super) fn commit(positions: u32) -> usize {
        let stage = 2 * OBJECT_BYTES + 11 * positions as usize;
        (LATER_STAGES * stage).min(MAX_STAGE_RECEIPTS_REPLY_BYTES)
    }

    fn leaf(&self) -> usize {
        LEAF_ROWS * self.row + 3 * PATH_BYTES + 8 * OBJECT_BYTES
    }

    /// Bytes the later stages may answer `challenge` with.
    pub(super) fn opening(&self, challenge: &Challenge, later: &LaterStages) -> usize {
        let positions = challenge.positions().len();
        let boundary = PATH_BYTES + OBJECT_BYTES;
        let last = later.commitments.len().saturating_sub(1);
        let mut total = OBJECT_BYTES;
        for (index, stage) in later.commitments.iter().enumerate() {
            let layers = stage.layer_start..stage.layer_end;
            let leaves: usize = challenge
                .picks
                .iter()
                .filter(|pick| layers.contains(&pick.layer))
                .map(|pick| pick.positions.len())
                .sum();
            let attention = challenge
                .attention
                .iter()
                .filter(|pick| layers.contains(&pick.layer))
                .count();
            total += OBJECT_BYTES + positions * 2 * boundary + leaves * self.leaf();
            // a picked position of a cached prefix: a hash per leaf of the
            // stage and per stored leaf of its layers, and its path
            let hashes = slots_per_position(stage.layer_start, stage.layer_end) as usize
                + (stage.layer_end - stage.layer_start) as usize;
            total += PREFIX_POSITIONS * (hashes * HASH_BYTES + PATH_BYTES + OBJECT_BYTES);
            if index == last {
                let head = self.row + LOGITS_BYTES + PATH_BYTES + OBJECT_BYTES;
                total += challenge.head.len() * head;
            }
            // one head's partials: per block of attended positions, a
            // logit, a sum and a value row
            let partials = blocks(later.position_end as usize)
                * base64_len((2 + self.width) * size_of::<f32>());
            total += attention * (partials + OBJECT_BYTES);
            if attention > 0 && stage.recurrent_state.is_some() {
                total += self.region(stage, later.position_end);
            }
        }
        total.saturating_mul(2).min(MAX_STAGE_RECEIPTS_REPLY_BYTES)
    }

    /// Bytes a recurrent region of `stage` may take: a mixing leaf per
    /// position of its largest ubatch and one more, and two states.
    fn region(&self, stage: &StageCommitment, position_end: u32) -> usize {
        let ends = stage.ubatches.iter().skip(1).copied().chain([position_end]);
        let largest = stage
            .ubatches
            .iter()
            .zip(ends)
            .map(|(&start, end)| end.saturating_sub(start))
            .max()
            .unwrap_or(0) as usize;
        let leaf = MIXING_LEAF_ROWS * self.row + PATH_BYTES + OBJECT_BYTES;
        (largest + 1) * leaf + REGION_STATES_BYTES
    }

    /// Bytes the later stages may answer `blocks` with: the stored rows of
    /// every position of every picked block, each with its proof.
    pub(super) fn rows(&self, blocks: &BlockChallenge) -> usize {
        let row = STORED_ROWS * self.row + 2 * PATH_BYTES + OBJECT_BYTES;
        let positions: usize = blocks
            .picks
            .iter()
            .map(|pick| pick.blocks.len() * BLOCK_POSITIONS)
            .sum();
        (LATER_STAGES * OBJECT_BYTES + positions * row)
            .saturating_mul(2)
            .min(MAX_STAGE_RECEIPTS_REPLY_BYTES)
    }
}

/// Refuses the commitments of the stages after `own` unless they continue
/// the plan: stage after stage, each holding the layers that follow the
/// previous stage's, with the leaves its layers make over `positions`, and
/// with fields no longer than a commitment's.
pub(super) fn check_commitments(
    own: &StageCommitment,
    later: &[StageCommitment],
    positions: &Range<u32>,
) -> Result<()> {
    if later.is_empty() {
        bail!("the next stage committed nothing");
    }
    if later.len() > LATER_STAGES {
        bail!("the next stage answered for {} stages", later.len());
    }
    let span = u64::from(positions.end - positions.start);
    let mut previous = own;
    for stage in later {
        if stage.stage_index != previous.stage_index + 1
            || stage.layer_start != previous.layer_end
            || stage.layer_end <= stage.layer_start
        {
            bail!(
                "stage {} commits layers {}..{}, after stage {} with layers {}..{}",
                stage.stage_index,
                stage.layer_start,
                stage.layer_end,
                previous.stage_index,
                previous.layer_start,
                previous.layer_end
            );
        }
        let leaves = span * slots_per_position(stage.layer_start, stage.layer_end);
        if stage.leaf_count != leaves {
            bail!(
                "stage {} commits {} leaves; its layers make {leaves} over positions {positions:?}",
                stage.stage_index,
                stage.leaf_count
            );
        }
        let hex = |text: &str| text.len() == 64 && text.bytes().all(|b| b.is_ascii_hexdigit());
        let fields = hex(&stage.root)
            && stage.backend.len() <= 256
            && stage.cache_types.len() <= 16
            && stage.cache_types.iter().all(|t| t.len() <= 32)
            && stage.ubatches.len() as u64 <= span
            // a later stage holds no layer 0, so it commits no tokens
            && stage
                .prefix
                .as_ref()
                .is_none_or(|p| hex(&p.root) && p.tokens.is_none())
            && stage.recurrent_state.as_deref().is_none_or(hex);
        if !fields {
            bail!("stage {} sent a malformed commitment", stage.stage_index);
        }
        previous = stage;
    }
    Ok(())
}

/// Refuses an opening or rows of the later stages unless one part came
/// from each stage that committed the turn, in order.
pub(super) fn check_stages(later: &LaterStages, answered: impl Iterator<Item = u32>) -> Result<()> {
    let committed = later.commitments.iter().map(|stage| stage.stage_index);
    if !committed.eq(answered) {
        bail!("the next stage did not answer for the stages that committed the turn, in order");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use skippy_receipts::attention::opening::{AttentionPick, BlockPick};
    use skippy_receipts::wire::Pick;

    use super::*;

    fn commitment(index: u32, layers: Range<u32>, positions: u32) -> StageCommitment {
        StageCommitment {
            stage_index: index,
            layer_start: layers.start,
            layer_end: layers.end,
            root: "ab".repeat(32),
            leaf_count: u64::from(positions) * slots_per_position(layers.start, layers.end),
            backend: "test".into(),
            cache_types: vec!["f16".into()],
            prefix: None,
            ubatches: vec![0],
            recurrent_state: None,
        }
    }

    fn later(commitments: Vec<StageCommitment>) -> LaterStages {
        LaterStages {
            position_start: 0,
            position_end: 8,
            commitments: commitments.into(),
        }
    }

    fn challenge(layers: &[u32]) -> Challenge {
        Challenge {
            receipt_id: "r".into(),
            picks: layers
                .iter()
                .map(|&layer| Pick {
                    layer,
                    positions: vec![1, 2],
                })
                .collect(),
            proof: None,
            head: Vec::new(),
            attention: Vec::new(),
        }
    }

    // A later stage's commitment is kept only when it continues the plan.
    #[test]
    fn a_commitment_must_continue_the_plan() {
        let own = commitment(0, 0..2, 8);
        let next = commitment(1, 2..4, 8);
        let last = commitment(2, 4..6, 8);
        check_commitments(&own, &[next.clone(), last.clone()], &(0..8)).unwrap();
        type Change = fn(&mut StageCommitment);
        let changes: [Change; 9] = [
            // a later stage holds no layer 0: the prefix's tokens are not its
            |s| {
                s.prefix = Some(skippy_receipts::wire::PrefixCommitment {
                    root: "ab".repeat(32),
                    positions: 8,
                    tokens: Some("ab".repeat(32)),
                });
            },
            |s| s.stage_index = 3,
            |s| s.layer_start = 3,
            |s| s.layer_end = 2,
            |s| s.leaf_count += 1,
            |s| s.root = "zz".repeat(32),
            |s| s.backend = "b".repeat(1000),
            |s| s.ubatches = vec![0; 9],
            |s| s.recurrent_state = Some("short".into()),
        ];
        for (i, change) in changes.iter().enumerate() {
            let mut forged = next.clone();
            change(&mut forged);
            let result = check_commitments(&own, &[forged, last.clone()], &(0..8));
            assert!(result.is_err(), "change {i} was kept");
        }
        assert!(check_commitments(&own, &[], &(0..8)).is_err());
        assert!(check_commitments(&own, &[last], &(0..8)).is_err(), "a gap");
        // the leaves are those of the turn's positions
        assert!(check_commitments(&own, &[next], &(0..9)).is_err());
    }

    // What a later stage may answer follows from what the challenge asks
    // of it.
    #[test]
    fn a_reply_is_bounded_by_what_its_request_asks() {
        let limits = ReplyLimits::for_width(Some(8));
        let stages = later(vec![commitment(1, 2..4, 8), commitment(2, 4..6, 8)]);
        let none = limits.opening(&challenge(&[0, 1]), &stages);
        let one = limits.opening(&challenge(&[0, 3]), &stages);
        let two = limits.opening(&challenge(&[3, 5]), &stages);
        assert!(none < one && one < two, "{none} {one} {two}");
        assert!(none < 256 << 10, "boundary leaves alone: {none}");
        let mut heads = challenge(&[0]);
        heads.head = vec![1, 2];
        assert!(limits.opening(&heads, &stages) > none + 2 * LOGITS_BYTES);
        let mut attention = challenge(&[3]);
        attention.attention = vec![AttentionPick {
            position: 1,
            layer: 3,
            head_draw: 0,
        }];
        assert!(limits.opening(&attention, &stages) > one);
        // a wider model may answer more, up to the protocol's bound
        let wide = ReplyLimits::for_width(Some(1 << 20));
        assert!(wide.opening(&challenge(&[3]), &stages) > one);
        assert_eq!(
            wide.opening(&challenge(&[3, 5]), &stages),
            MAX_STAGE_RECEIPTS_REPLY_BYTES
        );

        let blocks = |n: u32| BlockChallenge {
            receipt_id: "r".into(),
            picks: vec![BlockPick {
                position: 1,
                layer: 3,
                blocks: (0..n).collect(),
            }],
            proof: None,
        };
        assert!(limits.rows(&blocks(0)) < limits.rows(&blocks(2)));
        assert!(ReplyLimits::commit(8) < ReplyLimits::commit(8000));
        assert!(ReplyLimits::commit(8) < 1 << 20);
    }

    #[test]
    fn an_answer_must_come_from_the_stages_that_committed() {
        let stages = later(vec![commitment(1, 2..4, 8), commitment(2, 4..6, 8)]);
        check_stages(&stages, [1, 2].into_iter()).unwrap();
        for answered in [vec![1], vec![2, 1], vec![1, 2, 3], vec![0, 1, 2]] {
            assert!(check_stages(&stages, answered.into_iter()).is_err());
        }
    }
}
