//! The requester's offline check of a split request (design §7.2): from the
//! coordinator's main record and the stage records it carried, work out each
//! stage's cell on the stage strip and whether each hop's two ends agree.
//!
//! What `agree` means is narrow on purpose: both ends of one hop committed to
//! the same frames on one lane. It says nothing about whether the two stages
//! are independent, or whether either computed its slice correctly (design
//! §0, §9). A missing end, or a required lane empty on both ends, is a `gap`,
//! never agreement.
//!
//! [`verify_split`] is the core, over already-extracted blocks; the Python
//! reference (`mesh_split_stage.py`) implements the same function and both
//! run over `vectors/split-stage/hop-cases.json`.
//! [`verify_split_records`] wraps it for sealed records: it recomputes every
//! `capsule_id` first. Signatures are checked by the receiver on receipt, not
//! repeated here.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use serde_json::Value;

use crate::producer::jcs::compute_capsule_id;
use crate::producer::stage::{
    stage_block_of, CoordinatorReceipt, DownstreamHop, Lane, LaneFold, ReturnMode, Side,
    StageBlock, StageError, UpstreamHop, COORDINATOR_RECEIPT_BLOCK,
};

/// One stage's cell on the stage strip.
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CellState {
    /// Stage 0: the coordinator's own slice. Never a `✓`: it would only
    /// compare the coordinator's claim against its own assignment.
    CoordinatorSlice,
    /// Present, well-formed, and matching what the coordinator assigned.
    Ok,
    /// Present but not matching the assignment (layers, package, return mode).
    Disagrees,
    /// Asked for and not received, or cited and not carried.
    NotReceived,
    NotRequested,
    /// Two distinct records under one stage key. Neither counts as present.
    Conflict,
    /// Carried, but not a valid record of this stage of this split.
    Rejected,
}

/// One lane of one hop.
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HopState {
    Agree,
    /// An end is missing, or a required lane is empty on both ends.
    Gap,
    /// Both ends present and different.
    Break,
    /// Equal digests with unequal frame counts.
    Malformed,
    /// A lane this request did not need, empty on both ends.
    NotApplicable,
}

#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
pub struct Cell {
    pub stage_index: u32,
    pub state: CellState,
}

#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
pub struct LaneVerdict {
    pub hop_index: u32,
    pub lane: &'static str,
    pub required: bool,
    pub state: HopState,
}

#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
pub struct SplitVerdict {
    pub cells: Vec<Cell>,
    pub lanes: Vec<LaneVerdict>,
    /// `true` only when every lane agrees or was not needed.
    pub handoffs_agree: bool,
    /// Always shown beside the hand-offs line: agreement is not completion.
    pub terminal_state: String,
}

/// A stage record as carried to the requester: its id and its raw block.
pub struct CarriedStageRecord {
    pub capsule_id: String,
    pub block: Value,
}

fn compare(a: Option<&LaneFold>, b: Option<&LaneFold>, required: bool) -> HopState {
    let (Some(a), Some(b)) = (a, b) else {
        return HopState::Gap;
    };
    if a.frames == 0 && b.frames == 0 {
        return if required {
            HopState::Gap
        } else {
            HopState::NotApplicable
        };
    }
    match (a.digest == b.digest, a.frames == b.frames) {
        (true, true) => HopState::Agree,
        (true, false) => HopState::Malformed,
        (false, _) => HopState::Break,
    }
}

/// Stage `k`'s cell and, when its record can be read, its block.
fn stage_cell<'a>(
    k: u32,
    receipt: &CoordinatorReceipt,
    own: &StageBlock,
    by_id: &'a BTreeMap<&str, StageBlock>,
    conflicted: &BTreeSet<u32>,
    rejected: &BTreeSet<&str>,
) -> (CellState, Option<&'a StageBlock>) {
    if conflicted.contains(&k) {
        return (CellState::Conflict, None);
    }
    let entry = receipt
        .entry(k)
        .expect("validated receipt has one entry per stage");
    let digest = match (entry.bundle.as_str(), &entry.bundle_ref) {
        ("present", Some(r)) => r.digest.as_str(),
        ("not_requested", _) => return (CellState::NotRequested, None),
        ("conflict", _) => return (CellState::Conflict, None),
        _ => return (CellState::NotReceived, None),
    };
    if rejected.contains(digest) {
        return (CellState::Rejected, None);
    }
    let Some(block) = by_id.get(digest) else {
        return (CellState::NotReceived, None);
    };
    let belongs = block.side == Side::Stage
        && block.stage_index == k
        && block.split_key() == receipt.split_key()
        && block.stage_count == receipt.stage_count();
    if !belongs {
        return (CellState::Rejected, None);
    }
    let assigned = receipt
        .assignment(k)
        .expect("validated receipt carries assignments");
    let matches = (
        block.layer_start,
        block.layer_end,
        block.package_id.as_str(),
    ) == (
        assigned.layer_start,
        assigned.layer_end,
        assigned.package_id.as_str(),
    ) && block.return_mode == own.return_mode;
    let state = if matches {
        CellState::Ok
    } else {
        CellState::Disagrees
    };
    (state, Some(block))
}

/// The core check, over the coordinator's own slice, its receipt, and the
/// stage records it carried.
pub fn verify_split(
    own: &StageBlock,
    receipt: &CoordinatorReceipt,
    carried: &[CarriedStageRecord],
) -> Result<SplitVerdict, StageError> {
    own.validate()?;
    receipt.validate()?;
    if own.side != Side::Coordinator
        || own.stage_index != 0
        || own.split_key() != receipt.split_key()
    {
        return Err(StageError::Main(
            "own slice is not stage 0 of this receipt's split".into(),
        ));
    }
    let n = receipt.stage_count();

    // Parse every carried record once. A record that does not validate is
    // rejected by id; two distinct valid records under one stage key of this
    // split mark that stage a conflict whatever the receipt says.
    let mut by_id: BTreeMap<&str, StageBlock> = BTreeMap::new();
    let mut rejected: BTreeSet<&str> = BTreeSet::new();
    let mut ids_by_stage: BTreeMap<u32, BTreeSet<&str>> = BTreeMap::new();
    for record in carried {
        match StageBlock::from_value(&record.block) {
            Ok(block) => {
                if block.side == Side::Stage && block.split_key() == receipt.split_key() {
                    ids_by_stage
                        .entry(block.stage_index)
                        .or_default()
                        .insert(record.capsule_id.as_str());
                }
                by_id.insert(record.capsule_id.as_str(), block);
            }
            Err(_) => {
                rejected.insert(record.capsule_id.as_str());
            }
        }
    }
    let conflicted: BTreeSet<u32> = ids_by_stage
        .iter()
        .filter(|(_, ids)| ids.len() > 1)
        .map(|(k, _)| *k)
        .collect();

    let mut cells = vec![Cell {
        stage_index: 0,
        state: CellState::CoordinatorSlice,
    }];
    let mut blocks: Vec<Option<&StageBlock>> = vec![Some(own)];
    for k in 1..n {
        let (state, block) = stage_cell(k, receipt, own, &by_id, &conflicted, &rejected);
        cells.push(Cell {
            stage_index: k,
            state,
        });
        blocks.push(block);
    }

    let completed = own.terminal_state == "completed";
    let relayed = own.return_mode == ReturnMode::Relayed;
    let mut lanes = Vec::new();
    for h in 0..n - 1 {
        let up: Option<&DownstreamHop> = blocks[h as usize].and_then(|b| b.downstream.as_ref());
        let down: Option<&UpstreamHop> = blocks[h as usize + 1].and_then(|b| b.upstream.as_ref());
        lanes.push(LaneVerdict {
            hop_index: h,
            lane: Lane::Forward.wire(),
            required: true,
            state: compare(up.map(|u| &u.sent), down.map(|d| &d.received), true),
        });
        let reply_required = relayed && completed;
        lanes.push(LaneVerdict {
            hop_index: h,
            lane: Lane::Reply.wire(),
            required: reply_required,
            state: compare(
                down.map(|d| &d.replies_sent),
                up.map(|u| &u.replies_received),
                reply_required,
            ),
        });
    }
    if !relayed {
        let final_block = blocks[n as usize - 1];
        let sent = final_block
            .and_then(|b| b.direct_return.as_ref())
            .and_then(|d| d.sent.as_ref());
        let received = own.direct_return.as_ref().and_then(|d| d.received.as_ref());
        lanes.push(LaneVerdict {
            hop_index: n - 1,
            lane: Lane::DirectReturn.wire(),
            required: completed,
            state: compare(sent, received, completed),
        });
    }
    let handoffs_agree = lanes
        .iter()
        .all(|l| matches!(l.state, HopState::Agree | HopState::NotApplicable))
        && lanes.iter().any(|l| l.state == HopState::Agree);
    Ok(SplitVerdict {
        cells,
        lanes,
        handoffs_agree,
        terminal_state: own.terminal_state.clone(),
    })
}

fn checked_id(record: &Value) -> Result<String, StageError> {
    let stored = record
        .get("capsule_id")
        .and_then(Value::as_str)
        .ok_or_else(|| StageError::Main("record carries no capsule_id".into()))?;
    if compute_capsule_id(record)? != stored {
        return Err(StageError::Main(format!(
            "capsule_id {stored} does not recompute"
        )));
    }
    Ok(stored.to_string())
}

/// [`verify_split`] over sealed records. The main record must recompute and
/// carry both split blocks. A carried stage record whose id does not
/// recompute is passed on with a null block, so its stage reads `rejected`.
pub fn verify_split_records(
    main_record: &Value,
    stage_records: &[Value],
) -> Result<SplitVerdict, StageError> {
    checked_id(main_record)?;
    let own = stage_block_of(main_record)?
        .ok_or_else(|| StageError::Main("main record carries no x-mesh-stage-v1".into()))?;
    let receipt_value = main_record
        .pointer(&format!(
            "/model_attestation/compute_attestation/{COORDINATOR_RECEIPT_BLOCK}"
        ))
        .ok_or_else(|| StageError::Main("main record carries no coordinator receipt".into()))?;
    let receipt = CoordinatorReceipt::from_value(receipt_value)?;
    let carried: Vec<CarriedStageRecord> = stage_records
        .iter()
        .map(|record| {
            let id = record
                .get("capsule_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let block = match checked_id(record) {
                Ok(_) => record
                    .pointer(&format!(
                        "/model_attestation/compute_attestation/{}",
                        crate::producer::stage::STAGE_BLOCK
                    ))
                    .cloned()
                    .unwrap_or(Value::Null),
                Err(_) => Value::Null,
            };
            CarriedStageRecord {
                capsule_id: id.to_string(),
                block,
            }
        })
        .collect();
    verify_split(&own, &receipt, &carried)
}
