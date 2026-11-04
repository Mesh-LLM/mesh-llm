//! Opens a challenge on one stage: the picked layers it holds, the input of
//! layer 0 when it holds layer 0, the residual the output head reads when
//! it holds the final layer, and the residual leaves at its boundaries.
//! Picked layers, the layer-0 input and the head residual are recomputed by
//! one replay of the turn; boundary leaves come from the committed tree.
//! When that replay does not reproduce a committed leaf, one that reports
//! the committed values alone supplies them ([`ReplayScope::Committed`]),
//! and a full one anchored to them the matmuls and selections
//! ([`ReplayScope::Anchored`]).
//!
//! A challenge may pick positions of the turn's cached prefix. The stage
//! did not compute those in the turn: it opens each by replaying the one
//! recorded ubatch that wrote the position's cell, from a copy of the
//! prefix cells below that ubatch ([`super::prefix_opening`]). A replayed
//! leaf of such a position is checked against the leaves the stage
//! committed when it computed the position, and opens without a path.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use anyhow::{Context, Result, bail};

use super::cells::CellRecord;
use super::event_log::{Event, Written, last_writes};
use super::prefix_opening::{PrefixPlan, lineage_replays};
use super::recurrent_opening::{RegionPlan, open_region, region_plan};
use super::turn_seed::Segment;
use super::{
    AnchorValue, CommittedTurn, DUMMY_TOKEN, MAX_REPLAYS_PER_RECEIPT, PrefixCell, Recorder,
    ReplayAnchor, ReplayScope, ReplaySeed, ReplayStep, ReplayedLayer, UbatchRecord, turn,
};
use crate::commitment::{
    Hash, encode_path, leaf_index, mixing_leaf, mixing_slot, residual_leaf, residual_slot,
};
use crate::layer_tensors::{GGML_F32, decode_f32_bytes};
use crate::wire::{
    AnchorOpening, BoundaryLeaf, CannotOpen, CannotOpenReason, Challenge, HeadOpening,
    LayerOpening, LeafOpening, MixingKind, RecurrentOpening, StageOpening, encode_bytes,
    mixing_bytes, mixing_values,
};

/// A leaf of the committed tree and its Merkle path.
pub(super) struct Committed {
    pub(super) hash: Hash,
    pub(super) path: Vec<String>,
}

/// One replay of the plan: the cells it starts from, and its steps.
pub(super) struct Replay {
    pub(super) seed: Option<ReplaySeed>,
    pub(super) steps: Vec<ReplayStep>,
}

/// What a replay must reproduce, and the replays that reproduce it.
pub(super) struct ReplayPlan {
    seq: i32,
    replays: Vec<Replay>,
    /// Layers to capture, sorted.
    layers: Vec<u32>,
    /// Token per opened position.
    tokens: BTreeMap<u32, i32>,
    /// By (position, slot): the committed leaf.
    leaves: BTreeMap<(u32, u64), Committed>,
    /// The recurrent ubatch the opening shows.
    pub(super) region: Option<RegionPlan>,
    /// The picked positions of the turn's cached prefix.
    prefix: Option<PrefixPlan>,
    /// The tokens of the turn's cached prefix, on the stage that holds
    /// layer 0.
    prefix_tokens: Vec<i32>,
}

/// Replayed values by (position, layer).
pub(super) type Replayed = BTreeMap<(u32, u32), ReplayedLayer>;

/// What a stage opens from one replay of the plan.
struct Opened {
    layers: Vec<LayerOpening>,
    anchors: Vec<AnchorOpening>,
    head: Vec<HeadOpening>,
    recurrent: Vec<RecurrentOpening>,
}

/// A replay that does not reproduce a committed leaf.
#[derive(Debug)]
pub(super) struct Divergence(pub(super) String);

impl std::fmt::Display for Divergence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Divergence {}

fn replay_plan(
    replayer: &dyn super::Replayer,
    plan: &ReplayPlan,
    scope: ReplayScope<'_>,
) -> Result<Replayed> {
    let mut replayed = Replayed::new();
    for replay in &plan.replays {
        let values = replayer.replay(
            plan.seq,
            &plan.layers,
            scope,
            replay.seed.as_ref(),
            &replay.steps,
        )?;
        for (position, layer, values) in values {
            replayed.insert((position, layer), values);
        }
    }
    Ok(replayed)
}

/// Replays the plan twice more: once for the committed values alone, then
/// in full from them. A backend that fused ops when serving computes them
/// apart in a full replay, so its matmuls drift from the committed values
/// layer after layer; anchored at every picked layer, they follow from the
/// committed values instead.
fn replay_anchored(
    replayer: &dyn super::Replayer,
    plan: &ReplayPlan,
    imported: Option<u32>,
) -> Result<Replayed> {
    let committed = replay_plan(replayer, plan, ReplayScope::Committed)?;
    let mut anchored = Replayed::new();
    for replay in &plan.replays {
        let anchors = anchors(plan, replay, &committed, imported)?;
        let values = replayer.replay(
            plan.seq,
            &plan.layers,
            ReplayScope::Anchored(&anchors),
            replay.seed.as_ref(),
            &replay.steps,
        )?;
        for (position, layer, values) in values {
            anchored.insert((position, layer), values);
        }
    }
    Ok(with_committed_values(anchored, committed))
}

/// The committed values a replay's captured tokens are anchored to: the
/// residual entering every replayed layer, except `imported`, whose input
/// the stage imports and the replay sets from the turn's own, and the
/// layer's token-mixing outputs.
fn anchors(
    plan: &ReplayPlan,
    replay: &Replay,
    committed: &Replayed,
    imported: Option<u32>,
) -> Result<Vec<ReplayAnchor>> {
    let mut anchors = Vec::new();
    for (step, replay_step) in replay.steps.iter().enumerate() {
        let ReplayStep::Decode { capture, .. } = replay_step else {
            continue;
        };
        for &position in capture {
            let pos = i32::try_from(position)?;
            for &layer in &plan.layers {
                let values = committed.get(&(position, layer)).with_context(|| {
                    format!("the committed replay did not reach position {position}, layer {layer}")
                })?;
                let anchor = |value, values| ReplayAnchor {
                    step,
                    seq: plan.seq,
                    pos,
                    layer,
                    value,
                    values,
                };
                if imported != Some(layer) {
                    anchors.push(anchor(
                        AnchorValue::Residual,
                        decode_f32_bytes(&values.residual_in),
                    ));
                }
                for mixing in values
                    .mixing
                    .iter()
                    .filter(|m| m.kind == MixingKind::Output)
                {
                    if let Some((GGML_F32, bytes)) = &mixing.value {
                        anchors.push(anchor(
                            AnchorValue::MixingOutput(mixing.index),
                            decode_f32_bytes(bytes),
                        ));
                    }
                }
            }
        }
    }
    Ok(anchors)
}

/// An anchored replay's matmuls and selections with the committed values
/// of a replay at [`ReplayScope::Committed`].
fn with_committed_values(mut full: Replayed, committed: Replayed) -> Replayed {
    for (key, values) in committed {
        if let Some(full) = full.get_mut(&key) {
            full.residual_in = values.residual_in;
            full.residual_out = values.residual_out;
            full.logits = values.logits;
            full.mixing = values.mixing;
        }
    }
    full
}

impl Recorder {
    /// Opens this stage's part of a challenge, by replaying the turn unless
    /// the same challenge was opened before.
    pub fn open(
        &self,
        challenge: &Challenge,
        replayer: &dyn super::Replayer,
    ) -> Result<StageOpening> {
        let root = |committed: &_| turn(committed, &challenge.receipt_id).map(|t| t.tree.root());
        let opened_root = {
            let mut state = self.lock();
            if let Some(opening) = state.openings.get(challenge) {
                return Ok(StageOpening::clone(&opening));
            }
            root(&state.committed)
        };
        let opening = self
            .open_uncached(challenge, replayer)
            .map_err(|error| name_receipt(error, &challenge.receipt_id))?;
        let opening = Arc::new(opening);
        let mut state = self.lock();
        // a turn committed under the same id while this one replayed must
        // not be answered with this opening
        if opened_root.is_some() && root(&state.committed) == opened_root {
            state.openings.insert(challenge.clone(), opening.clone());
        }
        drop(state);
        Ok(Arc::unwrap_or_clone(opening))
    }

    fn open_uncached(
        &self,
        challenge: &Challenge,
        replayer: &dyn super::Replayer,
    ) -> Result<StageOpening> {
        let positions = challenge.positions();
        if positions.is_empty() {
            bail!("the challenge has no positions");
        }
        let head = if self.holds_output {
            challenge.head.clone()
        } else {
            Vec::new()
        };
        let (plan, boundaries) = self.plan(challenge, &positions, &head)?;
        let seeded: Vec<PrefixCell> = plan
            .replays
            .iter()
            .filter_map(|replay| replay.seed.as_ref())
            .flat_map(|seed| seed.cells.iter().copied())
            .collect();
        self.verify_exposed(replayer, &seeded)?;
        let mut replayed = replay_plan(replayer, &plan, ReplayScope::Full)?;
        let opened = match self.open_replayed(challenge, &plan, &positions, &head, &replayed) {
            Err(error) if error.is::<Divergence>() => {
                let imported = (self.layers.start > 0).then_some(self.layers.start);
                replayed = replay_anchored(replayer, &plan, imported)?;
                self.open_replayed(challenge, &plan, &positions, &head, &replayed)?
            }
            opened => opened?,
        };
        let Opened {
            layers,
            anchors,
            head,
            recurrent,
        } = opened;
        let attention = self.open_attention(
            &challenge.receipt_id,
            &challenge.attention,
            &layers,
            &replayed,
            replayer,
        )?;
        let prefix = self.open_prefix(plan.prefix.as_ref(), replayer)?;
        Ok(StageOpening {
            stage_index: self.stage_index,
            layers,
            anchors,
            entry: boundaries.entry,
            exit: boundaries.exit,
            head,
            attention,
            recurrent,
            prefix,
            prefix_tokens: plan.prefix_tokens,
        })
    }

    /// Checks the replayed values against the committed leaves and opens
    /// the picked layers, the anchors, the head and the recurrent region.
    fn open_replayed(
        &self,
        challenge: &Challenge,
        plan: &ReplayPlan,
        positions: &[u32],
        head: &[u32],
        replayed: &Replayed,
    ) -> Result<Opened> {
        let replayed_at = |position: u32, layer: u32| {
            replayed
                .get(&(position, layer))
                .with_context(|| format!("replay did not reach position {position}, layer {layer}"))
        };
        let mut layers = Vec::new();
        for pick in challenge
            .picks
            .iter()
            .filter(|p| self.layers.contains(&p.layer))
        {
            let mut leaves = Vec::with_capacity(pick.positions.len());
            for &position in &pick.positions {
                let values = replayed_at(position, pick.layer)?;
                leaves.push(self.leaf_opening(plan, position, pick.layer, values)?);
            }
            layers.push(LayerOpening {
                layer: pick.layer,
                leaves,
            });
        }
        let mut anchors = Vec::new();
        if self.layers.start == 0 {
            for &position in positions {
                let values = replayed_at(position, 0)?;
                let token = plan.tokens[&position];
                let committed = plan.committed(position, residual_slot(0, 0))?;
                let leaf = residual_leaf(position, 0, Some(token), Some(&values.residual_in));
                if leaf != committed.hash {
                    return Err(Divergence(format!(
                        "replay diverged from the committed input of layer 0 at position {position}"
                    ))
                    .into());
                }
                anchors.push(AnchorOpening {
                    position,
                    token,
                    residual: encode_bytes(&values.residual_in),
                    path: committed.path.clone(),
                });
            }
        }
        let head = head
            .iter()
            .map(|&position| {
                self.head_opening(plan, position, replayed_at(position, self.layers.end - 1)?)
            })
            .collect::<Result<_>>()?;
        Ok(Opened {
            layers,
            anchors,
            head,
            recurrent: open_region(plan, replayed, self.layers.start)?,
        })
    }

    /// The residual leaving the final layer at a head position, checked
    /// against its committed leaf.
    fn head_opening(
        &self,
        plan: &ReplayPlan,
        position: u32,
        values: &ReplayedLayer,
    ) -> Result<HeadOpening> {
        let end = self.layers.end;
        let residual = values
            .residual_out
            .as_deref()
            .with_context(|| format!("position {position} is not an output row"))?;
        let committed = plan.committed(position, residual_slot(end, self.layers.start))?;
        if residual_leaf(position, end, None, Some(residual)) != committed.hash {
            return Err(Divergence(format!(
                "replay diverged from the committed output residual at position {position}"
            ))
            .into());
        }
        Ok(HeadOpening {
            position,
            residual: encode_bytes(residual),
            path: committed.path.clone(),
            logits: values.logits.as_deref().map(encode_bytes),
        })
    }

    /// Checks one replayed position of a picked layer against its committed
    /// leaves and builds its opening.
    fn leaf_opening(
        &self,
        plan: &ReplayPlan,
        position: u32,
        layer: u32,
        values: &ReplayedLayer,
    ) -> Result<LeafOpening> {
        let start = self.layers.start;
        let token = (layer == 0).then(|| plan.tokens[&position]);
        let mixing = values.mixing_openings();
        let decoded = mixing_bytes(&mixing)?;
        let recomputed = [
            (
                "residual-in",
                residual_slot(layer, start),
                residual_leaf(position, layer, token, Some(&values.residual_in)),
            ),
            (
                "mixing",
                mixing_slot(layer, start),
                mixing_leaf(position, layer, &mixing_values(&decoded)),
            ),
            (
                "residual-out",
                residual_slot(layer + 1, start),
                residual_leaf(position, layer + 1, None, values.residual_out.as_deref()),
            ),
        ];
        let mut paths = Vec::with_capacity(3);
        for (name, slot, leaf) in recomputed {
            let committed = plan.committed(position, slot)?;
            if leaf != committed.hash {
                return Err(Divergence(format!(
                    "replay diverged from the committed turn at position {position}, layer {layer}: \
                     the {name} leaf differs"
                ))
                .into());
            }
            paths.push(committed.path.clone());
        }
        let [residual_in_path, mixing_path, residual_out_path] =
            <[Vec<String>; 3]>::try_from(paths).expect("three leaves");
        Ok(LeafOpening {
            position,
            residual_in: encode_bytes(&values.residual_in),
            residual_out: values.residual_out.as_deref().map(encode_bytes),
            residual_in_path,
            mixing_path,
            residual_out_path,
            mixing,
            matrices: values.matrices(),
            selections: values.selections.clone(),
        })
    }

    /// Finds the turn, collects the committed leaves the opening needs and
    /// the replay steps. Counts a replay against the receipt when one is
    /// needed.
    fn plan(
        &self,
        challenge: &Challenge,
        positions: &[u32],
        head: &[u32],
    ) -> Result<(ReplayPlan, Boundaries)> {
        let start = self.layers.start;
        let slots = self.slots();
        let mut layers: BTreeSet<u32> = challenge
            .picks
            .iter()
            .map(|p| p.layer)
            .filter(|layer| self.layers.contains(layer))
            .collect();
        if start == 0 {
            // the input of layer 0 is opened at every picked position
            layers.insert(0);
        }
        if !head.is_empty() {
            // the head reads the residual leaving the final layer
            layers.insert(self.layers.end - 1);
        }
        let mut guard = self.lock();
        let state = &mut *guard;
        let turn = self.opening_turn_mut(&mut state.committed, &challenge.receipt_id)?;
        // a picked position is one of the turn or of its prefix; a head
        // position is one the turn sampled from, never one of the prefix
        let in_turn = |p: &u32| (turn.position_start..turn.position_end).contains(p);
        let in_prefix = |p: &u32| (*p as usize) < turn.prefix.len() && *p < turn.position_start;
        if let Some(&position) = positions
            .iter()
            .find(|p| !in_turn(p) && !in_prefix(p))
            .or_else(|| head.iter().find(|p| !in_turn(p)))
        {
            bail!("position {position} is outside the receipt");
        }
        if !layers.is_empty() {
            if turn.state.as_ref().is_some_and(|s| s.bytes.is_none()) {
                return Err(self.cannot_open(
                    &challenge.receipt_id,
                    CannotOpenReason::StateDropped,
                    "the recurrent state the turn started from is no longer held",
                ));
            }
            if turn.replays >= MAX_REPLAYS_PER_RECEIPT {
                bail!(
                    "receipt {} was already replayed for {MAX_REPLAYS_PER_RECEIPT} challenges",
                    challenge.receipt_id
                );
            }
            turn.replays += 1;
        }
        let turn: &CommittedTurn = turn;
        let lineage = self.lineage(turn, &state.cells, positions)?;
        let committed = |position: u32, slot: u64| -> Result<Committed> {
            if let Some(lineage) = lineage.get(&position) {
                // a position of the prefix: its leaves are in no tree of
                // the turn, and open against the position's digest
                let leaf = usize::try_from(slot)
                    .ok()
                    .and_then(|slot| lineage.leaves.get(slot))
                    .context("leaf outside the committed position")?;
                return Ok(Committed {
                    hash: *leaf,
                    path: Vec::new(),
                });
            }
            let index = leaf_index(position, slot, turn.position_start, slots);
            Ok(Committed {
                hash: turn
                    .tree
                    .leaf(index)
                    .context("leaf outside the committed tree")?,
                path: encode_path(&turn.tree.path(index)?),
            })
        };
        let mut leaves = BTreeMap::new();
        let written = last_writes(&turn.events, turn.seq);
        let mut tokens = BTreeMap::new();
        let mut capture: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        let mut boundaries = Boundaries::default();
        for &position in positions {
            if lineage.contains_key(&position) {
                if start == 0 {
                    tokens.insert(position, turn.prefix.tokens[position as usize]);
                }
            } else {
                let step = computing_step(&written, position)?;
                capture.entry(step).or_default().push(position);
                if start == 0 {
                    tokens.insert(position, recorded_token(turn, &written, position)?);
                }
            }
            for &layer in &layers {
                for slot in [
                    residual_slot(layer, start),
                    mixing_slot(layer, start),
                    residual_slot(layer + 1, start),
                ] {
                    leaves.insert((position, slot), committed(position, slot)?);
                }
            }
            if start > 0 {
                boundaries
                    .entry
                    .push(boundary_leaf(position, committed(position, 0)?));
            }
            let exit = committed(position, residual_slot(self.layers.end, start))?;
            boundaries.exit.push(boundary_leaf(position, exit));
        }
        for &position in head {
            let step = computing_step(&written, position)?;
            let captured = capture.entry(step).or_default();
            if !captured.contains(&position) {
                captured.push(position);
            }
            let slot = residual_slot(self.layers.end, start);
            leaves.insert((position, slot), committed(position, slot)?);
        }
        let region = region_plan(
            turn,
            &written,
            &challenge.attention,
            &self.layers,
            &state.state_layers,
        )?;
        if let Some(region) = &region {
            let slot = mixing_slot(region.layer, start);
            for position in region.positions() {
                leaves.insert((position, slot), committed(position, slot)?);
            }
        }
        let last = capture
            .keys()
            .chain(region.iter().flat_map(|r| r.mixing.keys()))
            .max()
            .copied();
        let mut planned = Vec::new();
        if let Some(last) = last.filter(|_| !layers.is_empty()) {
            // with the turn's own cells copied, only some ubatches run; a
            // recurrent layer's state is known only at the turn's start
            let segments = if turn.state.is_some() || region.is_some() {
                None
            } else {
                super::turn_seed::segments(turn, &capture, last, &state.cells)?
            };
            let segments = segments.unwrap_or_else(|| {
                vec![Segment {
                    cells: Vec::new(),
                    events: (0..=last).collect(),
                }]
            });
            planned = replays(turn, &capture, region.as_ref(), segments);
        }
        if !layers.is_empty() {
            planned.extend(lineage_replays(turn, &lineage));
        }
        share_context(&mut planned);
        let plan = ReplayPlan {
            seq: turn.seq,
            replays: planned,
            layers: layers.into_iter().collect(),
            tokens,
            leaves,
            region,
            prefix: PrefixPlan::of(turn, &lineage),
            prefix_tokens: if start == 0 {
                turn.prefix.tokens.clone()
            } else {
                Vec::new()
            },
        };
        Ok((plan, boundaries))
    }

    /// Fails unless every cell of `seed` still holds what it held when the
    /// turn started. Call it with the stage idle and its reports applied, so
    /// no ubatch rewrites a cell before the seed is copied.
    pub fn check_seed(&self, seed: &ReplaySeed) -> Result<()> {
        self.check_cells(&seed.cells)
    }

    /// Fails unless every cell still holds what the turn read or wrote. The
    /// error is a [`CannotOpen`]: the stage's own cache no longer holds the
    /// turn, whatever the challenge asks. An exposed cell fails here: an
    /// opening checks its rows first ([`Self::verify_exposed`]).
    pub fn check_cells(&self, cells: &[PrefixCell]) -> Result<()> {
        self.check_records(cells, |record| record == CellRecord::Held)
    }

    /// [`Self::check_cells`] for cells whose rows are read to be checked
    /// against their digests: an exposed cell passes.
    pub fn check_unseen_overwrite(&self, cells: &[PrefixCell]) -> Result<()> {
        self.check_records(cells, |record| {
            matches!(record, CellRecord::Held | CellRecord::Exposed)
        })
    }

    fn check_records(
        &self,
        cells: &[PrefixCell],
        readable: impl Fn(CellRecord) -> bool,
    ) -> Result<()> {
        let state = self.lock();
        let lost = cells.iter().find_map(|cell| {
            let record = state.cells.record(cell);
            (!readable(record)).then_some((cell, record))
        });
        match lost {
            Some((cell, record)) => Err(self.lost_cell(cell, record)),
            None => Ok(()),
        }
    }

    /// The error of an opening that reads `cell`, which no longer holds
    /// what the turn read or wrote.
    pub(super) fn lost_cell(&self, cell: &PrefixCell, record: CellRecord) -> anyhow::Error {
        let (reason, what) = match record {
            CellRecord::Overwritten => (
                CannotOpenReason::CellsOverwritten,
                "was reused by a later request",
            ),
            CellRecord::Dropped => (
                CannotOpenReason::CellsUnknown,
                "was computed by a ubatch the stage no longer holds",
            ),
            CellRecord::Exposed => (
                CannotOpenReason::CellsUnknown,
                "was free when the stage computed a ubatch it could not record, and its \
                 rows were not checked since",
            ),
            _ => (
                CannotOpenReason::CellsUnknown,
                "may have been reused: the stage could not follow a change of its KV cache",
            ),
        };
        self.cannot_open(
            "",
            reason,
            format!(
                "the KV cell that held position {} of the turn or its prefix {what}",
                cell.pos
            ),
        )
    }

    /// The token at every position of a committed turn, from its first
    /// position. Meaningful on the stage that holds layer 0.
    pub fn turn_tokens(&self, receipt_id: &str) -> Result<Vec<i32>> {
        let state = self.lock();
        let turn = turn(&state.committed, receipt_id)
            .with_context(|| format!("receipt {receipt_id} is not committed"))?;
        let written = last_writes(&turn.events, turn.seq);
        (turn.position_start..turn.position_end)
            .map(|position| recorded_token(turn, &written, position))
            .collect()
    }
}

/// Names the receipt in a [`CannotOpen`] made where it was not known.
pub(super) fn name_receipt(error: anyhow::Error, receipt_id: &str) -> anyhow::Error {
    match error.downcast::<CannotOpen>() {
        Ok(mut cannot) => {
            if cannot.receipt_id.is_empty() {
                receipt_id.clone_into(&mut cannot.receipt_id);
            }
            anyhow::Error::new(cannot)
        }
        Err(error) => error,
    }
}

impl ReplayPlan {
    pub(super) fn committed(&self, position: u32, slot: u64) -> Result<&Committed> {
        self.leaves
            .get(&(position, slot))
            .with_context(|| format!("no committed leaf for position {position}, slot {slot}"))
    }
}

#[derive(Default)]
struct Boundaries {
    entry: Vec<BoundaryLeaf>,
    exit: Vec<BoundaryLeaf>,
}

fn boundary_leaf(position: u32, committed: Committed) -> BoundaryLeaf {
    BoundaryLeaf {
        position,
        leaf: hex::encode(committed.hash),
        path: committed.path,
    }
}

/// The replays of `segments` of `turn`, each from the prefix and the
/// segment's own cells, in replay contexts of one size.
fn replays(
    turn: &CommittedTurn,
    capture: &BTreeMap<usize, Vec<u32>>,
    region: Option<&RegionPlan>,
    segments: Vec<Segment>,
) -> Vec<Replay> {
    let captured = |index: usize| capture.get(&index).cloned().unwrap_or_default();
    // a captured position reports its mixing values with the rest
    let mixing = |index: usize, captured: &[u32]| -> Vec<u32> {
        region
            .and_then(|r| r.mixing.get(&index))
            .map(|positions| {
                positions
                    .iter()
                    .copied()
                    .filter(|p| !captured.contains(p))
                    .collect()
            })
            .unwrap_or_default()
    };
    let state = turn.state.as_ref().and_then(|state| state.bytes.clone());
    segments
        .into_iter()
        .map(|segment| {
            let steps: Vec<ReplayStep> = segment
                .events
                .iter()
                .map(|&index| match &turn.events[index] {
                    Event::Ubatch(record) => {
                        let capture = captured(index);
                        ReplayStep::Decode {
                            record: replay_columns(record, turn.seq),
                            mixing: mixing(index, &capture),
                            capture,
                        }
                    }
                    Event::SeqRm { p0, p1 } => ReplayStep::SeqRm { p0: *p0, p1: *p1 },
                })
                .collect();
            let cells: Vec<PrefixCell> = turn
                .prefix
                .cells
                .iter()
                .copied()
                .chain(segment.cells)
                .collect();
            let seed = (!cells.is_empty() || state.is_some())
                .then(|| seed_for(cells, &steps, state.clone()));
            Replay { seed, steps }
        })
        .collect()
}

/// Gives every seeded replay a context of one size, so the replay context
/// is not made again for each.
fn share_context(replays: &mut [Replay]) {
    let n_cells = replays
        .iter()
        .filter_map(|r| r.seed.as_ref())
        .map(|s| s.n_cells)
        .max();
    for seed in replays.iter_mut().filter_map(|r| r.seed.as_mut()) {
        seed.n_cells = n_cells.unwrap_or(seed.n_cells);
    }
}

/// A replay that starts from `cells`, in a context with room for them and
/// for every cell the steps use.
pub(super) fn seed_for(
    cells: Vec<PrefixCell>,
    steps: &[ReplayStep],
    state: Option<Arc<[u8]>>,
) -> ReplaySeed {
    let seeded = cells.iter().map(|c| c.cell + 1).max().unwrap_or(0);
    let used = steps
        .iter()
        .filter_map(|step| match step {
            ReplayStep::Decode { record, .. } => Some(record.n_kv),
            ReplayStep::SeqRm { .. } => None,
        })
        .max()
        .unwrap_or(0);
    ReplaySeed {
        cells,
        n_cells: seeded.max(used),
        state,
    }
}

/// The event of the ubatch that computed `position` of the turn's sequence
/// last, by the turn's [`last_writes`].
pub(super) fn computing_step(written: &BTreeMap<u32, Written>, position: u32) -> Result<usize> {
    written
        .get(&position)
        .map(|written| written.event)
        .with_context(|| format!("no recorded ubatch computed position {position}"))
}

/// The token the turn fed at `position`: the one of the ubatch that
/// computed the position last.
fn recorded_token(
    turn: &CommittedTurn,
    written: &BTreeMap<u32, Written>,
    position: u32,
) -> Result<i32> {
    written
        .get(&position)
        .and_then(|written| Some(turn.events[written.event].ubatch()?.token[written.col]))
        .with_context(|| format!("no recorded token at position {position}"))
}

/// A recorded ubatch as `seq` replays it: other sequences' columns become
/// dummy tokens at positions 0.. of their own sequence, with zero inputs. A
/// ubatch of `seq` alone is shared as recorded.
pub(super) fn replay_columns(record: &Arc<UbatchRecord>, seq: i32) -> Arc<UbatchRecord> {
    if record.seq.iter().all(|&s| s == seq) {
        return Arc::clone(record);
    }
    let mut out = UbatchRecord::clone(record);
    let mut next_pos: BTreeMap<i32, i32> = BTreeMap::new();
    for col in 0..record.seq.len() {
        if record.seq[col] == seq {
            continue;
        }
        out.token[col] = DUMMY_TOKEN;
        let pos = next_pos.entry(record.seq[col]).or_insert(0);
        out.pos[col] = *pos;
        *pos += 1;
        if let Some(input) = &mut out.input {
            let row = input.row_bytes;
            input.rows[col * row..(col + 1) * row].fill(0);
        }
    }
    Arc::new(out)
}
