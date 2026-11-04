//! The attention check of an audited turn ([`skippy_receipts::attention`]):
//! the partials the node opened with the challenge must combine to the
//! attention output it committed, and the blocks the client then draws
//! from the opening must recompute to those partials from the committed q
//! and the K and V rows the node opens for them.

use std::collections::BTreeMap;
use std::io::Write;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Serialize;
use skippy_receipts::attention::opening::{AttentionPick, BlockChallenge, RowsOpening};
use skippy_receipts::attention::verify::{
    OpenedHead, block_challenge, block_passed, check_blocks, check_partials, verify_rows,
};
use skippy_receipts::attention::{
    AttentionDescriptor, AttentionLayout, COMBINE_TOLERANCE, PARTIAL_TOLERANCE, layer_layout,
};
use skippy_receipts::wire::{Challenge, LeafOpening, Opening, Receipt};

/// Attention rows as the node sent them: decoded, their bytes, and the
/// node's signature of them.
pub struct OpenedRows {
    pub rows: RowsOpening,
    pub bytes: Vec<u8>,
    pub signature: Option<String>,
    pub seconds: f64,
}

#[derive(Serialize, Default)]
pub struct AttentionOutput {
    pub picks: Vec<AttentionPickOutput>,
    /// Seconds from asking for the blocks' rows to receiving them.
    pub rows_seconds: Option<f64>,
    /// Why the node's rows failed: no answer, a bad signature, or a row
    /// that is not the one the turn stored.
    pub rows_error: Option<String>,
}

#[derive(Serialize)]
pub struct AttentionPickOutput {
    pub position: u32,
    pub layer: u32,
    pub head: Option<u32>,
    /// Positions the query attends to.
    pub context: usize,
    /// How far the partials combine from the opened output, relative to
    /// its largest value.
    pub combine_error: Option<f64>,
    pub blocks: Vec<u32>,
    /// The largest error of a recomputed block against the node's partial.
    pub block_error: Option<f64>,
    /// Why the check cannot follow the layer's attention; says nothing
    /// about the node.
    pub unchecked: Option<String>,
    /// Why the pick failed.
    pub failure: Option<String>,
    pub pass: bool,
}

impl AttentionPickOutput {
    fn new(pick: &AttentionPick) -> Self {
        Self {
            position: pick.position,
            layer: pick.layer,
            head: None,
            context: 0,
            combine_error: None,
            blocks: Vec::new(),
            block_error: None,
            unchecked: None,
            failure: None,
            pass: false,
        }
    }

    fn fail(&mut self, reason: impl Into<String>) {
        self.failure = Some(reason.into());
        self.pass = false;
    }
}

impl AttentionOutput {
    pub fn pass(&self) -> bool {
        self.rows_error.is_none()
            && self
                .picks
                .iter()
                .all(|p| p.pass || (p.unchecked.is_some() && p.failure.is_none()))
    }

    pub fn failures(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .picks
            .iter()
            .filter_map(|p| {
                let failure = p.failure.as_ref()?;
                Some(format!(
                    "attention at position {}, layer {}: {failure}",
                    p.position, p.layer
                ))
            })
            .collect();
        if let Some(error) = &self.rows_error {
            out.push(format!("attention rows: {error}"));
        }
        out
    }

    pub fn unchecked(&self) -> Vec<String> {
        self.picks
            .iter()
            .filter_map(|p| {
                Some(format!(
                    "attention of layer {}: {}",
                    p.layer,
                    p.unchecked.as_ref()?
                ))
            })
            .collect()
    }
}

/// A head whose partials combined, with what its blocks are checked with.
struct Pending {
    pick: usize,
    head: OpenedHead,
    layout: AttentionLayout,
    q: Vec<f32>,
}

/// Runs the attention check: the partials in `opening`, then the rows
/// `open_rows` returns for the blocks drawn from `rows_seed`. The rows
/// are returned as the node sent them, for evidence.
pub(super) fn check_attention(
    receipt: &Receipt,
    challenge: &Challenge,
    opening: &Opening,
    layouts: &BTreeMap<u32, Vec<AttentionDescriptor>>,
    rows_seed: &[u8],
    open_rows: &dyn Fn(&BlockChallenge) -> Result<OpenedRows>,
) -> (AttentionOutput, Option<OpenedRows>) {
    let mut output = AttentionOutput::default();
    let mut pending = Vec::new();
    for pick in &challenge.attention {
        let mut checked = AttentionPickOutput::new(pick);
        match open_head(opening, layouts, pick) {
            Ok(None) => checked.unchecked = Some("the layer computes no attention".into()),
            Ok(Some(Opened::Unchecked(reason))) => checked.unchecked = Some(reason),
            Ok(Some(Opened::Head(head, layout, q))) => {
                checked.head = Some(head.head);
                checked.context = head.context.len();
                checked.combine_error = Some(head.combine_error);
                if head.passed() {
                    pending.push(Pending {
                        pick: output.picks.len(),
                        head,
                        layout: *layout,
                        q,
                    });
                } else {
                    checked.fail(format!(
                        "the partials combine {:.1e} from the output, past {COMBINE_TOLERANCE:.0e}",
                        head.combine_error
                    ));
                }
            }
            Err(error) => checked.fail(format!("{error:#}")),
        }
        output.picks.push(checked);
    }
    if pending.is_empty() {
        return (output, None);
    }
    let heads: Vec<OpenedHead> = pending.iter().map(|p| p.head.clone()).collect();
    let blocks = block_challenge(&challenge.receipt_id, &heads, rows_seed);
    let started = Instant::now();
    let opened = match open_rows(&blocks) {
        Ok(opened) => opened,
        Err(error) => {
            output.rows_error = Some(format!("{error:#}"));
            return (output, None);
        }
    };
    output.rows_seconds = Some(started.elapsed().as_secs_f64());
    let by_layer: BTreeMap<u32, AttentionLayout> = pending
        .iter()
        .map(|p| (p.layout.layer, p.layout.clone()))
        .collect();
    let rows = match verify_rows(receipt, &by_layer, &opened.rows) {
        Ok(rows) => rows,
        Err(error) => {
            output.rows_error = Some(format!("{error:#}"));
            return (output, Some(opened));
        }
    };
    for (pending, pick) in pending.iter().zip(&blocks.picks) {
        let checked = &mut output.picks[pending.pick];
        checked.blocks = pick.blocks.clone();
        match check_blocks(&pending.layout, &pending.head, pick, &pending.q, &rows) {
            Ok(error) if block_passed(error) => {
                checked.block_error = Some(error);
                checked.pass = true;
            }
            Ok(error) => {
                checked.block_error = Some(error);
                checked.fail(format!(
                    "a block recomputes {error:.1e} from its partial, past {PARTIAL_TOLERANCE:.0e}"
                ));
            }
            Err(error) => checked.fail(format!("{error:#}")),
        }
    }
    (output, Some(opened))
}

enum Opened {
    Unchecked(String),
    Head(OpenedHead, Box<AttentionLayout>, Vec<f32>),
}

/// The pick's head after its partials are checked against the output, or
/// why its layer's attention is not checked; `None` for a layer without
/// attention.
fn open_head(
    opening: &Opening,
    layouts: &BTreeMap<u32, Vec<AttentionDescriptor>>,
    pick: &AttentionPick,
) -> Result<Option<Opened>> {
    let Some(descriptors) = layouts.get(&pick.layer) else {
        return Ok(Some(Opened::Unchecked(
            "the layer's glue did not run".into(),
        )));
    };
    let layout = match layer_layout(descriptors) {
        Ok(Some(layout)) => layout,
        Ok(None) => return Ok(None),
        Err(reason) => return Ok(Some(Opened::Unchecked(format!("{reason:#}")))),
    };
    let leaf = opened_leaf(opening, pick).context("the attention position is not opened")?;
    let partials = opening
        .stages
        .iter()
        .flat_map(|s| &s.attention)
        .find(|a| a.position == pick.position && a.layer == pick.layer)
        .context("the node opened no partials")?;
    let head = check_partials(layout, pick, leaf, partials)?;
    let q = layout.q_values(&leaf.mixing)?;
    Ok(Some(Opened::Head(head, Box::new(layout.clone()), q)))
}

fn opened_leaf<'a>(opening: &'a Opening, pick: &AttentionPick) -> Option<&'a LeafOpening> {
    opening
        .stages
        .iter()
        .flat_map(|s| &s.layers)
        .find(|l| l.layer == pick.layer)?
        .leaves
        .iter()
        .find(|l| l.position == pick.position)
}

pub(super) fn render(out: &mut dyn Write, attention: &AttentionOutput) -> std::io::Result<()> {
    if attention.picks.is_empty() {
        return Ok(());
    }
    writeln!(out, "attention, per picked head:")?;
    for p in &attention.picks {
        let what = format!("position {} layer {}", p.position, p.layer);
        if let Some(reason) = &p.unchecked {
            writeln!(out, "  {what:<26} cannot check ({reason})")?;
            continue;
        }
        writeln!(
            out,
            "  {what:<26} head {:<3} {} positions  combine {}  blocks {:?} {}  {}{}",
            p.head.map_or("-".into(), |h| h.to_string()),
            p.context,
            p.combine_error.map_or("-".into(), |e| format!("{e:.1e}")),
            p.blocks,
            p.block_error.map_or("-".into(), |e| format!("{e:.1e}")),
            if p.pass { "ok" } else { "FAIL" },
            p.failure
                .as_ref()
                .map(|f| format!(" ({f})"))
                .unwrap_or_default()
        )?;
    }
    if let Some(seconds) = attention.rows_seconds {
        writeln!(out, "  rows opened in {seconds:.2} s")?;
    }
    if let Some(error) = &attention.rows_error {
        writeln!(out, "  rows: FAIL ({error})")?;
    }
    Ok(())
}
