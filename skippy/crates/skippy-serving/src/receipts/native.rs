//! Adapters between the native receipt recorder and replay, and the
//! [`Recorder`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use skippy_receipts::attention::{
    AttentionDescriptor, AttentionLayout, AttentionOp, OutputRotation, Window, q_rounding,
};
use skippy_receipts::commitment::ValueDigest;
use skippy_receipts::layer_tensors::{TensorColumn, TensorKind, UbatchColumns};
use skippy_receipts::recorder::Recorder;
use skippy_receipts::recorder::{
    AnchorValue, KvCell, PrefixCell, ReplayAnchor, ReplayScope, ReplaySeed, ReplayStep,
    ReplayedLayer, ReplayedPosition, Replayer, StageInput, StoredRows, UbatchRecord,
};
use skippy_runtime::{
    ReceiptAnchorRef, ReceiptAttentionOp, ReceiptAttentionWindow, ReceiptMemoryEvent,
    ReceiptRecorderSink, ReceiptReplayScope, ReceiptReplayStepRef, ReceiptReplayer,
    ReceiptTensorView, RecordedUbatch,
};

use super::report_queue::Report;
use crate::runtime_state::RuntimeState;
use crate::runtime_state::panic_recovery::lock_runtime;

/// The tensors of a native ubatch, by layer, kind and index. A tensor with
/// fewer columns than tokens covers only the output rows.
struct ViewColumns<'a> {
    output_index: Vec<Option<usize>>,
    tensors: HashMap<(u32, TensorKind, u32), ReceiptTensorView<'a>>,
    counts: HashMap<(u32, TensorKind), u32>,
}

impl<'a> ViewColumns<'a> {
    fn new(output: &[i8], tensors: impl Iterator<Item = ReceiptTensorView<'a>>) -> Self {
        let mut next = 0;
        let output_index = output
            .iter()
            .map(|&flag| {
                (flag != 0).then(|| {
                    next += 1;
                    next - 1
                })
            })
            .collect();
        let tensors: HashMap<_, _> = tensors
            .filter_map(|t| Some(((t.layer, TensorKind::from_code(t.kind)?, t.index), t)))
            .collect();
        let mut counts: HashMap<(u32, TensorKind), u32> = HashMap::new();
        for &(layer, kind, index) in tensors.keys() {
            let count = counts.entry((layer, kind)).or_default();
            *count = (*count).max(index + 1);
        }
        Self {
            output_index,
            tensors,
            counts,
        }
    }
}

impl UbatchColumns for ViewColumns<'_> {
    fn count(&self, layer: u32, kind: TensorKind) -> u32 {
        self.counts.get(&(layer, kind)).copied().unwrap_or(0)
    }

    fn column(
        &self,
        layer: u32,
        kind: TensorKind,
        index: u32,
        col: usize,
    ) -> Option<TensorColumn<'_>> {
        let tensor = self.tensors.get(&(layer, kind, index))?;
        let j = if kind == TensorKind::MixingState {
            sequence_column(self.output_index.len(), tensor.n_cols, col)?
        } else if tensor.n_cols == self.output_index.len() {
            col
        } else {
            (*self.output_index.get(col)?)?
        };
        (j < tensor.n_cols).then(|| TensorColumn {
            ggml_type: tensor.ggml_type,
            flags: tensor.flags,
            name: tensor.name,
            bytes: tensor.column(j),
            digest: tensor.column_digest(j).map(|digest| ValueDigest {
                len: tensor.row_bytes as u32,
                digest,
            }),
        })
    }
}

/// The state column of the token at `col`: a ubatch's sequences are runs of
/// `n_tokens / n_seqs` tokens, and each holds its state at its first token.
fn sequence_column(n_tokens: usize, n_seqs: usize, col: usize) -> Option<usize> {
    let per_seq = n_tokens.checked_div(n_seqs).filter(|&n| n > 0)?;
    col.is_multiple_of(per_seq).then_some(col / per_seq)
}

/// Records one native ubatch. The ubatch is dropped once its leaves are
/// hashed, which lets the runtime deliver the next one.
pub(super) fn apply_ubatch(recorder: &Recorder, mut ubatch: RecordedUbatch) {
    let record = UbatchRecord {
        token: std::mem::take(&mut ubatch.token),
        seq: std::mem::take(&mut ubatch.seq),
        pos: std::mem::take(&mut ubatch.pos),
        output: std::mem::take(&mut ubatch.output),
        cell: std::mem::take(&mut ubatch.cell),
        n_kv: ubatch.n_kv,
        input: ubatch
            .input
            .take()
            .map(|(rows, row_bytes, ggml_type)| StageInput {
                rows,
                row_bytes,
                ggml_type,
            }),
    };
    let columns = ViewColumns::new(&record.output, ubatch.tensors());
    recorder.record_ubatch(record, &columns);
}

/// Queues the native recorder's reports for the stage's [`super::StageReceipts`].
pub(super) struct RecorderSink(pub(super) Arc<super::StageReceipts>);

impl ReceiptRecorderSink for RecorderSink {
    fn ubatch(&self, ubatch: RecordedUbatch) {
        self.0.reports.push(Report::Ubatch(ubatch));
    }

    fn unreceiptable(&self, reason: &str) {
        self.0
            .reports
            .push(Report::Unreceiptable(reason.to_string()));
    }

    fn memory(&self, event: &ReceiptMemoryEvent) {
        self.0.reports.push(Report::Memory(*event));
    }

    fn state(&self, seq: i32, state: Option<&[u8]>) {
        self.0.reports.push(Report::State {
            seq,
            state: state.map(<[u8]>::to_vec),
        });
    }
}

/// The scratch context is freed after this long without a replay.
const SCRATCH_IDLE: Duration = Duration::from_secs(60);
const IDLE_POLL: Duration = Duration::from_secs(5);

/// Replays turns in the stage model's scratch context, and frees the
/// context once replays stop, since it holds a second KV cache.
pub(super) struct NativeReplayer(Arc<Scratch>);

struct Scratch {
    replayer: ReceiptReplayer,
    /// The stage, kept idle while a replay's prefix is copied from it.
    runtime: Weak<Mutex<RuntimeState>>,
    receipts: Weak<super::StageReceipts>,
    /// Held from seeding the scratch context to the end of the replay, so
    /// another replay does not take the seeded cells.
    serial: Mutex<()>,
    /// When the last replay finished, until the context is freed.
    last_replay: Mutex<Option<Instant>>,
}

impl NativeReplayer {
    pub(super) fn new(
        replayer: ReceiptReplayer,
        runtime: Weak<Mutex<RuntimeState>>,
        receipts: Weak<super::StageReceipts>,
    ) -> Self {
        let scratch = Arc::new(Scratch {
            replayer,
            runtime,
            receipts,
            serial: Mutex::new(()),
            last_replay: Mutex::new(None),
        });
        let weak = Arc::downgrade(&scratch);
        let spawned = std::thread::Builder::new()
            .name("receipt-scratch".into())
            .spawn(move || release_when_idle(&weak));
        if let Err(error) = spawned {
            tracing::warn!("receipts: the replay context will not be freed when idle: {error}");
        }
        Self(scratch)
    }

    /// Copies the prefix cells of `seed` into the scratch context, with the
    /// stage idle and every report of what it computed applied, once the
    /// recorder confirms the cells still hold the prefix.
    fn seed(&self, seq: i32, seed: &ReplaySeed) -> Result<()> {
        let (Some(runtime), Some(receipts)) = (self.0.runtime.upgrade(), self.0.receipts.upgrade())
        else {
            bail!("the stage is shutting down");
        };
        let _idle = lock_runtime(&runtime);
        receipts.reports.wait_applied();
        receipts.reports.recorder().check_seed(seed)?;
        self.0
            .replayer
            .seed(seq, &seed.indices(), &seed.positions(), seed.n_cells)?;
        match &seed.state {
            Some(state) => self.0.replayer.seed_state(seq, state),
            None => Ok(()),
        }
    }
}

fn release_when_idle(scratch: &Weak<Scratch>) {
    loop {
        std::thread::sleep(IDLE_POLL);
        let Some(scratch) = scratch.upgrade() else {
            return;
        };
        let mut last = scratch
            .last_replay
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if last.is_some_and(|at| at.elapsed() >= SCRATCH_IDLE) {
            // a replay finishing after this sets the time again
            *last = None;
            if let Err(error) = scratch.replayer.release_scratch() {
                tracing::warn!("receipts: cannot free the replay context: {error:#}");
            }
        }
    }
}

impl Replayer for NativeReplayer {
    fn replay(
        &self,
        seq: i32,
        layers: &[u32],
        scope: ReplayScope<'_>,
        seed: Option<&ReplaySeed>,
        steps: &[ReplayStep],
    ) -> Result<Vec<ReplayedPosition>> {
        let _serial = self.0.serial.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(seed) = seed {
            self.seed(seq, seed)?;
        }
        let refs: Vec<ReceiptReplayStepRef<'_>> = steps
            .iter()
            .map(|step| match step {
                ReplayStep::Decode {
                    record,
                    capture,
                    mixing,
                } => ReceiptReplayStepRef::Decode {
                    token: &record.token,
                    seq: &record.seq,
                    pos: &record.pos,
                    output: &record.output,
                    cell: &record.cell,
                    n_kv: record.n_kv,
                    input: record
                        .input
                        .as_ref()
                        .map(|input| (input.rows.as_slice(), input.row_bytes, input.ggml_type)),
                    capture: !capture.is_empty() || !mixing.is_empty(),
                },
                ReplayStep::SeqRm { p0, p1 } => ReceiptReplayStepRef::SeqRm { p0: *p0, p1: *p1 },
            })
            .collect();
        let anchors: Vec<ReceiptAnchorRef<'_>> = match scope {
            ReplayScope::Anchored(anchors) => anchors.iter().map(anchor_ref).collect(),
            _ => Vec::new(),
        };
        let scope = match scope {
            ReplayScope::Full => ReceiptReplayScope::Full,
            ReplayScope::Committed => ReceiptReplayScope::Committed,
            ReplayScope::Anchored(_) => ReceiptReplayScope::Anchored(&anchors),
        };
        let mut out = Vec::new();
        let replayed = self
            .0
            .replayer
            .replay(seq, layers, &scope, &refs, &mut |view| {
                let Some(ReplayStep::Decode {
                    capture, mixing, ..
                }) = view.step.and_then(|i| steps.get(i))
                else {
                    bail!("replay captured an unexpected step");
                };
                let columns = ViewColumns::new(view.output, view.tensors.iter().copied());
                let attention = attention_by_layer(&view.attention);
                for (col, (&s, &p)) in view.seq.iter().zip(view.pos).enumerate() {
                    let Ok(position) = u32::try_from(p) else {
                        continue;
                    };
                    if s == seq && capture.contains(&position) {
                        for &layer in layers {
                            let mut values = ReplayedLayer::from_columns(layer, &columns, col)?;
                            values.attention = attention.get(&layer).cloned().unwrap_or_default();
                            out.push((position, layer, values));
                        }
                    } else if s == seq && mixing.contains(&position) {
                        for &layer in layers {
                            let values = ReplayedLayer::from_mixing_columns(layer, &columns, col);
                            out.push((position, layer, values));
                        }
                    }
                }
                Ok(())
            });
        *self
            .0
            .last_replay
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(Instant::now());
        replayed?;
        Ok(out)
    }

    fn read_rows(&self, layer: u32, cells: &[PrefixCell]) -> Result<Vec<StoredRows>> {
        let (Some(runtime), Some(receipts)) = (self.0.runtime.upgrade(), self.0.receipts.upgrade())
        else {
            bail!("the stage is shutting down");
        };
        let _idle = lock_runtime(&runtime);
        receipts.reports.wait_applied();
        receipts.reports.recorder().check_cells(cells)?;
        let stream = cells.first().map_or(0, |c| c.stream);
        if cells.iter().any(|c| c.stream != stream) {
            bail!("the cells to read are in several streams");
        }
        // a sequence of the stream: stream s holds sequence s when the cache
        // has a stream per sequence, and a unified cache's one holds them all
        let seq = i32::try_from(stream)?;
        let indices: Vec<u32> = cells.iter().map(|c| c.cell).collect();
        let rows = self.0.replayer.read_rows(layer, seq, &indices)?;
        Ok(rows
            .into_iter()
            .map(|r| StoredRows {
                ggml_type: r.ggml_type,
                row_bytes: r.row_bytes,
                values: r.values,
            })
            .collect())
    }

    fn pin_cells(&self, cells: &[KvCell]) -> Result<()> {
        let streams: Vec<u32> = cells.iter().map(|c| c.stream).collect();
        let cells: Vec<u32> = cells.iter().map(|c| c.cell).collect();
        self.0.replayer.pin_cells(&streams, &cells)
    }
}

fn anchor_ref(anchor: &ReplayAnchor) -> ReceiptAnchorRef<'_> {
    let (kind, index) = match anchor.value {
        AnchorValue::Residual => (TensorKind::Residual, 0),
        AnchorValue::MixingOutput(index) => (TensorKind::MixingOutput, index),
    };
    ReceiptAnchorRef {
        step: anchor.step,
        seq: anchor.seq,
        pos: anchor.pos,
        layer: anchor.layer,
        kind: kind.code(),
        index,
        values: &anchor.values,
    }
}

/// A replay's attention ops, by layer.
fn attention_by_layer(ops: &[ReceiptAttentionOp]) -> HashMap<u32, Vec<AttentionDescriptor>> {
    let mut by_layer: HashMap<u32, Vec<AttentionDescriptor>> = HashMap::new();
    for op in ops {
        by_layer
            .entry(op.layer)
            .or_default()
            .push(attention_descriptor(op));
    }
    by_layer
}

/// An attention op as the receipt check describes it.
fn attention_descriptor(op: &ReceiptAttentionOp) -> AttentionDescriptor {
    let a = op.op.as_ref().map_err(Clone::clone)?;
    Ok(AttentionLayout {
        layer: op.layer,
        op: AttentionOp {
            n_head: a.n_head,
            n_head_kv: a.n_head_kv,
            head_dim_k: a.head_dim_k,
            head_dim_v: a.head_dim_v,
            scale: f64::from(a.scale),
            softcap: f64::from(a.softcap),
            sinks: a.sinks.clone(),
            output_rotation: a
                .output_rotation
                .clone()
                .map(|(n, matrix)| OutputRotation { n, matrix }),
            q_rounding: a
                .q_rounding
                .map(q_rounding)
                .transpose()
                .map_err(|e| e.to_string())?,
            window: match a.window {
                ReceiptAttentionWindow::Causal => Window::Causal,
                ReceiptAttentionWindow::Sliding(n) => Window::Sliding(n),
                ReceiptAttentionWindow::Chunked(n) => Window::Chunked(n),
            },
        },
        q_input: a.q_input,
        k_store: a.k_store,
        v_store: a.v_store,
        output: a.output,
        output_gate: a.output_gate,
    })
}
