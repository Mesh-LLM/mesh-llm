//! Adapters between the native receipt recorder and replay, and the
//! [`Recorder`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use skippy_receipts::layer_tensors::{TensorColumn, TensorKind, UbatchColumns};
use skippy_receipts::recorder::Recorder;
use skippy_receipts::recorder::{ReplayStep, ReplayedLayer, Replayer, StageInput, UbatchRecord};
use skippy_runtime::{
    ReceiptMemoryEvent, ReceiptRecorderSink, ReceiptReplayStepRef, ReceiptReplayer,
    ReceiptTensorView, RecordedUbatch,
};

use super::report_queue::Report;

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
        let j = if tensor.n_cols == self.output_index.len() {
            col
        } else {
            (*self.output_index.get(col)?)?
        };
        (j < tensor.n_cols).then(|| TensorColumn {
            ggml_type: tensor.ggml_type,
            flags: tensor.flags,
            name: tensor.name,
            bytes: tensor.column(j),
        })
    }
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
    let output = record.output.clone();
    let columns = ViewColumns::new(&output, ubatch.tensors());
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
}

/// The scratch context is freed after this long without a replay.
const SCRATCH_IDLE: Duration = Duration::from_secs(60);
const IDLE_POLL: Duration = Duration::from_secs(5);

/// Replays turns in the stage model's scratch context, and frees the
/// context once replays stop, since it holds a second KV cache.
pub(super) struct NativeReplayer(Arc<Scratch>);

struct Scratch {
    replayer: ReceiptReplayer,
    /// When the last replay finished, until the context is freed.
    last_replay: Mutex<Option<Instant>>,
}

impl NativeReplayer {
    pub(super) fn new(replayer: ReceiptReplayer) -> Self {
        let scratch = Arc::new(Scratch {
            replayer,
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
        layer: u32,
        steps: &[ReplayStep],
    ) -> Result<Vec<(u32, ReplayedLayer)>> {
        let refs: Vec<ReceiptReplayStepRef<'_>> = steps
            .iter()
            .map(|step| match step {
                ReplayStep::Decode { record, capture } => ReceiptReplayStepRef::Decode {
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
                    capture: !capture.is_empty(),
                },
                ReplayStep::SeqRm { p0, p1 } => ReceiptReplayStepRef::SeqRm { p0: *p0, p1: *p1 },
            })
            .collect();
        let mut out = Vec::new();
        let replayed = self.0.replayer.replay(seq, layer, &refs, &mut |view| {
            let Some(ReplayStep::Decode { capture, .. }) = view.step.and_then(|i| steps.get(i))
            else {
                bail!("replay captured an unexpected step");
            };
            let columns = ViewColumns::new(view.output, view.tensors.iter().copied());
            for (col, (&s, &p)) in view.seq.iter().zip(view.pos).enumerate() {
                let Ok(position) = u32::try_from(p) else {
                    continue;
                };
                if s == seq && capture.contains(&position) {
                    out.push((position, ReplayedLayer::from_columns(layer, &columns, col)?));
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
}
