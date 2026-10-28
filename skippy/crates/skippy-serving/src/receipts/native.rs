//! Adapters between the native receipt recorder and replay, and the
//! [`Recorder`].

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use skippy_receipts::recorder::Recorder;
use skippy_receipts::recorder::{
    ReplayStep, ReplayedLayer, Replayer, StageInput, TensorKind, UbatchColumns, UbatchRecord,
};
use skippy_runtime::{
    ReceiptMemoryEvent, ReceiptRecorderSink, ReceiptReplayStepRef, ReceiptReplayer,
    ReceiptTensorView, RecordedUbatch,
};

use super::report_queue::Report;

/// The tensors of a native ubatch, by layer and kind. A tensor with fewer
/// columns than tokens covers only the output rows.
struct ViewColumns<'a> {
    output_index: Vec<Option<usize>>,
    tensors: HashMap<(u32, TensorKind), ReceiptTensorView<'a>>,
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
        let tensors = tensors
            .filter_map(|t| Some(((t.layer, TensorKind::from_code(t.kind)?), t)))
            .collect();
        Self {
            output_index,
            tensors,
        }
    }
}

impl UbatchColumns for ViewColumns<'_> {
    fn column(&self, layer: u32, kind: TensorKind, col: usize) -> Option<&[f32]> {
        let tensor = self.tensors.get(&(layer, kind))?;
        let j = if tensor.n_cols == self.output_index.len() {
            col
        } else {
            (*self.output_index.get(col)?)?
        };
        (j < tensor.n_cols).then(|| tensor.column(j))
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

/// Replays turns in the stage model's scratch context.
pub(super) struct NativeReplayer(pub(super) ReceiptReplayer);

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
        self.0.replay(seq, layer, &refs, &mut |view| {
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
        })?;
        Ok(out)
    }
}
