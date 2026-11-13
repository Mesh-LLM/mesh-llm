//! Adapters between the native receipt recorder and replay, and the
//! [`Recorder`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::{Duration, Instant};

use super::StageReplayer;

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

/// Queues the native recorder's reports for the stage's
/// [`super::StageReceipts`]. The model owns the sink for as long as it
/// lives, and the receipts hold the model for their replays, so the sink
/// only points at the receipts: the stage owns them, and a model that is
/// being unloaded reports into nothing.
pub(super) struct RecorderSink(Weak<super::StageReceipts>);

impl RecorderSink {
    pub(super) fn new(receipts: &Arc<super::StageReceipts>) -> Self {
        Self(Arc::downgrade(receipts))
    }

    fn push(&self, report: Report) {
        if let Some(receipts) = self.0.upgrade() {
            receipts.reports.push(report);
        }
    }
}

impl ReceiptRecorderSink for RecorderSink {
    fn ubatch(&self, ubatch: RecordedUbatch) {
        self.push(Report::Ubatch(ubatch));
    }

    fn unreceiptable(&self, reason: &str) {
        self.push(Report::Unreceiptable(reason.to_string()));
    }

    fn unrecorded_ubatch(&self, reason: &str) {
        self.push(Report::UnrecordedUbatch(reason.to_string()));
    }

    fn memory(&self, event: &ReceiptMemoryEvent) {
        self.push(Report::Memory(*event));
    }

    fn state(&self, seq: i32, state: Option<&[u8]>) {
        self.push(Report::State {
            seq,
            state: state.map(<[u8]>::to_vec),
        });
    }
}

/// The scratch context is freed after this long without a replay.
const SCRATCH_IDLE: Duration = Duration::from_secs(60);

/// When the scratch context was last used, while it may exist.
#[derive(Default)]
struct ScratchUse(Mutex<Option<Instant>>);

/// What to do about the scratch context at some time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScratchDue {
    /// Free it: it has been idle long enough.
    Free,
    /// Look again then.
    At(Instant),
    /// Nothing used it since it was freed.
    Unused,
}

impl ScratchUse {
    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Instant>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A seed or replay may have created the context, or just used it.
    fn used(&self, now: Instant) {
        *self.lock() = Some(now);
    }

    /// What is due at `now`. A context that is due to be freed counts as
    /// unused from here; a use after this marks it again.
    fn due(&self, now: Instant) -> ScratchDue {
        let mut used = self.lock();
        match *used {
            None => ScratchDue::Unused,
            Some(at) if now.saturating_duration_since(at) >= SCRATCH_IDLE => {
                *used = None;
                ScratchDue::Free
            }
            Some(at) => ScratchDue::At(at + SCRATCH_IDLE),
        }
    }
}

/// Replays turns in the stage model's scratch context, and frees the
/// context once replays stop, since it holds a second KV cache.
pub(super) struct NativeReplayer {
    replayer: ReceiptReplayer,
    /// The stage, kept idle while a replay's prefix is copied from it.
    runtime: Weak<Mutex<RuntimeState>>,
    receipts: Weak<super::StageReceipts>,
    /// Held from seeding the scratch context to the end of the replay, so
    /// another replay does not take the seeded cells.
    serial: Mutex<()>,
    scratch: ScratchUse,
}

impl NativeReplayer {
    pub(super) fn new(
        replayer: ReceiptReplayer,
        runtime: Weak<Mutex<RuntimeState>>,
        receipts: Weak<super::StageReceipts>,
    ) -> Self {
        Self {
            replayer,
            runtime,
            receipts,
            serial: Mutex::new(()),
            scratch: ScratchUse::default(),
        }
    }

    /// The stage and its receipts, while the stage is loaded.
    fn stage(&self) -> Result<(Arc<Mutex<RuntimeState>>, Arc<super::StageReceipts>)> {
        match (self.runtime.upgrade(), self.receipts.upgrade()) {
            (Some(runtime), Some(receipts)) => Ok((runtime, receipts)),
            _ => bail!("the stage is shutting down"),
        }
    }

    /// Notes that the scratch context may exist from now, so the stage's
    /// upkeep frees it once it has been idle.
    fn scratch_used(&self, receipts: &super::StageReceipts) {
        self.scratch.used(Instant::now());
        receipts.poke_upkeep();
    }

    /// Copies the prefix cells of `seed` into the scratch context, with the
    /// stage idle and every report of what it computed applied, once the
    /// recorder confirms the cells still hold the prefix.
    ///
    /// Every lane's decode step needs the runtime lock this holds, so only
    /// what must see an idle stage runs under it: the last reports, the
    /// check of the cells, and the native copy. The copy creates the
    /// scratch context when there is none, or one too small, which the
    /// native call does not let a caller do beforehand.
    fn seed(&self, seq: i32, seed: &ReplaySeed) -> Result<()> {
        let (runtime, receipts) = self.stage()?;
        let (cells, positions) = (seed.indices(), seed.positions());
        // the reports made so far are applied without the lock; only those
        // made while it is awaited are left for under it
        receipts.reports.wait_applied();
        // the seed below may create the context and then fail
        self.scratch_used(&receipts);
        let mut state = lock_runtime(&runtime);
        receipts.reports.wait_applied();
        receipts.reports.recorder().check_seed(seed)?;
        let idle = state.model.idle();
        self.replayer
            .seed(&idle, seq, &cells, &positions, seed.n_cells)?;
        match &seed.state {
            Some(state) => self.replayer.seed_state(&idle, seq, state),
            None => Ok(()),
        }
    }
}

impl NativeReplayer {
    /// The rows of `layer` in `cells`, read [`READ_CHUNK_CELLS`] cells at a
    /// time with the stage idle, each chunk once `check` confirms its cells
    /// still hold what the turn read or wrote.
    ///
    /// Every lane's decode step needs the runtime lock, so a long context is
    /// not read under one hold of it: the lanes decode between chunks. That
    /// is safe because a chunk's rows are copied out under the check that
    /// covers them; a later write of a cell read before does not change
    /// what was read.
    fn read_checked_rows(
        &self,
        layer: u32,
        cells: &[PrefixCell],
        check: fn(&Recorder, &[PrefixCell]) -> Result<()>,
    ) -> Result<Vec<StoredRows>> {
        let (runtime, receipts) = self.stage()?;
        let stream = cells.first().map_or(0, |c| c.stream);
        if cells.iter().any(|c| c.stream != stream) {
            bail!("the cells to read are in several streams");
        }
        // a sequence of the stream: stream s holds sequence s when the cache
        // has a stream per sequence, and a unified cache's one holds them all
        let seq = i32::try_from(stream)?;
        let mut out: Vec<StoredRows> = Vec::new();
        for chunk in cells.chunks(READ_CHUNK_CELLS) {
            let indices: Vec<u32> = chunk.iter().map(|c| c.cell).collect();
            // as for a seed: the lock every decode step needs is held only
            // for the last reports, the check of the cells and the read
            receipts.reports.wait_applied();
            let rows = {
                let mut state = lock_runtime(&runtime);
                receipts.reports.wait_applied();
                check(receipts.reports.recorder(), chunk)?;
                let idle = state.model.idle();
                self.replayer.read_rows(&idle, layer, seq, &indices)?
            };
            append_rows(&mut out, rows)?;
        }
        Ok(out)
    }
}

/// Cells whose rows are read under one hold of the runtime lock: with
/// 8 KV heads of 128 dimensions in f16, 2 MiB of K and V rows a layer.
const READ_CHUNK_CELLS: usize = 1024;

/// Appends a chunk's rows, one [`StoredRows`] per stored row of the layer,
/// to those of the chunks before it.
fn append_rows(
    out: &mut Vec<StoredRows>,
    rows: Vec<skippy_runtime::ReceiptStoredRows>,
) -> Result<()> {
    if out.is_empty() {
        out.extend(rows.into_iter().map(|r| StoredRows {
            ggml_type: r.ggml_type,
            row_bytes: r.row_bytes,
            values: r.values,
        }));
        return Ok(());
    }
    if rows.len() != out.len() {
        bail!("the stage read a different number of stored rows per chunk");
    }
    for (kept, r) in out.iter_mut().zip(rows) {
        if (kept.ggml_type, kept.row_bytes) != (r.ggml_type, r.row_bytes) {
            bail!("the stage read stored rows of another type in a later chunk");
        }
        kept.values.extend_from_slice(&r.values);
    }
    Ok(())
}

impl StageReplayer for NativeReplayer {
    fn release_idle(&self, now: Instant) -> Option<Instant> {
        release_between_replays(&self.serial, &self.scratch, now, || {
            self.replayer.release_scratch()
        })
    }
}

/// Frees the scratch context once it has been idle, never during a replay.
///
/// A replay holds `serial` from its seed to its end, and marks the context
/// used only at those two points, so a replay longer than [`SCRATCH_IDLE`]
/// would otherwise see the context freed between its seed and its replay.
/// The idle time is checked under `serial`; while a replay holds it, the
/// context is left alone and looked at again later.
fn release_between_replays(
    serial: &Mutex<()>,
    scratch: &ScratchUse,
    now: Instant,
    release: impl FnOnce() -> Result<()>,
) -> Option<Instant> {
    let _serial = match serial.try_lock() {
        Ok(guard) => guard,
        Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => return Some(now + SCRATCH_IDLE),
    };
    match scratch.due(now) {
        ScratchDue::Unused => None,
        ScratchDue::At(at) => Some(at),
        ScratchDue::Free => {
            if let Err(error) = release() {
                tracing::warn!("receipts: cannot free the replay context: {error:#}");
            }
            None
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
        let _serial = self.serial.lock().unwrap_or_else(PoisonError::into_inner);
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
        if let Some(receipts) = self.receipts.upgrade() {
            self.scratch_used(&receipts);
        }
        replayed?;
        Ok(out)
    }

    fn read_rows(&self, layer: u32, cells: &[PrefixCell]) -> Result<Vec<StoredRows>> {
        self.read_checked_rows(layer, cells, Recorder::check_cells)
    }

    fn read_rows_to_check(&self, layer: u32, cells: &[PrefixCell]) -> Result<Vec<StoredRows>> {
        self.read_checked_rows(layer, cells, Recorder::check_unseen_overwrite)
    }

    fn pin_cells(&self, cells: &[KvCell]) -> Result<()> {
        let streams: Vec<u32> = cells.iter().map(|c| c.stream).collect();
        let cells: Vec<u32> = cells.iter().map(|c| c.cell).collect();
        self.replayer.pin_cells(&streams, &cells)
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

#[cfg(test)]
mod tests {
    use super::*;

    // The scratch context is freed once it has been idle, whether a replay
    // or only a seed used it: a seed that fails after it created the
    // context marks it used all the same.
    #[test]
    fn a_used_scratch_context_is_due_once_idle() {
        let scratch = ScratchUse::default();
        let start = Instant::now();
        assert_eq!(scratch.due(start + SCRATCH_IDLE), ScratchDue::Unused);
        scratch.used(start);
        let early = start + SCRATCH_IDLE / 2;
        assert_eq!(scratch.due(early), ScratchDue::At(start + SCRATCH_IDLE));
        // a later use moves the time
        scratch.used(early);
        assert_eq!(
            scratch.due(start + SCRATCH_IDLE),
            ScratchDue::At(early + SCRATCH_IDLE)
        );
        assert_eq!(scratch.due(early + SCRATCH_IDLE), ScratchDue::Free);
        assert_eq!(scratch.due(early + SCRATCH_IDLE), ScratchDue::Unused);
    }

    // A replay that outlasts the idle time keeps the context it seeded:
    // the upkeep does not free it until the replay lets go.
    #[test]
    fn a_running_replay_keeps_its_scratch_context() {
        let (serial, scratch) = (Mutex::new(()), ScratchUse::default());
        let start = Instant::now();
        scratch.used(start);
        let late = start + SCRATCH_IDLE * 2;
        let freed = std::cell::Cell::new(0);
        let release = || {
            freed.set(freed.get() + 1);
            Ok(())
        };
        {
            let _replay = serial.lock().unwrap();
            let next = release_between_replays(&serial, &scratch, late, release);
            assert_eq!(next, Some(late + SCRATCH_IDLE));
            assert_eq!(freed.get(), 0);
        }
        // once it ends, the idle context is freed
        assert_eq!(
            release_between_replays(&serial, &scratch, late, release),
            None
        );
        assert_eq!(freed.get(), 1);
        assert_eq!(
            release_between_replays(&serial, &scratch, late, release),
            None
        );
        assert_eq!(freed.get(), 1);
    }
}
