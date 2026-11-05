//! Delivers the native receipt recorder to a Rust sink, and replays a
//! recorded sequence to open a receipt challenge.

use std::ffi::{CStr, c_void};
use std::ptr;
use std::sync::{Arc, Condvar, Mutex, PoisonError, Weak};

use anyhow::{Result, anyhow, bail};
use skippy_ffi::{ReceiptAnchor, ReceiptReplayStep, ReceiptTensor, ReceiptUbatch};

use super::{StageModel, StageModelInner};

use crate::error::ensure_ok;
pub use skippy_ffi::ReceiptMemoryEvent;

/// One reported tensor, column per token (or per output row).
#[derive(Clone, Copy)]
pub struct ReceiptTensorView<'a> {
    pub layer: u32,
    /// A `skippy_ffi::RECEIPT_TENSOR_*` kind.
    pub kind: i32,
    /// Ordinal among the layer's tensors of this kind.
    pub index: u32,
    /// ggml type of the values: F32, I32, or a stored row's cache type.
    pub ggml_type: i32,
    /// `skippy_ffi::RECEIPT_TENSOR_ROUTER` and other flags.
    pub flags: i32,
    /// Elements per column.
    pub len: usize,
    pub row_bytes: usize,
    pub n_cols: usize,
    /// Column `j` at `values[j * row_bytes..]`; empty when the recorder
    /// reported digests instead.
    pub values: &'a [u8],
    /// Per column, the BLAKE3 digest of its `row_bytes` bytes, when the
    /// recorder computed it where the values are; empty otherwise.
    pub digests: &'a [[u8; 32]],
    /// The weight's name for a matmul, else the graph node's name.
    pub name: &'a str,
}

impl<'a> ReceiptTensorView<'a> {
    /// Column `j` as raw bytes; empty when only its digest was reported.
    pub fn column(&self, j: usize) -> &'a [u8] {
        self.values
            .get(j * self.row_bytes..(j + 1) * self.row_bytes)
            .unwrap_or(&[])
    }

    /// The digest of column `j`, when the recorder reported digests.
    pub fn column_digest(&self, j: usize) -> Option<[u8; 32]> {
        self.digests.get(j).copied()
    }
}

/// One computed ubatch, borrowed for the duration of the callback.
pub struct ReceiptUbatchView<'a> {
    /// Replay step index, or `None` while serving.
    pub step: Option<usize>,
    pub token: &'a [i32],
    pub seq: &'a [i32],
    pub pos: &'a [i32],
    pub output: &'a [i8],
    pub cell: &'a [u32],
    pub n_kv: u32,
    /// Stage input rows and their ggml type, for a stage after the first.
    pub input: Option<(&'a [u8], usize, i32)>,
    pub tensors: Vec<ReceiptTensorView<'a>>,
    /// Replay: the attention ops of the reported layers.
    pub attention: Vec<super::ReceiptAttentionOp>,
}

/// Receives the recorder's reports, in order: on the decoding thread, or on a
/// thread that calls [`ReceiptFlusher::flush`].
pub trait ReceiptRecorderSink: Send + Sync {
    /// A computed ubatch. It can be kept and read on any thread, but the next
    /// ubatch is not delivered until it is dropped: decoding waits for it.
    fn ubatch(&self, ubatch: RecordedUbatch);
    /// The ubatch could not be receipted; the sequences it touched cannot be opened.
    fn unreceiptable(&self, reason: &str);
    fn memory(&self, event: &ReceiptMemoryEvent);
    /// The recurrent state a sequence's next turn starts from, on a model
    /// with hybrid memory; `None` when it could not be read. Decoding waits
    /// for the call.
    fn state(&self, seq: i32, state: Option<&[u8]>) {
        let _ = (seq, state);
    }
}

/// Whether a delivered ubatch still borrows the native tensor buffers.
#[derive(Default)]
struct ReleaseGate {
    held: Mutex<bool>,
    released: Condvar,
}

impl ReleaseGate {
    fn hold(&self) {
        *self.held.lock().unwrap_or_else(PoisonError::into_inner) = true;
    }

    fn release(&self) {
        *self.held.lock().unwrap_or_else(PoisonError::into_inner) = false;
        self.released.notify_all();
    }

    fn wait_released(&self) {
        let held = self.held.lock().unwrap_or_else(PoisonError::into_inner);
        drop(
            self.released
                .wait_while(held, |held| *held)
                .unwrap_or_else(PoisonError::into_inner),
        );
    }
}

/// What the native recorder's user data points at.
struct RecorderBinding {
    sink: Arc<dyn ReceiptRecorderSink>,
    model: Weak<StageModelInner>,
    gate: Arc<ReleaseGate>,
}

/// Keeps the sink alive while the native recorder points at it.
#[derive(Default)]
pub(super) struct InstalledRecorder(Mutex<Option<Box<RecorderBinding>>>);

struct RecordedTensor {
    layer: u32,
    kind: i32,
    index: u32,
    ggml_type: i32,
    flags: i32,
    len: usize,
    row_bytes: usize,
    n_cols: usize,
    // pointer and length of the values, and of the digests
    values: (*const u8, usize),
    digests: (*const [u8; 32], usize),
    // the native name lives in the graph, which the next ubatch may rebuild
    name: String,
}

/// One computed ubatch, owned. The native recorder keeps the tensor values
/// of a ubatch until the callback for the next ubatch returns, and the
/// runtime does not deliver the next ubatch until this is dropped, so the
/// values stay valid for its lifetime. It keeps the model alive.
pub struct RecordedUbatch {
    pub token: Vec<i32>,
    pub seq: Vec<i32>,
    pub pos: Vec<i32>,
    pub output: Vec<i8>,
    pub cell: Vec<u32>,
    pub n_kv: u32,
    /// Stage input rows, row size in bytes, and ggml type.
    pub input: Option<(Vec<u8>, usize, i32)>,
    tensors: Vec<RecordedTensor>,
    gate: Arc<ReleaseGate>,
    _model: Arc<StageModelInner>,
}

// SAFETY: the tensor pointers are read-only and stay valid until drop (see
// the type docs); everything else is owned.
unsafe impl Send for RecordedUbatch {}
unsafe impl Sync for RecordedUbatch {}

impl RecordedUbatch {
    fn new(
        view: &ReceiptUbatchView<'_>,
        gate: Arc<ReleaseGate>,
        model: Arc<StageModelInner>,
    ) -> Self {
        gate.hold();
        Self {
            token: view.token.to_vec(),
            seq: view.seq.to_vec(),
            pos: view.pos.to_vec(),
            output: view.output.to_vec(),
            cell: view.cell.to_vec(),
            n_kv: view.n_kv,
            input: view
                .input
                .map(|(rows, row, ggml_type)| (rows.to_vec(), row, ggml_type)),
            tensors: view
                .tensors
                .iter()
                .map(|t| RecordedTensor {
                    layer: t.layer,
                    kind: t.kind,
                    index: t.index,
                    ggml_type: t.ggml_type,
                    flags: t.flags,
                    len: t.len,
                    row_bytes: t.row_bytes,
                    n_cols: t.n_cols,
                    values: (t.values.as_ptr(), t.values.len()),
                    digests: (t.digests.as_ptr(), t.digests.len()),
                    name: t.name.to_owned(),
                })
                .collect(),
            gate,
            _model: model,
        }
    }

    /// The reported tensors.
    pub fn tensors(&self) -> impl Iterator<Item = ReceiptTensorView<'_>> {
        self.tensors.iter().map(|t| ReceiptTensorView {
            layer: t.layer,
            kind: t.kind,
            index: t.index,
            ggml_type: t.ggml_type,
            flags: t.flags,
            len: t.len,
            row_bytes: t.row_bytes,
            n_cols: t.n_cols,
            // SAFETY: valid until self is dropped, see the type docs.
            values: unsafe { std::slice::from_raw_parts(t.values.0, t.values.1) },
            digests: unsafe { std::slice::from_raw_parts(t.digests.0, t.digests.1) },
            name: &t.name,
        })
    }
}

impl Drop for RecordedUbatch {
    fn drop(&mut self) {
        self.gate.release();
    }
}

/// Safe views of the tensors of a native ubatch.
pub(super) unsafe fn tensor_views<'a>(
    u: &'a ReceiptUbatch,
) -> Result<Vec<ReceiptTensorView<'a>>, String> {
    let raw: &[ReceiptTensor] = if u.n_tensors == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(u.tensors, u.n_tensors) }
    };
    let mut tensors = Vec::with_capacity(raw.len());
    for t in raw {
        let (Ok(layer), Ok(index), Ok(len), Ok(row_bytes), Ok(n_cols)) = (
            u32::try_from(t.layer),
            u32::try_from(t.index),
            usize::try_from(t.len),
            usize::try_from(t.row_bytes),
            usize::try_from(t.n_cols),
        ) else {
            return Err("malformed receipt tensor".to_string());
        };
        let name = if t.name.is_null() {
            ""
        } else {
            unsafe { CStr::from_ptr(t.name) }.to_str().unwrap_or("")
        };
        tensors.push(ReceiptTensorView {
            layer,
            kind: t.kind,
            index,
            ggml_type: t.ggml_type,
            flags: t.flags,
            len,
            row_bytes,
            n_cols,
            values: if t.values.is_null() || row_bytes * n_cols == 0 {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(t.values.cast::<u8>(), row_bytes * n_cols) }
            },
            digests: if t.digests.is_null() || n_cols == 0 {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(t.digests.cast::<[u8; 32]>(), n_cols) }
            },
            name,
        });
    }
    Ok(tensors)
}

/// Builds a safe view of a native ubatch; `Err` carries the native reason
/// the ubatch cannot be receipted.
unsafe fn view<'a>(u: &'a ReceiptUbatch) -> Result<ReceiptUbatchView<'a>, String> {
    if !u.error.is_null() {
        return Err(unsafe { CStr::from_ptr(u.error) }
            .to_string_lossy()
            .into_owned());
    }
    let n = usize::try_from(u.n_tokens).map_err(|_| "negative token count".to_string())?;
    let slice = |p: *const i32| unsafe { std::slice::from_raw_parts(p, n) };
    let input = if u.input.is_null() {
        None
    } else {
        let row = usize::try_from(u.input_row_bytes).map_err(|_| "bad input row".to_string())?;
        Some((
            unsafe { std::slice::from_raw_parts(u.input.cast::<u8>(), row * n) },
            row,
            u.input_type,
        ))
    };
    let tensors = unsafe { tensor_views(u) }?;
    let attention = unsafe { super::receipt_attention::attention_ops(u) }?;
    Ok(ReceiptUbatchView {
        step: usize::try_from(u.step).ok(),
        token: slice(u.token),
        seq: slice(u.seq),
        pos: slice(u.pos),
        output: unsafe { std::slice::from_raw_parts(u.output, n) },
        cell: unsafe { std::slice::from_raw_parts(u.cell, n) },
        n_kv: u.n_kv,
        input,
        tensors,
        attention,
    })
}

unsafe extern "C" fn deliver_ubatch(user_data: *mut c_void, ubatch: *const ReceiptUbatch) {
    if user_data.is_null() || ubatch.is_null() {
        return;
    }
    // SAFETY: user_data is the binding owned by InstalledRecorder, and the
    // native side keeps the ubatch buffers alive for the call.
    let binding = unsafe { &*user_data.cast::<RecorderBinding>() };
    // Once this returns, the native side may overwrite the previous ubatch's
    // buffers.
    binding.gate.wait_released();
    let Some(model) = binding.model.upgrade() else {
        binding.sink.unreceiptable("the model is being freed");
        return;
    };
    match unsafe { view(&*ubatch) } {
        Ok(view) => binding
            .sink
            .ubatch(RecordedUbatch::new(&view, binding.gate.clone(), model)),
        Err(reason) => binding.sink.unreceiptable(&reason),
    }
}

unsafe extern "C" fn deliver_memory(user_data: *mut c_void, event: *const ReceiptMemoryEvent) {
    if user_data.is_null() || event.is_null() {
        return;
    }
    // SAFETY: as in deliver_ubatch.
    let (binding, event) = unsafe { (&*user_data.cast::<RecorderBinding>(), &*event) };
    binding.sink.memory(event);
}

unsafe extern "C" fn deliver_state(
    user_data: *mut c_void,
    seq: i32,
    bytes: *const c_void,
    n_bytes: usize,
) {
    if user_data.is_null() {
        return;
    }
    // SAFETY: as in deliver_ubatch; the native side keeps `bytes` alive for
    // the call.
    let binding = unsafe { &*user_data.cast::<RecorderBinding>() };
    let state = (!bytes.is_null() && n_bytes > 0)
        .then(|| unsafe { std::slice::from_raw_parts(bytes.cast::<u8>(), n_bytes) });
    binding.sink.state(seq, state);
}

/// One step of a receipt replay.
pub enum ReceiptReplayStepRef<'a> {
    Decode {
        token: &'a [i32],
        seq: &'a [i32],
        pos: &'a [i32],
        output: &'a [i8],
        cell: &'a [u32],
        n_kv: u32,
        /// Stage input rows, row size in bytes, and ggml type.
        input: Option<(&'a [u8], usize, i32)>,
        capture: bool,
    },
    /// Removes positions `[p0, p1)` of the replayed sequence; `p1 < 0` has no end.
    SeqRm { p0: i32, p1: i32 },
}

/// A committed value an anchored replay writes into one token's column.
pub struct ReceiptAnchorRef<'a> {
    /// The decode step, as an index into the replay's steps.
    pub step: usize,
    pub seq: i32,
    pub pos: i32,
    pub layer: u32,
    /// `RECEIPT_TENSOR_RESIDUAL`, the residual entering `layer`, or
    /// `RECEIPT_TENSOR_MIXING_OUTPUT`.
    pub kind: i32,
    pub index: u32,
    pub values: &'a [f32],
}

/// Which tensors a replay reports, and from which values.
pub enum ReceiptReplayScope<'a> {
    /// The committed tensors, the matmuls and the selections.
    Full,
    /// Only the tensors the live recorder marks, so a backend fuses its
    /// kernels as it did live; no matmuls or selections.
    Committed,
    /// As [`Self::Full`], with the anchors' values written into their
    /// tokens' columns as soon as they are computed.
    Anchored(&'a [ReceiptAnchorRef<'a>]),
}

/// Replays recorded sequences in the model's scratch context. Cheap to clone
/// and usable from any thread; it keeps the model alive.
#[derive(Clone)]
pub struct ReceiptReplayer {
    inner: Arc<StageModelInner>,
}

struct CaptureState<'f> {
    on_capture: &'f mut dyn FnMut(&ReceiptUbatchView<'_>) -> Result<()>,
    error: Option<anyhow::Error>,
}

unsafe extern "C" fn deliver_capture(user_data: *mut c_void, ubatch: *const ReceiptUbatch) {
    if user_data.is_null() || ubatch.is_null() {
        return;
    }
    // SAFETY: user_data is the CaptureState on replay()'s stack.
    let state = unsafe { &mut *user_data.cast::<CaptureState<'_>>() };
    if state.error.is_some() {
        return;
    }
    let result = match unsafe { view(&*ubatch) } {
        Ok(view) => (state.on_capture)(&view),
        Err(reason) => Err(anyhow!("replay cannot capture: {reason}")),
    };
    if let Err(error) = result {
        state.error = Some(error);
    }
}

impl ReceiptReplayer {
    pub(super) fn raw(&self) -> *mut skippy_ffi::Model {
        self.inner.raw
    }

    /// Replays `seq` and calls `on_capture` for every captured step, which
    /// reports each of the sorted `layers` as `scope` says.
    pub fn replay(
        &self,
        seq: i32,
        layers: &[u32],
        scope: &ReceiptReplayScope<'_>,
        steps: &[ReceiptReplayStepRef<'_>],
        on_capture: &mut dyn FnMut(&ReceiptUbatchView<'_>) -> Result<()>,
    ) -> Result<()> {
        let raw_steps = steps.iter().map(raw_step).collect::<Result<Vec<_>>>()?;
        let layers = layers
            .iter()
            .map(|&layer| i32::try_from(layer))
            .collect::<Result<Vec<_>, _>>()?;
        let mut state = CaptureState {
            on_capture,
            error: None,
        };
        let mut error = ptr::null_mut();
        let user_data = ptr::from_mut(&mut state).cast::<c_void>();
        let (model, steps) = (self.inner.raw, raw_steps.as_ptr());
        let (n_layers, n_steps, layers) = (layers.len(), raw_steps.len(), layers.as_ptr());
        let on_capture: skippy_ffi::ReceiptUbatchFn = Some(deliver_capture);
        let status = match scope {
            ReceiptReplayScope::Full => unsafe {
                skippy_ffi::skippy_model_replay_receipt(
                    model, seq, layers, n_layers, steps, n_steps, on_capture, user_data, &mut error,
                )
            },
            ReceiptReplayScope::Committed => unsafe {
                skippy_ffi::skippy_model_replay_receipt_committed(
                    model, seq, layers, n_layers, steps, n_steps, on_capture, user_data, &mut error,
                )
            },
            ReceiptReplayScope::Anchored(anchors) => {
                let anchors = anchors.iter().map(raw_anchor).collect::<Result<Vec<_>>>()?;
                unsafe {
                    skippy_ffi::skippy_model_replay_receipt_anchored(
                        model,
                        seq,
                        layers,
                        n_layers,
                        steps,
                        n_steps,
                        anchors.as_ptr(),
                        anchors.len(),
                        on_capture,
                        user_data,
                        &mut error,
                    )
                }
            }
        };
        ensure_ok(status, error)?;
        state.error.map_or(Ok(()), Err)
    }

    /// Seeds the next replay of `seq` with the stage context's `cells`, which
    /// must hold positions `pos` (of any sequence); they become positions of
    /// `seq` in the scratch context, which is made to hold `n_cells` cells.
    /// The caller keeps the stage context idle meanwhile: no decode or memory
    /// operation may run on it.
    pub fn seed(&self, seq: i32, cells: &[u32], pos: &[i32], n_cells: u32) -> Result<()> {
        if cells.len() != pos.len() {
            bail!("a seeded cell needs a position");
        }
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_model_seed_receipt_replay(
                self.inner.raw,
                seq,
                cells.as_ptr(),
                pos.as_ptr(),
                cells.len(),
                n_cells,
                &mut error,
            )
        };
        ensure_ok(status, error)
    }

    /// Imports the recurrent state a turn started from into `seq` of the
    /// scratch context, after [`Self::seed`] copied its KV cells.
    pub fn seed_state(&self, seq: i32, state: &[u8]) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_model_seed_receipt_state(
                self.inner.raw,
                seq,
                state.as_ptr().cast::<c_void>(),
                state.len(),
                &mut error,
            )
        };
        ensure_ok(status, error)
    }

    /// Frees the scratch context; the next replay creates it again. A
    /// running replay keeps it.
    pub fn release_scratch(&self) -> Result<()> {
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_model_release_receipt_scratch(self.inner.raw, &mut error) };
        ensure_ok(status, error)
    }
}

/// Reports the ubatches the receipt recorder holds back: a device that copies
/// asynchronously reports a ubatch two ubatches late, or before the next
/// memory event. Call [`ReceiptFlusher::flush`] before reading what the sink
/// received.
#[derive(Clone)]
pub struct ReceiptFlusher {
    inner: Weak<StageModelInner>,
}

impl ReceiptFlusher {
    /// Delivers the held-back ubatches, if any, to the sink on this thread.
    /// A freed model has nothing held back.
    pub fn flush(&self) -> Result<()> {
        let Some(inner) = self.inner.upgrade() else {
            return Ok(());
        };
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_model_flush_receipt_recorder(inner.raw, &mut error) };
        ensure_ok(status, error)
    }

    /// Starts a new turn of `seq`: on a model with hybrid memory, its next
    /// ubatch reports its recurrent state first.
    pub fn arm_state(&self, seq: i32) -> Result<()> {
        let Some(inner) = self.inner.upgrade() else {
            return Ok(());
        };
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_model_arm_receipt_state(inner.raw, seq, &mut error) };
        ensure_ok(status, error)
    }
}

fn raw_anchor(anchor: &ReceiptAnchorRef<'_>) -> Result<ReceiptAnchor> {
    Ok(ReceiptAnchor {
        step: anchor.step,
        seq: anchor.seq,
        pos: anchor.pos,
        layer: i32::try_from(anchor.layer)?,
        kind: anchor.kind,
        index: i32::try_from(anchor.index)?,
        len: i64::try_from(anchor.values.len())?,
        values: anchor.values.as_ptr(),
    })
}

fn raw_step(step: &ReceiptReplayStepRef<'_>) -> Result<ReceiptReplayStep> {
    Ok(match *step {
        ReceiptReplayStepRef::Decode {
            token,
            seq,
            pos,
            output,
            cell,
            n_kv,
            input,
            capture,
        } => {
            let n = token.len();
            if [seq.len(), pos.len(), output.len(), cell.len()]
                .iter()
                .any(|&len| len != n)
                || input.is_some_and(|(rows, row, _)| rows.len() != row * n)
            {
                return Err(anyhow!("replay step columns disagree in length"));
            }
            let (input, row, input_type) = input.map_or((ptr::null(), 0, 0), |(rows, row, t)| {
                (rows.as_ptr().cast::<c_void>(), row, t)
            });
            ReceiptReplayStep {
                kind: skippy_ffi::RECEIPT_REPLAY_DECODE,
                n_tokens: i32::try_from(n)?,
                token: token.as_ptr(),
                seq: seq.as_ptr(),
                pos: pos.as_ptr(),
                output: output.as_ptr(),
                cell: cell.as_ptr(),
                n_kv,
                input,
                input_row_bytes: i64::try_from(row)?,
                input_type,
                p0: 0,
                p1: 0,
                capture: i32::from(capture),
            }
        }
        ReceiptReplayStepRef::SeqRm { p0, p1 } => ReceiptReplayStep {
            kind: skippy_ffi::RECEIPT_REPLAY_SEQ_RM,
            n_tokens: 0,
            token: ptr::null(),
            seq: ptr::null(),
            pos: ptr::null(),
            output: ptr::null(),
            cell: ptr::null(),
            n_kv: 0,
            input: ptr::null(),
            input_row_bytes: 0,
            input_type: 0,
            p0,
            p1,
            capture: 0,
        },
    })
}

impl StageModel {
    /// Reports every later ubatch and KV-cache change to `sink`, for
    /// receipts. Returns whether the model has hybrid memory, whose turns
    /// also report the recurrent state they start from.
    pub fn set_receipt_recorder(&self, sink: Arc<dyn ReceiptRecorderSink>) -> Result<bool> {
        if self.inner.raw.is_null() {
            return Err(anyhow!("receipt recorder needs a loaded model"));
        }
        let mut slot = self
            .inner
            .receipt_recorder
            .0
            .lock()
            .map_err(|_| anyhow!("receipt recorder lock poisoned"))?;
        if let Some(old) = slot.as_ref() {
            // the native side frees the old recorder's buffers
            old.gate.wait_released();
        }
        let boxed = Box::new(RecorderBinding {
            sink,
            model: Arc::downgrade(&self.inner),
            gate: Arc::default(),
        });
        let user_data = ptr::from_ref::<RecorderBinding>(&*boxed)
            .cast_mut()
            .cast::<c_void>();
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_model_set_receipt_recorder(
                self.inner.raw,
                Some(deliver_ubatch),
                Some(deliver_memory),
                user_data,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        let mut recurrent = false;
        let status = unsafe {
            skippy_ffi::skippy_model_set_receipt_state_observer(
                self.inner.raw,
                Some(deliver_state),
                user_data,
                &mut recurrent,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        *slot = Some(boxed);
        Ok(recurrent)
    }

    /// A handle that replays recorded sequences of this model.
    pub fn receipt_replayer(&self) -> ReceiptReplayer {
        ReceiptReplayer {
            inner: self.inner.clone(),
        }
    }

    /// A handle that reports the recorder's held-back ubatches before the
    /// recorder is read. It does not keep the model alive.
    pub fn receipt_flusher(&self) -> ReceiptFlusher {
        ReceiptFlusher {
            inner: Arc::downgrade(&self.inner),
        }
    }
}

/// The llama.cpp commit and patch queue the native runtime was built from,
/// as its receipts name them, or `unknown`.
pub fn receipt_runtime_source() -> String {
    let source = unsafe { skippy_ffi::skippy_receipt_runtime_source() };
    if source.is_null() {
        return "unknown".to_owned();
    }
    unsafe { CStr::from_ptr(source) }
        .to_string_lossy()
        .into_owned()
}
