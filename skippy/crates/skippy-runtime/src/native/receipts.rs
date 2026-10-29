//! Delivers the native receipt recorder to a Rust sink, and replays a
//! recorded sequence to open a receipt challenge.

use std::ffi::{CStr, c_void};
use std::ptr;
use std::sync::{Arc, Condvar, Mutex, PoisonError, Weak};

use anyhow::{Result, anyhow};
use skippy_ffi::{ReceiptReplayStep, ReceiptTensor, ReceiptUbatch};

use super::{StageModel, StageModelInner};

use crate::error::ensure_ok;
pub use skippy_ffi::ReceiptMemoryEvent;

/// One reported tensor, column per token (or per output row).
#[derive(Clone, Copy)]
pub struct ReceiptTensorView<'a> {
    pub layer: u32,
    /// A `skippy_ffi::RECEIPT_TENSOR_*` kind.
    pub kind: i32,
    pub len: usize,
    pub n_cols: usize,
    pub values: &'a [f32],
}

impl<'a> ReceiptTensorView<'a> {
    pub fn column(&self, j: usize) -> &'a [f32] {
        &self.values[j * self.len..(j + 1) * self.len]
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
}

/// Receives the recorder's reports on the decoding thread.
pub trait ReceiptRecorderSink: Send + Sync {
    /// A computed ubatch. It can be kept and read on any thread, but the next
    /// ubatch is not delivered until it is dropped: decoding waits for it.
    fn ubatch(&self, ubatch: RecordedUbatch);
    /// The ubatch could not be receipted; the sequences it touched cannot be opened.
    fn unreceiptable(&self, reason: &str);
    fn memory(&self, event: &ReceiptMemoryEvent);
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
    len: usize,
    n_cols: usize,
    values: *const f32,
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
                    len: t.len,
                    n_cols: t.n_cols,
                    values: t.values.as_ptr(),
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
            len: t.len,
            n_cols: t.n_cols,
            // SAFETY: valid until self is dropped, see the type docs.
            values: unsafe { std::slice::from_raw_parts(t.values, t.len * t.n_cols) },
        })
    }
}

impl Drop for RecordedUbatch {
    fn drop(&mut self) {
        self.gate.release();
    }
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
    let raw: &[ReceiptTensor] = if u.n_tensors == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(u.tensors, u.n_tensors) }
    };
    let mut tensors = Vec::with_capacity(raw.len());
    for t in raw {
        let (Ok(layer), Ok(len), Ok(n_cols)) = (
            u32::try_from(t.layer),
            usize::try_from(t.len),
            usize::try_from(t.n_cols),
        ) else {
            return Err("malformed receipt tensor".to_string());
        };
        tensors.push(ReceiptTensorView {
            layer,
            kind: t.kind,
            len,
            n_cols,
            values: unsafe { std::slice::from_raw_parts(t.values, len * n_cols) },
        });
    }
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
    })
}

unsafe extern "C" fn deliver_ubatch(user_data: *mut c_void, ubatch: *const ReceiptUbatch) {
    if user_data.is_null() || ubatch.is_null() {
        return;
    }
    // SAFETY: user_data is the binding owned by InstalledRecorder, and the
    // native side keeps the ubatch buffers alive for the call.
    let binding = unsafe { &*user_data.cast::<RecorderBinding>() };
    // The native side has just overwritten the buffers of the ubatch before
    // the previous one, and may overwrite the previous one's next time.
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
    /// Replays `seq` and calls `on_capture` for every captured step.
    pub fn replay(
        &self,
        seq: i32,
        layer: u32,
        steps: &[ReceiptReplayStepRef<'_>],
        on_capture: &mut dyn FnMut(&ReceiptUbatchView<'_>) -> Result<()>,
    ) -> Result<()> {
        let raw_steps = steps.iter().map(raw_step).collect::<Result<Vec<_>>>()?;
        let mut state = CaptureState {
            on_capture,
            error: None,
        };
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_model_replay_receipt(
                self.inner.raw,
                seq,
                i32::try_from(layer)?,
                raw_steps.as_ptr(),
                raw_steps.len(),
                Some(deliver_capture),
                ptr::from_mut(&mut state).cast::<c_void>(),
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        state.error.map_or(Ok(()), Err)
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
    /// Reports every later ubatch and KV-cache change to `sink`, for receipts.
    pub fn set_receipt_recorder(&self, sink: Arc<dyn ReceiptRecorderSink>) -> Result<()> {
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
        *slot = Some(boxed);
        Ok(())
    }

    /// A handle that replays recorded sequences of this model.
    pub fn receipt_replayer(&self) -> ReceiptReplayer {
        ReceiptReplayer {
            inner: self.inner.clone(),
        }
    }
}
