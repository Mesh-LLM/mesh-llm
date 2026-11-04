//! The attention ops a receipt replay or glue evaluation reports, and the
//! rows a stage stored in its KV cache, which the receipt attention check
//! reads and pins.

use std::ffi::{CStr, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use anyhow::{Result, anyhow, bail};
use skippy_ffi::{
    RECEIPT_ATTENTION_CAUSAL, RECEIPT_ATTENTION_CHUNKED, RECEIPT_ATTENTION_SLIDING, ReceiptUbatch,
};

use super::receipts::{ReceiptReplayer, StageIdle, tensor_views};
use crate::error::ensure_ok;

/// Which positions an attention op's queries read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReceiptAttentionWindow {
    /// Every earlier position, and its own.
    Causal,
    /// The last `n` positions up to its own.
    Sliding(u32),
    /// Earlier positions in its own chunk of `n`.
    Chunked(u32),
}

/// An attention op the check can follow, by the indices of its values
/// among the layer's reported tensors.
#[derive(Clone, Debug, PartialEq)]
pub struct ReceiptAttention {
    /// q, among the layer's mixing inputs.
    pub q_input: u32,
    /// K and V, among the layer's stored rows.
    pub k_store: u32,
    pub v_store: u32,
    /// The output, among the layer's mixing outputs.
    pub output: u32,
    /// A gate among the layer's mixing inputs that multiplies the output
    /// value by value, or `None`.
    pub output_gate: Option<u32>,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim_k: usize,
    pub head_dim_v: usize,
    pub scale: f32,
    /// Zero for none.
    pub softcap: f32,
    pub window: ReceiptAttentionWindow,
    /// A logit per query head that only adds to the softmax's denominator.
    pub sinks: Option<Vec<f32>>,
    /// A rotation of each head of the output in chunks of `n` values, row
    /// after row, before the gate: `(n, matrix)`.
    pub output_rotation: Option<(usize, Vec<f32>)>,
    /// A ggml type the kernel rounds each head of q to before its dot with
    /// K.
    pub q_rounding: Option<u32>,
}

/// One attention op of a reported layer, or why the check cannot follow it.
#[derive(Clone, Debug, PartialEq)]
pub struct ReceiptAttentionOp {
    pub layer: u32,
    pub op: std::result::Result<ReceiptAttention, String>,
}

/// The attention ops of a native ubatch.
pub(super) unsafe fn attention_ops(u: &ReceiptUbatch) -> Result<Vec<ReceiptAttentionOp>, String> {
    let raw: &[skippy_ffi::ReceiptAttention] = if u.n_attention == 0 || u.attention.is_null() {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(u.attention, u.n_attention) }
    };
    raw.iter().map(|a| unsafe { attention_op(a) }).collect()
}

unsafe fn attention_op(a: &skippy_ffi::ReceiptAttention) -> Result<ReceiptAttentionOp, String> {
    let layer = u32::try_from(a.layer).map_err(|_| "an attention op of no layer".to_string())?;
    if !a.error.is_null() {
        let reason = unsafe { CStr::from_ptr(a.error) }
            .to_string_lossy()
            .into_owned();
        return Ok(ReceiptAttentionOp {
            layer,
            op: Err(reason),
        });
    }
    let index = |i: i32| u32::try_from(i).map_err(|_| "a negative attention index".to_string());
    let count = |n: i32| usize::try_from(n).map_err(|_| "a negative attention shape".to_string());
    let size = u32::try_from(a.window).map_err(|_| "a negative attention window".to_string())?;
    let window = match a.window_kind {
        RECEIPT_ATTENTION_CAUSAL => ReceiptAttentionWindow::Causal,
        RECEIPT_ATTENTION_SLIDING => ReceiptAttentionWindow::Sliding(size),
        RECEIPT_ATTENTION_CHUNKED => ReceiptAttentionWindow::Chunked(size),
        kind => return Err(format!("attention window kind {kind}")),
    };
    let n_head = count(a.n_head)?;
    let sinks = (!a.sinks.is_null() && n_head > 0)
        .then(|| unsafe { std::slice::from_raw_parts(a.sinks, n_head) }.to_vec());
    let rotation_n = count(a.output_rotation_n)?;
    let output_rotation = (rotation_n > 0 && !a.output_rotation.is_null()).then(|| {
        let n = rotation_n * rotation_n;
        let matrix = unsafe { std::slice::from_raw_parts(a.output_rotation, n) };
        (rotation_n, matrix.to_vec())
    });
    Ok(ReceiptAttentionOp {
        layer,
        op: Ok(ReceiptAttention {
            q_input: index(a.q_input)?,
            k_store: index(a.k_store)?,
            v_store: index(a.v_store)?,
            output: index(a.output)?,
            output_gate: (a.output_gate >= 0)
                .then(|| index(a.output_gate))
                .transpose()?,
            n_head,
            n_head_kv: count(a.n_head_kv)?,
            head_dim_k: count(a.head_dim_k)?,
            head_dim_v: count(a.head_dim_v)?,
            scale: a.scale,
            softcap: a.softcap,
            window,
            sinks,
            output_rotation,
            q_rounding: u32::try_from(a.q_rounding).ok(),
        }),
    })
}

/// One stored row of a layer in several KV cells, in the cache's type.
#[derive(Clone, Debug, PartialEq)]
pub struct ReceiptStoredRows {
    /// Among the layer's stored rows.
    pub index: u32,
    pub ggml_type: i32,
    pub row_bytes: usize,
    /// Row `j`, for the `j`-th cell, at `values[j * row_bytes..]`.
    pub values: Vec<u8>,
}

struct RowsState {
    result: Option<std::result::Result<Vec<ReceiptStoredRows>, String>>,
}

unsafe extern "C" fn deliver_rows(user_data: *mut c_void, ubatch: *const ReceiptUbatch) {
    if user_data.is_null() || ubatch.is_null() {
        return;
    }
    // SAFETY: user_data is the RowsState on read_rows()'s stack, and the
    // ubatch is valid for the call.
    let (state, u) = unsafe { (&mut *user_data.cast::<RowsState>(), &*ubatch) };
    // unwinding out of a callback the native side called would abort
    let rows = catch_unwind(AssertUnwindSafe(|| {
        unsafe { tensor_views(u) }.map(|tensors| {
            tensors
                .into_iter()
                .map(|t| ReceiptStoredRows {
                    index: t.index,
                    ggml_type: t.ggml_type,
                    row_bytes: t.row_bytes,
                    values: t.values.to_vec(),
                })
                .collect()
        })
    }));
    state.result = Some(rows.unwrap_or_else(|_| Err("reading the stored rows panicked".into())));
}

impl ReceiptReplayer {
    /// The rows the stage context stored for `layer` in `cells` of the
    /// stream of `seq`, from its KV cache, one entry per stored row of the
    /// layer, by index. It reads the stage context's cache, which `idle`
    /// keeps still.
    pub fn read_rows(
        &self,
        idle: &StageIdle<'_>,
        layer: u32,
        seq: i32,
        cells: &[u32],
    ) -> Result<Vec<ReceiptStoredRows>> {
        self.check_idle(idle)?;
        let mut state = RowsState { result: None };
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_model_read_receipt_rows(
                self.raw(),
                i32::try_from(layer)?,
                seq,
                cells.as_ptr(),
                cells.len(),
                Some(deliver_rows),
                ptr::from_mut(&mut state).cast::<c_void>(),
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        let mut rows = state
            .result
            .ok_or_else(|| anyhow!("the stage reported no rows"))?
            .map_err(|reason| anyhow!("cannot read stored rows: {reason}"))?;
        rows.sort_by_key(|r| r.index);
        Ok(rows)
    }

    /// Pins `cells` of the stage context, cell i in stream `streams[i]`,
    /// replacing the cells pinned before: while one is empty, ubatches go
    /// to other cells as long as they fit, so a finished turn's rows stay
    /// readable for its challenge window. Safe while the stage context
    /// computes.
    pub fn pin_cells(&self, streams: &[u32], cells: &[u32]) -> Result<()> {
        if streams.len() != cells.len() {
            bail!("a pinned cell needs its stream");
        }
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_model_pin_receipt_cells(
                self.raw(),
                streams.as_ptr(),
                cells.as_ptr(),
                cells.len(),
                &mut error,
            )
        };
        ensure_ok(status, error)
    }

    /// The streams of the stage context's KV cache: 1 for a unified cache,
    /// else one per sequence, stream s holding the cells of sequence s.
    pub fn kv_streams(&self) -> Result<u32> {
        let mut streams = 0;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_model_receipt_kv_streams(self.raw(), &mut streams, &mut error)
        };
        ensure_ok(status, error)?;
        Ok(streams)
    }
}
