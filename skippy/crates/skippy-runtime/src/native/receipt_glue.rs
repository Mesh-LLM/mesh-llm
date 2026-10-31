//! Evaluates the glue of one layer on a receipt opening, for an auditor.

use std::ffi::{CStr, c_void};
use std::ptr;

use anyhow::{Result, anyhow};
use skippy_ffi::{ReceiptGlueRequest, ReceiptGlueValue, ReceiptUbatch};

use super::StageModel;
use super::receipts::tensor_views;
use crate::error::ensure_ok;

/// Values of an opened tensor, column after column.
#[derive(Clone, Copy)]
pub enum ReceiptGlueValues<'a> {
    F32(&'a [f32]),
    I32(&'a [i32]),
}

/// An opened value the evaluation uses instead of computing it.
#[derive(Clone, Copy)]
pub struct ReceiptGlueInput<'a> {
    /// `RECEIPT_TENSOR_MATMUL_OUTPUT`, `_MIXING_OUTPUT` or `_SELECTION`.
    pub kind: i32,
    pub index: u32,
    /// Elements per column.
    pub len: usize,
    pub values: ReceiptGlueValues<'a>,
}

/// One layer at a few positions.
pub struct ReceiptGlueRequestRef<'a> {
    pub layer: u32,
    pub positions: &'a [i32],
    /// Residual stream entering the layer, column after column.
    pub residual: &'a [f32],
    pub inputs: &'a [ReceiptGlueInput<'a>],
}

/// A computed tensor, column after column, as F32 or I32 bytes.
#[derive(Clone, Debug)]
pub struct ReceiptGlueTensor {
    pub layer: u32,
    pub kind: i32,
    pub index: u32,
    pub ggml_type: i32,
    pub flags: i32,
    pub len: usize,
    pub n_cols: usize,
    pub name: String,
    pub values: Vec<u8>,
}

#[derive(Clone, Debug, Default)]
pub struct ReceiptGlueResult {
    pub tensors: Vec<ReceiptGlueTensor>,
    /// Weights read inside token-mixing regions, which nothing checks.
    pub unchecked_weights: Vec<String>,
}

/// How far a stored KV row is from the row an auditor computed.
#[derive(Clone, Copy, Debug)]
pub struct StoredRowError {
    /// L2 distance of the dequantized stored row from the computed row.
    pub stored: f64,
    /// L2 distance of the computed row from its own round trip through the cache type.
    pub rounding: f64,
    /// L2 norm of the computed row.
    pub norm: f64,
}

struct GlueState {
    result: Option<std::result::Result<ReceiptGlueResult, String>>,
}

unsafe extern "C" fn deliver_glue(user_data: *mut c_void, ubatch: *const ReceiptUbatch) {
    if user_data.is_null() || ubatch.is_null() {
        return;
    }
    // SAFETY: user_data is the GlueState on eval_receipt_glue()'s stack, and
    // the ubatch is valid for the call.
    let (state, u) = unsafe { (&mut *user_data.cast::<GlueState>(), &*ubatch) };
    state.result = Some(unsafe { collect(u) });
}

unsafe fn collect(u: &ReceiptUbatch) -> std::result::Result<ReceiptGlueResult, String> {
    if !u.error.is_null() {
        return Err(unsafe { CStr::from_ptr(u.error) }
            .to_string_lossy()
            .into_owned());
    }
    let tensors = unsafe { tensor_views(u) }?
        .into_iter()
        .map(|t| ReceiptGlueTensor {
            layer: t.layer,
            kind: t.kind,
            index: t.index,
            ggml_type: t.ggml_type,
            flags: t.flags,
            len: t.len,
            n_cols: t.n_cols,
            name: t.name.to_owned(),
            values: t.values.to_vec(),
        })
        .collect();
    let names: &[*const std::ffi::c_char] = if u.n_unchecked_weights == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(u.unchecked_weights, u.n_unchecked_weights) }
    };
    let unchecked_weights = names
        .iter()
        .map(|&name| {
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    Ok(ReceiptGlueResult {
        tensors,
        unchecked_weights,
    })
}

impl StageModel {
    /// Computes one layer of this model on opened receipt values: every
    /// weight matmul, token-mixing output and selection takes the given
    /// value, so the layer's large weights are not read.
    pub fn eval_receipt_glue(
        &self,
        request: &ReceiptGlueRequestRef<'_>,
    ) -> Result<ReceiptGlueResult> {
        let n_cols = request.positions.len();
        if n_cols == 0 || !request.residual.len().is_multiple_of(n_cols) {
            return Err(anyhow!(
                "a glue evaluation needs a residual column per position"
            ));
        }
        let values = request
            .inputs
            .iter()
            .map(|input| {
                let (ggml_type, values, count) = match input.values {
                    ReceiptGlueValues::F32(v) => (0, v.as_ptr().cast::<c_void>(), v.len()),
                    ReceiptGlueValues::I32(v) => (26, v.as_ptr().cast::<c_void>(), v.len()),
                };
                if count != input.len * n_cols {
                    return Err(anyhow!(
                        "glue input {}:{} has {count} values, not {} columns of {}",
                        input.kind,
                        input.index,
                        n_cols,
                        input.len
                    ));
                }
                Ok(ReceiptGlueValue {
                    kind: input.kind,
                    index: i32::try_from(input.index)?,
                    ggml_type,
                    len: i64::try_from(input.len)?,
                    values,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let raw = ReceiptGlueRequest {
            layer: i32::try_from(request.layer)?,
            n_cols: i32::try_from(n_cols)?,
            pos: request.positions.as_ptr(),
            residual: request.residual.as_ptr(),
            residual_len: i64::try_from(request.residual.len() / n_cols)?,
            n_values: values.len(),
            values: values.as_ptr(),
        };
        let mut state = GlueState { result: None };
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_model_eval_receipt_glue(
                self.inner.raw,
                &raw,
                Some(deliver_glue),
                ptr::from_mut(&mut state).cast::<c_void>(),
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        match state.result {
            Some(Ok(result)) => Ok(result),
            Some(Err(reason)) => Err(anyhow!("the layer cannot be evaluated: {reason}")),
            None => Err(anyhow!("the glue evaluation reported nothing")),
        }
    }
}

/// Compares a stored KV row with the f32 row an auditor computed.
pub fn receipt_stored_row_error(
    ggml_type: i32,
    stored: &[u8],
    computed: &[f32],
) -> Result<StoredRowError> {
    let mut out = StoredRowError {
        stored: 0.0,
        rounding: 0.0,
        norm: 0.0,
    };
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_receipt_stored_row_error(
            ggml_type,
            stored.as_ptr().cast::<c_void>(),
            computed.as_ptr(),
            i64::try_from(computed.len())?,
            &mut out.stored,
            &mut out.rounding,
            &mut out.norm,
            &mut error,
        )
    };
    ensure_ok(status, error)?;
    Ok(out)
}
