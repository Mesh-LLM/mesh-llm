//! Evaluates the glue of one layer on a receipt opening, the recurrent
//! token mixing of one layer, and the input of layer 0 from tokens, for an
//! auditor.

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
    /// The attention ops of the layer.
    pub attention: Vec<super::ReceiptAttentionOp>,
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
    // SAFETY: user_data is the GlueState on the evaluating call's stack, and
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
    let attention = unsafe { super::receipt_attention::attention_ops(u) }?;
    Ok(ReceiptGlueResult {
        tensors,
        unchecked_weights,
        attention,
    })
}

/// Which evaluation of opened values a request runs.
#[derive(Clone, Copy)]
enum OpenedEval {
    Glue,
    Region,
}

impl OpenedEval {
    fn name(self) -> &'static str {
        match self {
            Self::Glue => "glue",
            Self::Region => "region",
        }
    }

    /// Columns of an input: a region takes one state, from before its
    /// first position.
    fn columns(self, kind: i32, n_cols: usize) -> usize {
        match self {
            Self::Region if kind == skippy_ffi::RECEIPT_TENSOR_MIXING_STATE => 1,
            _ => n_cols,
        }
    }
}

impl StageModel {
    /// Computes one layer of this model on opened receipt values: every
    /// weight matmul, token-mixing output and selection takes the given
    /// value, so the layer's large weights are not read.
    pub fn eval_receipt_glue(
        &self,
        request: &ReceiptGlueRequestRef<'_>,
    ) -> Result<ReceiptGlueResult> {
        self.eval_opened(request, OpenedEval::Glue)
    }

    /// Computes the recurrent token mixing of one layer from its opened
    /// inputs at consecutive positions of one sequence and the opened
    /// states before the first of them (one column each). The result holds
    /// the mixing outputs and the states the layer leaves.
    pub fn eval_receipt_region(
        &self,
        request: &ReceiptGlueRequestRef<'_>,
    ) -> Result<ReceiptGlueResult> {
        self.eval_opened(request, OpenedEval::Region)
    }

    fn eval_opened(
        &self,
        request: &ReceiptGlueRequestRef<'_>,
        eval: OpenedEval,
    ) -> Result<ReceiptGlueResult> {
        let n_cols = request.positions.len();
        if n_cols == 0 || !request.residual.len().is_multiple_of(n_cols) {
            return Err(anyhow!(
                "a {} evaluation needs a residual column per position",
                eval.name()
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
                let columns = eval.columns(input.kind, n_cols);
                if count != input.len * columns {
                    return Err(anyhow!(
                        "{} input {}:{} has {count} values, not {columns} columns of {}",
                        eval.name(),
                        input.kind,
                        input.index,
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
        let user_data = ptr::from_mut(&mut state).cast::<c_void>();
        let status = unsafe {
            match eval {
                OpenedEval::Glue => skippy_ffi::skippy_model_eval_receipt_glue(
                    self.inner.raw,
                    &raw,
                    Some(deliver_glue),
                    user_data,
                    &mut error,
                ),
                OpenedEval::Region => skippy_ffi::skippy_model_eval_receipt_region(
                    self.inner.raw,
                    &raw,
                    Some(deliver_glue),
                    user_data,
                    &mut error,
                ),
            }
        };
        ensure_ok(status, error)?;
        match state.result {
            Some(Ok(result)) => Ok(result),
            Some(Err(reason)) => Err(anyhow!("the layer cannot be evaluated: {reason}")),
            None => Err(anyhow!("the {} evaluation reported nothing", eval.name())),
        }
    }

    /// Computes the residual entering layer 0 for each token at its
    /// position. The model must hold layer 0. The result holds one RESIDUAL
    /// tensor of layer 0, a column per token.
    pub fn eval_receipt_entry(
        &self,
        tokens: &[i32],
        positions: &[i32],
    ) -> Result<ReceiptGlueResult> {
        if tokens.is_empty() || tokens.len() != positions.len() {
            return Err(anyhow!("an entry evaluation needs a position per token"));
        }
        let mut state = GlueState { result: None };
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_model_eval_receipt_entry(
                self.inner.raw,
                tokens.as_ptr(),
                positions.as_ptr(),
                i32::try_from(tokens.len())?,
                Some(deliver_glue),
                ptr::from_mut(&mut state).cast::<c_void>(),
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        match state.result {
            Some(Ok(result)) => Ok(result),
            Some(Err(reason)) => Err(anyhow!(
                "the input of layer 0 cannot be evaluated: {reason}"
            )),
            None => Err(anyhow!("the entry evaluation reported nothing")),
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
