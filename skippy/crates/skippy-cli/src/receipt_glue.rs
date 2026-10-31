//! The receipt auditor's glue evaluator: one layer of the claimed GGUF,
//! loaded alone on the CPU.

use std::path::Path;

use anyhow::{Context, Result, bail};
use skippy_receipts::glue::{
    GlueEvaluator, GlueInput, GlueOutput, GlueRequest, GlueTensor, GlueValues, StoredRowError,
};
use skippy_receipts::layer_tensors::{GGML_F32, GGML_I32, TensorKind, decode_f32_bytes};
use skippy_runtime::{
    ReceiptGlueInput, ReceiptGlueRequestRef, ReceiptGlueTensor, ReceiptGlueValues, RuntimeConfig,
    StageModel, plan_gguf_stage_runtime_plan_for_range, receipt_stored_row_error,
};

/// Context of the stage the planner sizes; the glue context sets its own.
const PLAN_CTX: u32 = 256;

pub struct NativeGlueEvaluator {
    model: StageModel,
}

impl NativeGlueEvaluator {
    /// Loads `layer` of the GGUF at `path` as a single-layer CPU stage.
    pub fn open(path: &Path, layer: u32) -> Result<Self> {
        let plan = plan_gguf_stage_runtime_plan_for_range(path, (layer, layer + 1), PLAN_CTX, 1)
            .with_context(|| format!("plan layer {layer} of {}", path.display()))?;
        let mut config = RuntimeConfig {
            layer_start: layer,
            layer_end: layer + 1,
            ctx_size: PLAN_CTX,
            // the CPU only: an auditor needs no GPU
            n_gpu_layers: 0,
            op_offload: Some(false),
            kv_offload: Some(false),
            ..RuntimeConfig::default()
        };
        plan.apply_to(&mut config);
        let model = StageModel::open(path, &config)
            .with_context(|| format!("load layer {layer} of {}", path.display()))?;
        Ok(Self { model })
    }
}

fn input_ref(input: &GlueInput) -> ReceiptGlueInput<'_> {
    ReceiptGlueInput {
        kind: input.kind.code(),
        index: input.index,
        len: input.len,
        values: match &input.values {
            GlueValues::F32(v) => ReceiptGlueValues::F32(v),
            GlueValues::I32(v) => ReceiptGlueValues::I32(v),
        },
    }
}

/// Keeps the first `n_cols` columns: the native evaluation pads to at least two.
fn glue_tensor(t: ReceiptGlueTensor, n_cols: usize) -> Result<Option<GlueTensor>> {
    let Some(kind) = TensorKind::from_code(t.kind) else {
        return Ok(None);
    };
    if t.n_cols < n_cols {
        bail!(
            "the evaluation reported {} columns of {}, not {n_cols}",
            t.n_cols,
            t.name
        );
    }
    let bytes = &t.values[..t.len * n_cols * 4];
    let values = match t.ggml_type {
        GGML_F32 => GlueValues::F32(decode_f32_bytes(bytes)),
        GGML_I32 => GlueValues::I32(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c))
                .collect(),
        ),
        other => bail!("the evaluation reported {} as ggml type {other}", t.name),
    };
    Ok(Some(GlueTensor {
        layer: t.layer,
        kind,
        index: t.index,
        flags: t.flags,
        name: t.name,
        len: t.len,
        values,
    }))
}

impl GlueEvaluator for NativeGlueEvaluator {
    fn evaluate(&self, request: &GlueRequest) -> Result<GlueOutput> {
        let inputs: Vec<ReceiptGlueInput<'_>> = request.inputs.iter().map(input_ref).collect();
        let result = self.model.eval_receipt_glue(&ReceiptGlueRequestRef {
            layer: request.layer,
            positions: &request.positions,
            residual: &request.residual,
            inputs: &inputs,
        })?;
        let n_cols = request.positions.len();
        let tensors = result
            .tensors
            .into_iter()
            .map(|t| glue_tensor(t, n_cols))
            .filter_map(Result::transpose)
            .collect::<Result<_>>()?;
        Ok(GlueOutput {
            tensors,
            unchecked_weights: result.unchecked_weights,
        })
    }

    fn stored_row_error(
        &self,
        ggml_type: i32,
        stored: &[u8],
        computed: &[f32],
    ) -> Result<StoredRowError> {
        let error = receipt_stored_row_error(ggml_type, stored, computed)?;
        Ok(StoredRowError {
            stored: error.stored,
            rounding: error.rounding,
            norm: error.norm,
        })
    }
}
