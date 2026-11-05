//! The receipt auditor's view of the claimed GGUF: one layer at a time,
//! loaded alone on the CPU, and the vocabulary of its first layer.

use std::cell::OnceCell;
use std::path::{Path, PathBuf};

use super::{CacheTypes, ClaimedModel};
use anyhow::{Context, Result, bail};
use skippy_receipts::attention::{
    AttentionDescriptor, AttentionLayout, AttentionOp, OutputRotation, Window, q_rounding,
};
use skippy_receipts::glue::{
    GlueEvaluator, GlueInput, GlueOutput, GlueRequest, GlueTensor, GlueValues, StoredRowError,
};
use skippy_receipts::layer_tensors::{GGML_F32, GGML_I32, TensorKind, decode_f32_bytes};
use skippy_runtime::{
    ReceiptAttentionOp, ReceiptAttentionWindow, ReceiptGlueInput, ReceiptGlueRequestRef,
    ReceiptGlueTensor, ReceiptGlueValues, RuntimeConfig, StageModel,
    plan_gguf_stage_runtime_plan_for_range, receipt_stored_row_error,
};

/// Context of the stage the planner sizes; the glue context sets its own.
const PLAN_CTX: u32 = 256;

pub struct NativeGlueEvaluator {
    model: StageModel,
}

impl NativeGlueEvaluator {
    /// Loads `layer` of the GGUF at `path` as a single-layer CPU stage, with
    /// a KV cache of `cache` types.
    pub fn open(path: &Path, layer: u32, cache: CacheTypes) -> Result<Self> {
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
        if let Some(k) = cache.k {
            config.cache_type_k = k;
        }
        if let Some(v) = cache.v {
            config.cache_type_v = v;
        }
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

/// The claimed GGUF. Layer 0 stays loaded for the vocabulary; every check
/// loads its layer for that check and frees it. A sparse copy holds more
/// bytes after each audit's fetches, and the loader copies a tensor when it
/// loads it, so a layer kept from an earlier check would miss rows fetched
/// since (the anchors' token embeddings, for one).
pub struct NativeClaimedModel {
    path: PathBuf,
    first: OnceCell<NativeGlueEvaluator>,
}

impl NativeClaimedModel {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            first: OnceCell::new(),
        }
    }

    fn first(&self) -> Result<&NativeGlueEvaluator> {
        if let Some(first) = self.first.get() {
            return Ok(first);
        }
        let first = NativeGlueEvaluator::open(&self.path, 0, CacheTypes::default())?;
        Ok(self.first.get_or_init(|| first))
    }
}

impl ClaimedModel for NativeClaimedModel {
    fn with_layer(
        &self,
        layer: u32,
        cache: CacheTypes,
        check: &mut dyn FnMut(&dyn GlueEvaluator) -> Result<()>,
    ) -> Result<()> {
        check(&NativeGlueEvaluator::open(&self.path, layer, cache)?)
    }

    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        self.first()?.model.detokenize(tokens)
    }

    fn is_end_of_generation(&self, token: i32) -> Result<bool> {
        self.first()?.model.token_is_eog(token)
    }

    fn runtime(&self) -> String {
        skippy_receipts::turn::runtime_identity(
            skippy_runtime::ABI_VERSION_MAJOR,
            skippy_runtime::ABI_VERSION_MINOR,
            skippy_runtime::ABI_VERSION_PATCH,
            &skippy_runtime::receipt_runtime_source(),
        )
    }
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
            attention: result.attention.iter().map(attention_descriptor).collect(),
        })
    }

    fn evaluate_region(&self, request: &GlueRequest) -> Result<GlueOutput> {
        let inputs: Vec<ReceiptGlueInput<'_>> = request.inputs.iter().map(input_ref).collect();
        let result = self.model.eval_receipt_region(&ReceiptGlueRequestRef {
            layer: request.layer,
            positions: &request.positions,
            residual: &request.residual,
            inputs: &inputs,
        })?;
        let n_cols = request.positions.len();
        let tensors = result
            .tensors
            .into_iter()
            .map(|t| {
                // a state has a column per sequence, and the region is the first
                let state = TensorKind::from_code(t.kind) == Some(TensorKind::MixingStateOut);
                glue_tensor(t, if state { 1 } else { n_cols })
            })
            .filter_map(Result::transpose)
            .collect::<Result<_>>()?;
        Ok(GlueOutput {
            tensors,
            unchecked_weights: result.unchecked_weights,
            attention: Vec::new(),
        })
    }

    fn entry(&self, tokens: &[i32], positions: &[i32]) -> Result<Vec<f32>> {
        let result = self.model.eval_receipt_entry(tokens, positions)?;
        let residual = result
            .tensors
            .into_iter()
            .find(|t| t.layer == 0 && TensorKind::from_code(t.kind) == Some(TensorKind::Residual))
            .context("the evaluation reported no input of layer 0")?;
        match glue_tensor(residual, tokens.len())? {
            Some(GlueTensor {
                values: GlueValues::F32(values),
                ..
            }) => Ok(values),
            _ => bail!("the input of layer 0 is not f32"),
        }
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

/// The claimed model the audit loads layers of from `path`.
pub fn load_claimed_model(path: PathBuf) -> Box<dyn ClaimedModel> {
    Box::new(NativeClaimedModel::new(path))
}

/// An attention op of the auditor's graph, as the receipt check describes it.
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
