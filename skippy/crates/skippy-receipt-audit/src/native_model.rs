//! The receipt auditor's view of the claimed GGUF: one layer at a time,
//! loaded alone on the CPU, and the vocabulary of its first layer.
//!
//! A node names the claimed GGUF, so its bytes are whatever that node
//! published. The native loader reads the file only after the header
//! passed the strict Rust reading an audit opens it with
//! ([`claimed_file::read_header`]): the runtime's loader then sees only
//! headers with each name once, in UTF-8, and tensors of known types laid
//! inside the file. The metadata values the native code asserts on, which
//! would end the process, are checked first too (`loader_asserts`), and so
//! are the token ids a receipt hands the tokenizer.

use std::cell::{OnceCell, RefCell};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::claimed_file;
use crate::claimed_model::{CacheTypes, ClaimedModel};
use anyhow::{Context, Result, bail};
use skippy_receipts::attention::{
    AttentionDescriptor, AttentionLayout, AttentionOp, OutputRotation, Window, q_rounding,
};
use skippy_receipts::glue::{
    GlueEvaluator, GlueInput, GlueOutput, GlueRequest, GlueTensor, GlueValues, StoredRowError,
};
use skippy_receipts::layer_tensors::{GGML_F32, GGML_I32, TensorKind, decode_f32_bytes};
use skippy_runtime::{
    GgufStageRuntimePlan, ReceiptAttentionOp, ReceiptAttentionWindow, ReceiptGlueInput,
    ReceiptGlueRequestRef, ReceiptGlueTensor, ReceiptGlueValues, RuntimeConfig, StageModel,
    plan_gguf_stage_runtime_plan_for_range, receipt_stored_row_error,
};

mod loader_asserts;

/// Context of the stage the planner sizes; the glue context sets its own.
const PLAN_CTX: u32 = 256;

pub struct NativeGlueEvaluator {
    model: StageModel,
}

/// The layer after `layer`, where its stage ends.
fn layer_end(layer: u32) -> Result<u32> {
    layer
        .checked_add(1)
        .with_context(|| format!("layer {layer} is the last a stage can name"))
}

/// The stage plan of `layer` alone. The native planner reads the header of
/// the GGUF and no tensor, so the plan of a layer is the same for as long
/// as the header is.
fn plan_layer(path: &Path, layer: u32) -> Result<GgufStageRuntimePlan> {
    plan_gguf_stage_runtime_plan_for_range(path, (layer, layer_end(layer)?), PLAN_CTX, 1)
        .with_context(|| format!("plan layer {layer} of {}", path.display()))
}

impl NativeGlueEvaluator {
    /// Loads `layer` of the GGUF at `path` as a single-layer CPU stage, by
    /// its `plan`, with a KV cache of `cache` types. Only
    /// [`NativeClaimedModel`] loads, after the header passed the strict
    /// check.
    fn load(
        path: &Path,
        layer: u32,
        cache: CacheTypes,
        plan: GgufStageRuntimePlan,
    ) -> Result<Self> {
        let mut config = RuntimeConfig {
            layer_start: layer,
            layer_end: layer_end(layer)?,
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
///
/// What does not change between loads is kept: that the header passed the
/// strict check, and each layer's stage plan, which the planner derives
/// from the header alone. The load itself reads the file again every time.
/// A header that passed every check before the native code.
struct CheckedHeader {
    weights: skippy_receipts::weights::WeightFile,
    checked: loader_asserts::Checked,
}

pub struct NativeClaimedModel {
    path: PathBuf,
    first: OnceCell<NativeGlueEvaluator>,
    /// The header, once it passed the strict check and the checks of what
    /// the native code asserts on.
    header: OnceCell<CheckedHeader>,
    /// The stage plan of each layer loaded so far.
    plans: RefCell<BTreeMap<u32, GgufStageRuntimePlan>>,
}

impl NativeClaimedModel {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            first: OnceCell::new(),
            header: OnceCell::new(),
            plans: RefCell::new(BTreeMap::new()),
        }
    }

    fn first(&self) -> Result<&NativeGlueEvaluator> {
        if let Some(first) = self.first.get() {
            return Ok(first);
        }
        let first = self.open_layer(0, CacheTypes::default())?;
        Ok(self.first.get_or_init(|| first))
    }

    /// Checks the header strictly, once: the native planner and loader run
    /// only on a file that passed.
    fn checked_header(&self) -> Result<&CheckedHeader> {
        if let Some(header) = self.header.get() {
            return Ok(header);
        }
        let weights = claimed_file::read_header(&self.path)
            .with_context(|| format!("check the GGUF header of {}", self.path.display()))?;
        let checked = loader_asserts::check(&weights)
            .with_context(|| format!("check the GGUF metadata of {}", self.path.display()))?;
        Ok(self
            .header
            .get_or_init(|| CheckedHeader { weights, checked }))
    }

    /// Refuses a token id the vocabulary does not hold: the native
    /// tokenizer throws on it where nothing catches the exception, which
    /// ends the process. A receipt names the ids.
    fn check_tokens(&self, tokens: &[i32]) -> Result<()> {
        let vocabulary = self.checked_header()?.checked.vocabulary;
        if let Some(token) = tokens
            .iter()
            .find(|&&token| !u64::try_from(token).is_ok_and(|id| id < vocabulary))
        {
            bail!("the token {token} is not in the claimed model's vocabulary of {vocabulary}");
        }
        Ok(())
    }

    /// Loads `layer` alone from the file as it is now, by a plan made once.
    fn open_layer(&self, layer: u32, cache: CacheTypes) -> Result<NativeGlueEvaluator> {
        let header = self.checked_header()?;
        loader_asserts::check_layer(&header.weights, layer)?;
        let known = self.plans.borrow().get(&layer).cloned();
        let plan = match known {
            Some(plan) => plan,
            None => {
                let plan = plan_layer(&self.path, layer)?;
                self.plans.borrow_mut().insert(layer, plan.clone());
                plan
            }
        };
        NativeGlueEvaluator::load(&self.path, layer, cache, plan)
    }
}

impl ClaimedModel for NativeClaimedModel {
    fn with_layer(
        &self,
        layer: u32,
        cache: CacheTypes,
        check: &mut dyn FnMut(&dyn GlueEvaluator) -> Result<()>,
    ) -> Result<()> {
        check(&self.open_layer(layer, cache)?)
    }

    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        self.check_tokens(tokens)?;
        self.first()?.model.detokenize(tokens)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        self.first()?.model.tokenize(text, false)
    }

    fn token_piece(&self, token: i32) -> Result<Vec<u8>> {
        self.check_tokens(&[token])?;
        self.first()?.model.detokenize_bytes(&[token])
    }

    fn is_end_of_generation(&self, token: i32) -> Result<bool> {
        self.check_tokens(&[token])?;
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

#[cfg(test)]
mod tests {
    use skippy_receipts::weights::limits::CannotAudit;

    use super::*;

    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    /// A GGUF with a header only: `tensors`, by name and offset, each two
    /// rows of four f32, and 64 bytes of data.
    fn gguf(tensors: &[(&[u8], u64)]) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend((tensors.len() as u64).to_le_bytes());
        out.extend(1u64.to_le_bytes());
        string(&mut out, "general.architecture");
        out.extend(8u32.to_le_bytes());
        string(&mut out, "llama");
        for (name, offset) in tensors {
            out.extend((name.len() as u64).to_le_bytes());
            out.extend(*name);
            out.extend(2u32.to_le_bytes());
            out.extend(4u64.to_le_bytes());
            out.extend(2u64.to_le_bytes());
            out.extend(0u32.to_le_bytes());
            out.extend(offset.to_le_bytes());
        }
        out.resize(out.len().div_ceil(32) * 32 + 64, 0);
        out
    }

    /// The native planner and loader would abort the test on these files,
    /// or read past them: the strict check answers first, with an error.
    #[test]
    fn a_header_that_fails_the_strict_check_is_never_loaded() {
        let dir = tempfile::tempdir().unwrap();
        let cases: [(&str, Vec<u8>); 5] = [
            ("a tensor twice", gguf(&[(b"blk.0.w", 0), (b"blk.0.w", 32)])),
            (
                "tensors that overlap",
                gguf(&[(b"blk.0.a", 0), (b"blk.0.b", 0)]),
            ),
            ("a tensor off the alignment", gguf(&[(b"blk.0.a", 16)])),
            (
                "a tensor past the file",
                gguf(&[(b"blk.0.a", 0), (b"blk.0.b", 32), (b"blk.0.c", 64)]),
            ),
            ("a name that is not UTF-8", gguf(&[(b"blk.0.\xff", 0)])),
        ];
        for (what, bytes) in cases {
            let path = dir.path().join("claimed.gguf");
            std::fs::write(&path, bytes).unwrap();
            let model = load_claimed_model(path.clone());
            let loaded = model.with_layer(0, CacheTypes::default(), &mut |_| Ok(()));
            let error = loaded.err().unwrap_or_else(|| panic!("{what} loaded"));
            assert!(CannotAudit::caused(&error), "{what}: {error:#}");
            let error = model.detokenize(&[1]).err().unwrap();
            assert!(CannotAudit::caused(&error), "{what}: {error:#}");
        }
    }

    /// A GGUF the strict header check passes whose `block_count` is 0.
    fn no_layers() -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend(1u64.to_le_bytes());
        out.extend(4u64.to_le_bytes());
        string(&mut out, "general.architecture");
        out.extend(8u32.to_le_bytes());
        string(&mut out, "llama");
        for (key, value) in [
            ("llama.context_length", 4096u32),
            ("llama.embedding_length", 4),
            ("llama.block_count", 0),
        ] {
            string(&mut out, key);
            out.extend(4u32.to_le_bytes());
            out.extend(value.to_le_bytes());
        }
        string(&mut out, "token_embd.weight");
        out.extend(2u32.to_le_bytes());
        out.extend(4u64.to_le_bytes());
        out.extend(2u64.to_le_bytes());
        out.extend(0u32.to_le_bytes());
        out.extend(0u64.to_le_bytes());
        out.resize(out.len().div_ceil(32) * 32 + 32, 0);
        out
    }

    /// The file the child of the test below opens.
    const HOSTILE_GGUF: &str = "SKIPPY_RECEIPT_AUDIT_TEST_HOSTILE_GGUF";

    /// In the child process only: opens a layer of the file, which the
    /// checks must refuse before the native planner aborts on it.
    #[test]
    fn hostile_metadata_child() {
        let Some(path) = std::env::var_os(HOSTILE_GGUF) else {
            return;
        };
        let model = load_claimed_model(PathBuf::from(path));
        let error = model
            .with_layer(0, CacheTypes::default(), &mut |_| Ok(()))
            .unwrap_err();
        assert!(CannotAudit::caused(&error), "{error:#}");
        assert!(format!("{error:#}").contains("block_count"), "{error:#}");
        let error = model.detokenize(&[0]).unwrap_err();
        assert!(CannotAudit::caused(&error), "{error:#}");
    }

    /// A file the strict header check passes, with a `block_count` of 0,
    /// on which the native planner asserts and the process ends. Run in a
    /// child process, so an abort fails this test and no other.
    #[test]
    fn hostile_metadata_never_reaches_the_native_code() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("claimed.gguf");
        std::fs::write(&path, no_layers()).unwrap();
        claimed_file::read_header(&path).unwrap();
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "native_model::tests::hostile_metadata_child",
                "--test-threads=1",
            ])
            .env(HOSTILE_GGUF, &path)
            .stdout(std::process::Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "the child ended with {status:?}");
    }

    #[test]
    fn the_last_layer_number_has_no_stage() {
        assert_eq!(layer_end(7).unwrap(), 8);
        assert!(layer_end(u32::MAX).is_err());
    }
}
