use std::path::Path;
use std::ptr;

use anyhow::{Context, Result, anyhow};

use crate::error::ensure_ok;
use crate::logging::write_native_log_note;
use crate::path_cstring::path_to_cstring;

/// Question types the Laya decision head distinguishes, in native order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayaQuestionType {
    Choice,
    Score,
    Noul,
}

impl LayaQuestionType {
    /// The native qtype index; also the index into [`LayaModelInfo::temperature`].
    pub(crate) fn index(self) -> usize {
        match self {
            Self::Choice => 0,
            Self::Score => 1,
            Self::Noul => 2,
        }
    }
}

/// Sequence budgets and special tokens a Laya GGUF was converted with.
#[derive(Debug, Clone, PartialEq)]
pub struct LayaModelInfo {
    pub max_len: usize,
    pub head_max_len: usize,
    pub max_markers: usize,
    pub action_classes: usize,
    pub cls_token_id: i32,
    pub sep_token_id: i32,
    pub mask_token_id: i32,
    pub embedding_size: u32,
    pub layer_count: u32,
    pub parameter_count: u64,
    /// Per-qtype softmax temperature; `0.0` when the GGUF carries none.
    pub temperature: [f32; skippy_ffi::LAYA_QTYPE_COUNT],
}

/// Measured memory of a loaded Laya model at its largest read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayaMemory {
    pub weights_bytes: u64,
    /// Scheduler compute buffers after a full-length read.
    pub compute_bytes: u64,
    /// Host-side attention masks one full pass builds.
    pub host_scratch_bytes: u64,
    pub on_accelerator: bool,
}

impl LayaMemory {
    /// Peak bytes while serving: weights, compute buffers, and one pass's
    /// host scratch.
    pub fn peak_bytes(&self) -> u64 {
        self.weights_bytes
            .saturating_add(self.compute_bytes)
            .saturating_add(self.host_scratch_bytes)
    }
}

/// One question sequence: `[CLS] head [SEP] ([MASK] option)* [SEP] state [SEP]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LayaSequence {
    pub(crate) tokens: Vec<i32>,
    pub(crate) question_type: LayaQuestionType,
    /// Sequence-local positions of each option's `[MASK]` marker.
    pub(crate) markers: Vec<u32>,
}

/// Raw per-question outputs of one Laya read.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LayaReadOutput {
    /// One scorer logit per option marker.
    pub(crate) logits: Vec<f32>,
}

/// A loaded Laya decision model (encoder + typed decision head).
///
/// Laya does not load through `skippy_model_open`: it owns its GGUF,
/// tokenizer, and execution context, and it only serves decision reads.
pub struct LayaModel {
    raw: *mut skippy_ffi::LayaModel,
    info: LayaModelInfo,
}

// The native handle serializes reads internally and the tokenizer is immutable
// after loading.
unsafe impl Send for LayaModel {}
unsafe impl Sync for LayaModel {}

impl LayaModel {
    /// Opens a Laya GGUF. `device` is `None` for the CPU backend, `"auto"` for
    /// the first GPU device, or a ggml backend device name such as `"CUDA0"`.
    pub fn open(path: impl AsRef<Path>, threads: usize, device: Option<&str>) -> Result<Self> {
        ensure_laya_supported()?;
        let path = path.as_ref();
        write_native_log_note(format!(
            "skippy_laya_model_open begin path={}",
            path.display()
        ));
        let c_path = path_to_cstring(path, "Laya model path")?;
        let threads = i32::try_from(threads.max(1)).unwrap_or(i32::MAX);
        let c_device = device
            .map(|device| {
                std::ffi::CString::new(device).context("Laya device name contains a NUL byte")
            })
            .transpose()?;
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_laya_model_open(
                c_path.as_ptr(),
                threads,
                c_device
                    .as_ref()
                    .map_or(ptr::null(), |device| device.as_ptr()),
                &mut raw,
                &mut error,
            )
        };
        ensure_ok(status, error).with_context(|| format!("open Laya model {}", path.display()))?;
        if raw.is_null() {
            return Err(anyhow!("skippy_laya_model_open returned a null handle"));
        }
        let mut model = Self {
            raw,
            info: LayaModelInfo {
                max_len: 0,
                head_max_len: 0,
                max_markers: 0,
                action_classes: 0,
                cls_token_id: 0,
                sep_token_id: 0,
                mask_token_id: 0,
                embedding_size: 0,
                layer_count: 0,
                parameter_count: 0,
                temperature: [0.0; skippy_ffi::LAYA_QTYPE_COUNT],
            },
        };
        model.info = model.query_info()?;
        write_native_log_note(format!(
            "skippy_laya_model_open returned layers={} max_len={}",
            model.info.layer_count, model.info.max_len
        ));
        Ok(model)
    }

    pub fn info(&self) -> &LayaModelInfo {
        &self.info
    }

    fn query_info(&self) -> Result<LayaModelInfo> {
        let mut raw = skippy_ffi::LayaInfoV1::default();
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_laya_model_info_v1(self.raw, &mut raw, &mut error) };
        ensure_ok(status, error)?;
        let positive = |value: i32, name: &str| {
            usize::try_from(value)
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| anyhow!("Laya model reports an invalid {name}: {value}"))
        };
        Ok(LayaModelInfo {
            max_len: positive(raw.max_len, "max_len")?,
            head_max_len: positive(raw.head_max_len, "head_max_len").and_then(|head| {
                // Native open enforces this too; a smaller sequence than
                // its head would cut option markers.
                if head > usize::try_from(raw.max_len).unwrap_or(0) {
                    Err(anyhow!(
                        "Laya model's head budget {head} exceeds max_len {}",
                        raw.max_len
                    ))
                } else {
                    Ok(head)
                }
            })?,
            max_markers: positive(raw.max_markers, "marker capacity")?,
            action_classes: positive(raw.n_act, "action class count")?,
            cls_token_id: raw.cls_token_id,
            sep_token_id: raw.sep_token_id,
            mask_token_id: raw.mask_token_id,
            embedding_size: u32::try_from(raw.n_embd).unwrap_or(0),
            layer_count: u32::try_from(raw.n_layer).unwrap_or(0),
            parameter_count: u64::try_from(raw.parameter_count).unwrap_or(0),
            temperature: raw.temperature,
        })
    }

    /// Measures the model's memory at its largest read. The first call runs one
    /// full-length warm-up read so the scheduler reserves its largest buffers.
    pub fn memory(&self) -> Result<LayaMemory> {
        let mut raw = skippy_ffi::LayaMemoryV1::default();
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_laya_model_memory_v1(self.raw, &mut raw, &mut error) };
        ensure_ok(status, error)?;
        Ok(LayaMemory {
            weights_bytes: raw.weights_bytes,
            compute_bytes: raw.compute_bytes,
            host_scratch_bytes: raw.host_scratch_bytes,
            on_accelerator: raw.on_accelerator,
        })
    }

    /// Tokenizes text without special tokens using the model's own tokenizer.
    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        // The tokenizer maps each space to U+2581 (three bytes) and prefixes
        // one, and every BPE token covers at least one byte of that text, so
        // this capacity always suffices; a larger reported count is an error,
        // never a reason to grow the buffer.
        let capacity = text.len().saturating_mul(3).saturating_add(8);
        let mut tokens = vec![0_i32; capacity];
        let mut count = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_laya_tokenize(
                self.raw,
                text.as_ptr().cast(),
                text.len(),
                tokens.as_mut_ptr(),
                tokens.len(),
                &mut count,
                &mut error,
            )
        };
        ensure_ok(status, error).with_context(|| {
            format!(
                "Laya tokenizer produced {count} tokens for {} bytes",
                text.len()
            )
        })?;
        tokens.truncate(count);
        Ok(tokens)
    }

    /// Runs one decision read over every question sequence.
    pub(crate) fn read(&self, sequences: &[LayaSequence]) -> Result<Vec<LayaReadOutput>> {
        if sequences.is_empty() {
            return Err(anyhow!("a Laya read needs at least one question"));
        }
        let max_markers = self.info.max_markers;
        let mut tokens = Vec::new();
        let mut markers = Vec::new();
        let mut raw_sequences = Vec::with_capacity(sequences.len());
        for sequence in sequences {
            if sequence.markers.is_empty() || sequence.markers.len() > max_markers {
                return Err(anyhow!(
                    "a Laya question needs 1 to {max_markers} options, got {}",
                    sequence.markers.len()
                ));
            }
            raw_sequences.push(skippy_ffi::LayaSequence {
                token_offset: tokens.len(),
                token_count: sequence.tokens.len(),
                qtype: sequence.question_type.index() as i32,
                marker_offset: markers.len(),
                marker_count: sequence.markers.len(),
            });
            tokens.extend_from_slice(&sequence.tokens);
            for marker in &sequence.markers {
                markers.push(i32::try_from(*marker).context("Laya marker position overflow")?);
            }
        }

        let action_classes = self.info.action_classes;
        let mut logits = vec![0.0_f32; sequences.len() * max_markers];
        let mut act_logits = vec![0.0_f32; sequences.len() * action_classes];
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_laya_read(
                self.raw,
                tokens.as_ptr(),
                tokens.len(),
                markers.as_ptr(),
                markers.len(),
                raw_sequences.as_ptr(),
                raw_sequences.len(),
                logits.as_mut_ptr(),
                logits.len(),
                act_logits.as_mut_ptr(),
                act_logits.len(),
                &mut error,
            )
        };
        ensure_ok(status, error)?;

        Ok(sequences
            .iter()
            .enumerate()
            .map(|(index, sequence)| {
                let row = index * max_markers;
                LayaReadOutput {
                    logits: logits[row..row + sequence.markers.len()].to_vec(),
                }
            })
            .collect())
    }
}

impl Drop for LayaModel {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { skippy_ffi::skippy_laya_model_free(self.raw) };
        }
    }
}

fn ensure_laya_supported() -> Result<()> {
    if skippy_ffi::try_abi_features()
        .is_none_or(|features| features & skippy_ffi::FEATURE_LAYA_DECISIONS == 0)
    {
        return Err(anyhow!(
            "native runtime does not support Laya decision reads"
        ));
    }
    Ok(())
}
