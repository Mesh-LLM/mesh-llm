//! DFlash block-diffusion speculative drafts beside a Skippy target context.
//!
//! A DFlash draft is a small upstream llama.cpp model (`general.architecture =
//! dflash`) trained for one target. It never sees target tokens directly:
//! after every target decode, the target's inputs to a few chosen layers are
//! encoded and injected into the draft's KV cache at the target positions.
//! Drafting then decodes one noise block `[anchor, <mask> * n]` and reads the
//! whole block at once. DFlash2 drafts add a candidate selector and return a
//! token lattice instead of logits.
//!
//! The draft context shares the target's token embeddings and output head
//! through `ctx_other`, so it is created with upstream llama.cpp calls rather
//! than through the Skippy model ABI. Draft state is advisory: a failed draft
//! operation clears that lane's draft cache and never fails the target.

mod draft_batch;
mod readout;

use std::ffi::CString;
use std::path::Path;
use std::ptr;
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use skippy_ffi::Opaque;
use skippy_ffi::llama_draft::{
    LLAMA_CONTEXT_TYPE_DEFAULT, LLAMA_ROPE_TYPE_MROPE, LlamaContextParams, LlamaDraftApi,
    LlamaModelParams,
};

use crate::path_cstring::path_to_cstring;
use crate::{RuntimeConfig, StageModel, StageSession, ensure_ok, write_native_log_note};

use draft_batch::{DraftBatch, target_feature_row};
use readout::{argmax_token, block_draft_capacity, walk_dflash2_lattice};

const DFLASH_ARCHITECTURE: &str = "dflash";
const DEFAULT_BLOCK_SIZE: usize = 16;
/// Draft cache and batch room reserved per lane for one scratch noise block.
/// Trained DFlash blocks are 8-16 tokens.
const MAX_BLOCK_SIZE: usize = 64;
/// DSpark drafts share the `dflash` architecture and add a Markov head.
const DSPARK_MARKOV_TENSOR: &str = "markov_w1.weight";

/// Which DFlash draft family a loaded model belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DFlashVariant {
    /// Greedy readout of every masked block position.
    DFlash,
    /// Local convolution plus a candidate selector lattice.
    DFlash2,
}

impl DFlashVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DFlash => "dflash",
            Self::DFlash2 => "dflash2",
        }
    }
}

/// Caller-selected draft placement and proposal bounds.
#[derive(Debug, Clone, Default)]
pub struct DFlashDraftOptions {
    /// Upper bound on draft tokens per block. Clamped to the trained block.
    pub max_draft_tokens: Option<usize>,
    /// Draft layers offloaded to the accelerator. Defaults to the target's.
    pub n_gpu_layers: Option<i32>,
    /// Backend device name for the draft. Defaults to the target's device.
    pub device: Option<String>,
}

/// Facts about an attached draft that callers use for policy and telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DFlashDraftInfo {
    pub variant: DFlashVariant,
    pub block_size: usize,
    pub max_draft_tokens: usize,
    pub target_layer_ids: Vec<u32>,
    pub selector_top_k: usize,
}

/// Draft tokens for the position after the session's last committed token.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DFlashProposal {
    pub tokens: Vec<i32>,
    pub proposal_compute_us: u64,
}

pub(crate) struct DFlashDraft {
    api: &'static LlamaDraftApi,
    target_ctx: *mut Opaque,
    model: *mut Opaque,
    ctx: *mut Opaque,
    info: DFlashDraftInfo,
    lanes: i32,
    n_embd_target: usize,
    n_embd_draft: usize,
    n_vocab: usize,
    n_ubatch: usize,
    target_n_batch: usize,
    mask_token: i32,
    mrope: bool,
    batch: DraftBatch,
}

// The draft model and context are only touched behind the owning Mutex, and
// Skippy serializes target-context use on the runtime worker.
unsafe impl Send for DFlashDraft {}

struct LoadedDraft {
    model: *mut Opaque,
    ctx: *mut Opaque,
}

impl DFlashDraft {
    /// Loads `path` as a draft for the target context and enables capture of
    /// the target layer inputs the draft was trained on.
    pub(crate) fn open(
        target_ctx: *mut Opaque,
        path: &Path,
        options: &DFlashDraftOptions,
        config: &RuntimeConfig,
    ) -> Result<Self> {
        let api = skippy_ffi::llama_draft::llama_draft_api()
            .ok_or_else(|| anyhow!("native runtime does not export the draft-context API"))?;
        reject_dspark(path)?;
        validate_default_params(api)?;
        let target_model = unsafe { (api.get_model)(target_ctx) };
        if target_model.is_null() {
            bail!("target context has no model");
        }
        let lanes = i32::try_from(config.lane_count.max(1)).context("lane count exceeds i32")?;
        let loaded = load_draft(api, target_ctx, path, options, config, lanes)?;
        let mut draft = Self {
            api,
            target_ctx,
            model: loaded.model,
            ctx: loaded.ctx,
            info: DFlashDraftInfo {
                variant: DFlashVariant::DFlash,
                block_size: DEFAULT_BLOCK_SIZE,
                max_draft_tokens: 0,
                target_layer_ids: Vec::new(),
                selector_top_k: 0,
            },
            lanes,
            n_embd_target: 0,
            n_embd_draft: 0,
            n_vocab: 0,
            n_ubatch: 0,
            target_n_batch: 0,
            mask_token: -1,
            mrope: false,
            batch: DraftBatch::default(),
        };
        // From here the Drop impl owns the native handles.
        draft.configure(target_model, options, config)?;
        Ok(draft)
    }

    fn configure(
        &mut self,
        target_model: *const Opaque,
        options: &DFlashDraftOptions,
        config: &RuntimeConfig,
    ) -> Result<()> {
        let api = self.api;
        let architecture =
            unsafe { skippy_ffi::llama_model_meta_val_str(self.model, "general.architecture") };
        if architecture.as_deref() != Some(DFLASH_ARCHITECTURE) {
            bail!(
                "draft model architecture is {}, expected {DFLASH_ARCHITECTURE}",
                architecture.as_deref().unwrap_or("unknown")
            );
        }
        if !(config.resident_tensor_names.is_empty() && config.layer_start == 0) {
            bail!("DFlash drafts require the complete target model in one stage");
        }
        let target_layers = unsafe { (api.model_n_layer)(target_model) };
        self.info.target_layer_ids = target_layer_ids(api, self.model, target_layers)?;
        self.n_embd_target =
            positive_usize(unsafe { (api.model_n_embd)(target_model) }, "target n_embd")?;
        self.n_embd_draft =
            positive_usize(unsafe { (api.model_n_embd)(self.model) }, "draft n_embd")?;
        if self.n_embd_draft != self.n_embd_target {
            bail!(
                "draft hidden size {} does not match target hidden size {}",
                self.n_embd_draft,
                self.n_embd_target
            );
        }
        let draft_vocab = unsafe { (api.model_get_vocab)(self.model) };
        let target_vocab = unsafe { (api.model_get_vocab)(target_model) };
        self.n_vocab = positive_usize(unsafe { (api.vocab_n_tokens)(draft_vocab) }, "draft vocab")?;
        let target_n_vocab = unsafe { (api.vocab_n_tokens)(target_vocab) };
        if usize::try_from(target_n_vocab).ok() != Some(self.n_vocab) {
            bail!(
                "draft vocabulary has {} tokens but the target has {target_n_vocab}",
                self.n_vocab
            );
        }
        self.mask_token = unsafe { (api.vocab_mask)(draft_vocab) };
        if self.mask_token < 0 {
            bail!("draft model has no mask token");
        }
        self.mrope = unsafe { (api.model_rope_type)(self.model) } == LLAMA_ROPE_TYPE_MROPE;
        self.n_ubatch = usize::try_from(unsafe { (api.n_ubatch)(self.ctx) })
            .context("draft n_ubatch exceeds usize")?
            .max(1);
        self.target_n_batch = usize::try_from(unsafe { (api.n_batch)(self.target_ctx) })
            .context("target n_batch exceeds usize")?;
        self.info.block_size =
            meta_usize(self.model, "dflash.block_size")?.unwrap_or(DEFAULT_BLOCK_SIZE);
        if !(2..=MAX_BLOCK_SIZE).contains(&self.info.block_size) {
            bail!(
                "draft block size {} is outside the supported 2..={MAX_BLOCK_SIZE}",
                self.info.block_size
            );
        }
        let selector_top_k = unsafe { (api.model_dflash_selector_top_k)(self.model) };
        if selector_top_k > 0 {
            self.info.variant = DFlashVariant::DFlash2;
            self.info.selector_top_k = positive_usize(selector_top_k, "selector top-k")?;
            let row_used = self.info.selector_top_k * (self.info.selector_top_k + 1);
            if row_used > self.n_embd_draft {
                bail!("draft hidden size is too small for the DFlash2 selector lattice");
            }
        }
        self.info.max_draft_tokens = block_draft_capacity(
            self.info.block_size,
            options.max_draft_tokens.unwrap_or(usize::MAX),
        );
        let causal =
            unsafe { skippy_ffi::llama_model_meta_val_str(self.model, "dflash.attention.causal") }
                .is_some_and(|value| value == "true");
        let dflash2 = self.info.variant == DFlashVariant::DFlash2;
        unsafe {
            // DFlash2 reads its lattice from every row of the nextn output.
            (api.set_embeddings_nextn)(self.ctx, true, !dflash2);
            (api.set_causal_attn)(self.ctx, causal);
            for &lid in &self.info.target_layer_ids {
                (api.set_embeddings_layer_inp)(self.target_ctx, lid, true);
            }
        }
        write_native_log_note(format!(
            "dflash draft attached variant={} block_size={} max_draft_tokens={} target_layers={:?} mask_token={} causal={causal} mrope={}",
            self.info.variant.as_str(),
            self.info.block_size,
            self.info.max_draft_tokens,
            self.info.target_layer_ids,
            self.mask_token,
            self.mrope,
        ));
        Ok(())
    }

    pub(crate) fn info(&self) -> &DFlashDraftInfo {
        &self.info
    }

    fn owns_lane(&self, seq_id: i32) -> bool {
        (0..self.lanes).contains(&seq_id)
    }

    /// Injects the target features for `positions.len()` rows of the last
    /// target decode, starting at batch row `first_row`.
    pub(crate) fn ingest(
        &mut self,
        seq_id: i32,
        first_row: usize,
        positions: &[i32],
    ) -> Result<()> {
        if positions.is_empty() {
            return Ok(());
        }
        // The capture buffer holds `n_batch` rows per layer, filled by the
        // target decode that just ran; reading past it would be out of bounds.
        let rows_needed = first_row
            .checked_add(positions.len())
            .filter(|rows| *rows <= self.target_n_batch)
            .ok_or_else(|| {
                anyhow!(
                    "{} target rows from row {first_row} exceed the capture buffer of {} rows",
                    positions.len(),
                    self.target_n_batch
                )
            })?;
        let layer_len = rows_needed * self.n_embd_target;
        let mut layers = Vec::with_capacity(self.info.target_layer_ids.len());
        for &lid in &self.info.target_layer_ids {
            // The native getter aborts unless the buffer is allocated. Capture
            // is enabled for the draft's whole lifetime and nothing else
            // toggles it, and sessions only ingest rows of a target decode
            // that succeeded, which allocated the buffer.
            let data = unsafe { (self.api.get_embeddings_layer_inp)(self.target_ctx, lid) };
            if data.is_null() {
                bail!("target layer {lid} input was not captured");
            }
            layers.push(unsafe { std::slice::from_raw_parts(data.cast_const(), layer_len) });
        }
        for (chunk_index, chunk) in positions.chunks(self.n_ubatch).enumerate() {
            self.batch.clear();
            let chunk_row = first_row + chunk_index * self.n_ubatch;
            for (offset, &position) in chunk.iter().enumerate() {
                self.batch.push_features(
                    target_feature_row(&layers, self.n_embd_target, chunk_row + offset),
                    position,
                    seq_id,
                );
            }
            let rc = unsafe { (self.api.decode)(self.ctx, self.batch.as_raw(self.mrope)) };
            if rc != 0 {
                bail!("draft feature injection failed with status {rc}");
            }
        }
        Ok(())
    }

    /// Drops draft cache entries at or after `position` for one lane.
    pub(crate) fn truncate(&mut self, seq_id: i32, position: i32) {
        unsafe {
            let memory = (self.api.get_memory)(self.ctx);
            (self.api.memory_seq_rm)(memory, seq_id, position.max(0), -1);
        }
    }

    /// Drops every lane's draft cache.
    fn clear_all(&mut self) {
        unsafe {
            let memory = (self.api.get_memory)(self.ctx);
            (self.api.memory_seq_rm)(memory, -1, -1, -1);
        }
    }

    pub(crate) fn clear(&mut self, seq_id: i32) {
        unsafe {
            let memory = (self.api.get_memory)(self.ctx);
            (self.api.memory_seq_rm)(memory, seq_id, -1, -1);
        }
    }

    /// Drafts up to `max_tokens` tokens that follow `anchor`, the token the
    /// target will consume next at `anchor_position`.
    pub(crate) fn propose(
        &mut self,
        seq_id: i32,
        anchor_position: i32,
        anchor: i32,
        max_tokens: usize,
    ) -> Result<Vec<i32>> {
        let draft_tokens = max_tokens.min(self.info.max_draft_tokens);
        if draft_tokens == 0 {
            return Ok(Vec::new());
        }
        let dflash2 = self.info.variant == DFlashVariant::DFlash2;
        // Every block row is an output, as upstream requests: marking only the
        // mask rows returns logits shifted by one row. DFlash2 reads its
        // lattice from the nextn output instead and needs no logits.
        self.batch.clear();
        for index in 0..=draft_tokens {
            let token = if index == 0 { anchor } else { self.mask_token };
            let position = anchor_position
                .checked_add(i32::try_from(index).context("draft index exceeds i32")?)
                .context("draft position overflow")?;
            self.batch.push_token(token, position, seq_id, !dflash2);
        }
        let rc = unsafe { (self.api.decode)(self.ctx, self.batch.as_raw(self.mrope)) };
        let tokens = if rc == 0 {
            self.read_block(draft_tokens)
        } else {
            Err(anyhow!("draft block decode failed with status {rc}"))
        };
        // The noise block is scratch; only target features persist.
        self.truncate(seq_id, anchor_position);
        tokens
    }

    fn read_block(&self, draft_tokens: usize) -> Result<Vec<i32>> {
        if self.info.variant == DFlashVariant::DFlash2 {
            let lattice = unsafe { (self.api.get_embeddings_nextn)(self.ctx) };
            if lattice.is_null() {
                bail!("DFlash2 selector produced no lattice");
            }
            let rows = draft_tokens + 1;
            let lattice = unsafe {
                std::slice::from_raw_parts(lattice.cast_const(), rows * self.n_embd_draft)
            };
            let draft_rows = lattice.chunks_exact(self.n_embd_draft).skip(1);
            return Ok(walk_dflash2_lattice(
                draft_rows,
                self.info.selector_top_k,
                self.n_vocab,
            ));
        }
        let mut tokens = Vec::with_capacity(draft_tokens);
        for row in 1..=draft_tokens {
            let row = i32::try_from(row).context("draft row exceeds i32")?;
            let logits = unsafe { (self.api.get_logits_ith)(self.ctx, row) };
            if logits.is_null() {
                bail!("draft produced no logits for block row {row}");
            }
            let logits = unsafe { std::slice::from_raw_parts(logits.cast_const(), self.n_vocab) };
            tokens.push(argmax_token(logits).context("draft logits row is empty")?);
        }
        Ok(tokens)
    }
}

impl Drop for DFlashDraft {
    fn drop(&mut self) {
        unsafe {
            for &lid in &self.info.target_layer_ids {
                (self.api.set_embeddings_layer_inp)(self.target_ctx, lid, false);
            }
            if !self.ctx.is_null() {
                (self.api.free)(self.ctx);
            }
            if !self.model.is_null() {
                (self.api.model_free)(self.model);
            }
        }
    }
}

fn reject_dspark(path: &Path) -> Result<()> {
    let catalog = skippy_model::gguf_catalog::read_gguf_tensor_catalog(path)
        .with_context(|| format!("read draft GGUF {}", path.display()))?;
    if catalog
        .tensors
        .iter()
        .any(|tensor| tensor.name == DSPARK_MARKOV_TENSOR)
    {
        bail!(
            "DSpark drafts are not supported; {} has a Markov head",
            path.display()
        );
    }
    Ok(())
}

/// Refuses a runtime whose upstream parameter defaults do not match the
/// pinned layout mirrored in `skippy_ffi::llama_draft`.
fn validate_default_params(api: &LlamaDraftApi) -> Result<()> {
    let context = unsafe { (api.context_default_params)() };
    let model = unsafe { (api.model_default_params)() };
    let context_matches = context.n_ubatch == 512
        && context.n_seq_max == 1
        && context.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT
        && context.flash_attn_type == -1
        && context.type_k == 1
        && context.type_v == 1
        && !context.embeddings
        && context.offload_kqv
        && context.n_samplers == 0
        && context.ctx_other.is_null();
    let model_matches = model.n_gpu_layers == -1
        && model.kv_overrides.is_null()
        && !model.vocab_only
        && model.use_extra_bufts
        && !model.load_mtp;
    if !(context_matches && model_matches) {
        bail!("native runtime llama.cpp parameter layout does not match this build");
    }
    Ok(())
}

fn target_layer_ids(
    api: &LlamaDraftApi,
    model: *const Opaque,
    target_layers: i32,
) -> Result<Vec<u32>> {
    let count = unsafe { (api.model_target_layer_ids_n)(model) };
    let ids = unsafe { (api.model_target_layer_ids)(model) };
    if count == 0 || ids.is_null() {
        bail!("draft model does not name any target layers");
    }
    let count = usize::try_from(count).context("target layer count exceeds usize")?;
    let ids = unsafe { std::slice::from_raw_parts(ids, count) };
    ids.iter()
        .map(|&lid| {
            // Skippy sizes layer-input capture to n_layer and aborts past it.
            if lid < 0 || lid >= target_layers {
                bail!("draft target layer {lid} is outside the target's {target_layers} layers");
            }
            u32::try_from(lid).context("target layer id exceeds u32")
        })
        .collect()
}

fn load_draft(
    api: &LlamaDraftApi,
    target_ctx: *mut Opaque,
    path: &Path,
    options: &DFlashDraftOptions,
    config: &RuntimeConfig,
    lanes: i32,
) -> Result<LoadedDraft> {
    let path = path_to_cstring(path, "DFlash draft model path")?;
    let device_name = options
        .device
        .as_deref()
        .or(config.selected_backend_device.as_deref());
    let device_name = device_name
        .map(|name| CString::new(name).context("draft device name contains a NUL byte"))
        .transpose()?;
    let mut devices = Vec::new();
    if let Some(name) = &device_name {
        let device = unsafe { (api.backend_dev_by_name)(name.as_ptr()) };
        if device.is_null() {
            bail!("draft device {} is not available", name.to_string_lossy());
        }
        devices.extend([device, ptr::null_mut()]);
    }
    let mut model_params: LlamaModelParams = unsafe { (api.model_default_params)() };
    model_params.n_gpu_layers = options.n_gpu_layers.unwrap_or(config.n_gpu_layers);
    if !devices.is_empty() {
        model_params.devices = devices.as_mut_ptr();
    }
    write_native_log_note(format!(
        "dflash draft load begin path={} device={:?} n_gpu_layers={}",
        path.to_string_lossy(),
        device_name,
        model_params.n_gpu_layers
    ));
    let model = unsafe { (api.model_load_from_file)(path.as_ptr(), model_params) };
    if model.is_null() {
        bail!(
            "failed to load DFlash draft model {}",
            path.to_string_lossy()
        );
    }
    let context_params = draft_context_params(api, target_ctx, config, lanes);
    let ctx = unsafe { (api.init_from_model)(model, context_params) };
    if ctx.is_null() {
        unsafe { (api.model_free)(model) };
        bail!(
            "failed to create the DFlash draft context (cache_k={} cache_v={} flash_attn={:?})",
            config.cache_type_k,
            config.cache_type_v,
            config.flash_attn_type
        );
    }
    Ok(LoadedDraft { model, ctx })
}

fn draft_context_params(
    api: &LlamaDraftApi,
    target_ctx: *mut Opaque,
    config: &RuntimeConfig,
    lanes: i32,
) -> LlamaContextParams {
    let mut params = unsafe { (api.context_default_params)() };
    let target_n_ctx = unsafe { (api.n_ctx)(target_ctx) };
    let target_n_batch = unsafe { (api.n_batch)(target_ctx) };
    let target_n_ubatch = unsafe { (api.n_ubatch)(target_ctx) };
    // Every lane can hold one scratch noise block beyond its target context.
    let block_headroom = MAX_BLOCK_SIZE as u32;
    params.n_ctx = target_n_ctx.saturating_add(block_headroom.saturating_mul(lanes as u32));
    (params.n_batch, params.n_ubatch) = draft_batch_sizes(target_n_batch, target_n_ubatch);
    params.n_seq_max = lanes as u32;
    // A quantized V cache turns on flash attention under `Auto`; the native
    // context refuses to initialize when the backend cannot provide it.
    params.type_k = config.cache_type_k as i32;
    params.type_v = config.cache_type_v as i32;
    params.flash_attn_type = config.flash_attn_type as i32;
    params.kv_unified = true;
    params.ctx_other = target_ctx;
    let threads = config
        .n_threads
        .or_else(|| {
            std::thread::available_parallelism()
                .ok()
                .and_then(|threads| u32::try_from(threads.get()).ok())
        })
        .and_then(|threads| i32::try_from(threads).ok());
    if let Some(threads) = threads {
        params.n_threads = threads;
        params.n_threads_batch = config
            .n_threads_batch
            .and_then(|threads| i32::try_from(threads).ok())
            .unwrap_or(threads);
    }
    params
}

/// Draft `(n_batch, n_ubatch)` for a target's batch sizes. Non-causal
/// attention needs the whole noise block in one microbatch, so a small target
/// batch or microbatch must not shrink the draft's below the largest block.
fn draft_batch_sizes(target_n_batch: u32, target_n_ubatch: u32) -> (u32, u32) {
    let block = MAX_BLOCK_SIZE as u32;
    let n_batch = target_n_batch.max(block);
    (n_batch, target_n_ubatch.max(block).min(n_batch))
}

fn meta_usize(model: *const Opaque, key: &str) -> Result<Option<usize>> {
    unsafe { skippy_ffi::llama_model_meta_val_str(model, key) }
        .map(|value| {
            value
                .trim()
                .parse::<usize>()
                .with_context(|| format!("draft metadata {key} is not an integer: {value}"))
        })
        .transpose()
}

fn positive_usize(value: i32, label: &str) -> Result<usize> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| anyhow!("{label} must be positive, got {value}"))
}

/// A session's handle to the model's draft. Weak so that the model frees the
/// draft before the target context it borrows from.
#[derive(Clone)]
pub(crate) struct DFlashSessionLink {
    draft: Weak<Mutex<DFlashDraft>>,
    seq_id: i32,
}

impl DFlashSessionLink {
    pub(crate) fn new(draft: &Arc<Mutex<DFlashDraft>>, seq_id: i32) -> Option<Self> {
        let owns_lane = lock_draft(draft).owns_lane(seq_id);
        owns_lane.then(|| Self {
            draft: Arc::downgrade(draft),
            seq_id,
        })
    }

    fn with_draft<R>(&self, operation: impl FnOnce(&mut DFlashDraft, i32) -> R) -> Option<R> {
        let draft = self.draft.upgrade()?;
        let mut draft = lock_draft(&draft);
        Some(operation(&mut draft, self.seq_id))
    }
}

/// Locks the draft, recovering it if a panic poisoned the lock. The panic may
/// have interrupted a cache update on any lane, so recovery drops every lane's
/// draft cache; lanes then draft from the context they decode next instead of
/// leaving DFlash off until restart.
fn lock_draft(draft: &Mutex<DFlashDraft>) -> MutexGuard<'_, DFlashDraft> {
    lock_recovering(draft, DFlashDraft::clear_all)
}

fn lock_recovering<T>(mutex: &Mutex<T>, reset: impl FnOnce(&mut T)) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| {
        let mut guard = poisoned.into_inner();
        reset(&mut guard);
        mutex.clear_poison();
        guard
    })
}

/// Rows of the last target decode that belong to one session.
pub(crate) enum DecodedRows<'a> {
    /// `count` rows ending at the session's current native position.
    Trailing { first_row: usize, count: usize },
    /// Rows at explicit sequence positions (the first M-RoPE axis).
    At {
        first_row: usize,
        positions: &'a [i32],
    },
}

impl<'a> DecodedRows<'a> {
    pub(crate) fn trailing(first_row: usize, count: usize) -> Self {
        Self::Trailing { first_row, count }
    }

    /// Rows for a prefill chunk. Explicit positions are laid out axis-major,
    /// so the first `count` entries are the sequence positions.
    pub(crate) fn for_prefill(count: usize, positions: &'a [i32]) -> Self {
        if positions.len() >= count && !positions.is_empty() {
            Self::At {
                first_row: 0,
                positions: &positions[..count],
            }
        } else {
            Self::trailing(0, count)
        }
    }

    pub(crate) fn offset(self, rows: usize) -> Self {
        match self {
            Self::Trailing { first_row, count } => Self::Trailing {
                first_row: first_row + rows,
                count,
            },
            Self::At {
                first_row,
                positions,
            } => Self::At {
                first_row: first_row + rows,
                positions,
            },
        }
    }
}

impl StageSession {
    pub(crate) fn link_dflash(&mut self, draft: &Arc<Mutex<DFlashDraft>>) -> Result<()> {
        let seq_id = self.native_sequence_id()?;
        self.dflash = DFlashSessionLink::new(draft, seq_id);
        self.dflash_after_replace();
        Ok(())
    }

    /// Whether this session can draft with an attached DFlash model.
    pub fn dflash_attached(&self) -> bool {
        self.dflash.is_some()
    }

    /// Drafts tokens that follow `anchor`, the token the target consumes
    /// next. Returns `None` when no draft is attached. A draft failure clears
    /// the lane's draft cache and yields an empty proposal.
    pub fn dflash_propose(
        &mut self,
        anchor: i32,
        max_tokens: usize,
    ) -> Result<Option<DFlashProposal>> {
        let Some(link) = self.dflash.clone() else {
            return Ok(None);
        };
        let anchor_position =
            i32::try_from(self.native_position()?).context("session position exceeds i32")?;
        let started = Instant::now();
        let tokens = link.with_draft(|draft, seq_id| {
            draft
                .propose(seq_id, anchor_position, anchor, max_tokens)
                .unwrap_or_else(|error| {
                    write_native_log_note(format!("dflash proposal failed: {error:#}"));
                    draft.clear(seq_id);
                    Vec::new()
                })
        });
        Ok(tokens.map(|tokens| DFlashProposal {
            tokens,
            proposal_compute_us: u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        }))
    }

    /// Feeds the target features of this session's rows in the last decode to
    /// the draft.
    pub(crate) fn dflash_after_decode(&mut self, rows: DecodedRows<'_>) {
        let Some(link) = self.dflash.clone() else {
            return;
        };
        let (first_row, trailing_positions);
        let positions = match rows {
            DecodedRows::At {
                first_row: row,
                positions,
            } => {
                first_row = row;
                positions
            }
            DecodedRows::Trailing {
                first_row: row,
                count,
            } => {
                first_row = row;
                let Some(end) = self
                    .native_position()
                    .ok()
                    .and_then(|position| i32::try_from(position).ok())
                else {
                    return;
                };
                let count = i32::try_from(count).unwrap_or(i32::MAX);
                trailing_positions = (end.saturating_sub(count)..end).collect::<Vec<_>>();
                &trailing_positions
            }
        };
        link.with_draft(|draft, seq_id| {
            if let Err(error) = draft.ingest(seq_id, first_row, positions) {
                write_native_log_note(format!("dflash feature injection failed: {error:#}"));
                draft.clear(seq_id);
            }
        });
    }

    /// Keeps the draft cache no longer than a rewound target session.
    pub(crate) fn dflash_after_rewind(&mut self) {
        let Some(link) = self.dflash.clone() else {
            return;
        };
        let Some(position) = self
            .native_position()
            .ok()
            .and_then(|position| i32::try_from(position).ok())
        else {
            link.with_draft(|draft, seq_id| draft.clear(seq_id));
            return;
        };
        link.with_draft(|draft, seq_id| draft.truncate(seq_id, position));
    }

    /// Forgets draft context after the target state was replaced without a
    /// decode (reset, cache restore, or imported state).
    pub(crate) fn dflash_after_replace(&mut self) {
        if let Some(link) = self.dflash.clone() {
            link.with_draft(|draft, seq_id| draft.clear(seq_id));
        }
    }
}

/// Draft storage owned by a target model.
pub(crate) type SharedDFlashDraft = Arc<Mutex<DFlashDraft>>;

impl StageModel {
    /// Attaches a DFlash or DFlash2 draft to this complete target model.
    ///
    /// Attach before creating serving sessions: only sessions created
    /// afterwards mirror their target positions into the draft.
    pub fn attach_dflash_draft(
        &mut self,
        path: impl AsRef<Path>,
        options: &DFlashDraftOptions,
        config: &RuntimeConfig,
    ) -> Result<DFlashDraftInfo> {
        if self.raw_model().is_null() {
            bail!("cannot attach a DFlash draft to a null model");
        }
        if self.dflash_draft().is_some() {
            bail!("a DFlash draft is already attached");
        }
        // Every lane shares one target context; a scratch session exposes it
        // and refreshes any installed stage program for layer capture.
        let probe = self.create_session()?;
        let target_ctx = unsafe { skippy_ffi::skippy_session_llama_context(probe.raw) };
        if target_ctx.is_null() {
            bail!("target session has no llama context");
        }
        let draft = DFlashDraft::open(target_ctx, path.as_ref(), options, config)?;
        let info = draft.info().clone();
        refresh_target_program(&probe)?;
        drop(probe);
        self.install_dflash_draft(Arc::new(Mutex::new(draft)))?;
        Ok(info)
    }

    /// The attached DFlash draft, if any.
    pub fn dflash_info(&self) -> Option<DFlashDraftInfo> {
        let draft = self.dflash_draft()?;
        Some(lock_draft(draft).info().clone())
    }
}

/// Reinstalls a stage program after layer capture changed the target's
/// computation; a no-op for targets without an installed program.
fn refresh_target_program(session: &StageSession) -> Result<()> {
    let mut error = ptr::null_mut();
    let status =
        unsafe { skippy_ffi::skippy_session_begin_external_decode(session.raw, &mut error) };
    ensure_ok(status, error)?;
    let mut error = ptr::null_mut();
    let status = unsafe { skippy_ffi::skippy_session_end_external_decode(session.raw, &mut error) };
    ensure_ok(status, error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_poisoned_lock_is_reset_and_reusable() {
        let mutex = Mutex::new(vec![1, 2, 3]);
        let _ = std::panic::catch_unwind(|| {
            let _guard = mutex.lock().unwrap();
            panic!("poison the lock");
        });
        assert!(mutex.is_poisoned());

        lock_recovering(&mutex, Vec::clear).push(7);

        assert!(!mutex.is_poisoned());
        assert_eq!(*lock_recovering(&mutex, |_| unreachable!()), vec![7]);
    }

    #[test]
    fn draft_microbatch_always_holds_a_whole_block() {
        assert_eq!(draft_batch_sizes(2048, 512), (2048, 512));
        assert_eq!(draft_batch_sizes(2048, 8), (2048, MAX_BLOCK_SIZE as u32));
        assert_eq!(
            draft_batch_sizes(8, 8),
            (MAX_BLOCK_SIZE as u32, MAX_BLOCK_SIZE as u32)
        );
    }
}
