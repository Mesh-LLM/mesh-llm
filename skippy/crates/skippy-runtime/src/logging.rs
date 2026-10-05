use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::io::Write;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

use serde_json::Value;
use tokio::sync::mpsc;

/// GGML_LLAMA_LOG_LEVEL values (set before llama_backend_init).
/// 0=silent, 1=error, 2=warn, 3=info (default), 4=debug.
pub const LLAMA_LOG_LEVEL_DEBUG: &str = "4";

/// Channel sender for filtered native log messages.
/// Messages matching key patterns (backend init, model load, VRAM, KV cache, tokenizer) are sent here.
static NATIVE_LOG_FILTERED_TX: OnceLock<Mutex<Option<mpsc::UnboundedSender<NativeLogEvent>>>> =
    OnceLock::new();

static NATIVE_LOG_AGGREGATOR: OnceLock<Mutex<NativeLogAggregator>> = OnceLock::new();
static NATIVE_LOG_FORWARDING_MASK: AtomicU8 = AtomicU8::new(0);

mod output;
mod parser_policy;

use output::native_log_file;
pub use output::{
    NativeLogSink, redirect_native_logs_to_file, restore_native_logs, set_native_log_sink,
    suppress_native_logs,
};

use parser_policy::{
    ALL_FORWARDING_CATEGORIES, MODEL_CATEGORY, MODEL_FALLBACK_NOTE, category_mask,
};
pub use parser_policy::{NativeLogParserMode, NativeLogParserPolicy};

#[cfg(test)]
static NATIVE_LOG_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) fn native_log_test_guard() -> std::sync::MutexGuard<'static, ()> {
    NATIVE_LOG_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeLogEvent {
    pub message: String,
    pub category: &'static str,
    pub params: Vec<(String, Value)>,
}

#[derive(Debug, Default)]
struct ProgressTracker {
    total: Option<usize>,
    completed: usize,
    next_percent: usize,
}

impl ProgressTracker {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn set_total(&mut self, total: usize) {
        self.total = Some(total);
        self.completed = 0;
        self.next_percent = 10;
    }

    fn advance(
        &mut self,
        delta: usize,
        category: &'static str,
        label: &'static str,
        unit: &'static str,
    ) -> Vec<NativeLogEvent> {
        let Some(total) = self.total else {
            return Vec::new();
        };
        if total == 0 {
            return Vec::new();
        }

        self.completed = self.completed.saturating_add(delta).min(total);
        let mut events = Vec::new();
        while self.next_percent <= 100 && self.completed * 100 >= total * self.next_percent {
            events.push(NativeLogEvent {
                message: format!(
                    "{label} {}% ({}/{} {unit})",
                    self.next_percent, self.completed, total
                ),
                category,
                params: Vec::new(),
            });
            self.next_percent += 10;
        }
        events
    }

    fn is_complete(&self) -> bool {
        matches!(self.total, Some(total) if total > 0 && self.completed >= total)
    }
}

/// Latest measured buffer sizes parsed from native log lines, keyed by the
/// line's kind (compute vs KV). These are what llama.cpp actually allocated
/// during `sched_reserve`, and are the ground truth the memory planner should
/// charge instead of the KV-scaled estimate.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MeasuredNativeBuffers {
    pub compute_mib: Option<f64>,
    pub kv_mib: Option<f64>,
    /// A CPU-resident compute or KV allocation was observed, so the device
    /// footprint is incomplete for a capacity pool that includes host RAM.
    pub host_memory_observed: bool,
}

/// Snapshot of the measured native buffer sizes observed so far in this
/// process. The native log callback is synchronous with model open, so by the
/// time `skippy_model_open` returns, the `sched_reserve` buffer lines have
/// already been parsed. The snapshot is returned whenever the aggregator is
/// reachable; its fields stay `None` until a buffer line is observed (e.g.
/// native log forwarding disabled).
pub fn measured_native_buffers() -> Option<MeasuredNativeBuffers> {
    let aggregator = native_log_aggregator().lock().ok()?;
    Some(aggregator.measured_snapshot())
}

#[derive(Debug, Default)]
struct ModelMetadataHighlights {
    architecture: Option<String>,
    name: Option<String>,
    model_type: Option<String>,
    size_label: Option<String>,
    context_length: Option<String>,
    block_count: Option<String>,
    embedding_length: Option<String>,
    feed_forward_length: Option<String>,
    attention_heads: Option<String>,
    attention_heads_kv: Option<String>,
    tokenizer_model: Option<String>,
    tokenizer_pre: Option<String>,
}

impl ModelMetadataHighlights {
    fn apply(&mut self, key: &str, value: &str) {
        let value = value.trim().trim_matches('"').to_string();
        if value.is_empty() {
            return;
        }

        match key {
            "general.architecture" => self.architecture = Some(value),
            "general.name" => self.name = Some(value),
            "general.type" => self.model_type = Some(value),
            "general.size_label" => self.size_label = Some(value),
            "tokenizer.ggml.model" => self.tokenizer_model = Some(value),
            "tokenizer.ggml.pre" => self.tokenizer_pre = Some(value),
            _ if key.ends_with(".context_length") => self.context_length = Some(value),
            _ if key.ends_with(".block_count") => self.block_count = Some(value),
            _ if key.ends_with(".embedding_length") => self.embedding_length = Some(value),
            _ if key.ends_with(".feed_forward_length") => self.feed_forward_length = Some(value),
            _ if key.ends_with(".attention.head_count") => self.attention_heads = Some(value),
            _ if key.ends_with(".attention.head_count_kv") => self.attention_heads_kv = Some(value),
            _ => {}
        }
    }

    fn summary_params(&self) -> Vec<(String, Value)> {
        let mut params = Vec::new();
        if let Some(value) = &self.architecture {
            params.push(("architecture".to_string(), Value::String(value.clone())));
        }
        if let Some(value) = &self.name {
            params.push(("name".to_string(), Value::String(value.clone())));
        }
        if let Some(value) = &self.model_type {
            params.push(("type".to_string(), Value::String(value.clone())));
        }
        if let Some(value) = &self.size_label {
            params.push(("size".to_string(), Value::String(value.clone())));
        }
        if let Some(value) = &self.context_length {
            params.push(("ctx".to_string(), json_value_from_text(value)));
        }
        if let Some(value) = &self.block_count {
            params.push(("blocks".to_string(), json_value_from_text(value)));
        }
        if let Some(value) = &self.embedding_length {
            params.push(("embed".to_string(), json_value_from_text(value)));
        }
        if let Some(value) = &self.feed_forward_length {
            params.push(("ffn".to_string(), json_value_from_text(value)));
        }
        if let Some(value) = &self.attention_heads {
            params.push(("heads".to_string(), json_value_from_text(value)));
        }
        if let Some(value) = &self.attention_heads_kv {
            params.push(("kv_heads".to_string(), json_value_from_text(value)));
        }
        if let Some(value) = &self.tokenizer_model {
            params.push(("tokenizer".to_string(), Value::String(value.clone())));
        }
        if let Some(value) = &self.tokenizer_pre {
            params.push(("tokenizer_pre".to_string(), Value::String(value.clone())));
        }
        params
    }
}

#[derive(Debug, Default)]
struct NativeLogAggregator {
    metadata_progress: ProgressTracker,
    tensor_progress: ProgressTracker,
    layer_assign_progress: ProgressTracker,
    layer_devices: BTreeMap<usize, String>,
    layer_devices_emitted: bool,
    kv_cache_progress: ProgressTracker,
    metadata_in_dump: bool,
    metadata_summary_emitted: bool,
    metadata_highlights: ModelMetadataHighlights,
    tensor_groups: Vec<(String, usize)>,
    tensor_groups_emitted: bool,
    kv_layers_seen: BTreeSet<usize>,
    /// Latest measured buffer sizes parsed from native log lines, keyed by
    /// backend device name (e.g. `CUDA0`, `Metal`). Updated by the
    /// memory/kv_cache arms of `summarize_native_log_line`; read via
    /// [`measured_native_buffers`] after model open completes. Multiple
    /// reserves on the same device keep the high-water mark; distinct devices
    /// are summed by [`measured_native_buffers`] (one buffer line is printed
    /// per device, so a plain per-kind max would under-measure multi-GPU by a
    /// factor of N). Host-pinned buffers (`CUDA_Host`) and CPU buffers are
    /// excluded at record time — they are not device memory and must never be
    /// charged against a VRAM budget.
    measured_compute_mib: BTreeMap<String, f64>,
    measured_kv_mib: BTreeMap<String, f64>,
    host_memory_observed: bool,
}

fn native_log_aggregator() -> &'static Mutex<NativeLogAggregator> {
    NATIVE_LOG_AGGREGATOR.get_or_init(|| Mutex::new(NativeLogAggregator::default()))
}

/// Register a channel receiver for filtered native log messages.
/// Returns the receiver end; call this once before model loading begins.
pub fn register_filtered_native_logs() -> mpsc::UnboundedReceiver<NativeLogEvent> {
    let (tx, rx) = mpsc::unbounded_channel();
    NATIVE_LOG_FILTERED_TX
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap()
        .replace(tx);
    if let Ok(mut aggregator) = native_log_aggregator().lock() {
        aggregator.reset();
    }
    rx
}

pub fn unregister_filtered_native_logs() {
    if let Some(sender) = NATIVE_LOG_FILTERED_TX.get() {
        sender.lock().unwrap().take();
    }
    if let Ok(mut aggregator) = native_log_aggregator().lock() {
        aggregator.reset();
    }
}

pub fn set_filtered_native_logs_enabled(enabled: bool) {
    let mask = if enabled {
        ALL_FORWARDING_CATEGORIES
    } else {
        0
    };
    NATIVE_LOG_FORWARDING_MASK.store(mask, Ordering::Relaxed);
}

pub fn configure_native_log_parser(policy: NativeLogParserPolicy) {
    NATIVE_LOG_FORWARDING_MASK.store(policy.forwarding_mask, Ordering::Relaxed);
}

impl NativeLogAggregator {
    /// Per-device-summed measured buffer snapshot (see
    /// [`measured_native_buffers`] for the accounting rules).
    fn measured_snapshot(&self) -> MeasuredNativeBuffers {
        let compute_mib = (!self.measured_compute_mib.is_empty())
            .then(|| self.measured_compute_mib.values().sum::<f64>());
        let kv_mib =
            (!self.measured_kv_mib.is_empty()).then(|| self.measured_kv_mib.values().sum::<f64>());
        MeasuredNativeBuffers {
            compute_mib,
            kv_mib,
            host_memory_observed: self.host_memory_observed,
        }
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn reset_model_loading_state(&mut self) {
        self.metadata_progress.reset();
        self.tensor_progress.reset();
        self.layer_assign_progress.reset();
        self.layer_devices.clear();
        self.layer_devices_emitted = false;
        self.kv_cache_progress.reset();
        self.metadata_in_dump = false;
        self.metadata_summary_emitted = false;
        self.metadata_highlights = ModelMetadataHighlights::default();
        self.tensor_groups.clear();
        self.tensor_groups_emitted = false;
        self.kv_layers_seen.clear();
        // A new model load invalidates the previous model's measured buffer
        // sizes: buffer scales are model/shape-specific, and charging one
        // model's HWM against another's budget would be wrong both directions.
        self.measured_compute_mib.clear();
        self.measured_kv_mib.clear();
        self.host_memory_observed = false;
    }

    fn process_line(&mut self, line: &str) -> Vec<NativeLogEvent> {
        let s = line.trim();
        if s.is_empty() {
            return Vec::new();
        }

        let mut events = Vec::new();
        let metadata_kv = parse_metadata_kv_line(s);
        let tensor_summary = parse_tensor_type_summary(s);
        let layer_assignment = parse_layer_assignment(s);
        if metadata_kv.is_none() {
            events.extend(self.flush_metadata_summary());
        }
        if tensor_summary.is_none() {
            events.extend(self.flush_tensor_group_summary());
        }
        if layer_assignment.is_none() {
            events.extend(self.flush_layer_device_summary());
        }

        if let Some((metadata_rows, tensor_rows)) = parse_loaded_metadata_counts(s) {
            self.reset_model_loading_state();
            self.metadata_progress.set_total(metadata_rows);
            self.tensor_progress.set_total(tensor_rows);
            events.push(NativeLogEvent {
                message: format!(
                    "model load plan: metadata rows={metadata_rows}, tensor rows={tensor_rows}"
                ),
                category: "model",
                params: Vec::new(),
            });
            return events;
        }

        if let Some((key, value)) = metadata_kv {
            self.metadata_in_dump = true;
            self.metadata_highlights.apply(key, value);
            events.extend(
                self.metadata_progress
                    .advance(1, "model", "metadata", "rows"),
            );
            return events;
        }

        if let Some((tensor_type, count)) = tensor_summary {
            self.record_tensor_group(tensor_type, count);
            events.extend(
                self.tensor_progress
                    .advance(count, "model", "tensors", "tensors"),
            );
            if self.tensor_progress.is_complete() {
                events.extend(self.flush_tensor_group_summary());
            }
            return events;
        }

        if let Some(layers) = parse_kv_cache_layers_total(s) {
            self.kv_cache_progress.set_total(layers);
            self.kv_layers_seen.clear();
            events.push(NativeLogEvent {
                message: format!("kv cache plan: layer rows={layers}"),
                category: "kv_cache",
                params: Vec::new(),
            });
            return events;
        }

        if let Some(layer_index) = parse_kv_cache_layer_index(s) {
            if self.kv_layers_seen.insert(layer_index) {
                events.extend(
                    self.kv_cache_progress
                        .advance(1, "kv_cache", "kv cache", "layers"),
                );
            }
            return events;
        }

        if let Some((layer_index, device)) = layer_assignment {
            events.extend(self.record_layer_assignment(layer_index, device));
            return events;
        }

        if should_suppress_native_log_line(s) {
            return events;
        }

        self.record_measured_buffer_size(s);

        if let Some(event) = summarize_native_log_line(s) {
            events.push(event);
        }

        events
    }

    /// Track the largest measured buffer size per kind. A later, smaller line
    /// (e.g. a per-graph reserve for a shorter context) must not lower the
    /// high-water mark recorded at full context init. CPU-offload lines are
    /// skipped: the snapshot feeds VRAM planning, and a CPU-resident buffer
    /// larger than the accelerator's must not be charged against VRAM.
    fn record_measured_buffer_size(&mut self, line: &str) {
        if !line.contains("buffer size") {
            return;
        }
        let Some(device) = buffer_size_device(line) else {
            return;
        };
        let is_compute = line.contains("compute buffer size");
        let is_kv = line.contains("KV buffer size");
        if !is_compute && !is_kv {
            return;
        }
        if device == "CPU" || device.starts_with("CPU_") {
            self.host_memory_observed = true;
            return;
        }
        let Some(mib) = parse_buffer_size_mib(line) else {
            return;
        };
        // Host-pinned staging buffers (CUDA_Host and friends) are host RAM,
        // not device memory — never charge them against a VRAM budget.
        if is_host_pinned_device_name(&device) {
            return;
        }
        let field = if is_compute {
            &mut self.measured_compute_mib
        } else {
            &mut self.measured_kv_mib
        };
        // One line per device per reserve: keep the high-water mark within a
        // device (larger of repeated reserves) so a smaller re-reserve on the
        // same device cannot shrink the measured footprint.
        let slot = field.entry(device).or_insert(0.0);
        if mib > *slot {
            *slot = mib;
        }
    }

    fn record_layer_assignment(&mut self, layer_index: usize, device: &str) -> Vec<NativeLogEvent> {
        if self.layer_devices.get(&layer_index).map(String::as_str) != Some(device) {
            self.layer_devices.insert(layer_index, device.to_string());
            self.layer_devices_emitted = false;
        }
        if self.layer_assign_progress.total.is_none()
            && let Some(total) = self
                .metadata_highlights
                .block_count
                .as_deref()
                .and_then(|s| s.parse::<usize>().ok())
        {
            self.layer_assign_progress.set_total(total);
        }
        let new_completed = layer_index.saturating_add(1);
        let delta = new_completed.saturating_sub(self.layer_assign_progress.completed);
        self.layer_assign_progress
            .advance(delta, "model", "layers", "layers")
    }

    fn flush_layer_device_summary(&mut self) -> Vec<NativeLogEvent> {
        if self.layer_devices.is_empty() || self.layer_devices_emitted {
            return Vec::new();
        }
        self.layer_devices_emitted = true;
        let mut counts = BTreeMap::new();
        for device in self.layer_devices.values() {
            *counts.entry(device.as_str()).or_insert(0_u64) += 1;
        }
        vec![NativeLogEvent {
            message: "Model layers by device".to_string(),
            category: "model",
            params: counts
                .into_iter()
                .map(|(device, count)| (device.to_string(), Value::from(count)))
                .collect(),
        }]
    }

    fn flush_metadata_summary(&mut self) -> Vec<NativeLogEvent> {
        if !self.metadata_in_dump || self.metadata_summary_emitted {
            return Vec::new();
        }
        self.metadata_in_dump = false;
        self.metadata_summary_emitted = true;
        let params = self.metadata_highlights.summary_params();
        if params.is_empty() {
            Vec::new()
        } else {
            vec![NativeLogEvent {
                message: "Reading model metadata...".to_string(),
                category: "model",
                params,
            }]
        }
    }

    fn record_tensor_group(&mut self, tensor_type: &str, count: usize) {
        let tensor_type = canonical_tensor_group_key(tensor_type);
        if let Some((_, existing_count)) = self
            .tensor_groups
            .iter_mut()
            .find(|(existing_type, _)| existing_type == &tensor_type)
        {
            *existing_count = count;
        } else {
            self.tensor_groups.push((tensor_type, count));
        }
        self.tensor_groups_emitted = false;
    }

    fn flush_tensor_group_summary(&mut self) -> Vec<NativeLogEvent> {
        if self.tensor_groups.is_empty() || self.tensor_groups_emitted {
            return Vec::new();
        }
        self.tensor_groups_emitted = true;
        vec![NativeLogEvent {
            message: "Reading tensor groups...".to_string(),
            category: "model",
            params: self
                .tensor_groups
                .iter()
                .map(|(tensor_type, count)| (tensor_type.clone(), Value::from(*count as u64)))
                .collect(),
        }]
    }
}

fn json_value_from_text(value: &str) -> Value {
    value
        .parse::<u64>()
        .map(Value::from)
        .unwrap_or_else(|_| Value::String(value.to_string()))
}

fn canonical_tensor_group_key(tensor_type: &str) -> String {
    let trimmed = tensor_type.trim();
    if trimmed.eq_ignore_ascii_case("q4_k") {
        "q4_K".to_string()
    } else if trimmed.eq_ignore_ascii_case("q5_k") {
        "q5_K".to_string()
    } else {
        trimmed.to_string()
    }
}

fn should_suppress_native_log_line(line: &str) -> bool {
    line.starts_with("llama_model_loader:") && (line.contains(": - kv") || line.contains("- kv"))
        || (line.starts_with("clip_model_loader:") && line.contains(": tensor["))
        || line.contains("tokenizer.ggml.tokens arr")
        || line.contains("tokenizer.ggml.merges arr")
        || line.contains("tokenizer.ggml.token_type arr")
        || line.starts_with("print_info:")
        || (line.starts_with("llama_kv_cache:")
            && (line.contains(": filtered") || line.contains(": dev =")))
}

fn parse_buffer_size_mib(line: &str) -> Option<f64> {
    // Native buffer-size lines print the value with a fixed-width field, e.g.
    // `sched_reserve:        CUDA0 compute buffer size =   579.83 MiB` or
    // `llama_kv_cache:        CUDA0 KV buffer size =  1088.00 MiB`. Capture the
    // last `<number> MiB` occurrence on the line.
    let rest = line.rfind("MiB")?;
    let prefix = line[..rest].trim_end();
    let start = prefix
        .rfind(|c: char| !(c.is_ascii_digit() || c == '.'))
        .map(|idx| idx + 1)
        .unwrap_or(0);
    prefix[start..].trim().parse::<f64>().ok()
}

/// Host-pinned buffer names some CUDA backends report (e.g. `CUDA_Host`).
/// Their memory is host RAM pinned for device transfers, not device memory —
/// it must not be charged against a VRAM budget.
fn is_host_pinned_device_name(device: &str) -> bool {
    device == "CUDA_Host" || device.ends_with("_Host")
}

/// Backend device a buffer-size line belongs to, from the buffer name token
/// that precedes `KV buffer size` / `compute buffer size` (e.g. `CUDA0`,
/// `CUDA1`, `CUDA_Host`, `Metal`, `CPU`). `None` when the device cannot be
/// determined.
fn buffer_size_device(line: &str) -> Option<String> {
    let marker = if line.contains("KV buffer size") {
        "KV buffer size"
    } else if line.contains("compute buffer size") {
        "compute buffer size"
    } else {
        return None;
    };
    let idx = line.find(marker)?;
    let name = line[..idx].trim();
    name.rsplit(' ').next().map(str::to_string)
}

fn buffer_size_params(line: &str) -> Vec<(String, Value)> {
    // Structured facts for memory-planning telemetry: the measured buffer size
    // (the number llama.cpp actually allocated) plus the device the line names.
    // These are the inputs the topology planner will consume in place of its
    // KV-scaled compute-buffer estimate.
    let mut params = Vec::new();
    if let Some(mib) = parse_buffer_size_mib(line) {
        params.push(("buffer_mib".to_string(), Value::from(mib)));
    }
    if let Some(device) = buffer_size_device(line) {
        params.push(("backend_device".to_string(), Value::String(device)));
    }
    params
}

fn summarize_native_log_line(line: &str) -> Option<NativeLogEvent> {
    if let Some((category, params)) = cpu_offload_diagnostic_params(line) {
        return Some(NativeLogEvent {
            message: line.to_string(),
            category,
            params,
        });
    }

    let lower = line.to_ascii_lowercase();
    if line.contains("backend_init")
        || line.contains("llama_backend_init")
        || line.contains("GGML_CUDA")
        || line.contains("GGML_HIP")
        || line.contains("GGML_ROCM")
        || ((lower.contains("cuda")
            || lower.contains("hip")
            || lower.contains("rocm")
            || lower.contains("metal"))
            && (lower.contains("init") || lower.contains("device") || lower.contains("backend")))
    {
        return Some(NativeLogEvent {
            message: line.to_string(),
            category: "backend",
            params: Vec::new(),
        });
    }

    if line.contains(".gguf loaded")
        || line.starts_with("llm_load_print_meta")
        || line.starts_with("llm_load_tensors")
        || (line.contains("loading model") && !line.contains("clip_model"))
        || (line.contains("loaded model") && !line.starts_with("llama_model_loader:"))
    {
        return Some(NativeLogEvent {
            message: line.to_string(),
            category: "model",
            params: Vec::new(),
        });
    }

    if line.starts_with("llama_context: n_ubatch") || line.starts_with("llama_context: flash_attn")
    {
        // Forward the resolved micro-batch size and flash-attention mode so a live
        // deployment can prove which values the runtime actually constructed with.
        // These lines come from the llama_context parameter dump
        // (llama-context.cpp, `n_ubatch = ...` / `flash_attn = ...`).
        return Some(NativeLogEvent {
            message: line.to_string(),
            category: "runtime",
            params: Vec::new(),
        });
    }

    if line.contains("VRAM")
        || line.contains("vram")
        || line.contains("mem_alloc")
        || line.contains("model buffer size")
        || (line.contains("GPU") && line.contains("memory"))
        || line.contains("compute buffer size")
        || line.contains("scratch buffer")
    {
        let params = if line.contains("buffer size") {
            buffer_size_params(line)
        } else {
            Vec::new()
        };
        return Some(NativeLogEvent {
            message: line.to_string(),
            category: "memory",
            params,
        });
    }

    if line.starts_with("llama_kv_cache:")
        && (line.contains("buffer size") || line.contains("size = ") || line.contains("attn_rot"))
    {
        let params = if line.contains("buffer size") {
            buffer_size_params(line)
        } else {
            Vec::new()
        };
        return Some(NativeLogEvent {
            message: line.to_string(),
            category: "kv_cache",
            params,
        });
    }

    if line.starts_with("init_tokenizer:")
        || line.starts_with("load: special tokens cache size")
        || line.starts_with("load: token to piece cache size")
    {
        return Some(NativeLogEvent {
            message: line.to_string(),
            category: "tokenizer",
            params: Vec::new(),
        });
    }

    None
}

fn cpu_offload_diagnostic_params(line: &str) -> Option<(&'static str, Vec<(String, Value)>)> {
    let lower = line.to_ascii_lowercase();
    let (category, surface) = if lower.contains("cpu_mapped model buffer size") {
        ("memory", "model_buffer")
    } else if lower.contains("cpu kv buffer size") {
        ("kv_cache", "kv_buffer")
    } else if lower.contains("cpu compute buffer size") {
        ("memory", "compute_buffer")
    } else {
        return None;
    };
    Some((
        category,
        vec![
            (
                "offload_device".to_string(),
                Value::String("CPU".to_string()),
            ),
            (
                "offload_surface".to_string(),
                Value::String(surface.to_string()),
            ),
        ],
    ))
}

fn parse_loaded_metadata_counts(line: &str) -> Option<(usize, usize)> {
    let (_, remainder) = line.split_once("loaded meta data with ")?;
    let (metadata_rows, remainder) = remainder.split_once(" key-value pairs and ")?;
    let metadata_rows = metadata_rows.trim().parse().ok()?;
    let (tensor_rows, _) = remainder.split_once(" tensors")?;
    let tensor_rows = tensor_rows.trim().parse().ok()?;
    Some((metadata_rows, tensor_rows))
}

fn parse_metadata_kv_line(line: &str) -> Option<(&str, &str)> {
    if !line.starts_with("llama_model_loader:") || !line.contains("- kv") {
        return None;
    }
    let (_, remainder) = line.split_once(": - kv")?;
    let (_, remainder) = remainder.split_once(':')?;
    let remainder = remainder.trim();
    let (lhs, value) = remainder.split_once(" = ")?;
    let key = lhs.split_whitespace().next()?;
    Some((key, value.trim()))
}

fn parse_tensor_type_summary(line: &str) -> Option<(&str, usize)> {
    if !line.starts_with("llama_model_loader:") || !line.contains("- type") {
        return None;
    }
    let (_, remainder) = line.split_once("- type")?;
    let remainder = remainder.trim();
    let (tensor_type, count_and_suffix) = remainder.split_once(':')?;
    let count = count_and_suffix.split_whitespace().next()?.parse().ok()?;
    Some((tensor_type.trim(), count))
}

fn parse_layer_assignment(line: &str) -> Option<(usize, &str)> {
    let remainder = line.strip_prefix("load_tensors: layer")?;
    let (index, device) = remainder.split_once("assigned to device")?;
    let index = index.trim().parse().ok()?;
    let device = device.split(',').next()?.trim();
    (!device.is_empty()).then_some((index, device))
}

fn parse_kv_cache_layers_total(line: &str) -> Option<usize> {
    if !line.starts_with("llama_kv_cache:") || !line.contains(" layers") {
        return None;
    }
    let prefix = line.split_once(" layers")?.0;
    let digits = prefix
        .chars()
        .rev()
        .skip_while(|ch| ch.is_whitespace())
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn parse_kv_cache_layer_index(line: &str) -> Option<usize> {
    if !line.starts_with("llama_kv_cache: layer") {
        return None;
    }
    let (_, remainder) = line.split_once("layer")?;
    let digits = remainder
        .trim_start()
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn sanitize_native_log_note(note: &str) -> String {
    note.chars()
        .map(|ch| if matches!(ch, '\n' | '\r') { ' ' } else { ch })
        .collect()
}

pub fn write_native_log_note(note: impl AsRef<str>) {
    write_native_log_note_with_mask(note, MODEL_CATEGORY);
}

/// Writes a native-log note that remains visible in `auto` mode even when
/// structured model-open events cover ordinary parsed model summaries.
///
/// This is reserved for source-specific compatibility gaps such as the
/// SafeTensors loader, which does not enter the native model-open callback.
pub(crate) fn write_native_log_fallback_note(note: impl AsRef<str>) {
    write_native_log_note_with_mask(note, MODEL_FALLBACK_NOTE);
}

fn write_native_log_note_with_mask(note: impl AsRef<str>, required_mask: u8) {
    let note = sanitize_native_log_note(note.as_ref());
    if let Ok(mut guard) = native_log_file().lock()
        && let Some(writer) = guard.as_mut()
    {
        let _ = writeln!(writer, "mesh-llm: {note}");
        let _ = writer.flush();
    }
    forward_native_log_note(note, required_mask);
}

fn forward_native_log_note(note: String, required_mask: u8) {
    let forwarding_mask = NATIVE_LOG_FORWARDING_MASK.load(Ordering::Relaxed);
    if forwarding_mask & required_mask == 0 {
        return;
    }
    let event = NativeLogEvent {
        message: format!("mesh-llm: {note}"),
        category: "model",
        params: Vec::new(),
    };
    if let Some(sender) = NATIVE_LOG_FILTERED_TX.get()
        && let Ok(guard) = sender.lock()
        && let Some(tx) = guard.as_ref()
    {
        let _ = tx.send(event);
    }
}

/// Enable verbose llama.cpp logging. Call before `llama_backend_init()` / model loading.
/// Sets GGML_LLAMA_LOG_LEVEL=4 so LLAMA_LOG_DEBUG macros produce output.
pub fn enable_verbose_native_logs() {
    // SAFETY: UNSAFE CONTRACT — callers must invoke this before concurrent
    // runtime work can access the process environment. The API does not yet
    // enforce that startup boundary; retain the audit TODO below.
    // TODO: Audit that the environment access only happens in single-threaded code.
    unsafe { std::env::set_var("GGML_LLAMA_LOG_LEVEL", LLAMA_LOG_LEVEL_DEBUG) };
}

/// Disable verbose llama.cpp logging (restore default level).
pub fn disable_verbose_native_logs() {
    // SAFETY: UNSAFE CONTRACT — callers must invoke this before concurrent
    // runtime work can access the process environment. The API does not yet
    // enforce that startup boundary; retain the audit TODO below.
    // TODO: Audit that the environment access only happens in single-threaded code.
    unsafe { std::env::remove_var("GGML_LLAMA_LOG_LEVEL") };
}

unsafe extern "C" fn write_native_log(level: c_int, text: *const c_char, _user_data: *mut c_void) {
    if text.is_null() {
        return;
    }

    let bytes = unsafe { CStr::from_ptr(text) }.to_bytes();
    output::forward_raw_native_log(level, bytes);
    if let Ok(mut guard) = native_log_file().lock()
        && let Some(writer) = guard.as_mut()
    {
        let _ = writer.write_all(bytes);
    }

    // Also send aggregated messages through the channel when runtime forwarding is enabled.
    let forwarding_mask = NATIVE_LOG_FORWARDING_MASK.load(Ordering::Relaxed);
    if forwarding_mask == 0 {
        return;
    }

    if let Ok(text_str) = core::str::from_utf8(bytes) {
        let events = match native_log_aggregator().lock() {
            Ok(mut aggregator) => aggregator.process_line(text_str.trim()),
            _ => Vec::new(),
        };
        if let Some(tx) = NATIVE_LOG_FILTERED_TX.get()
            && let Ok(guard) = tx.lock()
            && let Some(ref sender) = *guard
        {
            for event in events {
                if category_mask(event.category).is_some_and(|mask| forwarding_mask & mask != 0) {
                    let _ = sender.send(event);
                }
            }
        }
    }
}

unsafe extern "C" fn discard_native_log(
    _level: c_int,
    _text: *const c_char,
    _user_data: *mut c_void,
) {
}

#[cfg(test)]
#[path = "logging/tests/mod.rs"]
mod tests;
