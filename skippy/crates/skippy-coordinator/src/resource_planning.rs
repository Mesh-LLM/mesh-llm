use skippy_model_artifact::gguf::{GgufCompactMeta, GgufKvCacheQuant};

const DEFAULT_CONTEXT_LENGTH: u32 = 4096;
const DEFAULT_PARALLEL_SLOTS: usize = 4;
const MIN_AUTO_CONTEXT_LENGTH: u32 = 512;
/// Auto-planner ceiling on concurrent lanes.
///
/// Matches upstream llama-server: when `--parallel` is left to auto,
/// llama-server picks `n_parallel = 4` and turns on `kv_unified = true`
/// (see `tools/server/server.cpp`,
/// `"n_parallel is set to auto, using n_parallel = 4 and kv_unified = true"`).
///
/// With the default `kv_unified = auto`, Skippy enables a shared KV pool when
/// more than one lane is selected. It allocates `context_length × lane_count`
/// cells so every lane can reach its requested context concurrently. The
/// ceiling of four limits default concurrency independently of pool cost.
///
/// Concrete failure mode that prompted this change: Qwen3-8B on a
/// 32k `n_ctx` got `slots = 16`. Three concurrent agent-shape
/// requests (~14k tokens each — OpenCode system prompt plus tools
/// plus a tool-result follow-up) need ~45k cells in the shared 32k
/// pool; llama's `find_slot` fails on the third request and skippy
/// surfaces it as an HTTP 502 with body `RuntimeError: llama_decode failed`.
///
/// 4 is the same conservative ceiling llama-server uses for the
/// same `kv_unified = true` reason. Operators who know their
/// workload (e.g. all short chat turns, or a single-user MoA host)
/// can still go higher via `parallel_override` /
/// `[models.throughput] parallel = N` in the TOML config.
const MAX_AUTO_PARALLEL_SLOTS: usize = 4;
/// Default ceiling on auto-planned context length (128k).
///
/// Some published GGUFs advertise a native window far larger than is useful on
/// a mesh — e.g. the Nemotron family ships 1,048,576-token artifacts. Left
/// unclamped, the auto-planner would try to drive a 1M context, spend the whole
/// KV budget on depth, and starve the parallel lanes that agentic serving needs
/// (or shrink context per-lane below what an agent can use). 128k is the
/// agent-serving sweet spot: deep enough for real tool loops and replay
/// corpora, shallow enough to keep multiple lanes and usable decode throughput.
///
/// This clamp is a `min`, so native windows at or below 128k keep their full
/// native size. It is a *default* only: an explicit `--ctx-size` /
/// `[models] ctx_size` override bypasses planning entirely and can still request
/// the full native window.
///
/// Deepening the default past 128k (toward 256k) is deliberately **not** done
/// here: it is memory-bandwidth-bound, not capacity-bound, and picking the
/// depth safely needs a populated-KV tok/s calibration we do not have yet. That
/// work is tracked as a follow-up (bandwidth-aware context/lane planning).
const MAX_AUTO_CONTEXT_LENGTH: u32 = 131_072;
const KV_CACHE_BUDGET_NUMERATOR: u64 = 85;
const KV_CACHE_BUDGET_DENOMINATOR: u64 = 100;
const FALLBACK_CONTEXT_8K_FREE_BYTES: u64 = 3_000_000_000;
const FALLBACK_CONTEXT_16K_FREE_BYTES: u64 = 6_000_000_000;
const FALLBACK_CONTEXT_32K_FREE_BYTES: u64 = 12_000_000_000;
const FALLBACK_CONTEXT_64K_FREE_BYTES: u64 = 30_000_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeResourcePlanningProfile {
    /// A dedicated-local launch (no shared mesh-serving surface requested).
    DedicatedLocal,
    /// A shared mesh-serving launch (`--auto` / `--publish` / `--discover` /
    /// `--join`).
    ///
    /// Both profiles currently reserve the same per-lane context in one shared
    /// pool. The distinction is retained for the bandwidth-aware planner.
    SharedMesh,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeResourcePlan {
    pub context_length: u32,
    pub slots: usize,
    /// Structured breakdown of the inputs and intermediate results the planner
    /// used, emitted once per model start so every later memory-planning change
    /// is measurable in production logs. `None` only for plans built outside
    /// [`plan_runtime_resources`].
    pub breakdown: Option<RuntimeResourcePlanBreakdown>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeResourcePlanSource {
    ExplicitOverride,
    StaticEstimate,
    MeasuredFootprint,
}

impl RuntimeResourcePlanSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitOverride => "explicit_override",
            Self::StaticEstimate => "static_estimate",
            Self::MeasuredFootprint => "measured_footprint",
        }
    }
}

/// The accounting selected at plan time. Static plans carry metadata-derived
/// estimates; measured plans carry the prior compatible load's native KV rate
/// and lane-scaled compute charge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeResourcePlanBreakdown {
    pub vram_bytes: u64,
    pub model_bytes: u64,
    /// Projector bytes charged to the same pool as the weights (`0` when the
    /// model has no projector). Reported separately from `model_bytes` so a
    /// fit can be read back from production logs.
    pub projector_bytes: u64,
    /// KV budget selected by the active planner. Static planning applies the
    /// 85% utilization tax; measured planning applies its utilization target
    /// and then subtracts the lane-scaled measured compute charge.
    pub kv_budget_bytes: u64,
    /// Planned unified KV allocation: context_length x slots x kv_bytes_per_token.
    pub planned_kv_bytes: u64,
    /// Per-token KV cost used by the plan (layer-fraction scaled).
    pub kv_bytes_per_token: u64,
    /// Compute charge held outside `kv_budget_bytes` by the selected planner.
    pub compute_charge_bytes: u64,
    pub planning_source: RuntimeResourcePlanSource,
    /// `Some(false)` means a valid measured footprint proved that even the
    /// minimum context cannot fit. Static and explicit plans use `None`.
    pub measured_fit: Option<bool>,
    pub slots: usize,
    pub context_length: u32,
    /// True when slots came from the flat auto default rather than an
    /// explicit override.
    pub slots_auto: bool,
    /// True when the context length came from planning rather than an
    /// explicit override.
    pub context_auto: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct RuntimeResourcePlanInput<'a> {
    pub ctx_size_override: Option<u32>,
    pub parallel_override: Option<usize>,
    /// Model weight bytes **local to this node**.  For a split/layer-package
    /// load, pass only this node's share of the model weights.
    pub model_bytes: u64,
    /// Bytes the multimodal projector will hold in the same device pool as the
    /// model weights. `0` when the model has no projector.
    ///
    /// The projector is loaded *after* the text model on the direct `--gguf`
    /// path, and nothing charged it to the fit before this field existed: a
    /// plan that spent the last of the pool on weights and KV left the
    /// projector's allocation to fail, and upstream `mtmd` dereferences the
    /// NULL buffer instead of returning an error (mesh-llm#1166). Charging it
    /// here reserves the room, which is what upstream `llama-server` does with
    /// `mtmd_get_memory_usage` charged into its fit target.
    pub projector_bytes: u64,
    pub vram_bytes: u64,
    pub metadata: Option<&'a GgufCompactMeta>,
    /// The KV cache quant that will be used. The default is F16.
    /// Only differs when the user explicitly selects another cache type.
    pub kv_cache_quant: GgufKvCacheQuant,
    /// Fraction of the model's layers that reside on this node (0.0–1.0).
    /// `None` means the whole model is local (fraction = 1.0).
    pub local_layer_fraction: Option<f64>,
    pub planning_profile: RuntimeResourcePlanningProfile,
    /// Measured native buffer footprint from a prior context init of the same
    /// shape on this node (compute + KV buffer sizes and the context length
    /// they were measured at). When present, the planner charges these
    /// measured sizes instead of the 85% KV tax (budget-driven sizing); when
    /// absent it falls back to the static tax ladder.
    pub measured_buffers: Option<MeasuredBufferFootprint>,
}

/// Measured native buffer sizes from one context init, the ground truth the
/// budget-driven planner charges in place of the KV-scaled tax.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeasuredBufferFootprint {
    /// Compute-graph buffer(s) at context init, bytes.
    pub compute_bytes: u64,
    /// Total KV buffer for `lane_count` lanes at `context_length` per lane, bytes.
    pub kv_bytes: u64,
    /// Per-lane context length the KV buffer was measured at.
    pub context_length: u32,
    /// Realized lane count the buffers were measured at, rather than a requested
    /// count that the native workload may have clamped. Omit the footprint when
    /// that allocation cannot be established. Compute buffers scale
    /// ~linearly with lanes (measured 399/783/1551 MiB at 2/4/8 lanes on the
    /// 5080 for granite), so a plan resolving a different lane count scales
    /// the compute charge by the lane ratio.
    pub lane_count: u32,
}

/// Plan context length and parallel slots.
///
/// Strategy: maximise context up to `min(native, MAX_AUTO_CONTEXT_LENGTH)` (the
/// 128k agent-serving default ceiling) using the provided KV quant (default
/// F16) and resolved lane count. No negotiation —
/// the quant is decided upstream, and an explicit `--ctx-size` override
/// bypasses context planning entirely.
pub fn plan_runtime_resources(input: RuntimeResourcePlanInput<'_>) -> RuntimeResourcePlan {
    let context_auto = input.ctx_size_override.is_none();
    let slots_auto = input.parallel_override.is_none();
    let slots = input
        .parallel_override
        .unwrap_or_else(planned_parallel_slots)
        .max(1);
    let estimated_kv_bytes_per_token = input
        .metadata
        .and_then(|metadata| {
            input
                .kv_cache_quant
                .kv_cache_bytes_per_token(metadata)
                .map(|bytes| scale_by_layer_fraction(bytes, &input))
        })
        .unwrap_or(0);
    let estimated_kv_budget =
        usable_kv_cache_budget(input.vram_bytes, input.model_bytes, input.projector_bytes);
    let estimated_compute_charge =
        free_bytes_after_weights(input.vram_bytes, input.model_bytes, input.projector_bytes)
            .saturating_sub(estimated_kv_budget);

    let (
        context_length,
        kv_budget_bytes,
        kv_bytes_per_token,
        compute_charge_bytes,
        planning_source,
        measured_fit,
    ) = if let Some(context_length) = input.ctx_size_override {
        (
            context_length,
            estimated_kv_budget,
            estimated_kv_bytes_per_token,
            estimated_compute_charge,
            RuntimeResourcePlanSource::ExplicitOverride,
            None,
        )
    } else if let Some(measured) = measured_context_plan(&input, slots) {
        (
            measured.context_length,
            measured.kv_budget_bytes,
            measured.kv_bytes_per_token,
            measured.compute_charge_bytes,
            RuntimeResourcePlanSource::MeasuredFootprint,
            Some(measured.fits),
        )
    } else {
        (
            planned_context_length(&input, slots),
            estimated_kv_budget,
            estimated_kv_bytes_per_token,
            estimated_compute_charge,
            RuntimeResourcePlanSource::StaticEstimate,
            None,
        )
    };
    let planned_kv_bytes = kv_bytes_per_token
        .saturating_mul(u64::from(context_length))
        .saturating_mul(slots as u64);

    RuntimeResourcePlan {
        context_length,
        slots,
        breakdown: Some(RuntimeResourcePlanBreakdown {
            vram_bytes: input.vram_bytes,
            model_bytes: input.model_bytes,
            projector_bytes: input.projector_bytes,
            kv_budget_bytes,
            planned_kv_bytes,
            kv_bytes_per_token,
            compute_charge_bytes,
            planning_source,
            measured_fit,
            slots,
            context_length,
            slots_auto,
            context_auto,
        }),
    }
}

fn planned_context_length(input: &RuntimeResourcePlanInput<'_>, slots: usize) -> u32 {
    let fallback_context = fallback_context_length(input);
    let Some(metadata) = input.metadata else {
        return fallback_context;
    };
    let native_context = metadata.context_length;
    if native_context == 0 {
        return fallback_context;
    }
    // Clamp the *native* window (read per-artifact from this GGUF's header, not
    // from a model-name lookup) to the default auto-context ceiling before KV
    // planning. Keeps 1M-token natives from over-committing KV to a single very
    // deep context while leaving smaller native windows untouched.
    let native_context = native_context.min(MAX_AUTO_CONTEXT_LENGTH);
    let Some(kv_bytes_per_token_full) = input.kv_cache_quant.kv_cache_bytes_per_token(metadata)
    else {
        return fallback_context.min(native_context);
    };

    // In a pipeline-parallel split each stage only holds KV state for its
    // own layers.  Scale the per-token cost by the local layer fraction.
    let kv_bytes_per_token = scale_by_layer_fraction(kv_bytes_per_token_full, input);

    let kv_budget =
        usable_kv_cache_budget(input.vram_bytes, input.model_bytes, input.projector_bytes);
    if kv_bytes_per_token == 0 {
        return native_context;
    }
    let Some(kv_bytes_for_target_slots) = kv_bytes_per_token.checked_mul(slots as u64) else {
        return MIN_AUTO_CONTEXT_LENGTH.min(native_context);
    };
    let max_affordable_context = kv_budget / kv_bytes_for_target_slots;
    if max_affordable_context == 0 {
        return MIN_AUTO_CONTEXT_LENGTH.min(native_context);
    }

    let planned = max_affordable_context
        .min(u64::from(native_context))
        .min(u64::from(u32::MAX)) as u32;
    let minimum = MIN_AUTO_CONTEXT_LENGTH.min(native_context);
    if planned < minimum {
        minimum
    } else {
        snap_context_length_down(planned).max(minimum)
    }
}

/// Plan the number of concurrent lanes to run at the chosen context depth.
///
/// Auto uses four lanes; context depth is subsequently sized against the
/// resulting unified pool. Operators can override the lane count.
///
/// Recurrent/SSM layers do keep per-lane state; that per-lane cost is accounted
/// for by the split topology planner and is bounded here by the 4-lane cap. A
/// finer single-node recurrent-aware bound is left to the bandwidth-aware
/// planner follow-up.
fn planned_parallel_slots() -> usize {
    // llama-server's conservative unified-KV auto default, never above our
    // safety cap. Context planning accounts for the resulting pool size.
    DEFAULT_PARALLEL_SLOTS.min(MAX_AUTO_PARALLEL_SLOTS)
}

fn scale_by_layer_fraction(kv_bytes_per_token: u64, input: &RuntimeResourcePlanInput<'_>) -> u64 {
    let fraction = input.local_layer_fraction.unwrap_or(1.0).clamp(0.0, 1.0);
    if fraction < 1.0 && fraction > 0.0 {
        ((kv_bytes_per_token as f64) * fraction).ceil() as u64
    } else {
        kv_bytes_per_token
    }
}

/// Device bytes left once the resident weights are accounted for: the model
/// weights and the multimodal projector share one pool, and both are committed
/// before any KV is allocated.
fn free_bytes_after_weights(vram_bytes: u64, model_bytes: u64, projector_bytes: u64) -> u64 {
    vram_bytes
        .saturating_sub(model_bytes)
        .saturating_sub(projector_bytes)
}

fn usable_kv_cache_budget(vram_bytes: u64, model_bytes: u64, projector_bytes: u64) -> u64 {
    let free_bytes = free_bytes_after_weights(vram_bytes, model_bytes, projector_bytes);
    let budget = u128::from(free_bytes) * u128::from(KV_CACHE_BUDGET_NUMERATOR)
        / u128::from(KV_CACHE_BUDGET_DENOMINATOR);
    budget.min(u128::from(u64::MAX)) as u64
}

fn fallback_context_length(input: &RuntimeResourcePlanInput<'_>) -> u32 {
    let free_bytes =
        free_bytes_after_weights(input.vram_bytes, input.model_bytes, input.projector_bytes);
    if free_bytes >= FALLBACK_CONTEXT_64K_FREE_BYTES {
        65_536
    } else if free_bytes >= FALLBACK_CONTEXT_32K_FREE_BYTES {
        32_768
    } else if free_bytes >= FALLBACK_CONTEXT_16K_FREE_BYTES {
        16_384
    } else if free_bytes >= FALLBACK_CONTEXT_8K_FREE_BYTES {
        8192
    } else {
        DEFAULT_CONTEXT_LENGTH
    }
}

fn snap_context_length_down(value: u32) -> u32 {
    const CONTEXT_STEPS: &[u32] = &[512, 1024, 2048, 4096, 8192, 16_384, 32_768, 65_536, 131_072];
    CONTEXT_STEPS
        .iter()
        .rev()
        .copied()
        .find(|step| *step <= value)
        .unwrap_or(value)
}

/// Fraction of usable memory the budget-driven planner targets (Step 2).
///
/// vLLM reserves ~8% of free memory after weights (`gpu_memory_utilization`
/// 0.92), SGLang ~10% (`mem-fraction-static` 0.9). Ours is deliberately a
/// little more conservative: mesh nodes can co-host other stages, and Metal
/// unified-memory nodes share the pool with the OS (where an over-commit
/// page-out stalls decode rather than failing loudly like a CUDA OOM).
const DEFAULT_UTILIZATION_TARGET_NUMERATOR: u64 = 88;
const DEFAULT_UTILIZATION_TARGET_DENOMINATOR: u64 = 100;

/// Budget-driven context planning (Step 2) over the measured footprint from a
/// prior context init.
///
/// Model: `budget = (vram - model) × utilization - measured_compute`, then
/// solve for the deepest context whose *scaled* KV cost fits the budget. KV
/// scales linearly with per-lane context (unified pool of `n_ctx` cells), so the
/// total measured KV bytes give `kv_bytes_per_token_measured =
/// kv_bytes / (measured_ctx × measured_lanes)`, and the deepest affordable context is
/// `budget / (kv_bytes_per_token_measured × resolved_lanes)`.
///
/// Returns `None` only when the measurement is structurally unusable (zero
/// context, KV, or lanes), letting the static tax ladder answer instead. A
/// valid measurement that cannot fit the minimum context returns an explicit
/// `fits = false` plan so the caller fails closed. Compute buffers do not scale
/// linearly with context (ubatch/graph shape dominates), so the measured value
/// is charged as-is apart from the measured-to-requested lane ratio.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MeasuredContextPlan {
    context_length: u32,
    kv_budget_bytes: u64,
    kv_bytes_per_token: u64,
    compute_charge_bytes: u64,
    fits: bool,
}

fn measured_context_plan(
    input: &RuntimeResourcePlanInput<'_>,
    resolved_lanes: usize,
) -> Option<MeasuredContextPlan> {
    let measured = input.measured_buffers?;
    if measured.context_length == 0 || measured.kv_bytes == 0 || measured.lane_count == 0 {
        return None;
    }
    let metadata = input.metadata?;
    let native_context = metadata.context_length;
    if native_context == 0 {
        return None;
    }
    let native_context = native_context.min(MAX_AUTO_CONTEXT_LENGTH);

    let post_weight =
        free_bytes_after_weights(input.vram_bytes, input.model_bytes, input.projector_bytes);
    let utilised = u128::from(post_weight) * u128::from(DEFAULT_UTILIZATION_TARGET_NUMERATOR)
        / u128::from(DEFAULT_UTILIZATION_TARGET_DENOMINATOR);
    // Scale the measured compute charge by the lane ratio when the plan's
    // resolved lane count differs from the one the footprint was measured at:
    // compute buffers scale ~linearly with lanes (CUDA_Host + device graphs),
    // and the unified KV pool scales with its reserved lane capacity.
    // Linear-through-origin
    // slightly overestimates when scaling up (~2-3% at 4→8 on measured data),
    // which is the conservative direction; scaling down is symmetric.
    let lane_ratio_num = u128::from(resolved_lanes.max(1) as u64);
    let lane_ratio_den = u128::from(measured.lane_count);
    let compute_charge = u128::from(measured.compute_bytes) * lane_ratio_num / lane_ratio_den;
    let budget = utilised.saturating_sub(compute_charge);
    let measured_cells = u128::from(measured.context_length) * u128::from(measured.lane_count);
    let kv_per_token = u128::from(measured.kv_bytes).div_ceil(measured_cells);
    if kv_per_token == 0 {
        return None;
    }
    let max_affordable = (budget / kv_per_token / u128::from(resolved_lanes.max(1) as u64))
        .min(u128::from(u32::MAX)) as u32;
    let minimum = MIN_AUTO_CONTEXT_LENGTH.min(native_context);
    let fits = max_affordable >= minimum;
    let context_length = if fits {
        snap_context_length_down(max_affordable.min(native_context)).max(minimum)
    } else {
        minimum
    };
    Some(MeasuredContextPlan {
        context_length,
        kv_budget_bytes: budget.min(u128::from(u64::MAX)) as u64,
        kv_bytes_per_token: kv_per_token.min(u128::from(u64::MAX)) as u64,
        compute_charge_bytes: compute_charge.min(u128::from(u64::MAX)) as u64,
        fits,
    })
}

#[cfg(test)]
mod tests;
