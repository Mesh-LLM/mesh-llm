//! Local serving of Laya decision models.
//!
//! Laya opens through its own native entry point, not the llama loader, and
//! has no KV cache, sessions, or stages. What it does hold beyond its weights
//! is per-read activation memory — dense attention masks and scores over one
//! packed pass — so that worst case is charged to the same capacity ledger
//! and memory-plan record as every other local load.

use anyhow::{Context, Result};
use mesh_llm_events::{OutputEvent, emit_event};

use super::capacity::runtime_model_required_bytes;
use super::context_planning::{RuntimeResourcePlanBreakdown, RuntimeResourcePlanSource};
use super::local::{
    LocalOpenAiModelStartSpec, LocalRuntimeBackendHandle, LocalRuntimeModelHandle, ModelLoadSource,
    report_model_loaded_analytics,
};
use super::local_memory_plan::{MemoryPlanStartPath, emit_memory_plan_resolved};
use super::split_planning::format_gb;
use crate::inference::skippy;
use crate::mesh;
use crate::models;

/// `general.architecture` of a Laya decision model GGUF.
pub(super) const LAYA_ARCHITECTURE: &str = "laya";

/// Opts the Laya runtime into the node's accelerator. Laya runs on the CPU
/// backend by default: a 322M encoder answers in tens of milliseconds there,
/// and keeping it off the accelerator leaves that memory to the models the
/// split planner accounts for.
const LAYA_ACCELERATOR_ENV: &str = "MESH_LLM_LAYA_ACCELERATOR";

/// Default `laya.max_len` when a GGUF omits it, matching the native runtime.
const DEFAULT_MAX_LEN: u64 = 1024;
/// Native ceiling on `laya.max_len` (`SKIPPY_LAYA_MAX_SEQUENCE_TOKENS`).
const MAX_SEQUENCE_TOKENS: u64 = 4096;
const DEFAULT_EMBEDDING: u64 = 768;
const DEFAULT_HEAD_DIM: u64 = 64;

pub(super) fn is_laya(meta: Option<&models::gguf::GgufCompactMeta>) -> bool {
    meta.is_some_and(|meta| meta.architecture == LAYA_ARCHITECTURE)
}

/// Worst-case bytes one Laya read holds beyond its weights.
///
/// The native runtime packs sequences into passes of at most `max_len` tokens
/// (`P`) and, per pass, allocates:
/// - two dense f32 attention masks on the host and again in the backend
///   buffer: `4 · 4 · P²`;
/// - one layer's attention scores and their softmax, live together:
///   `2 · 4 · H · P²` for `H` heads;
/// - hidden-state and feed-forward activations: about `16 · 4 · P · E` for
///   embedding width `E`.
pub(super) fn laya_activation_reserve_bytes(meta: &models::gguf::GgufCompactMeta) -> u64 {
    let pass = match u64::from(meta.max_len) {
        0 => DEFAULT_MAX_LEN,
        max_len => max_len.min(MAX_SEQUENCE_TOKENS),
    };
    let embedding = match u64::from(meta.embedding_size) {
        0 => DEFAULT_EMBEDDING,
        embedding => embedding,
    };
    let heads = match u64::from(meta.head_count) {
        0 => (embedding / DEFAULT_HEAD_DIM).max(1),
        heads => heads,
    };
    let square = pass * pass;
    16 * square + 8 * heads * square + 64 * pass * embedding
}

/// Bytes a Laya GGUF of `weight_bytes` occupies once serving: its weights plus
/// the worst-case read reserve.
pub(super) fn laya_resident_bytes(weight_bytes: u64, meta: &models::gguf::GgufCompactMeta) -> u64 {
    weight_bytes.saturating_add(laya_activation_reserve_bytes(meta))
}

fn accelerator_requested(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

/// Loads a Laya decision model. It answers System One reads only and
/// advertises the decision workload class.
pub(super) async fn start_local_laya_model(
    spec: LocalOpenAiModelStartSpec<'_>,
    model_name: String,
    weight_bytes: u64,
    compact_meta: &models::gguf::GgufCompactMeta,
) -> Result<(
    String,
    LocalRuntimeModelHandle,
    tokio::sync::oneshot::Receiver<()>,
)> {
    let reserve = laya_activation_reserve_bytes(compact_meta);
    let resident = laya_resident_bytes(weight_bytes, compact_meta);
    let required = runtime_model_required_bytes(resident);
    anyhow::ensure!(
        spec.capacity_budget_bytes >= required,
        "Laya model requires {} (weights {} plus {} for its largest read), local capacity is {}",
        format_gb(required),
        format_gb(weight_bytes),
        format_gb(reserve),
        format_gb(spec.capacity_budget_bytes)
    );
    let max_len = match compact_meta.max_len {
        0 => DEFAULT_MAX_LEN as u32,
        max_len => max_len,
    };
    emit_memory_plan_resolved(
        &model_name,
        Some(&RuntimeResourcePlanBreakdown {
            vram_bytes: spec.capacity_budget_bytes,
            model_bytes: weight_bytes,
            projector_bytes: 0,
            kv_budget_bytes: 0,
            planned_kv_bytes: 0,
            kv_bytes_per_token: 0,
            compute_charge_bytes: reserve,
            planning_source: RuntimeResourcePlanSource::StaticEstimate,
            measured_fit: None,
            slots: 1,
            context_length: max_len,
            slots_auto: false,
            context_auto: false,
        }),
        // A direct GGUF load; the Laya reserve is the plan's compute charge,
        // and the load line below names the backend.
        MemoryPlanStartPath::Direct,
    );

    let use_accelerator =
        accelerator_requested(std::env::var(LAYA_ACCELERATOR_ENV).ok().as_deref());
    let _ = emit_event(OutputEvent::ModelLoading {
        model: model_name.clone(),
        source: None,
    });
    let model_path = spec.model_path.to_path_buf();
    let model =
        tokio::task::spawn_blocking(move || skippy::load_laya_model(&model_path, use_accelerator))
            .await
            .context("join load Laya model task")??;
    tracing::info!(
        model = model_name,
        laya.accelerator = use_accelerator,
        laya.max_len = model.info().max_len,
        laya.reserve_bytes = reserve,
        "Laya decision model loaded"
    );
    let _ = emit_event(OutputEvent::ModelLoaded {
        model: model_name.clone(),
        bytes: None,
    });
    report_model_loaded_analytics(&model_name, ModelLoadSource::DirectGguf, Some(compact_meta));
    let context_length = u32::try_from(model.info().max_len).unwrap_or(u32::MAX);
    let capabilities = models::runtime_verified_model_capabilities(
        &model_name,
        spec.model_path,
        models::runtime_media_capability_evidence(None).await,
    );
    let http = skippy::start_laya_http_on(&model_name, model.clone(), spec.http_bind_addr);
    let (death_tx, death_rx) = tokio::sync::oneshot::channel();
    Ok((
        model_name,
        LocalRuntimeModelHandle {
            port: http.port(),
            backend: "skippy".into(),
            context_length,
            // The native runtime serializes reads on one lane per model.
            slots: 1,
            capabilities,
            workload_class: mesh::ModelWorkloadClass::Decision,
            inner: LocalRuntimeBackendHandle::Laya {
                _model: model,
                http,
                _death_tx: death_tx,
            },
        },
        death_rx,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(max_len: u32, embedding: u32, heads: u32) -> models::gguf::GgufCompactMeta {
        models::gguf::GgufCompactMeta {
            architecture: LAYA_ARCHITECTURE.into(),
            max_len,
            embedding_size: embedding,
            head_count: heads,
            ..Default::default()
        }
    }

    #[test]
    fn reserve_for_laya_multilingual_is_about_170_mb() {
        // mmBERT-base: max_len 1024, width 768, 12 heads.
        let reserve = laya_activation_reserve_bytes(&meta(1024, 768, 12));
        assert_eq!(
            reserve,
            16 * 1024 * 1024 + 8 * 12 * 1024 * 1024 + 64 * 1024 * 768
        );
        assert!((150_000_000..200_000_000).contains(&reserve), "{reserve}");
    }

    #[test]
    fn reserve_is_capped_at_the_native_sequence_ceiling() {
        assert_eq!(
            laya_activation_reserve_bytes(&meta(1_000_000, 768, 12)),
            laya_activation_reserve_bytes(&meta(4096, 768, 12))
        );
    }

    #[test]
    fn missing_metadata_falls_back_to_the_runtime_defaults() {
        assert_eq!(
            laya_activation_reserve_bytes(&meta(0, 0, 0)),
            laya_activation_reserve_bytes(&meta(1024, 768, 12))
        );
    }

    #[test]
    fn the_accelerator_is_opt_in() {
        assert!(!accelerator_requested(None));
        assert!(!accelerator_requested(Some("0")));
        assert!(!accelerator_requested(Some("")));
        assert!(accelerator_requested(Some("1")));
        assert!(accelerator_requested(Some(" TRUE ")));
    }
}
