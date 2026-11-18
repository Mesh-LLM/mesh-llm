//! One more load with a single lane when the native context of auto-planned
//! lanes cannot be created.
//!
//! The memory plan sizes the shared KV pool from the memory the device reports
//! free. On an integrated GPU that memory is the system's own, shared with
//! everything else running, so the pool can still fail to allocate: Qwen3-8B
//! Q4_K_M on a Radeon 780M (Vulkan, 16 GB laptop) plans four 4096-token lanes,
//! and Vulkan refuses the 1 GiB KV block with `ErrorOutOfDeviceMemory`. The
//! model then never loads, while a single lane of the same depth does.

use anyhow::Result;
use mesh_llm_events::{OutputEvent, emit_event};

use super::{LocalOpenAiModelStartSpec, load_direct_skippy_model_with_dflash_fallback};
use crate::inference::skippy;
use crate::runtime::context_planning::RuntimeResourcePlan;

/// What the native runtime reports when `llama_init_from_model` returns no
/// context, which is how a KV or compute buffer allocation failure surfaces.
const CONTEXT_CREATION_FAILED: &str = "failed to create llama context";

/// Loads as `resolved` plans it, and once more with one lane, at the same
/// context length, when the context of auto-planned lanes cannot be created.
/// `plan` follows the lanes the model was loaded with.
pub(super) async fn load_direct_skippy_model_with_lane_fallback(
    spec: &LocalOpenAiModelStartSpec<'_>,
    resolved: &mut skippy::ResolvedSkippyConfig,
    plan: &mut RuntimeResourcePlan,
    model_name: &str,
    progress_ingress: Option<crate::runtime_events::engine::ScopedIngress>,
) -> Result<skippy::SkippyModelHandle> {
    let error = match load_direct_skippy_model_with_dflash_fallback(
        spec,
        resolved,
        model_name,
        progress_ingress.clone(),
    )
    .await
    {
        Err(error) if retries_with_one_lane(plan, &error) => error,
        loaded => return loaded,
    };
    tracing::warn!(
        model = model_name,
        lanes = plan.slots,
        context = plan.context_length,
        "retrying the load with one lane: {error:#}"
    );
    let _ = emit_event(OutputEvent::Warning {
        message: fallback_warning(model_name, plan),
        context: None,
    });
    resolved.throughput.parallel = 1;
    use_one_lane(plan);
    load_direct_skippy_model_with_dflash_fallback(spec, resolved, model_name, progress_ingress)
        .await
}

/// A context creation failure on more than one lane that the planner chose,
/// not the operator: a lane count the operator set is kept as asked.
fn retries_with_one_lane(plan: &RuntimeResourcePlan, error: &anyhow::Error) -> bool {
    plan.slots > 1
        && plan
            .breakdown
            .as_ref()
            .is_some_and(|breakdown| breakdown.slots_auto)
        && skippy_runtime::NativeError::find(error).is_some_and(|native| {
            native.status == skippy_runtime::Status::RuntimeError
                && native.message == CONTEXT_CREATION_FAILED
        })
}

/// What the operator sees, said before the retry and limited to what the
/// runtime knows: the context failed, which is most often device memory.
fn fallback_warning(model_name: &str, plan: &RuntimeResourcePlan) -> String {
    format!(
        "{model_name}: could not create the context for {} parallel requests of {} tokens; retrying with one lane (set --parallel to choose)",
        plan.slots, plan.context_length,
    )
}

fn use_one_lane(plan: &mut RuntimeResourcePlan) {
    let lanes = plan.slots as u64;
    plan.slots = 1;
    if let Some(breakdown) = plan.breakdown.as_mut() {
        breakdown.slots = 1;
        breakdown.planned_kv_bytes /= lanes.max(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::context_planning::{
        RuntimeResourcePlanBreakdown, RuntimeResourcePlanSource,
    };

    fn plan(slots: usize, slots_auto: bool) -> RuntimeResourcePlan {
        RuntimeResourcePlan {
            context_length: 4096,
            slots,
            breakdown: Some(RuntimeResourcePlanBreakdown {
                vram_bytes: 8_062_000_000,
                model_bytes: 5_027_783_488,
                projector_bytes: 0,
                speculative_bytes: 0,
                kv_budget_bytes: 2_579_000_000,
                planned_kv_bytes: 147_456 * 4096 * slots as u64,
                kv_bytes_per_token: 147_456,
                compute_charge_bytes: 455_000_000,
                planning_source: RuntimeResourcePlanSource::StaticEstimate,
                measured_fit: None,
                slots,
                context_length: 4096,
                slots_auto,
                context_auto: true,
            }),
        }
    }

    fn native(status: skippy_runtime::Status, message: &str) -> anyhow::Error {
        anyhow::Error::new(skippy_runtime::NativeError {
            status,
            message: message.to_string(),
        })
        .context("load Skippy runtime for model qwen3-8b")
    }

    #[test]
    fn a_context_that_auto_planned_lanes_cannot_create_retries_with_one_lane() {
        let error = native(
            skippy_runtime::Status::RuntimeError,
            CONTEXT_CREATION_FAILED,
        );
        assert!(retries_with_one_lane(&plan(4, true), &error));
    }

    #[test]
    fn a_failure_carrying_its_native_log_lines_still_retries_with_one_lane() {
        // The shape `with_last_native_error` gives an open failure when native
        // logging captured the cause.
        let error =
            anyhow::anyhow!("ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory")
                .context(skippy_runtime::NativeError {
                    status: skippy_runtime::Status::RuntimeError,
                    message: CONTEXT_CREATION_FAILED.to_string(),
                })
                .context("load Skippy runtime for model qwen3-8b");
        assert!(retries_with_one_lane(&plan(4, true), &error));
    }

    #[test]
    fn a_lane_count_the_operator_set_is_kept() {
        let error = native(
            skippy_runtime::Status::RuntimeError,
            CONTEXT_CREATION_FAILED,
        );
        assert!(!retries_with_one_lane(&plan(4, false), &error));
    }

    #[test]
    fn a_single_lane_has_nothing_to_give_up() {
        let error = native(
            skippy_runtime::Status::RuntimeError,
            CONTEXT_CREATION_FAILED,
        );
        assert!(!retries_with_one_lane(&plan(1, true), &error));
    }

    #[test]
    fn other_load_failures_do_not_retry() {
        for error in [
            native(
                skippy_runtime::Status::RuntimeError,
                "failed to load model weights",
            ),
            native(
                skippy_runtime::Status::InvalidArgument,
                CONTEXT_CREATION_FAILED,
            ),
            anyhow::anyhow!("{CONTEXT_CREATION_FAILED}"),
        ] {
            assert!(!retries_with_one_lane(&plan(4, true), &error), "{error:#}");
        }
    }

    #[test]
    fn a_plan_without_its_breakdown_is_not_known_to_be_auto() {
        let error = native(
            skippy_runtime::Status::RuntimeError,
            CONTEXT_CREATION_FAILED,
        );
        let mut plan = plan(4, true);
        plan.breakdown = None;
        assert!(!retries_with_one_lane(&plan, &error));
    }

    #[test]
    fn the_warning_names_the_lanes_given_up_and_the_override() {
        assert_eq!(
            fallback_warning("Qwen3-8B-Q4_K_M", &plan(4, true)),
            "Qwen3-8B-Q4_K_M: could not create the context for 4 parallel requests of 4096 tokens; retrying with one lane (set --parallel to choose)"
        );
    }

    #[test]
    fn the_plan_follows_the_single_lane() {
        let mut plan = plan(4, true);
        use_one_lane(&mut plan);
        assert_eq!(plan.slots, 1);
        assert_eq!(plan.context_length, 4096);
        let breakdown = plan.breakdown.expect("breakdown");
        assert_eq!(breakdown.slots, 1);
        assert_eq!(breakdown.planned_kv_bytes, 147_456 * 4096);
    }
}
