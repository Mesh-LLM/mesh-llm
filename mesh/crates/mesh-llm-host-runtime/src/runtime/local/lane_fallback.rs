//! One more load with a single lane when the native context of auto-planned
//! lanes cannot be created.
//!
//! The memory plan sizes the shared KV pool from the memory the device reports
//! free. On an integrated GPU that memory is the system's own, shared with
//! everything else running, so the pool can still fail to allocate: Qwen3-8B
//! Q4_K_M on a Radeon 780M (Vulkan, 16 GB laptop) plans four 4096-token lanes,
//! and Vulkan refuses the 1 GiB KV block with `ErrorOutOfDeviceMemory`. The
//! model then never loads, while a single lane of the same depth does.
//!
//! The footprint that load measures is reused for the next load of the model,
//! scaled to its lane count, so the plan also gives up auto-planned lanes when
//! that measurement leaves them no room for the minimum context.

use anyhow::Result;
use mesh_llm_events::{OutputEvent, emit_event};

use super::{LocalOpenAiModelStartSpec, load_direct_skippy_model_with_dflash_fallback};
use crate::inference::skippy;
use crate::runtime::context_planning::{
    RuntimeResourcePlan, RuntimeResourcePlanInput, plan_runtime_resources,
};

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

/// The widest plan under `plan.slots` auto-planned lanes whose measured
/// footprint fits, when `plan`'s leaves no room for the minimum context.
/// After a one-lane retry, the measurement is one lane's compute charge, which
/// the next plan multiplies by the lanes it picks again: that can overrun a
/// device the single lane fits. `plan` is kept otherwise, and always for a
/// lane count the operator set.
pub(super) fn fewer_lanes_for_measured_fit(
    input: RuntimeResourcePlanInput<'_>,
    plan: RuntimeResourcePlan,
) -> RuntimeResourcePlan {
    let measured_no_fit = plan
        .breakdown
        .is_some_and(|breakdown| breakdown.measured_fit == Some(false));
    if !measured_no_fit || input.parallel_override.is_some() {
        return plan;
    }
    (1..plan.slots)
        .rev()
        .find_map(|lanes| {
            let mut fewer = plan_runtime_resources(RuntimeResourcePlanInput {
                parallel_override: Some(lanes),
                ..input
            });
            let breakdown = fewer.breakdown.as_mut()?;
            // The planner chose this lane count, not the operator.
            breakdown.slots_auto = true;
            (breakdown.measured_fit != Some(false)).then_some(fewer)
        })
        .unwrap_or(plan)
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

    /// The device of the module example, as the measurement a one-lane retry
    /// left: 300 MiB of compute and a 4096-token KV, with 1,200 MiB beside
    /// the weights. Four lanes charge 1,200 MiB of compute against the
    /// 1,056 MiB the utilization target leaves; two lanes still fit.
    #[test]
    fn a_measurement_from_one_lane_plans_the_auto_lanes_it_fits() {
        use crate::runtime::context_planning::{
            MeasuredBufferFootprint, RuntimeResourcePlanSource, RuntimeResourcePlanningProfile,
        };
        const MIB: u64 = 1024 * 1024;
        let meta = crate::models::gguf::GgufCompactMeta {
            context_length: 40_960,
            ..Default::default()
        };
        let input = |parallel_override| RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override,
            model_bytes: 5_027_783_488,
            projector_bytes: 0,
            vram_bytes: 5_027_783_488 + 1_200 * MIB,
            metadata: Some(&meta),
            kv_cache_quant: crate::models::gguf::GgufKvCacheQuant::F16,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: Some(MeasuredBufferFootprint {
                compute_bytes: 300 * MIB,
                kv_bytes: 576 * MIB,
                context_length: 4096,
                lane_count: 1,
            }),
        };

        let four = plan_runtime_resources(input(None));
        assert_eq!(four.slots, 4);
        assert_eq!(four.breakdown.unwrap().measured_fit, Some(false));

        let fewer = fewer_lanes_for_measured_fit(input(None), four);
        let breakdown = fewer.breakdown.unwrap();
        assert_eq!(fewer.slots, 2);
        assert_eq!(breakdown.measured_fit, Some(true));
        assert!(breakdown.slots_auto);
        assert_eq!(
            breakdown.planning_source,
            RuntimeResourcePlanSource::MeasuredFootprint
        );

        // Four lanes the operator asked for are kept, and so is their refusal.
        let asked = plan_runtime_resources(input(Some(4)));
        assert_eq!(fewer_lanes_for_measured_fit(input(Some(4)), asked), asked);
        // A plan whose measurement fits is left alone.
        assert_eq!(fewer_lanes_for_measured_fit(input(None), fewer), fewer);
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
    #[ignore = "needs Skippy #189 (a native failure stays typed once its log lines are attached); the pinned v0.80.1 lacks it"]
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
