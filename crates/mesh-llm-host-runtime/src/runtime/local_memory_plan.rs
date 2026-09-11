use std::sync::Mutex;

use super::context_planning::{
    MeasuredBufferFootprint, RuntimeResourcePlan, RuntimeResourcePlanBreakdown,
    reconcile_memory_plan_with_measurements,
};

#[derive(Clone, Copy)]
pub(super) enum MemoryPlanStartPath {
    Direct,
    PackageV2,
}

/// Host-side tie between this process's measured native buffers and the plan
/// that produced them. The model key, context length, and lane count are
/// written together after model open so later planning cannot combine state
/// from two different starts.
#[derive(Clone, Copy, Debug, PartialEq)]
struct MeasuredPlanSnapshot {
    model_bytes: u64,
    context_length: u32,
    lane_count: u32,
}

static MEASURED_PLAN_SNAPSHOT: Mutex<Option<MeasuredPlanSnapshot>> = Mutex::new(None);

/// Return native buffer measurements only when they belong to this model.
pub(super) fn measured_buffers_footprint(model_bytes: u64) -> Option<MeasuredBufferFootprint> {
    let measured = skippy_runtime::measured_native_buffers()?;
    let snapshot = (*MEASURED_PLAN_SNAPSHOT.lock().ok()?)?;
    if snapshot.model_bytes != model_bytes || snapshot.context_length == 0 {
        return None;
    }
    let compute_bytes = measured.compute_mib.map(mib_to_bytes)?;
    let kv_bytes = measured.kv_mib.map(mib_to_bytes)?;
    Some(MeasuredBufferFootprint {
        compute_bytes,
        kv_bytes,
        context_length: snapshot.context_length,
        lane_count: snapshot.lane_count,
    })
}

fn mib_to_bytes(mib: f64) -> u64 {
    (mib * 1024.0 * 1024.0).round() as u64
}

/// Emit the structured plan-time estimate while preserving the package-v2
/// discriminator used by existing telemetry queries.
pub(super) fn emit_memory_plan_resolved(
    model_name: &str,
    breakdown: Option<&RuntimeResourcePlanBreakdown>,
    start_path: MemoryPlanStartPath,
) {
    let Some(breakdown) = breakdown else {
        return;
    };
    let slots_source = plan_value_source(breakdown.slots_auto);
    let context_source = plan_value_source(breakdown.context_auto);

    macro_rules! emit {
        ($($package_field:tt)*) => {
            tracing::info!(
                model = model_name,
                $($package_field)*
                memory_plan.vram_bytes = breakdown.vram_bytes,
                memory_plan.model_bytes = breakdown.model_bytes,
                memory_plan.kv_budget_bytes = breakdown.kv_budget_bytes,
                memory_plan.planned_kv_bytes = breakdown.planned_kv_bytes,
                memory_plan.kv_bytes_per_token = breakdown.kv_bytes_per_token,
                memory_plan.context_length = breakdown.context_length,
                memory_plan.slots = breakdown.slots,
                memory_plan.slots_source = slots_source,
                memory_plan.context_source = context_source,
                "memory plan resolved: charged estimates at plan time; compare with measured buffer_mib native events"
            )
        };
    }

    match start_path {
        MemoryPlanStartPath::Direct => emit!(),
        MemoryPlanStartPath::PackageV2 => emit!(memory_plan.package = "v2",),
    }
}

fn plan_value_source(automatic: bool) -> &'static str {
    if automatic { "auto" } else { "override" }
}

/// Reconcile a resolved plan against the native buffers captured during open.
pub(super) fn emit_measured_memory_reconciliation(model_name: &str, plan: &RuntimeResourcePlan) {
    let Some(breakdown) = plan.breakdown.as_ref() else {
        return;
    };
    let measured = skippy_runtime::measured_native_buffers();
    let reconciliation = reconcile_memory_plan_with_measurements(breakdown, measured);
    if (reconciliation.measured_compute_bytes.is_some()
        || reconciliation.measured_kv_bytes.is_some())
        && let Ok(mut snapshot) = MEASURED_PLAN_SNAPSHOT.lock()
    {
        *snapshot = Some(MeasuredPlanSnapshot {
            model_bytes: breakdown.model_bytes,
            context_length: breakdown.context_length,
            lane_count: breakdown.slots as u32,
        });
    }
    let memory_plan_measured =
        measured.is_some_and(|m| m.compute_mib.is_some() || m.kv_mib.is_some());
    tracing::info!(
        model = model_name,
        memory_plan.measured_available = memory_plan_measured,
        memory_plan.charged_compute_reserve_bytes = reconciliation.charged_compute_reserve_bytes,
        memory_plan.measured_compute_bytes = reconciliation.measured_compute_bytes.unwrap_or(0),
        memory_plan.measured_kv_bytes = reconciliation.measured_kv_bytes.unwrap_or(0),
        memory_plan.residual_free_bytes = reconciliation.residual_free_bytes.unwrap_or(0),
        memory_plan.measured_residual_available = reconciliation.residual_free_bytes.is_some(),
        "memory plan reconciled with measured native buffers"
    );
}

#[cfg(test)]
mod tests {
    use super::{MEASURED_PLAN_SNAPSHOT, MeasuredPlanSnapshot, measured_buffers_footprint};

    #[test]
    fn measured_footprint_reads_one_coherent_plan_snapshot() {
        let mut snapshot = MEASURED_PLAN_SNAPSHOT.lock().unwrap();
        *snapshot = None;
        drop(snapshot);
        assert!(measured_buffers_footprint(1000).is_none());

        snapshot = MEASURED_PLAN_SNAPSHOT.lock().unwrap();
        *snapshot = Some(MeasuredPlanSnapshot {
            model_bytes: 2000,
            context_length: 32768,
            lane_count: 4,
        });
        drop(snapshot);
        assert!(measured_buffers_footprint(1000).is_none());

        snapshot = MEASURED_PLAN_SNAPSHOT.lock().unwrap();
        *snapshot = Some(MeasuredPlanSnapshot {
            model_bytes: 1000,
            context_length: 8192,
            lane_count: 2,
        });
        drop(snapshot);
        if let Some(footprint) = measured_buffers_footprint(1000) {
            assert_eq!(footprint.context_length, 8192);
            assert_eq!(footprint.lane_count, 2);
        }

        snapshot = MEASURED_PLAN_SNAPSHOT.lock().unwrap();
        *snapshot = Some(MeasuredPlanSnapshot {
            model_bytes: 1000,
            context_length: 0,
            lane_count: 4,
        });
        drop(snapshot);
        assert!(measured_buffers_footprint(1000).is_none());

        *MEASURED_PLAN_SNAPSHOT.lock().unwrap() = None;
    }
}
