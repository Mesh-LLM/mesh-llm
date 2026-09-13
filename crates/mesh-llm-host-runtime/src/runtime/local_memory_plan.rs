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
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MemoryPlanMeasurementKey(String);

impl MemoryPlanMeasurementKey {
    pub(super) fn new(value: String) -> Self {
        Self(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MeasuredPlanSnapshot {
    key: MemoryPlanMeasurementKey,
    footprint: MeasuredBufferFootprint,
}

static MEASURED_PLAN_SNAPSHOT: Mutex<Option<MeasuredPlanSnapshot>> = Mutex::new(None);

fn completed_measurement_snapshot(
    key: &MemoryPlanMeasurementKey,
    breakdown: &RuntimeResourcePlanBreakdown,
    measured: Option<skippy_runtime::MeasuredNativeBuffers>,
) -> Option<MeasuredPlanSnapshot> {
    let measured = measured?;
    if measured.host_memory_observed {
        return None;
    }
    let mib_to_bytes = |mib: f64| (mib * 1024.0 * 1024.0).round() as u64;
    Some(MeasuredPlanSnapshot {
        key: key.clone(),
        footprint: MeasuredBufferFootprint {
            compute_bytes: mib_to_bytes(measured.compute_mib?),
            kv_bytes: mib_to_bytes(measured.kv_mib?),
            context_length: breakdown.context_length,
            lane_count: breakdown.slots as u32,
        },
    })
}

/// Return a completed native buffer measurement only when its model identity,
/// capacity pool, and allocation-affecting configuration all match.
pub(super) fn measured_buffers_footprint(
    key: &MemoryPlanMeasurementKey,
) -> Option<MeasuredBufferFootprint> {
    let snapshot = MEASURED_PLAN_SNAPSHOT.lock().ok()?.clone()?;
    if snapshot.key != *key || snapshot.footprint.context_length == 0 {
        return None;
    }
    Some(snapshot.footprint)
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
                memory_plan.compute_charge_bytes = breakdown.compute_charge_bytes,
                memory_plan.planning_source = breakdown.planning_source.as_str(),
                memory_plan.measured_fit = breakdown.measured_fit.unwrap_or(true),
                memory_plan.measured_fit_available = breakdown.measured_fit.is_some(),
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
pub(super) fn emit_measured_memory_reconciliation(
    model_name: &str,
    measurement_key: &MemoryPlanMeasurementKey,
    plan: &RuntimeResourcePlan,
) {
    let Some(breakdown) = plan.breakdown.as_ref() else {
        return;
    };
    let measured = skippy_runtime::measured_native_buffers();
    let reconciliation = reconcile_memory_plan_with_measurements(breakdown, measured);
    if let Ok(mut snapshot) = MEASURED_PLAN_SNAPSHOT.lock() {
        *snapshot = completed_measurement_snapshot(measurement_key, breakdown, measured);
    }
    let memory_plan_measured =
        measured.is_some_and(|m| m.compute_mib.is_some() || m.kv_mib.is_some());
    let measurement_reusable = measured.is_some_and(|measurement| {
        !measurement.host_memory_observed
            && measurement.compute_mib.is_some()
            && measurement.kv_mib.is_some()
    });
    tracing::info!(
        model = model_name,
        memory_plan.measured_available = memory_plan_measured,
        memory_plan.charged_compute_reserve_bytes = reconciliation.charged_compute_reserve_bytes,
        memory_plan.measured_compute_bytes = reconciliation.measured_compute_bytes.unwrap_or(0),
        memory_plan.measured_kv_bytes = reconciliation.measured_kv_bytes.unwrap_or(0),
        memory_plan.residual_free_bytes = reconciliation.residual_free_bytes.unwrap_or(0),
        memory_plan.measured_residual_available = reconciliation.residual_free_bytes.is_some(),
        memory_plan.measurement_reusable = measurement_reusable,
        "memory plan reconciled with measured native buffers"
    );
}

#[cfg(test)]
mod tests {
    use super::{
        MEASURED_PLAN_SNAPSHOT, MeasuredPlanSnapshot, MemoryPlanMeasurementKey,
        completed_measurement_snapshot, measured_buffers_footprint,
    };
    use crate::runtime::context_planning::{
        MeasuredBufferFootprint, RuntimeResourcePlanBreakdown, RuntimeResourcePlanSource,
    };

    #[test]
    fn measured_footprint_reads_one_coherent_plan_snapshot() {
        let mut snapshot = MEASURED_PLAN_SNAPSHOT.lock().unwrap();
        *snapshot = None;
        drop(snapshot);
        let first_key = MemoryPlanMeasurementKey::new("first".to_string());
        let second_key = MemoryPlanMeasurementKey::new("second".to_string());
        assert!(measured_buffers_footprint(&first_key).is_none());

        snapshot = MEASURED_PLAN_SNAPSHOT.lock().unwrap();
        *snapshot = Some(MeasuredPlanSnapshot {
            key: second_key,
            footprint: MeasuredBufferFootprint {
                compute_bytes: 10,
                kv_bytes: 20,
                context_length: 32768,
                lane_count: 4,
            },
        });
        drop(snapshot);
        assert!(measured_buffers_footprint(&first_key).is_none());

        snapshot = MEASURED_PLAN_SNAPSHOT.lock().unwrap();
        *snapshot = Some(MeasuredPlanSnapshot {
            key: first_key.clone(),
            footprint: MeasuredBufferFootprint {
                compute_bytes: 30,
                kv_bytes: 40,
                context_length: 8192,
                lane_count: 2,
            },
        });
        drop(snapshot);
        let footprint = measured_buffers_footprint(&first_key).expect("matching snapshot");
        assert_eq!(footprint.compute_bytes, 30);
        assert_eq!(footprint.kv_bytes, 40);
        assert_eq!(footprint.context_length, 8192);
        assert_eq!(footprint.lane_count, 2);

        snapshot = MEASURED_PLAN_SNAPSHOT.lock().unwrap();
        *snapshot = Some(MeasuredPlanSnapshot {
            key: first_key.clone(),
            footprint: MeasuredBufferFootprint {
                compute_bytes: 30,
                kv_bytes: 40,
                context_length: 0,
                lane_count: 4,
            },
        });
        drop(snapshot);
        assert!(measured_buffers_footprint(&first_key).is_none());

        *MEASURED_PLAN_SNAPSHOT.lock().unwrap() = None;
    }

    #[test]
    fn completed_snapshot_rejects_host_offload_and_incomplete_measurements() {
        let key = MemoryPlanMeasurementKey::new("config".to_string());
        let breakdown = RuntimeResourcePlanBreakdown {
            vram_bytes: 10_000,
            model_bytes: 2_000,
            kv_budget_bytes: 6_000,
            planned_kv_bytes: 4_000,
            kv_bytes_per_token: 4,
            compute_charge_bytes: 2_000,
            planning_source: RuntimeResourcePlanSource::StaticEstimate,
            measured_fit: None,
            slots: 4,
            context_length: 1_000,
            slots_auto: true,
            context_auto: true,
        };
        let measured = skippy_runtime::MeasuredNativeBuffers {
            compute_mib: Some(1.0),
            kv_mib: Some(2.0),
            host_memory_observed: false,
        };
        assert!(completed_measurement_snapshot(&key, &breakdown, Some(measured)).is_some());
        assert!(
            completed_measurement_snapshot(
                &key,
                &breakdown,
                Some(skippy_runtime::MeasuredNativeBuffers {
                    host_memory_observed: true,
                    ..measured
                })
            )
            .is_none()
        );
        assert!(
            completed_measurement_snapshot(
                &key,
                &breakdown,
                Some(skippy_runtime::MeasuredNativeBuffers {
                    kv_mib: None,
                    ..measured
                })
            )
            .is_none()
        );
    }
}
