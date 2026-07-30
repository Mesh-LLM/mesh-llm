use crate::system::hardware::HardwareSurvey;

pub(super) fn mesh_capacity_bytes(hw: &HardwareSurvey) -> u64 {
    let unified_memory_only =
        hw.is_soc && (hw.gpus.is_empty() || hw.gpus.iter().all(|gpu| gpu.unified_memory));
    if unified_memory_only {
        return hw.vram_bytes;
    }

    let gpu_capacity = hw
        .gpus
        .iter()
        .map(|gpu| mesh_llm_system::vram::allocatable_bytes(gpu.vram_bytes, gpu.reserved_bytes))
        .sum();
    if gpu_capacity > 0 {
        return gpu_capacity;
    }

    let legacy_gpu_capacity = hw
        .gpu_vram
        .iter()
        .enumerate()
        .map(|(index, &vram)| {
            mesh_llm_system::vram::allocatable_bytes(
                vram,
                hw.gpu_reserved.get(index).copied().flatten(),
            )
        })
        .sum();
    if legacy_gpu_capacity > 0 {
        legacy_gpu_capacity
    } else {
        // A non-SoC node without enumerated accelerator memory cannot host a
        // GPU stage. Keep the broader RAM/offload budget local-only instead
        // of advertising it as accelerator capacity.
        0
    }
}

pub(super) fn capped_capacity_bytes(capacity_bytes: u64, max_vram_gb: Option<f64>) -> u64 {
    max_vram_gb
        .map(|cap| capacity_bytes.min((cap * 1e9) as u64))
        .unwrap_or(capacity_bytes)
}

pub(super) fn advertised_capacity_bytes(hw: &HardwareSurvey, max_vram_gb: Option<f64>) -> u64 {
    let detected = mesh_capacity_bytes(hw);
    match (detected, max_vram_gb) {
        (0, Some(cap)) => hw.vram_bytes.min((cap * 1e9) as u64),
        _ => capped_capacity_bytes(detected, max_vram_gb),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{NodeRole, node::hardware_snapshot_for_start};
    use crate::system::hardware::GpuFacts;

    fn gpu(vram_bytes: u64, reserved_bytes: Option<u64>, unified_memory: bool) -> GpuFacts {
        GpuFacts {
            vram_bytes,
            reserved_bytes,
            unified_memory,
            ..GpuFacts::default()
        }
    }

    #[test]
    fn discrete_gpu_mesh_capacity_excludes_host_ram_offload_budget() {
        let hw = HardwareSurvey {
            vram_bytes: 491_000_000_000,
            gpu_vram: vec![40_000_000_000],
            gpu_reserved: vec![Some(1_000_000_000)],
            gpus: vec![gpu(40_000_000_000, Some(1_000_000_000), false)],
            ..HardwareSurvey::default()
        };

        let snapshot = hardware_snapshot_for_start(hw, &NodeRole::Worker, None);

        assert_eq!(snapshot.vram_bytes, 39_000_000_000);
        assert_eq!(snapshot.local_runtime_capacity_bytes, 491_000_000_000);
    }

    #[test]
    fn unified_memory_mesh_capacity_keeps_recommended_working_set() {
        let hw = HardwareSurvey {
            vram_bytes: 96_000_000_000,
            is_soc: true,
            gpu_vram: vec![128_000_000_000],
            gpu_reserved: vec![Some(16_000_000_000)],
            gpus: vec![gpu(128_000_000_000, Some(16_000_000_000), true)],
            ..HardwareSurvey::default()
        };

        let snapshot = hardware_snapshot_for_start(hw, &NodeRole::Worker, None);

        assert_eq!(snapshot.vram_bytes, 96_000_000_000);
        assert_eq!(snapshot.local_runtime_capacity_bytes, 96_000_000_000);
    }

    #[test]
    fn missing_discrete_gpu_facts_do_not_advertise_host_ram_as_stage_capacity() {
        let hw = HardwareSurvey {
            vram_bytes: 491_000_000_000,
            is_soc: false,
            ..HardwareSurvey::default()
        };

        let snapshot = hardware_snapshot_for_start(hw, &NodeRole::Worker, None);

        assert_eq!(snapshot.vram_bytes, 0);
        assert_eq!(snapshot.local_runtime_capacity_bytes, 491_000_000_000);
    }

    #[test]
    fn explicit_cpu_budget_advertises_bounded_stage_capacity() {
        let hw = HardwareSurvey {
            vram_bytes: 16_000_000_000,
            is_soc: false,
            ..HardwareSurvey::default()
        };

        let snapshot = hardware_snapshot_for_start(hw, &NodeRole::Worker, Some(1.0));

        assert_eq!(snapshot.vram_bytes, 1_000_000_000);
        assert_eq!(snapshot.local_runtime_capacity_bytes, 1_000_000_000);
    }

    #[test]
    fn max_vram_caps_mesh_and_local_runtime_capacities() {
        let hw = HardwareSurvey {
            vram_bytes: 491_000_000_000,
            gpu_vram: vec![40_000_000_000],
            gpu_reserved: vec![Some(1_000_000_000)],
            gpus: vec![gpu(40_000_000_000, Some(1_000_000_000), false)],
            ..HardwareSurvey::default()
        };

        let snapshot = hardware_snapshot_for_start(hw, &NodeRole::Worker, Some(32.0));

        assert_eq!(snapshot.vram_bytes, 32_000_000_000);
        assert_eq!(snapshot.local_runtime_capacity_bytes, 32_000_000_000);
    }
}
