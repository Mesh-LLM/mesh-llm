//! The capacity a node announces, and the itemized view behind it.
//!
//! The arithmetic lives in `mesh_llm_system::capacity` so the runtime, the
//! console and the `gpus` command all derive the announced budget the same
//! way. This module re-exports it under the names the mesh code already uses,
//! and keeps the tests that exercise it through a node's startup snapshot.

pub use mesh_llm_system::capacity::AdvertisedMemory;
pub(super) use mesh_llm_system::capacity::{
    advertised_capacity_bytes, advertised_memory, capped_capacity_bytes,
};

#[cfg(test)]
mod tests {
    use crate::mesh::{NodeRole, node::hardware_snapshot_for_start};
    use crate::system::hardware::{GpuFacts, HardwareSurvey};

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

        let snapshot = hardware_snapshot_for_start(hw, &NodeRole::Worker, None, 0);

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

        let snapshot = hardware_snapshot_for_start(hw, &NodeRole::Worker, None, 0);

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

        let snapshot = hardware_snapshot_for_start(hw, &NodeRole::Worker, None, 0);

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

        let snapshot = hardware_snapshot_for_start(hw, &NodeRole::Worker, Some(1.0), 0);

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

        let snapshot = hardware_snapshot_for_start(hw, &NodeRole::Worker, Some(32.0), 0);

        assert_eq!(snapshot.vram_bytes, 32_000_000_000);
        assert_eq!(snapshot.local_runtime_capacity_bytes, 32_000_000_000);
    }

    #[test]
    fn snapshot_carries_the_breakdown_next_to_the_budget() {
        let hw = HardwareSurvey {
            vram_bytes: 30_000_000_000,
            gpu_vram: vec![12_000_000_000],
            gpu_reserved: vec![Some(500_000_000)],
            gpus: vec![gpu(12_000_000_000, Some(500_000_000), false)],
            ..HardwareSurvey::default()
        };

        let snapshot = hardware_snapshot_for_start(hw, &NodeRole::Worker, None, 2_000_000_000);

        assert_eq!(snapshot.vram_bytes, 11_500_000_000);
        assert_eq!(snapshot.memory.total_bytes, 12_000_000_000);
        assert_eq!(snapshot.memory.usable_bytes, 9_500_000_000);
    }
}
