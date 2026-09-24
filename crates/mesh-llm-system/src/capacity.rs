//! Capacity a node advertises to the mesh, and how that single number breaks
//! down.
//!
//! The survey reports raw device memory; a node announces one placement
//! budget (`vram_bytes`). This module derives that budget and the itemized
//! view behind it, so the runtime, the console and the CLI all read the same
//! arithmetic instead of each recomputing its own.

use crate::hardware::HardwareSurvey;
use std::sync::atomic::{AtomicBool, Ordering};

pub fn mesh_capacity_bytes(hw: &HardwareSurvey) -> u64 {
    if unified_memory_only(hw) {
        return hw.vram_bytes;
    }

    let gpu_capacity = hw
        .gpus
        .iter()
        .map(|gpu| crate::vram::allocatable_bytes(gpu.vram_bytes, gpu.reserved_bytes))
        .sum();
    if gpu_capacity > 0 {
        return gpu_capacity;
    }

    let legacy_gpu_capacity = hw
        .gpu_vram
        .iter()
        .enumerate()
        .map(|(index, &vram)| {
            crate::vram::allocatable_bytes(vram, hw.gpu_reserved.get(index).copied().flatten())
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

/// A host whose accelerator memory is the system memory: its budget comes
/// from a platform working set (Metal) or a platform policy (the Tegra
/// collector), not from device VRAM minus a driver reserve.
fn unified_memory_only(hw: &HardwareSurvey) -> bool {
    hw.is_soc && (hw.gpus.is_empty() || hw.gpus.iter().all(|gpu| gpu.unified_memory))
}

pub fn capped_capacity_bytes(capacity_bytes: u64, max_vram_gb: Option<f64>) -> u64 {
    max_vram_gb
        .map(|cap| capacity_bytes.min((cap * 1e9) as u64))
        .unwrap_or(capacity_bytes)
}

pub fn advertised_capacity_bytes(hw: &HardwareSurvey, max_vram_gb: Option<f64>) -> u64 {
    let detected = mesh_capacity_bytes(hw);
    match (detected, max_vram_gb) {
        (0, Some(cap)) => hw.vram_bytes.min((cap * 1e9) as u64),
        _ => capped_capacity_bytes(detected, max_vram_gb),
    }
}

/// Budget the local fit and auto-join plan against. By default it is what the
/// accelerators can hold, the same device memory the node advertises: a model
/// that only fits by spilling into system RAM decodes an order of magnitude
/// slower, so that path is opt-in through `host_ram_offload`. A host without
/// enumerated accelerator memory has nothing to spill from; system RAM is the
/// only memory it serves from, so it keeps its RAM-backed budget either way.
pub fn local_fit_capacity_bytes(
    hw: &HardwareSurvey,
    max_vram_gb: Option<f64>,
    host_ram_offload: bool,
) -> u64 {
    let accelerator_bytes = mesh_capacity_bytes(hw);
    if host_ram_offload || accelerator_bytes == 0 {
        capped_capacity_bytes(hw.vram_bytes, max_vram_gb)
    } else {
        capped_capacity_bytes(accelerator_bytes, max_vram_gb)
    }
}

/// The share of the local fit budget that system RAM backs, after any
/// `max_vram_gb` cap: the RAM credit on an accelerator host that opted into
/// host-RAM offload, zero on one that did not, the whole budget on a host
/// without accelerator memory. A pinned GPU adds this share to its own
/// device memory, so pinning honours the same setting as the whole node.
pub fn local_ram_share_bytes(
    hw: &HardwareSurvey,
    max_vram_gb: Option<f64>,
    host_ram_offload: bool,
) -> u64 {
    let (device_vram, _) = enumerated_device_memory(hw);
    hw.ram_offload_bytes.min(
        local_fit_capacity_bytes(hw, max_vram_gb, host_ram_offload).saturating_sub(device_vram),
    )
}

/// What turning `host_ram_offload` on would add to the local fit budget.
/// Zero when it is already on, and on hosts where the setting changes
/// nothing: without accelerator memory, or with unified memory.
pub fn host_ram_offload_gain_bytes(
    hw: &HardwareSurvey,
    max_vram_gb: Option<f64>,
    host_ram_offload: bool,
) -> u64 {
    if host_ram_offload {
        return 0;
    }
    local_fit_capacity_bytes(hw, max_vram_gb, true).saturating_sub(local_fit_capacity_bytes(
        hw,
        max_vram_gb,
        false,
    ))
}

static PROCESS_HOST_RAM_OFFLOAD: AtomicBool = AtomicBool::new(false);

/// Records this process's `gpu.host_ram_offload` for the local-fit consumers
/// that do not carry the config: model resolution, search and the model CLI.
/// Set from the loaded config at startup; unset reads as off, the default.
pub fn set_process_host_ram_offload(enabled: bool) {
    PROCESS_HOST_RAM_OFFLOAD.store(enabled, Ordering::Relaxed);
}

pub fn process_host_ram_offload() -> bool {
    PROCESS_HOST_RAM_OFFLOAD.load(Ordering::Relaxed)
}

/// This host's local fit budget under the process setting, for consumers
/// that size or select models without a config in hand.
pub fn local_fit_budget_bytes(hw: &HardwareSurvey) -> u64 {
    local_fit_capacity_bytes(hw, None, process_host_ram_offload())
}

/// Itemized view of the capacity a node advertises. The announcement's
/// `vram_bytes` stays the placement budget; this block explains how that
/// number was derived. Invariant: `total_bytes == reserved_bytes +
/// platform_reserve_bytes + configured_reserve_bytes + usable_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AdvertisedMemory {
    /// Enumerated accelerator memory: the sum of device VRAM, or the unified
    /// working set on SoCs.
    pub total_bytes: u64,
    /// Driver/runtime reserved or unavailable bytes when the platform reports
    /// a true value.
    pub reserved_bytes: u64,
    /// Withheld by platform policy before the owner configures anything: the
    /// share of unified memory the platform keeps for the system (the Tegra
    /// collector budgets 90% of physical RAM). Zero for discrete GPUs, whose
    /// budget is the device memory minus the driver reserve, and zero on
    /// Metal while the survey reports the working set as the device memory.
    pub platform_reserve_bytes: u64,
    /// Withheld by the node owner: the effective safety margin plus whatever
    /// a `max_vram_gb` cap leaves out.
    pub configured_reserve_bytes: u64,
    /// What remains for mesh placement after both reserves.
    pub usable_bytes: u64,
    /// Total system RAM when the platform reports it.
    pub system_ram_bytes: Option<u64>,
    /// Portion of the local fit budget backed by system RAM, after any
    /// `max_vram_gb` cap. Zero on an accelerator host unless the owner opted
    /// into host-RAM offload. Never advertised as accelerator capacity.
    pub ram_offload_bytes: u64,
}

pub fn advertised_memory(
    hw: &HardwareSurvey,
    max_vram_gb: Option<f64>,
    safety_margin_bytes: u64,
    host_ram_offload: bool,
) -> AdvertisedMemory {
    let (device_vram, driver_reserved) = enumerated_device_memory(hw);
    let budget = advertised_capacity_bytes(hw, max_vram_gb);
    // A host without enumerated accelerator memory only reaches a non-zero
    // budget through an explicit `max_vram_gb` cap on its CPU budget; that
    // bounded budget is then the whole of what it offers.
    let total_bytes = if device_vram == 0 {
        budget
    } else {
        device_vram
    };
    let reserved_bytes = driver_reserved.min(total_bytes);
    // On a unified-memory host the platform's own budget is a policy (Metal's
    // working set, Tegra's 90% of RAM): whatever it keeps back from the
    // enumerated memory is a platform reserve, not something the owner set.
    // Discrete hosts budget the device memory minus the driver reserve, so
    // nothing lands here.
    let platform_reserve_bytes = if unified_memory_only(hw) {
        total_bytes
            .saturating_sub(reserved_bytes)
            .saturating_sub(mesh_capacity_bytes(hw))
    } else {
        0
    };
    let owner_ceiling = total_bytes
        .saturating_sub(reserved_bytes)
        .saturating_sub(platform_reserve_bytes);
    // The budget never exceeds what the platform leaves to the owner; the
    // clamp only keeps the invariant if a survey ever reports otherwise.
    let usable_bytes = budget
        .saturating_sub(safety_margin_bytes)
        .min(owner_ceiling);
    let configured_reserve_bytes = owner_ceiling.saturating_sub(usable_bytes);
    // The survey derives its RAM-backed share from the uncapped budget. A
    // `max_vram_gb` cap shrinks the local budget first, and only what that
    // capped budget still carries beyond the device memory is RAM; without
    // host-RAM offload an accelerator host's budget carries none.
    let ram_offload_bytes = local_ram_share_bytes(hw, max_vram_gb, host_ram_offload);
    AdvertisedMemory {
        total_bytes,
        reserved_bytes,
        platform_reserve_bytes,
        configured_reserve_bytes,
        usable_bytes,
        system_ram_bytes: hw.system_ram_bytes,
        ram_offload_bytes,
    }
}

/// Sum of the enumerated device memory and of the reserved bytes the platform
/// reported for it, with the same precedence as `mesh_capacity_bytes`: the
/// per-device facts first, the legacy per-GPU lists otherwise.
fn enumerated_device_memory(hw: &HardwareSurvey) -> (u64, u64) {
    if !hw.gpus.is_empty() {
        let vram = hw.gpus.iter().map(|gpu| gpu.vram_bytes).sum();
        let reserved = hw.gpus.iter().filter_map(|gpu| gpu.reserved_bytes).sum();
        return (vram, reserved);
    }
    let vram = hw.gpu_vram.iter().sum();
    let reserved = hw.gpu_reserved.iter().flatten().sum();
    (vram, reserved)
}

/// Margin the local fit withholds on top of the driver reserve when the owner
/// configures none.
pub const BUILTIN_SAFETY_MARGIN_GB: f64 = 2.0;

/// The configured safety margin in bytes, rounded to whole MiB the way the fit
/// target rounds it, so the advertised reserve matches the memory the fit
/// actually withholds. `None` means the owner configured no margin and the
/// built-in default applies. Absurd or negative values saturate to zero rather
/// than overflowing.
pub fn safety_margin_bytes(safety_margin_gb: Option<f64>) -> u64 {
    let gb = safety_margin_gb.unwrap_or(BUILTIN_SAFETY_MARGIN_GB);
    safety_margin_mib(gb).saturating_mul(1024 * 1024)
}

/// Whole MiB withheld for a margin expressed in GB.
pub fn safety_margin_mib(safety_margin_gb: f64) -> u64 {
    (safety_margin_gb * 1024.0).round().max(0.0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::GpuFacts;

    fn gpu(vram_bytes: u64, reserved_bytes: Option<u64>, unified_memory: bool) -> GpuFacts {
        GpuFacts {
            vram_bytes,
            reserved_bytes,
            unified_memory,
            ..GpuFacts::default()
        }
    }

    #[test]
    fn the_built_in_margin_is_withheld_when_the_owner_configures_none() {
        assert_eq!(safety_margin_bytes(None), 2 * 1024 * 1024 * 1024);
        assert_eq!(safety_margin_bytes(Some(0.5)), 512 * 1024 * 1024);
        assert_eq!(safety_margin_bytes(Some(0.0)), 0);
    }

    #[test]
    fn an_impossible_margin_saturates_instead_of_overflowing() {
        assert_eq!(safety_margin_bytes(Some(-1.0)), 0);
        assert_eq!(safety_margin_bytes(Some(f64::MAX)), u64::MAX);
    }

    #[test]
    fn discrete_gpu_breakdown_itemizes_driver_reserve_margin_and_offload() {
        // 12 GB device, 0.5 GB driver reserve, 2 GB margin, 32 GB host:
        // 9.5 GB usable, and the 18 GB RAM credit stays a local-only item.
        let hw = HardwareSurvey {
            vram_bytes: 30_000_000_000,
            gpu_vram: vec![12_000_000_000],
            gpu_reserved: vec![Some(500_000_000)],
            gpus: vec![gpu(12_000_000_000, Some(500_000_000), false)],
            system_ram_bytes: Some(32_000_000_000),
            ram_offload_bytes: 18_000_000_000,
            ..HardwareSurvey::default()
        };

        let memory = advertised_memory(&hw, None, 2_000_000_000, true);

        assert_eq!(
            memory,
            AdvertisedMemory {
                total_bytes: 12_000_000_000,
                reserved_bytes: 500_000_000,
                platform_reserve_bytes: 0,
                configured_reserve_bytes: 2_000_000_000,
                usable_bytes: 9_500_000_000,
                system_ram_bytes: Some(32_000_000_000),
                ram_offload_bytes: 18_000_000_000,
            }
        );
        assert_breakdown_adds_up(&memory);
    }

    #[test]
    fn max_vram_cap_remainder_lands_in_the_configured_reserve() {
        // 40 GB device, 1 GB driver reserve, capped at 32 GB, 2 GB margin:
        // 30 GB usable, and the owner withholds the 7 GB cap remainder plus
        // the 2 GB margin.
        let hw = HardwareSurvey {
            vram_bytes: 491_000_000_000,
            gpu_vram: vec![40_000_000_000],
            gpu_reserved: vec![Some(1_000_000_000)],
            gpus: vec![gpu(40_000_000_000, Some(1_000_000_000), false)],
            ..HardwareSurvey::default()
        };

        let memory = advertised_memory(&hw, Some(32.0), 2_000_000_000, true);

        assert_eq!(memory.total_bytes, 40_000_000_000);
        assert_eq!(memory.reserved_bytes, 1_000_000_000);
        assert_eq!(memory.configured_reserve_bytes, 9_000_000_000);
        assert_eq!(memory.usable_bytes, 30_000_000_000);
        assert_breakdown_adds_up(&memory);
    }

    #[test]
    fn max_vram_cap_bounds_the_ram_backed_share_of_the_local_budget() {
        // 12 GB device credited to 30 GB with RAM: a 20 GB cap leaves 8 GB of
        // the capped local budget beyond the device, a cap under the device
        // memory leaves none.
        let hw = HardwareSurvey {
            vram_bytes: 30_000_000_000,
            gpu_vram: vec![12_000_000_000],
            gpu_reserved: vec![None],
            gpus: vec![gpu(12_000_000_000, None, false)],
            system_ram_bytes: Some(32_000_000_000),
            ram_offload_bytes: 18_000_000_000,
            ..HardwareSurvey::default()
        };

        assert_eq!(
            advertised_memory(&hw, None, 0, true).ram_offload_bytes,
            18_000_000_000
        );
        assert_eq!(
            advertised_memory(&hw, Some(20.0), 0, true).ram_offload_bytes,
            8_000_000_000
        );
        assert_eq!(
            advertised_memory(&hw, Some(8.0), 0, true).ram_offload_bytes,
            0
        );
    }

    #[test]
    fn unified_memory_breakdown_reports_the_working_set_as_total() {
        // The Metal survey reports the recommended working set as the device
        // memory with no driver reserve, so only the margin is withheld.
        let hw = HardwareSurvey {
            vram_bytes: 96_000_000_000,
            is_soc: true,
            gpu_vram: vec![96_000_000_000],
            gpu_reserved: vec![None],
            gpus: vec![gpu(96_000_000_000, None, true)],
            ..HardwareSurvey::default()
        };

        let memory = advertised_memory(&hw, None, 2_000_000_000, true);

        assert_eq!(memory.total_bytes, 96_000_000_000);
        assert_eq!(memory.reserved_bytes, 0);
        assert_eq!(memory.platform_reserve_bytes, 0);
        assert_eq!(memory.configured_reserve_bytes, 2_000_000_000);
        assert_eq!(memory.usable_bytes, 94_000_000_000);
        assert_eq!(memory.ram_offload_bytes, 0);
        assert_breakdown_adds_up(&memory);
    }

    #[test]
    fn tegra_shaped_survey_reports_the_platform_share_as_platform_reserve() {
        // The Tegra collector reports physical RAM as the device memory and
        // budgets 90% of it without a driver reserve: the 10% it keeps back
        // is platform policy, not an owner setting.
        let hw = HardwareSurvey {
            vram_bytes: 57_600_000_000,
            is_soc: true,
            gpu_vram: vec![64_000_000_000],
            gpu_reserved: Vec::new(),
            system_ram_bytes: Some(64_000_000_000),
            ..HardwareSurvey::default()
        };

        let memory = advertised_memory(&hw, None, 0, true);

        assert_eq!(memory.total_bytes, 64_000_000_000);
        assert_eq!(memory.reserved_bytes, 0);
        assert_eq!(memory.platform_reserve_bytes, 6_400_000_000);
        assert_eq!(memory.configured_reserve_bytes, 0);
        assert_eq!(memory.usable_bytes, 57_600_000_000);
        assert_eq!(memory.ram_offload_bytes, 0);
        assert_breakdown_adds_up(&memory);

        // The owner's margin and cap still land in the configured reserve,
        // on top of the platform share.
        let memory = advertised_memory(&hw, Some(32.0), 2_000_000_000, true);

        assert_eq!(memory.platform_reserve_bytes, 6_400_000_000);
        assert_eq!(memory.usable_bytes, 30_000_000_000);
        assert_eq!(memory.configured_reserve_bytes, 27_600_000_000);
        assert_breakdown_adds_up(&memory);
    }

    #[test]
    fn cpu_only_host_without_cap_advertises_an_empty_breakdown() {
        // No accelerator memory to itemize; the RAM-backed local budget is
        // still reported, as an informational item.
        let hw = HardwareSurvey {
            vram_bytes: 24_000_000_000,
            system_ram_bytes: Some(32_000_000_000),
            ram_offload_bytes: 24_000_000_000,
            ..HardwareSurvey::default()
        };

        let memory = advertised_memory(&hw, None, 2_000_000_000, true);

        assert_eq!(
            memory,
            AdvertisedMemory {
                total_bytes: 0,
                reserved_bytes: 0,
                platform_reserve_bytes: 0,
                configured_reserve_bytes: 0,
                usable_bytes: 0,
                system_ram_bytes: Some(32_000_000_000),
                ram_offload_bytes: 24_000_000_000,
            }
        );
    }

    #[test]
    fn explicit_cpu_budget_is_the_whole_total_of_a_cpu_only_host() {
        // A 1 GB bounded budget under a 2 GB margin: nothing usable, and the
        // whole bounded budget counts as withheld by the owner.
        let hw = HardwareSurvey {
            vram_bytes: 16_000_000_000,
            ..HardwareSurvey::default()
        };

        let memory = advertised_memory(&hw, Some(1.0), 2_000_000_000, true);

        assert_eq!(memory.total_bytes, 1_000_000_000);
        assert_eq!(memory.usable_bytes, 0);
        assert_eq!(memory.configured_reserve_bytes, 1_000_000_000);
        assert_breakdown_adds_up(&memory);
    }

    #[test]
    fn discrete_gpu_local_fit_stays_on_the_device_unless_ram_offload_is_opted_in() {
        // The survey of a 12 GB RTX 4070 Ti in a 31 GiB Windows host: the
        // RAM credit takes the budget to 31.4 GB against 12.9 GB of device.
        let hw = HardwareSurvey {
            vram_bytes: 31_427_447_193,
            gpu_vram: vec![12_878_610_432],
            gpu_reserved: vec![None],
            gpus: vec![gpu(12_878_610_432, None, false)],
            system_ram_bytes: Some(33_488_429_056),
            ram_offload_bytes: 18_548_836_761,
            ..HardwareSurvey::default()
        };

        assert_eq!(local_fit_capacity_bytes(&hw, None, false), 12_878_610_432);
        assert_eq!(local_fit_capacity_bytes(&hw, None, true), 31_427_447_193);
        assert_eq!(advertised_memory(&hw, None, 0, false).ram_offload_bytes, 0);
        assert_eq!(
            advertised_memory(&hw, None, 0, true).ram_offload_bytes,
            18_548_836_761
        );
        // The announcement does not depend on the local choice.
        assert_eq!(
            advertised_memory(&hw, None, 0, false).usable_bytes,
            advertised_memory(&hw, None, 0, true).usable_bytes
        );
    }

    #[test]
    fn a_max_vram_cap_still_bounds_the_device_only_local_fit() {
        let hw = HardwareSurvey {
            vram_bytes: 30_000_000_000,
            gpu_vram: vec![12_000_000_000],
            gpu_reserved: vec![Some(500_000_000)],
            gpus: vec![gpu(12_000_000_000, Some(500_000_000), false)],
            ram_offload_bytes: 18_000_000_000,
            ..HardwareSurvey::default()
        };

        assert_eq!(local_fit_capacity_bytes(&hw, None, false), 11_500_000_000);
        assert_eq!(
            local_fit_capacity_bytes(&hw, Some(8.0), false),
            8_000_000_000
        );
    }

    #[test]
    fn cpu_only_and_unified_hosts_keep_their_budget_without_ram_offload() {
        // No accelerator to spill from: RAM is the only memory a CPU host
        // serves from, and a unified host's budget already is its memory.
        let cpu_only = HardwareSurvey {
            vram_bytes: 24_000_000_000,
            system_ram_bytes: Some(32_000_000_000),
            ram_offload_bytes: 24_000_000_000,
            ..HardwareSurvey::default()
        };
        assert_eq!(
            local_fit_capacity_bytes(&cpu_only, None, false),
            24_000_000_000
        );
        assert_eq!(
            advertised_memory(&cpu_only, None, 0, false).ram_offload_bytes,
            24_000_000_000
        );

        let unified = HardwareSurvey {
            vram_bytes: 96_000_000_000,
            is_soc: true,
            gpu_vram: vec![96_000_000_000],
            gpu_reserved: vec![None],
            gpus: vec![gpu(96_000_000_000, None, true)],
            ..HardwareSurvey::default()
        };
        assert_eq!(
            local_fit_capacity_bytes(&unified, None, false),
            local_fit_capacity_bytes(&unified, None, true)
        );
    }

    #[test]
    fn the_offload_gain_and_ram_share_follow_the_setting_where_it_can_change_anything() {
        let discrete = HardwareSurvey {
            vram_bytes: 31_427_447_193,
            gpu_vram: vec![12_878_610_432],
            gpu_reserved: vec![None],
            gpus: vec![gpu(12_878_610_432, None, false)],
            system_ram_bytes: Some(33_488_429_056),
            ram_offload_bytes: 18_548_836_761,
            ..HardwareSurvey::default()
        };
        assert_eq!(
            host_ram_offload_gain_bytes(&discrete, None, false),
            18_548_836_761
        );
        assert_eq!(host_ram_offload_gain_bytes(&discrete, None, true), 0);
        assert_eq!(local_ram_share_bytes(&discrete, None, false), 0);
        assert_eq!(local_ram_share_bytes(&discrete, None, true), 18_548_836_761);

        // A CPU-only host, even with a `max_vram_gb` cap that gives it a
        // non-zero budget, has nothing to gain: RAM already is its budget.
        let cpu_only = HardwareSurvey {
            vram_bytes: 24_000_000_000,
            ram_offload_bytes: 24_000_000_000,
            ..HardwareSurvey::default()
        };
        assert_eq!(host_ram_offload_gain_bytes(&cpu_only, None, false), 0);
        assert_eq!(host_ram_offload_gain_bytes(&cpu_only, Some(8.0), false), 0);
        assert_eq!(
            local_ram_share_bytes(&cpu_only, None, false),
            24_000_000_000
        );

        let unified = HardwareSurvey {
            vram_bytes: 96_000_000_000,
            is_soc: true,
            gpu_vram: vec![96_000_000_000],
            gpu_reserved: vec![None],
            gpus: vec![gpu(96_000_000_000, None, true)],
            ..HardwareSurvey::default()
        };
        assert_eq!(host_ram_offload_gain_bytes(&unified, None, false), 0);
    }

    fn assert_breakdown_adds_up(memory: &AdvertisedMemory) {
        assert_eq!(
            memory.total_bytes,
            memory.reserved_bytes
                + memory.platform_reserve_bytes
                + memory.configured_reserve_bytes
                + memory.usable_bytes
        );
    }
}
