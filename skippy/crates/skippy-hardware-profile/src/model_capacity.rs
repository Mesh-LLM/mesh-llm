//! OS-reported local capacity for model discovery, without loading a native runtime.

/// Budget used by both model CLIs for fit labels and default variant selection.
/// Accelerator hosts use usable device memory; CPU-only hosts use 90% of RAM.
pub fn local_model_fit_budget_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        metal::recommended_working_set_bytes().unwrap_or(0)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        let ram = system.total_memory();
        nvidia_capacity()
            .or_else(|| rocm_capacity(ram))
            .or_else(intel_capacity)
            .or_else(platform_adapter_capacity)
            .filter(|bytes| *bytes > 0)
            .unwrap_or_else(|| ram.saturating_mul(9) / 10)
    }
}

#[cfg(not(target_os = "macos"))]
fn nvidia_capacity() -> Option<u64> {
    use crate::platform::command_output;
    #[cfg(not(target_os = "windows"))]
    if let Some(output) = command_output(
        "nvidia-smi",
        &[
            "--query-gpu=memory.total,memory.reserved",
            "--format=csv,noheader,nounits",
        ],
    ) {
        let bytes = nvidia_usable_bytes(&output);
        if bytes > 0 {
            return Some(bytes);
        }
    }
    let output = command_output(
        "nvidia-smi",
        &["--query-gpu=memory.total", "--format=csv,noheader,nounits"],
    )?;
    Some(nvidia_usable_bytes(&output)).filter(|bytes| *bytes > 0)
}

#[cfg(any(not(target_os = "macos"), test))]
fn nvidia_usable_bytes(output: &str) -> u64 {
    output
        .lines()
        .filter_map(|line| {
            let mut fields = line.split(',').map(str::trim);
            let total = fields.next()?.parse::<u64>().ok()?;
            let reserved = fields
                .next()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0);
            Some(total.saturating_sub(reserved).saturating_mul(1024 * 1024))
        })
        .sum()
}

#[cfg(not(target_os = "macos"))]
fn rocm_capacity(ram: u64) -> Option<u64> {
    use crate::platform::command_output;
    let output = command_output("rocm-smi", &["--showmeminfo", "vram", "--csv"])?;
    let vram = rocm_totals(&output);
    let gtt = command_output("rocm-smi", &["--showmeminfo", "gtt", "--csv"])
        .map(|output| rocm_totals(&output))
        .unwrap_or_default();
    let bytes = rocm_unified_capacity(&vram, &gtt, ram).unwrap_or_else(|| vram.iter().sum());
    (bytes > 0).then_some(bytes)
}

#[cfg(any(not(target_os = "macos"), test))]
fn rocm_totals(output: &str) -> Vec<u64> {
    let mut rows = output.lines();
    rows.find(|line| {
        let lower = line.to_ascii_lowercase();
        lower.contains("total") && lower.contains("memory")
    });
    rows.filter_map(|line| line.split(',').nth(1)?.trim().parse().ok())
        .collect()
}

#[cfg(any(not(target_os = "macos"), test))]
fn rocm_unified_capacity(vram: &[u64], gtt: &[u64], ram: u64) -> Option<u64> {
    if vram.len() != 1 || gtt.len() != 1 {
        return None;
    }
    let (vram, gtt) = (vram[0], gtt[0]);
    if vram == 0
        || vram > 2 * 1024 * 1024 * 1024
        || gtt < 8 * 1024 * 1024 * 1024
        || gtt < vram.saturating_mul(8)
    {
        return None;
    }
    let total = if ram > 0 { gtt.min(ram) } else { gtt };
    Some(total.saturating_mul(9) / 10)
}

#[cfg(not(target_os = "macos"))]
fn intel_capacity() -> Option<u64> {
    for args in [["discovery", "--json"], ["discovery", "-j"]] {
        if let Some(output) = crate::platform::command_output("xpu-smi", &args)
            && let Ok(value) = serde_json::from_str(&output)
        {
            let bytes = intel_memory_bytes(&value);
            if bytes > 0 {
                return Some(bytes);
            }
        }
    }
    None
}

#[cfg(any(not(target_os = "macos"), test))]
fn intel_memory_bytes(value: &serde_json::Value) -> u64 {
    match value {
        serde_json::Value::Object(map) => {
            let own = [
                "memory_physical_size_byte",
                "memoryPhysicalSizeByte",
                "memory_total_bytes",
                "memoryTotalBytes",
                "memory_size_byte",
                "memorySizeByte",
                "lmem_total_bytes",
                "lmemTotalBytes",
            ]
            .iter()
            .find_map(|key| {
                map.get(*key).and_then(|value| {
                    value
                        .as_u64()
                        .or_else(|| value.as_str()?.trim().parse::<u64>().ok())
                })
            })
            .unwrap_or(0);
            own + map.values().map(intel_memory_bytes).sum::<u64>()
        }
        serde_json::Value::Array(values) => values.iter().map(intel_memory_bytes).sum(),
        _ => 0,
    }
}

#[cfg(target_os = "windows")]
fn platform_adapter_capacity() -> Option<u64> {
    let bytes = windows::read_windows_video_controllers()
        .iter()
        .map(|(_, bytes)| bytes)
        .sum();
    (bytes > 0).then_some(bytes)
}
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn platform_adapter_capacity() -> Option<u64> {
    None
}
#[cfg(any(target_os = "windows", test))]
mod windows;

#[cfg(target_os = "macos")]
mod metal;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nvidia_budget_subtracts_reserved_memory_per_device() {
        assert_eq!(
            nvidia_usable_bytes("24576, 512\n24576, 1024"),
            47_616 * 1024 * 1024
        );
        assert_eq!(nvidia_usable_bytes("1024, 2048"), 0);
        assert_eq!(nvidia_usable_bytes("24576, [N/A]"), 24_576 * 1024 * 1024);
    }
    #[test]
    fn rocm_uses_capacity_without_subtracting_live_usage() {
        assert_eq!(
            rocm_totals("device,VRAM Total Memory (B),VRAM Total Used Memory (B)\ncard0,1000,800"),
            vec![1000]
        );
        assert_eq!(
            rocm_unified_capacity(&[1 << 30], &[16 << 30], 32 << 30),
            Some((16_u64 << 30) * 9 / 10)
        );
        assert_eq!(
            rocm_unified_capacity(&[8 << 30], &[16 << 30], 32 << 30),
            None
        );
    }
    #[test]
    fn intel_discovery_reads_nested_string_and_numeric_capacities() {
        let value = serde_json::json!({"devices": [
            {"device_name": "A", "memory_physical_size_byte": "1000", "memory_used_bytes": 100},
            {"device_name": "B", "memoryTotalBytes": 2000}
        ]});
        assert_eq!(intel_memory_bytes(&value), 3000);
    }
}
