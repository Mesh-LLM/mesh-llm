use anyhow::{Context, Result};
use mesh_llm_cli::{GpuCommand, benchmark::GpuBenchmarkBackend};
use mesh_llm_system::{
    benchmark::{self, SavedBenchmark},
    capacity::{self, AdvertisedMemory},
    hardware::{self, GpuFacts, HardwareSurvey},
    vram::VramCapacity,
};
use serde_json::{Value, json};
use std::path::Path;

pub mod tune;

pub(crate) mod tune_apply;
pub(crate) mod tune_hardware;
pub(crate) mod tune_resolver;
pub(crate) mod tune_runner;

pub fn dispatch_gpu_command(
    json_output: bool,
    command: Option<&GpuCommand>,
    config_path: Option<&Path>,
) -> Result<()> {
    match command {
        Some(command) => match command {
            GpuCommand::Detect { json } => run_gpu_benchmark(json_output || *json),
            GpuCommand::RunBenchmark { backend } => run_gpu_backend_benchmark(*backend),
        },
        None => run_gpus(json_output, config_path),
    }
}

fn run_gpu_backend_benchmark(backend: GpuBenchmarkBackend) -> Result<()> {
    let outputs = benchmark::run_backend_by_name(map_gpu_backend(backend))?;
    println!("{}", serde_json::to_string(&outputs)?);
    Ok(())
}

fn map_gpu_backend(backend: GpuBenchmarkBackend) -> &'static str {
    match backend {
        GpuBenchmarkBackend::Metal => "metal",
        GpuBenchmarkBackend::Cuda => "cuda",
        GpuBenchmarkBackend::Hip => "hip",
        GpuBenchmarkBackend::Intel => "intel",
    }
}

pub fn run_gpus(json_output: bool, config_path: Option<&Path>) -> Result<()> {
    let mut hw = hardware::survey();
    attach_cached_bandwidth(&mut hw);
    let margin = configured_safety_margin(config_path);

    if json_output {
        return print_json(gpus_json(&hw, &margin));
    }

    println!("{}", format_gpus(&hw, &margin));

    Ok(())
}

/// The safety margin the local fit withholds, and where its value came from.
///
/// `gpus` runs on hosts that have never been configured, so an unreadable or
/// absent config is not an error here: the built-in margin applies and the
/// output says so, rather than the command failing over a file it only needs
/// one number from.
struct SafetyMargin {
    bytes: u64,
    configured: bool,
}

fn configured_safety_margin(config_path: Option<&Path>) -> SafetyMargin {
    let safety_margin_gb = mesh_llm_config::load_config(config_path)
        .ok()
        .and_then(|config| config.defaults)
        .and_then(|defaults| defaults.hardware)
        .and_then(|hardware| hardware.safety_margin_gb);
    SafetyMargin {
        bytes: capacity::safety_margin_bytes(safety_margin_gb),
        configured: safety_margin_gb.is_some(),
    }
}

/// What this host would announce to a mesh, itemized.
///
/// The `--max-vram` ceiling belongs to `serve`, so it is not applied here; the
/// figures are what an uncapped node would advertise from this survey.
fn advertised_memory(hw: &HardwareSurvey, margin: &SafetyMargin) -> AdvertisedMemory {
    capacity::advertised_memory(hw, None, margin.bytes)
}

fn run_gpu_benchmark(json_output: bool) -> Result<()> {
    let hw = hardware::survey();
    if hw.gpus.is_empty() {
        if json_output {
            return print_json(gpu_benchmark_empty_json());
        }
        println!("⚠️ No GPUs detected on this node. Nothing to benchmark.");
        return Ok(());
    }

    let bin_dir = std::env::current_exe()
        .context("failed to resolve mesh-llm binary path")?
        .parent()
        .context("mesh-llm binary path has no parent directory")?
        .to_path_buf();

    let saved = benchmark::run_and_save(&hw, &bin_dir, benchmark::BENCHMARK_TIMEOUT)?;
    let total_bandwidth: f64 = saved.result.mem_bandwidth_gbps.iter().sum();

    if json_output {
        return print_json(gpu_benchmark_json(&hw, &saved));
    }

    println!("✅ Refreshed GPU benchmark fingerprint.");
    println!(
        "  GPUs benchmarked: {}",
        saved.result.mem_bandwidth_gbps.len()
    );
    println!("  Total bandwidth: {}", format_bandwidth(total_bandwidth));
    println!("  Cache path: {}", saved.path.display());

    Ok(())
}

fn gpus_json(hw: &HardwareSurvey, margin: &SafetyMargin) -> Value {
    json!({
        "gpu_count": hw.gpus.len(),
        "gpus": hw.gpus.iter().map(gpu_json).collect::<Vec<_>>(),
        "advertised_memory": advertised_memory_json(hw, margin),
    })
}

fn advertised_memory_json(hw: &HardwareSurvey, margin: &SafetyMargin) -> Value {
    let memory = advertised_memory(hw, margin);
    json!({
        "total_bytes": memory.total_bytes,
        "reserved_bytes": memory.reserved_bytes,
        "platform_reserve_bytes": memory.platform_reserve_bytes,
        "configured_reserve_bytes": memory.configured_reserve_bytes,
        "configured_reserve_source": if margin.configured { "config" } else { "built-in" },
        "usable_bytes": memory.usable_bytes,
        "system_ram_bytes": memory.system_ram_bytes,
        "ram_offload_bytes": memory.ram_offload_bytes,
    })
}

fn gpu_json(gpu: &GpuFacts) -> Value {
    let capacity = VramCapacity::new(gpu.vram_bytes, gpu.reserved_bytes);
    json!({
        "index": gpu.index,
        "name": gpu.display_name,
        "stable_id": gpu.stable_id,
        "backend_device": gpu.backend_device,
        "vram_bytes": gpu.vram_bytes,
        "rated_vram_gb": capacity.rated_capacity_gb(),
        "reserved_bytes": gpu.reserved_bytes,
        "allocatable_vram_bytes": capacity.allocatable_bytes(),
        "mem_bandwidth_gbps": gpu.mem_bandwidth_gbps,
        "compute_tflops_fp32": gpu.compute_tflops_fp32,
        "compute_tflops_fp16": gpu.compute_tflops_fp16,
        "unified_memory": gpu.unified_memory,
        "pci_bdf": gpu.pci_bdf,
        "vendor_uuid": gpu.vendor_uuid,
        "metal_registry_id": gpu.metal_registry_id,
        "dxgi_luid": gpu.dxgi_luid,
        "pnp_instance_id": gpu.pnp_instance_id,
        "runtime_offload": runtime_offload_json(gpu),
    })
}

fn runtime_offload_json(gpu: &GpuFacts) -> Value {
    let backend_device_visible = gpu.backend_device.is_some();
    json!({
        "backend_device_visible": backend_device_visible,
        "selectable": backend_device_visible,
        "diagnostic": if backend_device_visible {
            "embedded_backend_device_available"
        } else {
            "hardware_detected_without_embedded_backend_device"
        },
    })
}

fn gpu_benchmark_empty_json() -> Value {
    json!({
        "refreshed": false,
        "reason": "no_gpus_detected",
        "gpu_count": 0,
        "detected_gpu_count": 0,
        "total_bandwidth_gbps": 0.0,
        "cache_path": Value::Null,
        "gpus": [],
    })
}

fn gpu_benchmark_json(hw: &HardwareSurvey, saved: &SavedBenchmark) -> Value {
    let benchmarked_gpu_count = saved.result.mem_bandwidth_gbps.len();
    let gpus = hw
        .gpus
        .iter()
        .take(benchmarked_gpu_count)
        .enumerate()
        .map(|(index, gpu)| {
            let capacity = VramCapacity::new(gpu.vram_bytes, gpu.reserved_bytes);
            json!({
                "index": gpu.index,
                "name": gpu.display_name,
                "stable_id": gpu.stable_id,
                "backend_device": gpu.backend_device,
                "vram_bytes": gpu.vram_bytes,
                "rated_vram_gb": capacity.rated_capacity_gb(),
                "reserved_bytes": gpu.reserved_bytes,
                "allocatable_vram_bytes": capacity.allocatable_bytes(),
                "unified_memory": gpu.unified_memory,
                "pci_bdf": gpu.pci_bdf,
                "vendor_uuid": gpu.vendor_uuid,
                "metal_registry_id": gpu.metal_registry_id,
                "dxgi_luid": gpu.dxgi_luid,
                "pnp_instance_id": gpu.pnp_instance_id,
                "mem_bandwidth_gbps": saved.result.mem_bandwidth_gbps.get(index),
                "compute_tflops_fp32": saved
                    .result
                    .compute_tflops_fp32
                    .as_ref()
                    .and_then(|values| values.get(index)),
                "compute_tflops_fp16": saved
                    .result
                    .compute_tflops_fp16
                    .as_ref()
                    .and_then(|values| values.get(index)),
            })
        })
        .collect::<Vec<_>>();

    json!({
        "refreshed": true,
        "gpu_count": benchmarked_gpu_count,
        "detected_gpu_count": hw.gpus.len(),
        "total_bandwidth_gbps": saved.result.mem_bandwidth_gbps.iter().sum::<f64>(),
        "cache_path": saved.path,
        "gpus": gpus,
    })
}

fn print_json(value: Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

fn attach_cached_bandwidth(hw: &mut HardwareSurvey) {
    let path = benchmark::fingerprint_path();
    let Some(fingerprint) = benchmark::load_fingerprint(&path) else {
        return;
    };
    if benchmark::hardware_changed(&fingerprint, hw) {
        return;
    }

    for (gpu, cached) in hw.gpus.iter_mut().zip(fingerprint.gpus.iter()) {
        gpu.mem_bandwidth_gbps = Some(cached.p90_gbps);
    }
}

fn format_gpus(hw: &HardwareSurvey, margin: &SafetyMargin) -> String {
    if hw.gpus.is_empty() {
        return "⚠️ No runtime-selectable GPUs reported by the embedded inference backend. This node will run CPU-only until the backend exposes a selectable device.".to_string();
    }
    let mut sections = hw.gpus.iter().map(format_gpu).collect::<Vec<_>>();
    sections.push(format_advertised_memory(hw, margin));
    sections.join("\n\n")
}

/// Explains the single number a node announces: what it starts from, what each
/// party withholds, and what is left for the mesh to place work in.
fn format_advertised_memory(hw: &HardwareSurvey, margin: &SafetyMargin) -> String {
    let memory = advertised_memory(hw, margin);
    let margin_source = if margin.configured {
        "configured"
    } else {
        "built-in default"
    };
    let mut lines = vec![
        "📡 Advertised to the mesh".to_string(),
        format!(
            "  Total device memory: {}",
            format_bytes(memory.total_bytes)
        ),
        format!("  Driver reserved: {}", format_bytes(memory.reserved_bytes)),
    ];
    if memory.platform_reserve_bytes > 0 {
        lines.push(format!(
            "  Platform reserve: {}",
            format_bytes(memory.platform_reserve_bytes)
        ));
    }
    // A margin wider than the memory left cannot be withheld in full. Say so,
    // rather than printing a reserve that looks like the configured value.
    let clamped = if memory.configured_reserve_bytes < margin.bytes {
        format!(
            ", {} asked for but only this much was left",
            format_bytes(margin.bytes)
        )
    } else {
        String::new()
    };
    lines.push(format!(
        "  Configured reserve: {} ({margin_source}{clamped})",
        format_bytes(memory.configured_reserve_bytes)
    ));
    lines.push(format!(
        "  Usable for mesh placement: {}",
        format_bytes(memory.usable_bytes)
    ));
    if let Some(system_ram_bytes) = memory.system_ram_bytes {
        lines.push(format!("  System RAM: {}", format_bytes(system_ram_bytes)));
    }
    lines.push(format!(
        "  RAM-backed local budget: {} (local fit only, never advertised)",
        format_bytes(memory.ram_offload_bytes)
    ));
    if memory.usable_bytes > 0 {
        lines.push(
            "  A `serve --max-vram` ceiling would lower the usable share further.".to_string(),
        );
    }
    lines.join("\n")
}

fn format_gpu(gpu: &GpuFacts) -> String {
    let mut lines = vec![
        format!("🖥️ GPU {}", gpu.index),
        format!("  Name: {}", gpu.display_name),
    ];
    if let Some(stable_id) = gpu.stable_id.as_deref() {
        lines.push(format!("  Stable ID: {stable_id}"));
    }
    if let Some(backend_device) = gpu.backend_device.as_deref() {
        lines.push(format!("  Backend device: {backend_device}"));
    } else {
        lines.push("  Backend device: unavailable (hardware-visible only; embedded runtime did not report a selectable device)".to_string());
    }
    lines.push(format!("  VRAM: {}", format_vram(gpu.vram_bytes)));
    lines.push(format!(
        "  Bandwidth: {}",
        gpu.mem_bandwidth_gbps
            .map(format_bandwidth)
            .unwrap_or_else(|| "unavailable".to_string())
    ));
    lines.push(format!(
        "  Unified memory: {}",
        if gpu.unified_memory { "yes" } else { "no" }
    ));
    if let Some(pci_bdf) = gpu.pci_bdf.as_deref() {
        lines.push(format!("  PCI BDF: {pci_bdf}"));
    }
    if let Some(vendor_uuid) = gpu.vendor_uuid.as_deref() {
        lines.push(format!("  Vendor UUID: {vendor_uuid}"));
    }
    if let Some(metal_registry_id) = gpu.metal_registry_id.as_deref() {
        lines.push(format!("  Metal registry ID: {metal_registry_id}"));
    }
    if let Some(dxgi_luid) = gpu.dxgi_luid.as_deref() {
        lines.push(format!("  DXGI LUID: {dxgi_luid}"));
    }
    if let Some(pnp_instance_id) = gpu.pnp_instance_id.as_deref() {
        lines.push(format!("  PnP instance ID: {pnp_instance_id}"));
    }
    lines.join("\n")
}

fn format_vram(bytes: u64) -> String {
    mesh_llm_system::vram::format_rated_capacity(bytes)
}

/// Exact decimal GB, for itemized values that are not capacity classes. The
/// per-GPU `VRAM:` line keeps the rated class; a reserve of 0.5 GB has no
/// class to round to and must be shown as it is.
fn format_bytes(bytes: u64) -> String {
    format!("{:.1} GB", mesh_llm_system::vram::decimal_gb(bytes))
}

fn format_bandwidth(gbps: f64) -> String {
    format!("{gbps:.1} GB/s")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn built_in_margin() -> SafetyMargin {
        SafetyMargin {
            bytes: capacity::safety_margin_bytes(None),
            configured: false,
        }
    }

    fn sample_gpu(index: usize) -> GpuFacts {
        GpuFacts {
            index,
            display_name: format!("GPU {index}"),
            backend_device: Some(format!("CUDA{index}")),
            vram_bytes: 24_000_000_000,
            reserved_bytes: Some(1_000_000_000),
            mem_bandwidth_gbps: Some(1008.0),
            compute_tflops_fp32: Some(82.4),
            compute_tflops_fp16: Some(164.8),
            unified_memory: false,
            stable_id: Some(format!("stable-{index}")),
            pci_bdf: Some(format!("0000:{index:02x}:00.0")),
            vendor_uuid: Some(format!("uuid-{index}")),
            metal_registry_id: None,
            dxgi_luid: None,
            pnp_instance_id: None,
        }
    }

    #[test]
    fn human_output_itemizes_what_the_node_would_advertise() {
        // 12 GB device with a 0.5 GB driver reserve on a 32 GB host: the 2 GB
        // built-in margin leaves 9.5 GB for the mesh to place work in.
        let mut gpu = sample_gpu(0);
        gpu.vram_bytes = 12_000_000_000;
        gpu.reserved_bytes = Some(500_000_000);
        let hw = HardwareSurvey {
            vram_bytes: 30_000_000_000,
            gpus: vec![gpu],
            system_ram_bytes: Some(32_000_000_000),
            ram_offload_bytes: 18_000_000_000,
            ..HardwareSurvey::default()
        };

        let output = format_gpus(&hw, &built_in_margin());

        assert!(output.contains("📡 Advertised to the mesh"));
        assert!(output.contains("  Total device memory: 12.0 GB"));
        assert!(output.contains("  Driver reserved: 0.5 GB"));
        assert!(output.contains("  Configured reserve: 2.1 GB (built-in default)"));
        assert!(output.contains("  Usable for mesh placement: 9.4 GB"));
        assert!(output.contains("  System RAM: 32.0 GB"));
        assert!(output.contains("  RAM-backed local budget: 18.0 GB"));
        // A discrete GPU keeps nothing back by platform policy.
        assert!(!output.contains("Platform reserve"));
    }

    #[test]
    fn a_margin_wider_than_the_memory_left_is_reported_as_clamped() {
        // An integrated GPU with 0.5 GB enumerated cannot withhold the 2 GB
        // built-in margin: it keeps everything and advertises nothing.
        let mut gpu = sample_gpu(0);
        gpu.vram_bytes = 536_870_912;
        gpu.reserved_bytes = None;
        let hw = HardwareSurvey {
            vram_bytes: 14_848_231_424,
            gpus: vec![gpu],
            system_ram_bytes: Some(16_438_382_592),
            ram_offload_bytes: 14_311_360_512,
            ..HardwareSurvey::default()
        };

        let output = format_gpus(&hw, &built_in_margin());

        assert!(output.contains(
            "  Configured reserve: 0.5 GB (built-in default, 2.1 GB asked for but only this much was left)"
        ));
        assert!(output.contains("  Usable for mesh placement: 0.0 GB"));
        // Nothing is left to cap, so the `--max-vram` hint would be noise.
        assert!(!output.contains("--max-vram"));
    }

    #[test]
    fn human_output_names_the_source_of_the_configured_reserve() {
        let hw = HardwareSurvey {
            gpus: vec![sample_gpu(0)],
            ..HardwareSurvey::default()
        };
        let configured = SafetyMargin {
            bytes: capacity::safety_margin_bytes(Some(4.0)),
            configured: true,
        };

        let output = format_gpus(&hw, &configured);

        assert!(output.contains("  Configured reserve: 4.3 GB (configured)"));
    }

    #[test]
    fn human_output_shows_the_platform_reserve_only_when_one_is_withheld() {
        // A Tegra-shaped survey: the collector budgets 90% of physical RAM, so
        // the tenth the platform keeps is a reserve the owner never set.
        let mut gpu = sample_gpu(0);
        gpu.vram_bytes = 64_000_000_000;
        gpu.reserved_bytes = None;
        gpu.unified_memory = true;
        let hw = HardwareSurvey {
            vram_bytes: 57_600_000_000,
            is_soc: true,
            gpus: vec![gpu],
            system_ram_bytes: Some(64_000_000_000),
            ..HardwareSurvey::default()
        };

        let output = format_gpus(&hw, &built_in_margin());

        assert!(output.contains("  Platform reserve: 6.4 GB"));
    }

    #[test]
    fn machine_output_carries_the_breakdown_next_to_the_gpu_inventory() {
        let mut gpu = sample_gpu(0);
        gpu.vram_bytes = 12_000_000_000;
        gpu.reserved_bytes = Some(500_000_000);
        let hw = HardwareSurvey {
            vram_bytes: 30_000_000_000,
            gpus: vec![gpu],
            system_ram_bytes: Some(32_000_000_000),
            ram_offload_bytes: 18_000_000_000,
            ..HardwareSurvey::default()
        };

        let memory = &gpus_json(&hw, &built_in_margin())["advertised_memory"];

        assert_eq!(memory["total_bytes"], json!(12_000_000_000u64));
        assert_eq!(memory["reserved_bytes"], json!(500_000_000u64));
        assert_eq!(memory["platform_reserve_bytes"], json!(0));
        assert_eq!(memory["system_ram_bytes"], json!(32_000_000_000u64));
        assert_eq!(memory["ram_offload_bytes"], json!(18_000_000_000u64));
        assert_eq!(memory["configured_reserve_source"], json!("built-in"));
        // The itemized shares account for the whole total, as the announcement
        // invariant requires.
        let sum = memory["reserved_bytes"].as_u64().unwrap()
            + memory["platform_reserve_bytes"].as_u64().unwrap()
            + memory["configured_reserve_bytes"].as_u64().unwrap()
            + memory["usable_bytes"].as_u64().unwrap();
        assert_eq!(sum, memory["total_bytes"].as_u64().unwrap());
    }

    #[test]
    fn machine_output_reports_an_absent_system_ram_reading_as_null() {
        let hw = HardwareSurvey {
            gpus: vec![sample_gpu(0)],
            ..HardwareSurvey::default()
        };

        let memory = &gpus_json(&hw, &built_in_margin())["advertised_memory"];

        assert_eq!(memory["system_ram_bytes"], Value::Null);
    }

    #[test]
    fn an_unreadable_config_falls_back_to_the_built_in_margin() {
        let margin = configured_safety_margin(Some(Path::new(
            "/nonexistent/mesh-llm/config-that-is-not-there.toml",
        )));

        assert_eq!(margin.bytes, capacity::safety_margin_bytes(None));
        assert!(!margin.configured);
    }

    #[test]
    fn test_format_vram_unknown() {
        assert_eq!(format_vram(0), "unknown");
    }

    #[test]
    fn test_format_vram_gb() {
        assert_eq!(format_vram(24_000_000_000), "24 GB");
        assert_eq!(format_vram(32 * 1024 * 1024 * 1024), "32 GB");
    }

    #[test]
    fn test_format_bandwidth() {
        assert_eq!(format_bandwidth(1008.04), "1008.0 GB/s");
    }

    #[test]
    fn gpus_json_includes_gpu_fields() {
        let hw = HardwareSurvey {
            gpus: vec![sample_gpu(0)],
            ..HardwareSurvey::default()
        };

        let value = gpus_json(&hw, &built_in_margin());

        assert_eq!(value["gpu_count"], json!(1));
        assert_eq!(value["gpus"][0]["name"], json!("GPU 0"));
        assert_eq!(value["gpus"][0]["mem_bandwidth_gbps"], json!(1008.0));
        assert_eq!(value["gpus"][0]["stable_id"], json!("stable-0"));
        assert_eq!(
            value["gpus"][0]["runtime_offload"],
            json!({
                "backend_device_visible": true,
                "selectable": true,
                "diagnostic": "embedded_backend_device_available",
            })
        );
    }

    #[test]
    fn gpus_json_handles_no_gpus() {
        let value = gpus_json(&HardwareSurvey::default(), &built_in_margin());

        // A host with no enumerated accelerator memory advertises nothing: the
        // breakdown is present but empty, matching the zero capacity such a
        // node announces rather than offering its system RAM as VRAM.
        assert_eq!(
            value,
            json!({
                "gpu_count": 0,
                "gpus": [],
                "advertised_memory": {
                    "total_bytes": 0,
                    "reserved_bytes": 0,
                    "platform_reserve_bytes": 0,
                    "configured_reserve_bytes": 0,
                    "configured_reserve_source": "built-in",
                    "usable_bytes": 0,
                    "system_ram_bytes": Value::Null,
                    "ram_offload_bytes": 0,
                },
            })
        );
    }

    #[test]
    fn human_output_formats_rocm_gpu_without_omitting_backend_details() {
        let mut gpu = sample_gpu(0);
        gpu.display_name = "AMD Instinct MI300X".to_string();
        gpu.backend_device = Some("ROCm0".to_string());
        gpu.stable_id = Some("pci:0000:65:00.0".to_string());
        gpu.pci_bdf = Some("0000:65:00.0".to_string());
        gpu.vendor_uuid = None;
        gpu.mem_bandwidth_gbps = None;
        let hw = HardwareSurvey {
            gpus: vec![gpu],
            ..HardwareSurvey::default()
        };

        let output = format_gpus(&hw, &built_in_margin());
        let gpu_section = output.split("\n\n").next().expect("a GPU section");

        assert_eq!(
            gpu_section,
            "🖥️ GPU 0\n  Name: AMD Instinct MI300X\n  Stable ID: pci:0000:65:00.0\n  Backend device: ROCm0\n  VRAM: 24 GB\n  Bandwidth: unavailable\n  Unified memory: no\n  PCI BDF: 0000:65:00.0"
        );
    }

    #[test]
    fn human_output_keeps_every_rocm_gpu() {
        let mut first = sample_gpu(0);
        first.display_name = "AMD Instinct MI300X".to_string();
        first.backend_device = Some("ROCm0".to_string());
        let mut second = sample_gpu(1);
        second.display_name = "AMD Instinct MI300X".to_string();
        second.backend_device = Some("HIP1".to_string());
        let hw = HardwareSurvey {
            gpus: vec![first, second],
            ..HardwareSurvey::default()
        };

        let output = format_gpus(&hw, &built_in_margin());

        assert_eq!(output.matches("🖥️ GPU ").count(), 2);
        assert!(output.contains("Backend device: ROCm0"));
        assert!(output.contains("Backend device: HIP1"));
    }

    #[test]
    fn gpu_benchmark_json_includes_summary_and_gpu_metrics() {
        let hw = HardwareSurvey {
            gpus: vec![sample_gpu(0), sample_gpu(1)],
            ..HardwareSurvey::default()
        };
        let saved = SavedBenchmark {
            path: PathBuf::from("/tmp/benchmark-fingerprint.json"),
            result: benchmark::BenchmarkResult {
                mem_bandwidth_gbps: vec![1008.0, 912.5],
                compute_tflops_fp32: Some(vec![82.4, 70.2]),
                compute_tflops_fp16: Some(vec![164.8, 140.4]),
            },
        };

        let value = gpu_benchmark_json(&hw, &saved);

        assert_eq!(value["refreshed"], json!(true));
        assert_eq!(value["gpu_count"], json!(2));
        assert_eq!(value["detected_gpu_count"], json!(2));
        assert_eq!(value["total_bandwidth_gbps"], json!(1920.5));
        assert_eq!(
            value["cache_path"],
            json!("/tmp/benchmark-fingerprint.json")
        );
        assert_eq!(value["gpus"][1]["mem_bandwidth_gbps"], json!(912.5));
        assert_eq!(value["gpus"][1]["compute_tflops_fp16"], json!(140.4));
    }

    #[test]
    fn gpu_benchmark_json_truncates_gpu_entries_to_benchmarked_count() {
        let hw = HardwareSurvey {
            gpus: vec![sample_gpu(0), sample_gpu(1)],
            ..HardwareSurvey::default()
        };
        let saved = SavedBenchmark {
            path: PathBuf::from("/tmp/benchmark-fingerprint.json"),
            result: benchmark::BenchmarkResult {
                mem_bandwidth_gbps: vec![1008.0],
                compute_tflops_fp32: Some(vec![82.4]),
                compute_tflops_fp16: Some(vec![164.8]),
            },
        };

        let value = gpu_benchmark_json(&hw, &saved);

        assert_eq!(value["gpu_count"], json!(1));
        assert_eq!(value["detected_gpu_count"], json!(2));
        assert_eq!(value["gpus"].as_array().map(Vec::len), Some(1));
        assert_eq!(value["gpus"][0]["name"], json!("GPU 0"));
    }

    #[test]
    fn gpu_benchmark_empty_json_is_machine_readable() {
        assert_eq!(
            gpu_benchmark_empty_json(),
            json!({
                "refreshed": false,
                "reason": "no_gpus_detected",
                "gpu_count": 0,
                "detected_gpu_count": 0,
                "total_bandwidth_gbps": 0.0,
                "cache_path": Value::Null,
                "gpus": [],
            })
        );
    }
}
