//! Size a standalone model with the same pure planner used by Mesh.

use std::path::Path;

use skippy_coordinator::resource_planning::{
    RuntimeResourcePlan, RuntimeResourcePlanInput, RuntimeResourcePlanningProfile,
    plan_runtime_resources,
};
use skippy_model_artifact::gguf::{GgufKvCacheQuant, scan_gguf_compact_meta};
use skippy_runtime::{BackendDevice, BackendDeviceType, backend_devices};

pub(crate) struct LocalResourcePlanningInput<'a> {
    pub(crate) model_path: &'a Path,
    pub(crate) model_bytes: u64,
    pub(crate) projector_path: Option<&'a Path>,
    pub(crate) n_gpu_layers: i32,
    pub(crate) kv_offload: Option<bool>,
    pub(crate) selected_device: Option<&'a str>,
    pub(crate) ctx_size_override: Option<u32>,
    pub(crate) parallel_override: Option<usize>,
    pub(crate) cache_type_k: &'a str,
    pub(crate) cache_type_v: &'a str,
}

pub(crate) fn plan_local_resources(input: LocalResourcePlanningInput<'_>) -> RuntimeResourcePlan {
    let metadata = scan_gguf_compact_meta(input.model_path);
    let kv_cache_quant = GgufKvCacheQuant::from_llama_args(input.cache_type_k, input.cache_type_v)
        .unwrap_or(GgufKvCacheQuant::F16);
    let vram_bytes = metadata
        .as_ref()
        .and_then(|meta| kv_cache_quant.kv_cache_bytes_per_token(meta))
        .map(|_| {
            available_device_memory(input.n_gpu_layers, input.kv_offload, input.selected_device)
        })
        .unwrap_or(0);
    let projector_bytes = input
        .projector_path
        .and_then(|path| path.metadata().ok())
        .map(|metadata| metadata.len())
        .unwrap_or(0);

    plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: input.ctx_size_override,
        parallel_override: input.parallel_override,
        model_bytes: input.model_bytes,
        projector_bytes,
        vram_bytes,
        metadata: metadata.as_ref(),
        kv_cache_quant,
        local_layer_fraction: None,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        measured_buffers: None,
    })
}

fn available_device_memory(
    n_gpu_layers: i32,
    kv_offload: Option<bool>,
    selected_device: Option<&str>,
) -> u64 {
    let Ok(devices) = backend_devices() else {
        return 0;
    };
    available_memory_from_devices(&devices, n_gpu_layers, kv_offload, selected_device)
}

fn available_memory_from_devices(
    devices: &[BackendDevice],
    n_gpu_layers: i32,
    kv_offload: Option<bool>,
    selected_device: Option<&str>,
) -> u64 {
    let mut gpu_bytes = 0u64;
    let mut gpu_present = false;
    let mut cpu_bytes = 0u64;
    for device in devices {
        if device.device_type != BackendDeviceType::Cpu
            && selected_device.is_some_and(|name| name != device.name)
        {
            continue;
        }
        match device.device_type {
            BackendDeviceType::Gpu | BackendDeviceType::IntegratedGpu => {
                gpu_present = true;
                gpu_bytes = gpu_bytes.max(device.memory_free);
            }
            BackendDeviceType::Cpu => cpu_bytes = cpu_bytes.max(device.memory_free),
            BackendDeviceType::Accelerator | BackendDeviceType::Meta => {}
        }
    }
    if n_gpu_layers == 0 || kv_offload == Some(false) || !gpu_present {
        cpu_bytes
    } else {
        gpu_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(device_type: BackendDeviceType, memory_free: u64) -> BackendDevice {
        BackendDevice {
            name: "test".into(),
            description: None,
            device_id: None,
            memory_free,
            memory_total: memory_free,
            device_type,
            caps: 0,
        }
    }

    #[test]
    fn uses_one_gpu_budget_without_falling_back_to_cpu_when_gpu_is_exhausted() {
        let devices = [
            device(BackendDeviceType::Cpu, 64_000_000_000),
            device(BackendDeviceType::IntegratedGpu, 0),
            device(BackendDeviceType::Gpu, 12_000_000_000),
        ];
        assert_eq!(
            available_memory_from_devices(&devices, -1, None, None),
            12_000_000_000
        );
        assert_eq!(
            available_memory_from_devices(&devices[..2], -1, None, None),
            0
        );
        assert_eq!(
            available_memory_from_devices(&devices, 0, None, None),
            64_000_000_000
        );
    }

    #[test]
    fn selected_device_and_host_kv_controls_choose_the_correct_memory_budget() {
        let mut selected = device(BackendDeviceType::Gpu, 4_000_000_000);
        selected.name = "GPU0".into();
        let mut other = device(BackendDeviceType::Gpu, 16_000_000_000);
        other.name = "GPU1".into();
        let devices = [
            device(BackendDeviceType::Cpu, 64_000_000_000),
            selected,
            other,
        ];
        assert_eq!(
            available_memory_from_devices(&devices, -1, Some(true), Some("GPU0")),
            4_000_000_000
        );
        assert_eq!(
            available_memory_from_devices(&devices, -1, Some(false), Some("GPU0")),
            64_000_000_000
        );
    }
}
