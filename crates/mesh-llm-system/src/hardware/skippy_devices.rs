use super::GpuFacts;

#[cfg(test)]
use std::sync::{Mutex, OnceLock};

#[cfg(test)]
type TestGpuFactsResult = anyhow::Result<Vec<GpuFacts>, String>;

#[cfg(test)]
type TestGpuFactsOverride = Mutex<Option<TestGpuFactsResult>>;

#[cfg(test)]
static TEST_GPU_FACTS_RESULT: OnceLock<TestGpuFactsOverride> = OnceLock::new();

#[cfg(test)]
fn test_gpu_facts_result() -> &'static TestGpuFactsOverride {
    TEST_GPU_FACTS_RESULT.get_or_init(|| Mutex::new(None))
}

#[cfg(all(test, target_os = "linux"))]
pub(super) fn set_test_gpu_facts_result(result: anyhow::Result<Vec<GpuFacts>>) {
    *test_gpu_facts_result().lock().unwrap() = Some(result.map_err(|err| err.to_string()));
}

#[cfg(all(test, target_os = "linux"))]
pub(super) fn clear_test_gpu_facts_result() {
    *test_gpu_facts_result().lock().unwrap() = None;
}

pub fn gpu_facts() -> anyhow::Result<Vec<GpuFacts>> {
    #[cfg(test)]
    if let Some(result) = test_gpu_facts_result().lock().unwrap().take() {
        return result.map_err(anyhow::Error::msg);
    }

    let mut facts = gpu_facts_from_backend_devices(skippy_runtime::backend_devices()?);
    if !facts.is_empty() {
        super::enrichers::enrich_gpu_facts(&mut facts);
    }

    Ok(facts)
}

fn gpu_facts_from_backend_devices(
    backend_devices: Vec<skippy_runtime::BackendDevice>,
) -> Vec<GpuFacts> {
    let mut accelerator_index = 0usize;
    let mut facts = Vec::new();

    for device in backend_devices {
        if !is_runtime_accelerator(&device) {
            continue;
        }

        let index = accelerator_index;
        accelerator_index += 1;

        let backend_device = Some(device.name.clone());
        let pci_bdf = device.device_id.clone();
        let unified_memory = device.device_type == skippy_runtime::BackendDeviceType::IntegratedGpu
            || (cfg!(target_os = "macos") && device.name.starts_with("MTL"));
        let stable_id = if unified_memory && cfg!(target_os = "macos") {
            Some(format!("metal:{index}"))
        } else {
            pci_bdf
                .as_ref()
                .filter(|id| !super::is_placeholder_pci_bdf(id))
                .map(|id| format!("pci:{id}"))
                .or_else(|| Some(device.name.to_ascii_lowercase()))
        };
        let (vram_bytes, reserved_bytes) = if unified_memory && cfg!(target_os = "macos") {
            #[cfg(target_os = "macos")]
            {
                super::macos_metal_gpu_budget(super::query_metal_recommended_working_set_bytes())
                    .unwrap_or((device.memory_total, None))
            }
            #[cfg(not(target_os = "macos"))]
            {
                (device.memory_total, None)
            }
        } else {
            (device.memory_total, None)
        };

        facts.push(GpuFacts {
            index,
            display_name: device.description.unwrap_or(device.name),
            backend_device,
            vram_bytes,
            reserved_bytes,
            mem_bandwidth_gbps: None,
            compute_tflops_fp32: None,
            compute_tflops_fp16: None,
            unified_memory,
            stable_id,
            pci_bdf,
            vendor_uuid: None,
            metal_registry_id: None,
            dxgi_luid: None,
            pnp_instance_id: None,
        });
    }

    facts
}

fn is_runtime_accelerator(device: &skippy_runtime::BackendDevice) -> bool {
    match device.device_type {
        skippy_runtime::BackendDeviceType::Gpu
        | skippy_runtime::BackendDeviceType::IntegratedGpu => true,
        skippy_runtime::BackendDeviceType::Accelerator => {
            device.memory_total > 0 || looks_like_known_gpu_backend(device)
        }
        skippy_runtime::BackendDeviceType::Cpu | skippy_runtime::BackendDeviceType::Meta => false,
    }
}

fn looks_like_known_gpu_backend(device: &skippy_runtime::BackendDevice) -> bool {
    let text = format!(
        "{} {}",
        device.name,
        device.description.as_deref().unwrap_or_default()
    )
    .to_ascii_uppercase();
    [
        "CUDA", "HIP", "ROCM", "VULKAN", "SYCL", "METAL", "MTL", "GPU",
    ]
    .iter()
    .any(|needle| text.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_runtime::{BackendDevice, BackendDeviceType};

    fn backend_device(
        name: &str,
        device_type: BackendDeviceType,
        memory_total: u64,
    ) -> BackendDevice {
        BackendDevice {
            name: name.to_string(),
            description: Some(name.to_string()),
            device_id: None,
            memory_free: memory_total,
            memory_total,
            device_type,
            caps: 0,
        }
    }

    #[test]
    fn empty_backend_inventory_does_not_synthesize_platform_gpu_facts() {
        assert!(gpu_facts_from_backend_devices(Vec::new()).is_empty());
    }

    #[test]
    fn cpu_backend_inventory_does_not_synthesize_platform_gpu_facts() {
        let facts = gpu_facts_from_backend_devices(vec![backend_device(
            "CPU",
            BackendDeviceType::Cpu,
            64 * 1024 * 1024 * 1024,
        )]);

        assert!(facts.is_empty());
    }

    #[test]
    fn hip_backend_device_is_runtime_selectable_gpu_fact() {
        let facts = gpu_facts_from_backend_devices(vec![BackendDevice {
            name: "HIP0".to_string(),
            description: Some("AMD Instinct MI300X".to_string()),
            device_id: Some("0000:65:00.0".to_string()),
            memory_free: 200_000_000_000,
            memory_total: 206_158_430_208,
            device_type: BackendDeviceType::Accelerator,
            caps: 0,
        }]);

        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].display_name, "AMD Instinct MI300X");
        assert_eq!(facts[0].backend_device.as_deref(), Some("HIP0"));
        assert_eq!(facts[0].vram_bytes, 206_158_430_208);
        assert_eq!(facts[0].stable_id.as_deref(), Some("pci:0000:65:00.0"));
    }
}
