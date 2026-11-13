//! Whether a stage runs on a backend an audit covers.
//!
//! An audit checks a stage's matrix products against the rounding of its
//! backend, and has measured profiles for CPU and CUDA alone
//! ([`skippy_receipts::rounding_profiles::BackendFamily`]). On any other backend
//! (Vulkan, Metal, ROCm) a stage issues no receipts: an honest Vulkan node
//! with K-quant weights failed most audits, and Metal's prefill fits no
//! profile. So its recorder does not start, which also spares it the cost
//! of recording, the stage serves every request as usual, and an embedder
//! that asks whether the runtime issues receipts hears that it issues
//! none.

use skippy_protocol::StageConfig;
use skippy_receipts::rounding_profiles::BackendFamily;
use skippy_runtime::{BackendDevice, BackendDeviceType};

/// The backend the stage's layers run on, named as its device is.
///
/// It follows the stage's own placement, not the first device the runtime
/// registered: a selected CPU device or no offloaded layers is CPU, a
/// selected device is that device, and an offload to every layer runs on
/// the GPUs llama.cpp picks, which must share one audited family. A
/// placement that can mix backends, a partial offload or GPUs of two
/// families, gets a name no audit covers, so the stage issues no receipts.
pub(super) fn stage_backend(config: &StageConfig, devices: &[BackendDevice]) -> String {
    if let Some(device) = &config.selected_device {
        let name = device.backend_device.as_str();
        return if name.eq_ignore_ascii_case("CPU") || name.eq_ignore_ascii_case("CPU0") {
            "CPU".to_string()
        } else {
            name.to_string()
        };
    }
    if config.n_gpu_layers == 0 {
        return "CPU".to_string();
    }
    // llama.cpp offloads to its discrete GPUs, or its integrated ones if none
    let of_type = |kind: BackendDeviceType| -> Vec<&str> {
        devices
            .iter()
            .filter(|device| device.device_type == kind)
            .map(|device| device.name.as_str())
            .collect()
    };
    let mut gpus = of_type(BackendDeviceType::Gpu);
    if gpus.is_empty() {
        gpus = of_type(BackendDeviceType::IntegratedGpu);
    }
    let Some(first) = gpus.first() else {
        return "CPU".to_string();
    };
    if config.n_gpu_layers > 0 {
        // the stage's layers may sit on both the CPU and a GPU
        return format!(
            "CPU+{first} (partial offload of {} layers)",
            config.n_gpu_layers
        );
    }
    let family = BackendFamily::of(first);
    if gpus.len() == 1
        || family.is_some() && gpus.iter().all(|gpu| BackendFamily::of(gpu) == family)
    {
        (*first).to_string()
    } else {
        format!("mixed {}", gpus.join("+"))
    }
}

/// Why a stage on `backend` issues no receipts, when it issues none.
pub(super) fn unaudited(backend: &str) -> Option<String> {
    BackendFamily::of(backend).is_none().then(|| {
        format!("the stage runs on backend {backend}, and an audit covers CPU and CUDA alone")
    })
}

#[cfg(test)]
mod tests {
    use skippy_protocol::{StageConfig, StageDevice};
    use skippy_runtime::{BackendDevice, BackendDeviceType};

    use super::{stage_backend, unaudited};

    fn device(name: &str, device_type: BackendDeviceType) -> BackendDevice {
        BackendDevice {
            name: name.to_string(),
            description: None,
            device_id: None,
            memory_free: 0,
            memory_total: 0,
            device_type,
            caps: 0,
        }
    }

    fn placed(n_gpu_layers: i32, selected: Option<&str>) -> StageConfig {
        StageConfig {
            n_gpu_layers,
            selected_device: selected.map(|name| StageDevice {
                backend_device: name.to_string(),
                stable_id: None,
                index: None,
                vram_bytes: None,
            }),
            ..StageConfig::default()
        }
    }

    #[test]
    fn the_backend_is_where_the_stage_runs_not_the_first_device() {
        use BackendDeviceType::{Cpu, Gpu, IntegratedGpu};
        // a CPU registered first does not make a CUDA stage a CPU one
        let cuda = [
            device("CPU", Cpu),
            device("CUDA0", Gpu),
            device("CUDA1", Gpu),
        ];
        assert_eq!(stage_backend(&placed(-1, None), &cuda), "CUDA0");
        assert_eq!(stage_backend(&placed(-1, Some("CUDA1")), &cuda), "CUDA1");
        // nor does a CUDA registered first make a CPU stage a CUDA one
        let gpu_first = [device("CUDA0", Gpu), device("CPU", Cpu)];
        assert_eq!(stage_backend(&placed(0, None), &gpu_first), "CPU");
        assert_eq!(stage_backend(&placed(-1, Some("CPU0")), &gpu_first), "CPU");
        assert_eq!(stage_backend(&placed(-1, Some("cpu")), &gpu_first), "CPU");
        assert_eq!(
            stage_backend(&placed(-1, None), &[device("CPU", Cpu)]),
            "CPU"
        );
        let metal = [device("MTL0", IntegratedGpu), device("CPU", Cpu)];
        assert_eq!(stage_backend(&placed(-1, None), &metal), "MTL0");
        assert!(unaudited(&stage_backend(&placed(-1, None), &metal)).is_some());
    }

    #[test]
    fn a_placement_that_can_mix_backends_gets_no_receipts() {
        use BackendDeviceType::{Cpu, Gpu};
        let cuda = [device("CPU", Cpu), device("CUDA0", Gpu)];
        let partial = stage_backend(&placed(12, None), &cuda);
        assert!(unaudited(&partial).is_some(), "{partial}");
        let mixed = [
            device("CUDA0", Gpu),
            device("Vulkan0", Gpu),
            device("CPU", Cpu),
        ];
        let both = stage_backend(&placed(-1, None), &mixed);
        assert_eq!(both, "mixed CUDA0+Vulkan0");
        assert!(unaudited(&both).is_some());
    }

    #[test]
    fn only_a_backend_with_rounding_profiles_gets_receipts() {
        for backend in ["CPU", "CUDA0", "CUDA1"] {
            assert_eq!(unaudited(backend), None, "{backend}");
        }
        for backend in [
            "Vulkan0", "Vulkan1", "Metal", "MTL0", "ROCm0", "unknown", "",
        ] {
            let reason = unaudited(backend).unwrap();
            assert!(reason.contains("CPU and CUDA alone"), "{backend}: {reason}");
        }
        assert!(unaudited("Vulkan0").unwrap().contains("backend Vulkan0"));
    }
}
