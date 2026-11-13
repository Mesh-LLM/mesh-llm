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

/// The backend the stage's matrix products run on, named as its device is.
///
/// It follows the stage's own placement, not the first device the runtime
/// registered, and names a backend an audit covers only when every product
/// of the stage runs there:
///
/// - On the CPU (a selected CPU device, no offloaded layers, or no GPU),
///   no other backend may take a product: llama.cpp hands host-weight
///   products to an accelerator such as BLAS whenever one is registered,
///   and offloads large batches of them to a GPU unless `op_offload` is
///   off.
/// - On a GPU, selected or not, every layer of the stage must be offloaded
///   ([`offloads_every_layer`]) with no routed experts kept in host
///   memory: selecting a device does not offload a layer to it. The GPUs
///   llama.cpp picks must share one audited family.
///
/// Any other placement gets a name no audit covers, so the stage issues no
/// receipts. `block_count` is the model's layer count, which llama.cpp
/// counts `n_gpu_layers` against.
pub(super) fn stage_backend(
    config: &StageConfig,
    devices: &[BackendDevice],
    block_count: Option<u32>,
) -> String {
    let of_type = |kind: BackendDeviceType| -> Vec<&str> {
        devices
            .iter()
            .filter(|device| device.device_type == kind)
            .map(|device| device.name.as_str())
            .collect()
    };
    // llama.cpp offloads to its discrete GPUs, or its integrated ones if none
    let mut gpus = of_type(BackendDeviceType::Gpu);
    if gpus.is_empty() {
        gpus = of_type(BackendDeviceType::IntegratedGpu);
    }
    let on_cpu = || {
        let accelerators = of_type(BackendDeviceType::Accelerator);
        if !accelerators.is_empty() {
            format!("CPU+{} (accelerator)", accelerators.join("+"))
        } else if !gpus.is_empty() && config.op_offload != Some(false) {
            format!("CPU+{} (operation offload)", gpus.join("+"))
        } else {
            "CPU".to_string()
        }
    };
    let on_gpu = |gpu: &str| {
        if !offloads_every_layer(config, block_count) {
            // the stage's layers may sit on both the CPU and the GPU
            format!(
                "CPU+{gpu} (partial offload of {} layers)",
                config.n_gpu_layers
            )
        } else if config.host_expert_layers.is_some() {
            format!("CPU+{gpu} (routed experts in host memory)")
        } else {
            gpu.to_string()
        }
    };
    if let Some(device) = &config.selected_device {
        let name = device.backend_device.as_str();
        return if name.eq_ignore_ascii_case("CPU") || name.eq_ignore_ascii_case("CPU0") {
            // no layer is offloaded, but the model keeps every GPU, which
            // operation offload can reach
            on_cpu()
        } else {
            on_gpu(name)
        };
    }
    let Some(first) = gpus.first().copied() else {
        return on_cpu();
    };
    if config.n_gpu_layers == 0 {
        return on_cpu();
    }
    let family = BackendFamily::of(first);
    if gpus.len() == 1
        || family.is_some() && gpus.iter().all(|gpu| BackendFamily::of(gpu) == family)
    {
        on_gpu(first)
    } else {
        format!("mixed {}", gpus.join("+"))
    }
}

/// Whether `n_gpu_layers` puts every layer of the stage on a GPU.
///
/// llama.cpp numbers the model's layers `0..=block_count`, the output
/// layer last, and offloads the last `n_gpu_layers` of them; a negative
/// count offloads all. So the stage's layers from `layer_start` on are all
/// offloaded when `n_gpu_layers >= block_count + 1 - layer_start`. A count
/// such as 99 that operators use for "everything" is thus a full offload,
/// and an unknown `block_count` counts as a partial one.
fn offloads_every_layer(config: &StageConfig, block_count: Option<u32>) -> bool {
    let Ok(count) = u32::try_from(config.n_gpu_layers) else {
        return true;
    };
    block_count
        .is_some_and(|blocks| u64::from(count) + u64::from(config.layer_start) > u64::from(blocks))
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

    use super::unaudited;

    /// The layers of the model the stages below serve.
    const BLOCKS: Option<u32> = Some(36);

    /// The backend of a stage of that model.
    fn stage_backend(config: &StageConfig, devices: &[BackendDevice]) -> String {
        super::stage_backend(config, devices, BLOCKS)
    }

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

    /// `placed`, with operation offload turned off.
    fn kept(n_gpu_layers: i32, selected: Option<&str>) -> StageConfig {
        StageConfig {
            op_offload: Some(false),
            ..placed(n_gpu_layers, selected)
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
        assert_eq!(stage_backend(&kept(0, None), &gpu_first), "CPU");
        assert_eq!(stage_backend(&kept(-1, Some("CPU0")), &gpu_first), "CPU");
        assert_eq!(stage_backend(&kept(-1, Some("cpu")), &gpu_first), "CPU");
        let cpu_only = [device("CPU", Cpu)];
        assert_eq!(stage_backend(&placed(-1, None), &cpu_only), "CPU");
        assert_eq!(stage_backend(&placed(0, Some("CPU")), &cpu_only), "CPU");
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

    /// Selecting a GPU does not offload a layer to it: the stage runs there
    /// only with every layer offloaded, and no routed experts on the host.
    #[test]
    fn a_selected_gpu_runs_the_stage_only_with_every_layer_offloaded() {
        use BackendDeviceType::{Cpu, Gpu};
        let cuda = [device("CPU", Cpu), device("CUDA0", Gpu)];
        assert_eq!(stage_backend(&placed(-1, Some("CUDA0")), &cuda), "CUDA0");
        for layers in [0, 1, 12, 36] {
            let backend = stage_backend(&placed(layers, Some("CUDA0")), &cuda);
            assert!(unaudited(&backend).is_some(), "{layers}: {backend}");
        }
        // a count that covers every layer is an offload of every layer
        assert_eq!(stage_backend(&placed(999, Some("CUDA0")), &cuda), "CUDA0");
        let experts = |selected| StageConfig {
            host_expert_layers: Some(skippy_protocol::HostExpertLayers { start: 0, end: 4 }),
            ..placed(-1, selected)
        };
        for selected in [Some("CUDA0"), None] {
            let backend = stage_backend(&experts(selected), &cuda);
            assert!(unaudited(&backend).is_some(), "{selected:?}: {backend}");
        }
    }

    /// llama.cpp gives host-weight products to a registered accelerator
    /// (BLAS), and large batches of them to a GPU unless operation offload
    /// is off: neither runs with the CPU's rounding.
    #[test]
    fn a_cpu_stage_whose_products_can_leave_the_cpu_gets_no_receipts() {
        use BackendDeviceType::{Accelerator, Cpu, Gpu};
        let blas = [device("BLAS", Accelerator), device("CPU", Cpu)];
        for config in [placed(0, None), placed(-1, None), kept(0, Some("CPU"))] {
            let backend = stage_backend(&config, &blas);
            assert!(unaudited(&backend).is_some(), "{backend}");
            assert!(backend.contains("BLAS"), "{backend}");
        }
        let cuda = [device("CPU", Cpu), device("CUDA0", Gpu)];
        for config in [
            placed(0, None),
            placed(0, Some("CPU0")),
            StageConfig {
                op_offload: Some(true),
                ..placed(0, None)
            },
        ] {
            let backend = stage_backend(&config, &cuda);
            assert!(unaudited(&backend).is_some(), "{backend}");
        }
        assert_eq!(stage_backend(&kept(0, Some("CPU0")), &cuda), "CPU");
        assert_eq!(
            stage_backend(&placed(0, None), &[device("CPU", Cpu)]),
            "CPU"
        );
    }

    #[test]
    fn a_count_that_offloads_every_layer_of_the_stage_is_a_full_offload() {
        use BackendDeviceType::{Cpu, Gpu};
        let cuda = [device("CPU", Cpu), device("CUDA0", Gpu)];
        let backend = |n_gpu_layers, layer_start| {
            let config = StageConfig {
                layer_start,
                ..placed(n_gpu_layers, None)
            };
            stage_backend(&config, &cuda)
        };
        // the counts operators use for "everything"
        for count in [-1, 99, 999, i32::MAX] {
            assert_eq!(backend(count, 0), "CUDA0", "{count}");
            assert_eq!(backend(count, 18), "CUDA0", "{count}");
        }
        // 37 layers with the output: the first stage needs all 37
        assert_eq!(backend(37, 0), "CUDA0");
        let below = backend(36, 0);
        assert_eq!(below, "CPU+CUDA0 (partial offload of 36 layers)");
        assert!(unaudited(&below).is_some());
        // the second stage, layers 18 to 35, needs 19: its 18 and the output
        assert_eq!(backend(19, 18), "CUDA0");
        assert!(unaudited(&backend(18, 18)).is_some());
        assert!(unaudited(&backend(10, 18)).is_some());
        // a model whose layer count is unknown is not judged offloaded
        let unknown = super::stage_backend(&placed(99, None), &cuda, None);
        assert!(unaudited(&unknown).is_some(), "{unknown}");
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
