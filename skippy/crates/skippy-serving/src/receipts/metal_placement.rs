//! Metal qualification excludes ambiguous devices and partial host placement.

use skippy_protocol::{SplitMode, StageConfig};
use skippy_runtime::{BackendDevice, BackendDeviceType};

pub(super) fn device<'a>(
    config: &StageConfig,
    devices: &'a [BackendDevice],
) -> Option<&'a BackendDevice> {
    if let Some(selected) = &config.selected_device {
        return devices
            .iter()
            .find(|device| device.name == selected.backend_device);
    }
    let mut gpus = devices.iter().filter(|device| {
        matches!(
            device.device_type,
            BackendDeviceType::Gpu | BackendDeviceType::IntegratedGpu
        )
    });
    let first = gpus.next()?;
    gpus.next().is_none().then_some(first)
}

pub(super) fn unaudited(config: &StageConfig) -> Option<String> {
    (config.n_gpu_layers != -1
        || config.host_expert_layers.is_some()
        || config.main_gpu.is_some_and(|gpu| gpu != 0)
        || !matches!(config.split_mode, SplitMode::Auto | SplitMode::None)
        || config.op_offload == Some(false))
    .then(|| "Metal placement is unqualified: receipts require full GPU offload on one device without host experts".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(name: &str) -> BackendDevice {
        BackendDevice {
            name: name.into(),
            description: Some("Apple M5 Max".into()),
            device_id: None,
            memory_free: 0,
            memory_total: 0,
            device_type: BackendDeviceType::IntegratedGpu,
            caps: 0,
        }
    }

    #[test]
    fn selection_never_substitutes_the_first_device() {
        let devices = vec![gpu("MTL0"), gpu("MTL1")];
        let mut config = StageConfig::default();
        assert!(device(&config, &devices).is_none());
        config.selected_device = Some(skippy_protocol::StageDevice {
            backend_device: "MTL1".into(),
            stable_id: None,
            index: None,
            vram_bytes: None,
        });
        assert_eq!(device(&config, &devices).unwrap().name, "MTL1");
        config.selected_device.as_mut().unwrap().backend_device = "absent".into();
        assert!(device(&config, &devices).is_none());
        config.selected_device = None;
        assert_eq!(device(&config, &devices[..1]).unwrap().name, "MTL0");
    }

    #[test]
    fn partial_or_split_placement_stays_off() {
        let qualified = StageConfig {
            n_gpu_layers: -1,
            ..StageConfig::default()
        };
        assert!(unaudited(&qualified).is_none());
        for n_gpu_layers in [0, 1, 999] {
            assert!(
                unaudited(&StageConfig {
                    n_gpu_layers,
                    ..qualified.clone()
                })
                .is_some()
            );
        }
        assert!(
            unaudited(&StageConfig {
                split_mode: SplitMode::Layer,
                ..qualified.clone()
            })
            .is_some()
        );
        assert!(
            unaudited(&StageConfig {
                op_offload: Some(false),
                ..qualified
            })
            .is_some()
        );
    }
}
