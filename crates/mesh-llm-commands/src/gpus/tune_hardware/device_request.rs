use super::types::ConfiguredDeviceSource;
use mesh_llm_config::{HardwareConfig, MeshConfig, ModelConfigEntry};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EffectiveTuneHardware {
    pub device_request: Option<ConfiguredTuneDeviceRequest>,
    pub report_only_main_gpu: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ConfiguredTuneDeviceRequest {
    pub requested_value: String,
    pub source: ConfiguredDeviceSource,
}

pub(crate) fn effective_tune_hardware(
    config: &MeshConfig,
    target: &crate::gpus::tune_resolver::ResolvedTuneTarget,
) -> EffectiveTuneHardware {
    let model_entry = target
        .config_matches
        .first()
        .and_then(|config_match| config.models.get(config_match.row_index));
    let defaults_hardware = config
        .defaults
        .as_ref()
        .and_then(|value| value.hardware.as_ref());
    let model_hardware = model_entry.and_then(|entry| entry.hardware.as_ref());

    EffectiveTuneHardware {
        device_request: preferred_device_request(model_hardware, defaults_hardware, model_entry),
        report_only_main_gpu: model_hardware
            .and_then(|hardware| hardware.main_gpu)
            .or_else(|| defaults_hardware.and_then(|hardware| hardware.main_gpu)),
    }
}

fn preferred_device_request(
    model_hardware: Option<&HardwareConfig>,
    defaults_hardware: Option<&HardwareConfig>,
    model_entry: Option<&ModelConfigEntry>,
) -> Option<ConfiguredTuneDeviceRequest> {
    non_empty_owned(model_hardware.and_then(|hardware| hardware.device.clone()))
        .map(|requested_value| ConfiguredTuneDeviceRequest {
            requested_value,
            source: ConfiguredDeviceSource::ModelHardwareDevice,
        })
        .or_else(|| {
            non_empty_owned(defaults_hardware.and_then(|hardware| hardware.device.clone())).map(
                |requested_value| ConfiguredTuneDeviceRequest {
                    requested_value,
                    source: ConfiguredDeviceSource::DefaultsHardwareDevice,
                },
            )
        })
        .or_else(|| {
            non_empty_owned(model_entry.and_then(|entry| entry.gpu_id.clone())).map(
                |requested_value| ConfiguredTuneDeviceRequest {
                    requested_value,
                    source: ConfiguredDeviceSource::LegacyGpuId,
                },
            )
        })
}

fn non_empty_owned(value: Option<String>) -> Option<String> {
    value.and_then(|raw| {
        let trimmed = raw.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}
