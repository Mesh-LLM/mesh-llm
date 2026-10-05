use crate::gpus::tune::{
    TuneDiagnostic, TuneDiagnosticCode, TuneDiagnosticSeverity, TuneField, TuneFieldStatus,
    TuneRecommendation, TuneRecommendedValue,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TuneHardwareEvaluation {
    pub evaluated_device: EvaluatedTuneDevice,
    pub memory: TuneMemoryBudget,
    pub mlock: TuneMlockEvaluation,
}

#[derive(Clone, Debug)]
pub(crate) struct TuneHardwareEvaluationInput<'a> {
    pub config: &'a mesh_llm_config::MeshConfig,
    pub target: &'a crate::gpus::tune_resolver::ResolvedTuneTarget,
    pub survey: &'a mesh_llm_system::hardware::HardwareSurvey,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EvaluatedTuneDevice {
    pub target: TuneDeviceTarget,
    pub source: TuneDeviceSelectionSource,
    pub report_only_main_gpu: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TuneDeviceTarget {
    Gpu(TuneGpuTarget),
    Cpu,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TuneGpuTarget {
    pub index: usize,
    pub display_name: String,
    pub stable_id: Option<String>,
    pub backend_device: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TuneDeviceSelectionSource {
    ModelHardwareDevice,
    DefaultsHardwareDevice,
    LegacyGpuId,
    SurveyDefault,
    CpuSystemRamFallback,
}

/// The subset of [`TuneDeviceSelectionSource`] variants that represent explicit
/// user-configured device requests. This narrower enum eliminates unreachable
/// arms in functions that only operate on configured device requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConfiguredDeviceSource {
    ModelHardwareDevice,
    DefaultsHardwareDevice,
    LegacyGpuId,
}

impl From<ConfiguredDeviceSource> for TuneDeviceSelectionSource {
    fn from(source: ConfiguredDeviceSource) -> Self {
        match source {
            ConfiguredDeviceSource::ModelHardwareDevice => Self::ModelHardwareDevice,
            ConfiguredDeviceSource::DefaultsHardwareDevice => Self::DefaultsHardwareDevice,
            ConfiguredDeviceSource::LegacyGpuId => Self::LegacyGpuId,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TuneMemorySource {
    EvaluatedGpuVram,
    SystemRamFallback,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TuneMemoryBudget {
    pub source: TuneMemorySource,
    pub total_bytes: u64,
    pub reserved_bytes: Option<u64>,
    pub allocatable_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TuneMlockEvaluation {
    pub available: bool,
    pub reason: String,
}

impl TuneHardwareEvaluation {
    pub(crate) fn device_field_status(&self) -> TuneFieldStatus {
        match self.evaluated_device.source {
            TuneDeviceSelectionSource::ModelHardwareDevice
            | TuneDeviceSelectionSource::DefaultsHardwareDevice
            | TuneDeviceSelectionSource::LegacyGpuId => TuneFieldStatus::Preserved {
                field: TuneField::Device,
                reason: self.device_reason(),
            },
            TuneDeviceSelectionSource::SurveyDefault
            | TuneDeviceSelectionSource::CpuSystemRamFallback => TuneFieldStatus::ReportOnly {
                recommendation: TuneRecommendation {
                    field: TuneField::Device,
                    value: TuneRecommendedValue::Device(self.recommended_device_value()),
                    rationale: self.device_rationale(),
                },
                reason: self.device_reason(),
            },
        }
    }

    pub(crate) fn diagnostics(&self) -> Vec<TuneDiagnostic> {
        if self.mlock.available {
            return Vec::new();
        }
        vec![TuneDiagnostic {
            severity: TuneDiagnosticSeverity::Warning,
            code: TuneDiagnosticCode::MlockUnavailable,
            field: Some(TuneField::Mlock),
            message: self.mlock.reason.clone(),
        }]
    }

    pub(crate) fn recommended_device_value(&self) -> String {
        match &self.evaluated_device.target {
            TuneDeviceTarget::Gpu(gpu) => gpu
                .stable_id
                .clone()
                .or_else(|| gpu.backend_device.clone())
                .unwrap_or_else(|| gpu.display_name.clone()),
            TuneDeviceTarget::Cpu => "cpu".to_string(),
        }
    }

    fn device_rationale(&self) -> String {
        match self.evaluated_device.target {
            TuneDeviceTarget::Gpu(_) => {
                "report the evaluated GPU for tune planning without writing hardware.device in v1"
                    .to_string()
            }
            TuneDeviceTarget::Cpu => {
                "no runtime-selectable GPU was available, so tune planning falls back to CPU/system RAM"
                    .to_string()
            }
        }
    }

    fn device_reason(&self) -> String {
        let selection = match self.evaluated_device.source {
            TuneDeviceSelectionSource::ModelHardwareDevice => "per-model hardware.device",
            TuneDeviceSelectionSource::DefaultsHardwareDevice => "defaults.hardware.device",
            TuneDeviceSelectionSource::LegacyGpuId => "legacy gpu_id",
            TuneDeviceSelectionSource::SurveyDefault => "surveyed default GPU",
            TuneDeviceSelectionSource::CpuSystemRamFallback => "CPU/system-RAM fallback",
        };
        let main_gpu_note = self
            .evaluated_device
            .report_only_main_gpu
            .map(|main_gpu| {
                format!(
                    "; main_gpu={main_gpu} is recorded for reporting only and does not select the evaluated device in v1"
                )
            })
            .unwrap_or_default();
        match &self.evaluated_device.target {
            TuneDeviceTarget::Gpu(gpu) => format!(
                "{selection} selects GPU {} ({}) with {} allocatable after {} reserved{}",
                gpu.index,
                gpu_label(gpu),
                format_bytes(self.memory.allocatable_bytes),
                format_optional_bytes(self.memory.reserved_bytes),
                main_gpu_note,
            ),
            TuneDeviceTarget::Cpu => format!(
                "{selection} uses {} of system RAM for tune planning{}",
                format_bytes(self.memory.allocatable_bytes),
                main_gpu_note,
            ),
        }
    }
}

pub(super) fn display_list(values: &[String]) -> String {
    if values.is_empty() {
        return "none".to_string();
    }
    values.join(", ")
}

pub(super) fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const KIB: f64 = 1024.0;
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GiB", bytes as f64 / GIB)
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / MIB)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / KIB)
    } else {
        format!("{bytes} B")
    }
}

pub(super) fn format_optional_bytes(bytes: Option<u64>) -> String {
    bytes.map(format_bytes).unwrap_or_else(|| "0 B".to_string())
}

pub(super) fn gpu_label(gpu: &TuneGpuTarget) -> String {
    gpu.stable_id
        .clone()
        .or_else(|| gpu.backend_device.clone())
        .unwrap_or_else(|| gpu.display_name.clone())
}

pub(super) fn memory_label(source: TuneMemorySource) -> &'static str {
    match source {
        TuneMemorySource::EvaluatedGpuVram => "GPU VRAM",
        TuneMemorySource::SystemRamFallback => "system RAM",
    }
}

pub(super) fn is_pinnable_stable_id(stable_id: &str) -> bool {
    stable_id.starts_with("pci:")
        || stable_id.starts_with("uuid:")
        || stable_id.starts_with("metal:")
}
