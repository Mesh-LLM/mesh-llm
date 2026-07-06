use mesh_llm_config::ConfigPath;

use super::*;

impl TuneField {
    pub fn all() -> Vec<Self> {
        <Self as strum::IntoEnumIterator>::iter().collect()
    }

    pub fn spec(self) -> TuneFieldSpec {
        let (config_path, support) = match self {
            Self::CacheTypeK => (
                ConfigPath::from_fields(["models", "<model-ref>", "model_fit", "cache_type_k"]),
                TuneFieldSupport::Writable,
            ),
            Self::CacheTypeV => (
                ConfigPath::from_fields(["models", "<model-ref>", "model_fit", "cache_type_v"]),
                TuneFieldSupport::Writable,
            ),
            Self::FlashAttention => (
                ConfigPath::from_fields(["models", "<model-ref>", "model_fit", "flash_attention"]),
                TuneFieldSupport::Writable,
            ),
            Self::CtxSize => (
                ConfigPath::from_fields(["models", "<model-ref>", "model_fit", "ctx_size"]),
                TuneFieldSupport::Writable,
            ),
            Self::Batch => (
                ConfigPath::from_fields(["models", "<model-ref>", "model_fit", "batch"]),
                TuneFieldSupport::Writable,
            ),
            Self::Ubatch => (
                ConfigPath::from_fields(["models", "<model-ref>", "model_fit", "ubatch"]),
                TuneFieldSupport::Writable,
            ),
            Self::GpuLayers => (
                ConfigPath::from_fields(["models", "<model-ref>", "hardware", "gpu_layers"]),
                TuneFieldSupport::Writable,
            ),
            Self::FitTargetMib => (
                ConfigPath::from_fields(["models", "<model-ref>", "hardware", "fit_target_mib"]),
                TuneFieldSupport::Writable,
            ),
            Self::Device => (
                ConfigPath::from_fields(["models", "<model-ref>", "hardware", "device"]),
                TuneFieldSupport::PreserveOnly,
            ),
            Self::Mmap => (
                ConfigPath::from_fields(["models", "<model-ref>", "hardware", "mmap"]),
                TuneFieldSupport::Writable,
            ),
            Self::Mlock => (
                ConfigPath::from_fields(["models", "<model-ref>", "hardware", "mlock"]),
                TuneFieldSupport::Writable,
            ),
            Self::CpuMoe => (
                ConfigPath::from_fields(["models", "<model-ref>", "hardware", "cpu_moe"]),
                TuneFieldSupport::Unsupported,
            ),
            Self::NCpuMoe => (
                ConfigPath::from_fields(["models", "<model-ref>", "hardware", "n_cpu_moe"]),
                TuneFieldSupport::Unsupported,
            ),
            Self::TensorSplit => (
                ConfigPath::from_fields(["models", "<model-ref>", "hardware", "tensor_split"]),
                TuneFieldSupport::Unsupported,
            ),
            Self::Placement => (
                ConfigPath::from_fields(["models", "<model-ref>", "hardware", "placement"]),
                TuneFieldSupport::Unsupported,
            ),
            Self::Defaults => (
                ConfigPath::from_fields(["defaults"]),
                TuneFieldSupport::PreserveOnly,
            ),
        };
        TuneFieldSpec {
            field: self,
            config_path,
            support,
        }
    }
}

impl TunePlan {
    pub fn summary(&self) -> TunePlanSummary {
        self.field_statuses
            .iter()
            .fold(TunePlanSummary::default(), |mut summary, status| {
                match status {
                    TuneFieldStatus::Applied { .. } => summary.applied += 1,
                    TuneFieldStatus::Preserved { .. } => summary.preserved += 1,
                    TuneFieldStatus::ReportOnly { .. } => summary.report_only += 1,
                    TuneFieldStatus::Unsupported { .. } => summary.unsupported += 1,
                    TuneFieldStatus::Error { .. } => summary.error += 1,
                }
                summary
            })
    }

    pub fn config_edits(&self) -> Vec<TuneConfigEdit> {
        self.field_statuses
            .iter()
            .filter_map(|status| match status {
                TuneFieldStatus::Applied { edit, .. } => Some(edit.clone()),
                TuneFieldStatus::Preserved { .. }
                | TuneFieldStatus::ReportOnly { .. }
                | TuneFieldStatus::Unsupported { .. }
                | TuneFieldStatus::Error { .. } => None,
            })
            .collect()
    }
}
