impl TuneField {
    pub fn all() -> Vec<Self> {
        <Self as strum::IntoEnumIterator>::iter().collect()
    }

    pub fn spec(self) -> TuneFieldSpec {
        let config_path = match self {
            Self::CacheTypeK => {
                ConfigPath::from_fields(["models", "<model-ref>", "model_fit", "cache_type_k"])
            }
            Self::CacheTypeV => {
                ConfigPath::from_fields(["models", "<model-ref>", "model_fit", "cache_type_v"])
            }
            Self::FlashAttention => {
                ConfigPath::from_fields(["models", "<model-ref>", "model_fit", "flash_attention"])
            }
            Self::CtxSize => {
                ConfigPath::from_fields(["models", "<model-ref>", "model_fit", "ctx_size"])
            }
            Self::Batch => ConfigPath::from_fields(["models", "<model-ref>", "model_fit", "batch"]),
            Self::Ubatch => {
                ConfigPath::from_fields(["models", "<model-ref>", "model_fit", "ubatch"])
            }
            Self::GpuLayers => {
                ConfigPath::from_fields(["models", "<model-ref>", "hardware", "gpu_layers"])
            }
            Self::FitTargetMib => {
                ConfigPath::from_fields(["models", "<model-ref>", "hardware", "fit_target_mib"])
            }
            Self::Device => {
                ConfigPath::from_fields(["models", "<model-ref>", "hardware", "device"])
            }
            Self::Mmap => ConfigPath::from_fields(["models", "<model-ref>", "hardware", "mmap"]),
            Self::Mlock => ConfigPath::from_fields(["models", "<model-ref>", "hardware", "mlock"]),
            Self::CpuMoe => {
                ConfigPath::from_fields(["models", "<model-ref>", "hardware", "cpu_moe"])
            }
            Self::NCpuMoe => {
                ConfigPath::from_fields(["models", "<model-ref>", "hardware", "n_cpu_moe"])
            }
            Self::TensorSplit => {
                ConfigPath::from_fields(["models", "<model-ref>", "hardware", "tensor_split"])
            }
            Self::Placement => {
                ConfigPath::from_fields(["models", "<model-ref>", "hardware", "placement"])
            }
            Self::Defaults => ConfigPath::from_fields(["defaults"]),
        };
        let support = match self {
            Self::CacheTypeK
            | Self::CacheTypeV
            | Self::FlashAttention
            | Self::CtxSize
            | Self::Batch
            | Self::Ubatch
            | Self::GpuLayers
            | Self::FitTargetMib
            | Self::Mmap
            | Self::Mlock => TuneFieldSupport::Writable,
            Self::Device | Self::Defaults => TuneFieldSupport::PreserveOnly,
            Self::CpuMoe | Self::NCpuMoe | Self::TensorSplit | Self::Placement => {
                TuneFieldSupport::Unsupported
            }
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
