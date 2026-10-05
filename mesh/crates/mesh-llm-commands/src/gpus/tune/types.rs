use mesh_llm_config::ConfigPath;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct TuneTarget {
    pub requested: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_model_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived_profile: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TuneApplyMode {
    Review,
    ApplyMissing,
    ReplaceExisting,
    LaunchArgs,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash, strum::EnumIter)]
#[serde(rename_all = "snake_case")]
pub enum TuneField {
    CacheTypeK,
    CacheTypeV,
    FlashAttention,
    CtxSize,
    Batch,
    Ubatch,
    GpuLayers,
    FitTargetMib,
    Device,
    Mmap,
    Mlock,
    CpuMoe,
    NCpuMoe,
    TensorSplit,
    Placement,
    Defaults,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TuneFieldSupport {
    Writable,
    PreserveOnly,
    ReportOnly,
    Unsupported,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct TuneFieldSpec {
    pub field: TuneField,
    pub config_path: ConfigPath,
    pub support: TuneFieldSupport,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TuneKvCacheType {
    F16,
    Q8_0,
    Q4_0,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TuneFlashAttentionValue {
    Enabled,
    Disabled,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TuneGpuLayersValue {
    All,
    Count(u32),
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TuneBoolOrAutoValue {
    Enabled,
    Disabled,
    Auto,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum TuneRecommendedValue {
    KvCacheType(TuneKvCacheType),
    FlashAttention(TuneFlashAttentionValue),
    ContextSize(u32),
    Batch(u32),
    Ubatch(u32),
    GpuLayers(TuneGpuLayersValue),
    FitTargetMib(u64),
    Device(String),
    Bool(bool),
    BoolOrAuto(TuneBoolOrAutoValue),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct TuneRecommendation {
    pub field: TuneField,
    pub value: TuneRecommendedValue,
    pub rationale: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum TuneConfigEdit {
    SetModelFitCacheTypeK(TuneKvCacheType),
    SetModelFitCacheTypeV(TuneKvCacheType),
    SetModelFitFlashAttention(TuneFlashAttentionValue),
    SetModelFitCtxSize(u32),
    SetModelFitBatch(u32),
    SetModelFitUbatch(u32),
    SetHardwareGpuLayers(TuneGpuLayersValue),
    SetHardwareFitTargetMib(u64),
    SetHardwareMmap(TuneBoolOrAutoValue),
    SetHardwareMlock(bool),
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TuneDiagnosticSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TuneDiagnosticCode {
    PreservedExistingValue,
    ReportOnlyField,
    UnsupportedField,
    MissingConfiguredDevice,
    InvalidExistingValue,
    MlockUnavailable,
    InsufficientMemory,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct TuneDiagnostic {
    pub severity: TuneDiagnosticSeverity,
    pub code: TuneDiagnosticCode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field: Option<TuneField>,
    pub message: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TuneFieldStatus {
    Applied {
        recommendation: TuneRecommendation,
        edit: TuneConfigEdit,
    },
    Preserved {
        field: TuneField,
        reason: String,
    },
    ReportOnly {
        recommendation: TuneRecommendation,
        reason: String,
    },
    Unsupported {
        field: TuneField,
        reason: String,
    },
    Error {
        field: TuneField,
        diagnostic: TuneDiagnostic,
    },
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TunePlanSummary {
    pub applied: usize,
    pub preserved: usize,
    pub report_only: usize,
    pub unsupported: usize,
    pub error: usize,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TuneResultSummary {
    pub total_targets: usize,
    pub ready_targets: usize,
    pub failed_targets: usize,
    pub written_targets: usize,
    pub skipped_targets: usize,
    pub fields: TunePlanSummary,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct TunePlan {
    pub target: TuneTarget,
    pub apply_mode: TuneApplyMode,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub field_statuses: Vec<TuneFieldStatus>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<TuneDiagnostic>,
}
