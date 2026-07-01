#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TuneTargetFailure {
    pub requested_input: String,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TuneTargetStatus {
    Ready,
    Written,
    Skipped,
    Failed,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TuneRenderedSettingStatus {
    Applied,
    Preserved,
    ReportOnly,
    Unsupported,
    Error,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(crate) struct TuneRenderedSetting {
    pub field: TuneField,
    pub support: TuneFieldSupport,
    pub status: TuneRenderedSettingStatus,
    pub config_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<TuneRecommendedValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic: Option<TuneDiagnostic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edit: Option<TuneConfigEdit>,
    pub applied_write: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(crate) struct TuneLaunchSetting {
    pub config_path: String,
    pub field: TuneField,
    pub value: TuneRecommendedValue,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(crate) struct TuneLaunchPreview {
    pub argv: Vec<String>,
    pub shell: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub config_settings: Vec<TuneLaunchSetting>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub report_only: Vec<TuneRenderedSetting>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unsupported: Vec<TuneRenderedSetting>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(crate) struct TuneTargetReport {
    pub target: TuneTarget,
    pub status: TuneTargetStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical_model_ref: Option<String>,
    pub selection: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field_summary: Option<TunePlanSummary>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<TuneDiagnostic>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub settings: Vec<TuneRenderedSetting>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub config_edits: Vec<TuneRenderedSetting>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub launch: Option<TuneLaunchPreview>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(crate) struct TuneRunReport {
    pub command: &'static str,
    pub apply_mode: TuneApplyMode,
    pub summary: TuneResultSummary,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub global_blockers: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<TuneTargetReport>,
}
