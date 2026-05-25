use super::policy::GuardrailMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GuardrailTelemetryDecision {
    Eligible,
    Bypassed,
    Unsupported,
    Rejected,
}

impl GuardrailTelemetryDecision {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Eligible => "eligible",
            Self::Bypassed => "bypassed",
            Self::Unsupported => "unsupported",
            Self::Rejected => "rejected",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GuardrailTelemetryBypassReason {
    Disabled,
    Streaming,
    NoContract,
    UnsupportedSurface,
    ReservedCollision,
    MixedToolsStructured,
    AfterToolResult,
}

impl GuardrailTelemetryBypassReason {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Streaming => "streaming",
            Self::NoContract => "no_contract",
            Self::UnsupportedSurface => "unsupported_surface",
            Self::ReservedCollision => "reserved_collision",
            Self::MixedToolsStructured => "mixed_tools_structured",
            Self::AfterToolResult => "after_tool_result",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GuardrailTelemetryOutcome {
    PassThrough,
    Valid,
    Rescued,
    Retried,
    Failed,
    MetricsOnlyFailure,
}

impl GuardrailTelemetryOutcome {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::PassThrough => "pass_through",
            Self::Valid => "valid",
            Self::Rescued => "rescued",
            Self::Retried => "retried",
            Self::Failed => "failed",
            Self::MetricsOnlyFailure => "metrics_only_failure",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GuardrailTelemetryParserStage {
    None,
    JsonExact,
    JsonFenced,
    JsonSubstring,
}

impl GuardrailTelemetryParserStage {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::JsonExact => "json_exact",
            Self::JsonFenced => "json_fenced",
            Self::JsonSubstring => "json_substring",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GuardrailTelemetryAttemptBucket {
    One,
    Two,
    ThreePlus,
}

impl GuardrailTelemetryAttemptBucket {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::One => "1",
            Self::Two => "2",
            Self::ThreePlus => "3_plus",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GuardrailTelemetryContract {
    Tools,
    Structured,
}

impl GuardrailTelemetryContract {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Tools => "tools",
            Self::Structured => "structured",
        }
    }
}

pub trait GuardrailTelemetrySink: Send + Sync + 'static {
    fn record_decision(
        &self,
        mode: GuardrailMode,
        contract: Option<&'static str>,
        decision: &'static str,
        bypass_reason: Option<&'static str>,
    );

    fn record_outcome(
        &self,
        mode: GuardrailMode,
        contract: Option<&'static str>,
        outcome: &'static str,
        parser_stage: Option<&'static str>,
        attempt_bucket: Option<&'static str>,
    );
}
