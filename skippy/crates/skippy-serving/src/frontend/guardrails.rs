/// Compatibility behavior selected by the embedding application.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum InferenceGuardrailsMode {
    #[default]
    Disabled,
    Metrics,
    Enforce,
}
use serde::Serialize;
use skippy_inference_api::CompactingInferenceBackend;
use skippy_inference_api::CompactionConfig;
use skippy_inference_api::GuardedInferenceBackend;
use skippy_inference_api::GuardrailMode;
use skippy_inference_api::GuardrailPolicy;
use skippy_inference_api::GuardrailPolicyHandle;
use skippy_inference_api::GuardrailTelemetrySink;
use skippy_inference_api::InferenceBackend;
use skippy_inference_api::RetryExhaustionMode;
use skippy_inference_api::StreamingGuardrailMode;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InferenceGuardrailsTarget {
    Skippy,
}

impl InferenceGuardrailsTarget {
    const fn as_status_label(self) -> &'static str {
        match self {
            Self::Skippy => "skippy",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InferenceGuardrailsConfig {
    pub target: InferenceGuardrailsTarget,
    pub policy: GuardrailPolicyHandle,
    pub compaction: Option<CompactionConfig>,
}

impl InferenceGuardrailsConfig {
    /// Shared serving policy: automatic chat compaction and operator-selected guardrails.
    pub fn with_policy(policy: GuardrailPolicyHandle) -> Self {
        Self {
            target: InferenceGuardrailsTarget::Skippy,
            policy,
            compaction: Some(CompactionConfig {
                enabled: true,
                ..CompactionConfig::default()
            }),
        }
    }

    pub fn disabled_for_skippy() -> Self {
        Self::with_policy(GuardrailPolicyHandle::default())
    }

    pub fn compatibility_for_skippy() -> Self {
        Self::with_policy(
            GuardrailPolicy {
                mode: GuardrailMode::MetricsOnly,
                apply_to_all_models: true,
                retry_exhaustion_mode: RetryExhaustionMode::PassLastText,
                ..GuardrailPolicy::default()
            }
            .into(),
        )
    }

    pub fn for_standalone_mode(mode: InferenceGuardrailsMode) -> Self {
        match mode {
            InferenceGuardrailsMode::Disabled => Self::disabled_for_skippy(),
            InferenceGuardrailsMode::Metrics => Self::compatibility_for_skippy(),
            InferenceGuardrailsMode::Enforce => Self::with_policy(
                GuardrailPolicy {
                    mode: GuardrailMode::Enforce,
                    apply_to_all_models: true,
                    ..GuardrailPolicy::default()
                }
                .into(),
            ),
        }
    }

    pub fn status(&self) -> InferenceGuardrailsStatus {
        let policy = self.policy.snapshot();
        InferenceGuardrailsStatus {
            mode: guardrail_mode_label(policy.mode),
            target: self.target.as_status_label(),
            streaming: streaming_mode_label(policy.streaming_mode),
            retry_exhaustion: retry_exhaustion_label(&policy),
            small_model_policy: small_model_policy_label(&policy),
            small_param_threshold_b: policy.small_param_threshold_b,
            max_tool_retries: policy.max_tool_retries,
            max_structured_retries: policy.max_structured_retries,
        }
    }

    fn should_wrap_guardrail_backend(&self) -> bool {
        matches!(self.target, InferenceGuardrailsTarget::Skippy)
    }

    #[cfg(test)]
    pub(super) fn wrap_backend(
        &self,
        backend: Arc<dyn InferenceBackend>,
    ) -> Arc<dyn InferenceBackend> {
        self.wrap_backend_with_context_limit(backend, None)
    }

    pub(super) fn wrap_backend_with_context_limit(
        &self,
        backend: Arc<dyn InferenceBackend>,
        context_limit_tokens: Option<usize>,
    ) -> Arc<dyn InferenceBackend> {
        self.wrap_backend_with_telemetry(backend, context_limit_tokens, None)
    }

    /// Apply common compaction and guardrails with an optional caller-owned observer.
    pub fn wrap_backend_with_telemetry(
        &self,
        backend: Arc<dyn InferenceBackend>,
        context_limit_tokens: Option<usize>,
        telemetry: Option<Arc<dyn GuardrailTelemetrySink>>,
    ) -> Arc<dyn InferenceBackend> {
        let backend = self.wrap_compacting_backend(backend, context_limit_tokens);
        if !self.should_wrap_guardrail_backend() {
            return backend;
        }
        let guarded = GuardedInferenceBackend::with_policy_handle(backend, self.policy.clone());
        match telemetry {
            Some(telemetry) => Arc::new(guarded.with_telemetry(telemetry)),
            None => Arc::new(guarded),
        }
    }

    fn wrap_compacting_backend(
        &self,
        backend: Arc<dyn InferenceBackend>,
        context_limit_tokens: Option<usize>,
    ) -> Arc<dyn InferenceBackend> {
        let Some(mut compaction) = self.compaction else {
            return backend;
        };
        if compaction.context_limit_tokens.is_none() {
            compaction.context_limit_tokens = context_limit_tokens;
        }
        Arc::new(CompactingInferenceBackend::new(backend, compaction))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct InferenceGuardrailsStatus {
    pub mode: &'static str,
    pub target: &'static str,
    pub streaming: &'static str,
    pub retry_exhaustion: &'static str,
    pub small_model_policy: &'static str,
    pub small_param_threshold_b: f32,
    pub max_tool_retries: u8,
    pub max_structured_retries: u8,
}

fn guardrail_mode_label(mode: GuardrailMode) -> &'static str {
    match mode {
        GuardrailMode::Disabled => "disabled",
        GuardrailMode::MetricsOnly => "metrics",
        GuardrailMode::Enforce => "enforce",
    }
}

fn streaming_mode_label(mode: StreamingGuardrailMode) -> &'static str {
    match mode {
        StreamingGuardrailMode::PassThrough => "pass_through",
    }
}

fn retry_exhaustion_label(policy: &GuardrailPolicy) -> &'static str {
    match policy.retry_exhaustion_mode {
        RetryExhaustionMode::Error => "error",
        RetryExhaustionMode::PassLastText => "pass_last_text",
    }
}

fn small_model_policy_label(policy: &GuardrailPolicy) -> &'static str {
    if policy.apply_to_all_models {
        "all"
    } else {
        "small_models_only"
    }
}
