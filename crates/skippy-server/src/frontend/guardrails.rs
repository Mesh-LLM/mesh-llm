use crate::cli::OpenAiGuardrailsCliMode;
use openai_frontend::CompactingOpenAiBackend;
use openai_frontend::CompactionConfig;
use openai_frontend::GuardedOpenAiBackend;
use openai_frontend::GuardrailMode;
use openai_frontend::GuardrailPolicy;
use openai_frontend::GuardrailPolicyHandle;
use openai_frontend::OpenAiBackend;
use openai_frontend::RetryExhaustionMode;
use openai_frontend::StreamingGuardrailMode;
use serde::Serialize;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenAiGuardrailsTarget {
    Skippy,
}

impl OpenAiGuardrailsTarget {
    const fn as_status_label(self) -> &'static str {
        match self {
            Self::Skippy => "skippy",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpenAiGuardrailsConfig {
    pub target: OpenAiGuardrailsTarget,
    pub policy: GuardrailPolicyHandle,
    pub compaction: Option<CompactionConfig>,
}

impl OpenAiGuardrailsConfig {
    pub fn disabled_for_skippy() -> Self {
        Self {
            target: OpenAiGuardrailsTarget::Skippy,
            policy: GuardrailPolicyHandle::default(),
            compaction: None,
        }
    }

    pub fn compatibility_for_skippy() -> Self {
        Self {
            target: OpenAiGuardrailsTarget::Skippy,
            policy: GuardrailPolicy {
                mode: GuardrailMode::MetricsOnly,
                apply_to_all_models: true,
                retry_exhaustion_mode: RetryExhaustionMode::PassLastText,
                ..GuardrailPolicy::default()
            }
            .into(),
            compaction: None,
        }
    }

    pub fn for_standalone_mode(mode: OpenAiGuardrailsCliMode) -> Self {
        match mode {
            OpenAiGuardrailsCliMode::Disabled => Self::disabled_for_skippy(),
            OpenAiGuardrailsCliMode::Metrics => Self::compatibility_for_skippy(),
            OpenAiGuardrailsCliMode::Enforce => Self {
                target: OpenAiGuardrailsTarget::Skippy,
                policy: GuardrailPolicy {
                    mode: GuardrailMode::Enforce,
                    apply_to_all_models: true,
                    ..GuardrailPolicy::default()
                }
                .into(),
                compaction: None,
            },
        }
    }

    pub fn status(&self) -> OpenAiGuardrailsStatus {
        let policy = self.policy.snapshot();
        OpenAiGuardrailsStatus {
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
        matches!(self.target, OpenAiGuardrailsTarget::Skippy)
    }

    #[cfg(test)]
    pub(super) fn wrap_backend(&self, backend: Arc<dyn OpenAiBackend>) -> Arc<dyn OpenAiBackend> {
        self.wrap_backend_with_context_limit(backend, None)
    }

    pub(super) fn wrap_backend_with_context_limit(
        &self,
        backend: Arc<dyn OpenAiBackend>,
        context_limit_tokens: Option<usize>,
    ) -> Arc<dyn OpenAiBackend> {
        let backend = self.wrap_compacting_backend(backend, context_limit_tokens);
        if self.should_wrap_guardrail_backend() {
            Arc::new(GuardedOpenAiBackend::with_policy_handle(
                backend,
                self.policy.clone(),
            ))
        } else {
            backend
        }
    }

    fn wrap_compacting_backend(
        &self,
        backend: Arc<dyn OpenAiBackend>,
        context_limit_tokens: Option<usize>,
    ) -> Arc<dyn OpenAiBackend> {
        let Some(mut compaction) = self.compaction else {
            return backend;
        };
        if compaction.context_limit_tokens.is_none() {
            compaction.context_limit_tokens = context_limit_tokens;
        }
        Arc::new(CompactingOpenAiBackend::new(backend, compaction))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAiGuardrailsStatus {
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
