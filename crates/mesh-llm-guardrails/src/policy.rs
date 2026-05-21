use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GuardrailMode {
    #[default]
    Disabled,
    MetricsOnly,
    Enforce,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamingGuardrailMode {
    #[default]
    PassThrough,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RetryExhaustionMode {
    #[default]
    Error,
    PassLastText,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GuardrailPolicy {
    pub mode: GuardrailMode,
    pub streaming_mode: StreamingGuardrailMode,
    pub max_tool_retries: u8,
    pub max_structured_retries: u8,
    pub retry_exhaustion_mode: RetryExhaustionMode,
    pub apply_to_all_models: bool,
    pub small_param_threshold_b: f32,
    pub reserved_tool_prefix: String,
}

impl GuardrailPolicy {
    pub fn small_models_only(&self) -> bool {
        !self.apply_to_all_models
    }
}

impl Default for GuardrailPolicy {
    fn default() -> Self {
        Self {
            mode: GuardrailMode::Disabled,
            streaming_mode: StreamingGuardrailMode::PassThrough,
            max_tool_retries: 1,
            max_structured_retries: 2,
            retry_exhaustion_mode: RetryExhaustionMode::Error,
            apply_to_all_models: false,
            small_param_threshold_b: 9.0,
            reserved_tool_prefix: "_mesh_".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GuardrailPolicyHandle {
    inner: Arc<RwLock<GuardrailPolicy>>,
}

impl GuardrailPolicyHandle {
    pub fn new(policy: GuardrailPolicy) -> Self {
        Self {
            inner: Arc::new(RwLock::new(policy)),
        }
    }

    pub fn snapshot(&self) -> GuardrailPolicy {
        self.inner
            .read()
            .expect("guardrail policy lock poisoned")
            .clone()
    }

    pub fn update(&self, policy: GuardrailPolicy) {
        *self.inner.write().expect("guardrail policy lock poisoned") = policy;
    }

    pub fn set_mode(&self, mode: GuardrailMode) {
        self.inner
            .write()
            .expect("guardrail policy lock poisoned")
            .mode = mode;
    }
}

impl Default for GuardrailPolicyHandle {
    fn default() -> Self {
        Self::new(GuardrailPolicy::default())
    }
}

impl From<GuardrailPolicy> for GuardrailPolicyHandle {
    fn from(policy: GuardrailPolicy) -> Self {
        Self::new(policy)
    }
}

impl PartialEq for GuardrailPolicyHandle {
    fn eq(&self, other: &Self) -> bool {
        self.snapshot() == other.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guardrail_policy_default_is_conservative() {
        let policy = GuardrailPolicy::default();

        assert_eq!(policy.mode, GuardrailMode::Disabled);
        assert_eq!(policy.streaming_mode, StreamingGuardrailMode::PassThrough);
        assert_eq!(policy.max_tool_retries, 1);
        assert_eq!(policy.max_structured_retries, 2);
        assert_eq!(policy.retry_exhaustion_mode, RetryExhaustionMode::Error);
        assert!(policy.small_models_only());
        assert_eq!(policy.small_param_threshold_b, 9.0);
        assert_eq!(policy.reserved_tool_prefix, "_mesh_");
    }

    #[test]
    fn guardrail_policy_handle_shares_live_mode_across_clones() {
        let handle = GuardrailPolicyHandle::default();
        let clone = handle.clone();

        handle.set_mode(GuardrailMode::MetricsOnly);

        assert_eq!(clone.snapshot().mode, GuardrailMode::MetricsOnly);
    }

    #[test]
    fn guardrail_policy_handle_snapshot_is_stable_after_update() {
        let handle = GuardrailPolicyHandle::default();
        let snapshot = handle.snapshot();

        handle.set_mode(GuardrailMode::Enforce);

        assert_eq!(snapshot.mode, GuardrailMode::Disabled);
        assert_eq!(handle.snapshot().mode, GuardrailMode::Enforce);
    }
}
