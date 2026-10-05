use super::MeshGuardrailMode;
use crate::{api, network::nostr};

pub(super) fn single_quote_shell_arg(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

pub(super) fn mesh_guardrail_mode_to_openai(
    mode: MeshGuardrailMode,
) -> skippy_inference_api::GuardrailMode {
    match mode {
        MeshGuardrailMode::Disabled => skippy_inference_api::GuardrailMode::Disabled,
        MeshGuardrailMode::Metrics => skippy_inference_api::GuardrailMode::MetricsOnly,
        MeshGuardrailMode::Enforce => skippy_inference_api::GuardrailMode::Enforce,
    }
}

pub(super) fn current_time_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub(super) fn publication_state_from_update(
    update: nostr::PublishStateUpdate,
) -> api::PublicationState {
    match update {
        nostr::PublishStateUpdate::Public => api::PublicationState::Public,
        nostr::PublishStateUpdate::PublishFailed => api::PublicationState::PublishFailed,
    }
}
