//! Derive local serving status from native processes and available external inference.

use super::snapshots::RuntimeStatusDerivation;
use crate::api::status::NodeState;

pub(super) struct RuntimeStatusDerivationInput<'a> {
    pub(super) is_client: bool,
    pub(super) is_host: bool,
    pub(super) llama_ready: bool,
    pub(super) external_inference_ready: bool,
    pub(super) local_processes: &'a [crate::api::RuntimeProcessPayload],
    pub(super) hosted_models: &'a [String],
    pub(super) serving_models: &'a [String],
    pub(super) model_name: &'a str,
    pub(super) api_port: u16,
}

pub(super) fn derive_runtime_status(
    input: RuntimeStatusDerivationInput<'_>,
) -> RuntimeStatusDerivation {
    let has_local_processes = !input.local_processes.is_empty();
    let effective_llama_ready = input.llama_ready || has_local_processes;
    let effective_is_host = input.is_host || has_local_processes || input.external_inference_ready;
    let serving_ready = effective_llama_ready || input.external_inference_ready;
    let display_model_name = input
        .local_processes
        .first()
        .map(|process| process.name.clone())
        .or_else(|| input.hosted_models.first().cloned())
        .or_else(|| input.serving_models.first().cloned())
        .unwrap_or_else(|| input.model_name.to_string());
    let has_local_worker_activity = has_local_processes || !input.hosted_models.is_empty();
    let node_state = derive_local_node_state(
        input.is_client,
        effective_is_host,
        serving_ready,
        has_local_worker_activity,
        &display_model_name,
    );
    let launch_pi = if serving_ready {
        Some(format!(
            "mesh-llm pi --host 127.0.0.1:{} --model {}",
            input.api_port,
            single_quote_shell_arg(&display_model_name)
        ))
    } else {
        None
    };
    let launch_goose = if serving_ready {
        let api_port = input.api_port;
        let model = single_quote_shell_arg(&display_model_name);
        Some(format!(
            "GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:{api_port} OPENAI_API_KEY=mesh GOOSE_MODEL={model} goose session"
        ))
    } else {
        None
    };

    RuntimeStatusDerivation {
        effective_is_host,
        effective_llama_ready,
        display_model_name,
        node_state,
        node_status: node_state.node_status_alias().to_string(),
        launch_pi,
        launch_goose,
    }
}

fn single_quote_shell_arg(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn derive_local_node_state(
    is_client: bool,
    effective_is_host: bool,
    serving_ready: bool,
    has_local_worker_activity: bool,
    display_model_name: &str,
) -> NodeState {
    let has_declared_local_serving_work =
        (effective_is_host || has_local_worker_activity) && !display_model_name.trim().is_empty();

    if is_client {
        NodeState::Client
    } else if serving_ready && has_declared_local_serving_work {
        NodeState::Serving
    } else if has_declared_local_serving_work {
        NodeState::Loading
    } else {
        NodeState::Standby
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn derive(
        is_client: bool,
        native_ready: bool,
        external_ready: bool,
    ) -> RuntimeStatusDerivation {
        derive_runtime_status(RuntimeStatusDerivationInput {
            is_client,
            is_host: false,
            llama_ready: native_ready,
            external_inference_ready: external_ready,
            local_processes: &[],
            hosted_models: &["provider-model".into()],
            serving_models: &["provider-model".into()],
            model_name: "",
            api_port: 9337,
        })
    }

    #[test]
    fn available_external_provider_serves_without_native_readiness() {
        let status = derive(false, false, true);
        assert_eq!(status.node_state, NodeState::Serving);
        assert!(status.effective_is_host);
        assert!(!status.effective_llama_ready);
        assert_eq!(status.display_model_name, "provider-model");
        assert!(status.launch_pi.is_some());
        assert!(status.launch_goose.is_some());
    }

    #[test]
    fn declared_native_model_stays_loading_until_ready() {
        let status = derive_runtime_status(RuntimeStatusDerivationInput {
            is_client: false,
            is_host: true,
            llama_ready: false,
            external_inference_ready: false,
            local_processes: &[],
            hosted_models: &[],
            serving_models: &[],
            model_name: "native-model",
            api_port: 9337,
        });
        assert_eq!(status.node_state, NodeState::Loading);
        assert!(!status.effective_llama_ready);
        assert!(status.launch_pi.is_none());
    }

    #[test]
    fn mixed_providers_preserve_native_readiness() {
        let status = derive(false, true, true);
        assert_eq!(status.node_state, NodeState::Serving);
        assert!(status.effective_llama_ready);
    }

    #[test]
    fn unavailable_external_provider_cannot_make_declared_models_ready() {
        let status = derive(false, false, false);
        assert_eq!(status.node_state, NodeState::Loading);
        assert!(!status.effective_llama_ready);
        assert!(status.launch_goose.is_none());
    }

    #[test]
    fn client_role_remains_client_with_available_provider() {
        let status = derive(true, false, true);
        assert_eq!(status.node_state, NodeState::Client);
        assert!(!status.effective_llama_ready);
    }

    #[test]
    fn provider_launch_commands_quote_the_exact_model_id() {
        let models = ["provider's model$(echo unsafe)".to_string()];
        let status = derive_runtime_status(RuntimeStatusDerivationInput {
            is_client: false,
            is_host: false,
            llama_ready: false,
            external_inference_ready: true,
            local_processes: &[],
            hosted_models: &models,
            serving_models: &models,
            model_name: "",
            api_port: 9337,
        });
        assert_eq!(
            status.launch_goose.as_deref(),
            Some(
                "GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:9337 OPENAI_API_KEY=mesh GOOSE_MODEL='provider'\\''s model$(echo unsafe)' goose session"
            )
        );
        assert_eq!(
            status.launch_pi.as_deref(),
            Some("mesh-llm pi --host 127.0.0.1:9337 --model 'provider'\\''s model$(echo unsafe)'")
        );
    }

    #[test]
    fn empty_node_remains_standby_without_a_provider() {
        let status = derive_runtime_status(RuntimeStatusDerivationInput {
            is_client: false,
            is_host: false,
            llama_ready: false,
            external_inference_ready: false,
            local_processes: &[],
            hosted_models: &[],
            serving_models: &[],
            model_name: "",
            api_port: 9337,
        });
        assert_eq!(status.node_state, NodeState::Standby);
        assert!(!status.effective_is_host);
    }
}
