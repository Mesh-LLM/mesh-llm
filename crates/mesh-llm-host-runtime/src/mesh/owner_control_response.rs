use mesh_llm_protocol::proto::node;
use mesh_llm_protocol::protocol::NODE_PROTOCOL_GENERATION;

use crate::runtime::config_state::ConfigApplyMode;

pub(super) fn apply_response_envelope(
    request_id: u64,
    apply_config: node::OwnerControlApplyConfigResponse,
) -> node::OwnerControlEnvelope {
    node::OwnerControlEnvelope {
        r#gen: NODE_PROTOCOL_GENERATION,
        handshake: None,
        request: None,
        response: Some(node::OwnerControlResponse {
            request_id,
            get_config: None,
            watch_config: None,
            apply_config: Some(apply_config),
            refresh_inventory: None,
        }),
        error: None,
    }
}

pub(super) fn config_diagnostics_to_proto(
    diagnostics: &[mesh_llm_config::ConfigDiagnostic],
) -> Vec<node::ConfigDiagnostic> {
    diagnostics
        .iter()
        .map(crate::protocol::convert::config_diagnostic_to_proto)
        .collect()
}

pub(super) fn proto_apply_mode(apply_mode: ConfigApplyMode) -> i32 {
    match apply_mode {
        ConfigApplyMode::Staged => node::ConfigApplyMode::Staged as i32,
        ConfigApplyMode::Noop => node::ConfigApplyMode::Noop as i32,
    }
}
