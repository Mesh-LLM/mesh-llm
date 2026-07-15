pub(crate) mod scan_refresh;

use crate::proto::node::{
    OwnerControlApplyConfigRequest, OwnerControlGetConfigRequest, OwnerControlRequest,
    OwnerControlWatchConfigRequest,
};
use std::time::Duration;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OwnedNodeCommandExecutionShape {
    Unary,
    Watch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OwnedNodeCommandDeadline {
    Unary(Duration),
    Scan(Duration),
    Watch,
}

#[derive(Debug)]
pub(crate) enum OwnedNodeCommand {
    GetConfig {
        request_id: u64,
        request: OwnerControlGetConfigRequest,
    },
    WatchConfig {
        request_id: u64,
        request: OwnerControlWatchConfigRequest,
    },
    ApplyConfig {
        request_id: u64,
        request: OwnerControlApplyConfigRequest,
    },
    ScanRefresh {
        request_id: u64,
        request: crate::proto::node::OwnerControlRefreshInventoryRequest,
    },
}

impl OwnedNodeCommand {
    pub(crate) fn decode(request: OwnerControlRequest) -> Option<Self> {
        let request_id = request.request_id;
        if let Some(request) = request.get_config {
            return Some(Self::GetConfig {
                request_id,
                request,
            });
        }
        if let Some(request) = request.watch_config {
            return Some(Self::WatchConfig {
                request_id,
                request,
            });
        }
        if let Some(request) = request.apply_config {
            return Some(Self::ApplyConfig {
                request_id,
                request,
            });
        }
        request.refresh_inventory.map(|request| Self::ScanRefresh {
            request_id,
            request,
        })
    }

    pub(crate) fn request_id(&self) -> u64 {
        match self {
            Self::GetConfig { request_id, .. }
            | Self::WatchConfig { request_id, .. }
            | Self::ApplyConfig { request_id, .. }
            | Self::ScanRefresh { request_id, .. } => *request_id,
        }
    }

    pub(crate) fn requester_node_id(&self) -> &[u8] {
        match self {
            Self::GetConfig { request, .. } => &request.requester_node_id,
            Self::WatchConfig { request, .. } => &request.requester_node_id,
            Self::ApplyConfig { request, .. } => &request.requester_node_id,
            Self::ScanRefresh { request, .. } => &request.requester_node_id,
        }
    }

    pub(crate) fn target_node_id(&self) -> &[u8] {
        match self {
            Self::GetConfig { request, .. } => &request.target_node_id,
            Self::WatchConfig { request, .. } => &request.target_node_id,
            Self::ApplyConfig { request, .. } => &request.target_node_id,
            Self::ScanRefresh { request, .. } => &request.target_node_id,
        }
    }

    pub(crate) fn execution_shape(&self) -> OwnedNodeCommandExecutionShape {
        match self {
            Self::WatchConfig { .. } => OwnedNodeCommandExecutionShape::Watch,
            Self::GetConfig { .. } | Self::ApplyConfig { .. } | Self::ScanRefresh { .. } => {
                OwnedNodeCommandExecutionShape::Unary
            }
        }
    }

    pub(crate) fn deadline(&self) -> OwnedNodeCommandDeadline {
        match self {
            Self::GetConfig { .. } | Self::ApplyConfig { .. } => {
                OwnedNodeCommandDeadline::Unary(Duration::from_secs(5))
            }
            Self::ScanRefresh { .. } => OwnedNodeCommandDeadline::Scan(Duration::from_secs(30)),
            Self::WatchConfig { .. } => OwnedNodeCommandDeadline::Watch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[test]
    fn command_metadata_is_exhaustive_and_shared() {
        let command = OwnedNodeCommand::decode(OwnerControlRequest {
            request_id: 41,
            get_config: None,
            watch_config: None,
            apply_config: None,
            refresh_inventory: Some(crate::proto::node::OwnerControlRefreshInventoryRequest {
                requester_node_id: vec![1],
                target_node_id: vec![2],
            }),
        })
        .expect("typed command");

        assert_eq!(command.request_id(), 41);
        assert_eq!(command.requester_node_id(), [1]);
        assert_eq!(command.target_node_id(), [2]);
        assert_eq!(
            command.execution_shape(),
            OwnedNodeCommandExecutionShape::Unary
        );
        assert_eq!(
            command.deadline(),
            OwnedNodeCommandDeadline::Scan(Duration::from_secs(30))
        );
    }

    #[test]
    fn oversized_command_result_maps_to_control_unavailable() {
        let envelope = crate::proto::node::OwnerControlEnvelope {
            r#gen: crate::protocol::NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: None,
            response: Some(crate::proto::node::OwnerControlResponse {
                request_id: 73,
                get_config: None,
                watch_config: None,
                apply_config: None,
                refresh_inventory: Some(crate::proto::node::OwnerControlRefreshInventoryResponse {
                    snapshot: None,
                    inventory: Some(crate::proto::node::OwnerControlRefreshInventory {
                        entries: vec![crate::proto::node::OwnerControlInventoryEntry {
                            canonical_model_ref: "x"
                                .repeat(crate::protocol::MAX_CONTROL_FRAME_BYTES),
                            display_name: None,
                            total_size_bytes: 0,
                            metadata: None,
                        }],
                        disposition:
                            crate::proto::node::OwnerControlRefreshInventoryDisposition::Executed
                                as i32,
                    }),
                }),
            }),
            error: None,
        };
        assert!(
            envelope.encode_to_vec().len() > crate::protocol::MAX_CONTROL_FRAME_BYTES,
            "fixture must exceed the bound"
        );

        let bounded = super::super::bound_owner_control_envelope(envelope);
        let error = bounded
            .error
            .as_ref()
            .expect("oversized result becomes an error");
        assert_eq!(error.request_id, Some(73));
        assert_eq!(
            crate::proto::node::OwnerControlErrorCode::try_from(error.code),
            Ok(crate::proto::node::OwnerControlErrorCode::ControlUnavailable)
        );
        assert!(bounded.encode_to_vec().len() < crate::protocol::MAX_CONTROL_FRAME_BYTES);
    }
}
