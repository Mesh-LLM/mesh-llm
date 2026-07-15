pub(crate) mod scan_refresh;

use crate::proto::node::{
    OwnerControlApplyConfigRequest, OwnerControlGetConfigRequest, OwnerControlRequest,
    OwnerControlWatchConfigRequest,
};
use std::future::Future;
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

impl OwnedNodeCommandDeadline {
    pub(crate) fn timeout_message(self) -> String {
        match self {
            Self::Unary(duration) => format!(
                "owner-control unary command timed out after {}s",
                duration.as_secs()
            ),
            Self::Scan(duration) => format!(
                "owner-control inventory scan timed out after {}s",
                duration.as_secs()
            ),
            Self::Watch => "owner-control watch commands do not have a unary deadline".to_string(),
        }
    }
}

pub(crate) async fn await_command_deadline<T, F>(
    deadline: OwnedNodeCommandDeadline,
    future: F,
) -> Result<T, OwnedNodeCommandDeadline>
where
    F: Future<Output = T>,
{
    let duration = match deadline {
        OwnedNodeCommandDeadline::Unary(duration) | OwnedNodeCommandDeadline::Scan(duration) => {
            duration
        }
        OwnedNodeCommandDeadline::Watch => return Ok(future.await),
    };
    tokio::time::timeout(duration, future)
        .await
        .map_err(|_| deadline)
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

    #[tokio::test(start_paused = true)]
    async fn slow_scan_within_deadline_completes() {
        let deadline = OwnedNodeCommandDeadline::Scan(Duration::from_secs(30));
        let result = await_command_deadline(deadline, async {
            tokio::time::sleep(Duration::from_secs(29)).await;
            "complete"
        })
        .await;

        assert_eq!(result, Ok("complete"));
    }

    #[tokio::test(start_paused = true)]
    async fn scan_exceeding_deadline_is_cancelled_deterministically() {
        let deadline = OwnedNodeCommandDeadline::Scan(Duration::from_secs(30));
        let result = await_command_deadline(deadline, async {
            tokio::time::sleep(Duration::from_secs(31)).await;
        })
        .await;

        assert_eq!(result, Err(deadline));
        assert_eq!(
            deadline.timeout_message(),
            "owner-control inventory scan timed out after 30s"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn accepted_watch_does_not_inherit_unary_deadline() {
        let result = await_command_deadline(OwnedNodeCommandDeadline::Watch, async {
            tokio::time::sleep(Duration::from_secs(31)).await;
            "still-open"
        })
        .await;

        assert_eq!(result, Ok("still-open"));
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
