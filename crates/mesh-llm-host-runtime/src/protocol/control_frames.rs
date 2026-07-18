use anyhow::Result;

use super::MAX_CONTROL_FRAME_BYTES;
use super::NODE_PROTOCOL_GENERATION;

#[derive(Debug, PartialEq)]
pub(crate) enum ControlFrameError {
    OversizeFrame {
        size: usize,
    },
    BadGeneration {
        got: u32,
    },
    InvalidEndpointId {
        got: usize,
    },
    InvalidSenderId {
        got: usize,
    },
    MissingDirectPathAddress,
    MissingHttpPort,
    MissingControlOwnerId,
    InvalidConfigHashLength {
        got: usize,
    },
    InvalidSubprotocol,
    InvalidPublicKeyLength {
        got: usize,
    },
    MissingSignature,
    InvalidSignatureLength {
        got: usize,
    },
    MissingConfig,
    MissingControlEnvelope,
    MissingControlCommand,
    MissingControlResult,
    MissingControlOwnership,
    MissingRequestId,
    InvalidOwnerControlErrorCode {
        got: i32,
    },
    InvalidInventoryDisposition {
        got: i32,
    },
    MissingInventoryModelRef,
    InvalidInventoryOrder,
    #[cfg(test)]
    DecodeError(String),
    #[cfg(test)]
    WrongStreamType {
        expected: u8,
        got: u8,
    },
    ForgedSender,
}

impl std::fmt::Display for ControlFrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ControlFrameError::OversizeFrame { size } => write!(
                f,
                "control frame too large: {} bytes (max {})",
                size, MAX_CONTROL_FRAME_BYTES
            ),
            ControlFrameError::BadGeneration { got } => write!(
                f,
                "bad protocol generation: expected {}, got {}",
                NODE_PROTOCOL_GENERATION, got
            ),
            ControlFrameError::InvalidEndpointId { got } => {
                write!(f, "invalid endpoint_id length: expected 32, got {}", got)
            }
            ControlFrameError::InvalidSenderId { got } => {
                write!(f, "invalid sender_id length: expected 32, got {}", got)
            }
            ControlFrameError::MissingDirectPathAddress => {
                write!(f, "direct path request missing endpoint address")
            }
            ControlFrameError::MissingHttpPort => {
                write!(f, "HOST-role peer annotation missing http_port")
            }
            ControlFrameError::MissingControlOwnerId => {
                write!(f, "owner control handshake missing owner_id")
            }
            ControlFrameError::InvalidConfigHashLength { got } => {
                write!(f, "invalid config_hash length: expected 32, got {}", got)
            }
            ControlFrameError::InvalidSubprotocol => {
                write!(f, "subprotocol entries require a non-empty name and major")
            }
            ControlFrameError::InvalidPublicKeyLength { got } => {
                write!(f, "invalid public key length: expected 32, got {}", got)
            }
            ControlFrameError::MissingSignature => write!(f, "config push missing signature"),
            ControlFrameError::InvalidSignatureLength { got } => {
                write!(f, "invalid signature length: expected 64, got {got}")
            }
            ControlFrameError::MissingConfig => {
                write!(f, "config field is required but missing")
            }
            ControlFrameError::MissingControlEnvelope => {
                write!(f, "owner control envelope requires exactly one payload")
            }
            ControlFrameError::MissingControlCommand => {
                write!(
                    f,
                    "owner control request requires exactly one command variant"
                )
            }
            ControlFrameError::MissingControlResult => {
                write!(
                    f,
                    "owner control response requires exactly one result variant"
                )
            }
            ControlFrameError::MissingControlOwnership => {
                write!(f, "owner control handshake missing ownership attestation")
            }
            ControlFrameError::MissingRequestId => {
                write!(f, "owner control request_id must be non-zero")
            }
            ControlFrameError::InvalidOwnerControlErrorCode { got } => {
                write!(f, "invalid owner control error code: {got}")
            }
            ControlFrameError::InvalidInventoryDisposition { got } => {
                write!(f, "invalid inventory scan disposition: {got}")
            }
            ControlFrameError::MissingInventoryModelRef => {
                write!(f, "inventory entry requires a canonical model ref")
            }
            ControlFrameError::InvalidInventoryOrder => write!(
                f,
                "inventory entries must be strictly sorted by canonical model ref"
            ),
            #[cfg(test)]
            ControlFrameError::DecodeError(msg) => write!(f, "protobuf decode error: {}", msg),
            #[cfg(test)]
            ControlFrameError::WrongStreamType { expected, got } => write!(
                f,
                "wrong stream type: expected {:#04x}, got {:#04x}",
                expected, got
            ),
            ControlFrameError::ForgedSender => {
                write!(f, "frame peer_id does not match QUIC connection identity")
            }
        }
    }
}

impl std::error::Error for ControlFrameError {}

pub(crate) trait ValidateControlFrame: prost::Message + Default + Sized {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::GossipFrame {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.r#gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.r#gen });
        }
        if self.sender_id.len() != 32 {
            return Err(ControlFrameError::InvalidSenderId {
                got: self.sender_id.len(),
            });
        }
        for pa in &self.peers {
            validate_peer_announcement(pa)?;
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::TunnelMap {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.owner_peer_id.len() != 32 {
            return Err(ControlFrameError::InvalidEndpointId {
                got: self.owner_peer_id.len(),
            });
        }
        for entry in &self.entries {
            if entry.target_peer_id.len() != 32 {
                return Err(ControlFrameError::InvalidEndpointId {
                    got: entry.target_peer_id.len(),
                });
            }
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::RouteTableRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.r#gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.r#gen });
        }
        if !self.requester_id.is_empty() && self.requester_id.len() != 32 {
            return Err(ControlFrameError::InvalidEndpointId {
                got: self.requester_id.len(),
            });
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::RouteTable {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.r#gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.r#gen });
        }
        for entry in &self.entries {
            if entry.endpoint_id.len() != 32 {
                return Err(ControlFrameError::InvalidEndpointId {
                    got: entry.endpoint_id.len(),
                });
            }
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::PeerDown {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.r#gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.r#gen });
        }
        if self.peer_id.len() != 32 {
            return Err(ControlFrameError::InvalidEndpointId {
                got: self.peer_id.len(),
            });
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::PeerLeaving {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.r#gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.r#gen });
        }
        if self.peer_id.len() != 32 {
            return Err(ControlFrameError::InvalidEndpointId {
                got: self.peer_id.len(),
            });
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::DirectPathRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.r#gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.r#gen });
        }
        if self.requester_id.len() != 32 {
            return Err(ControlFrameError::InvalidEndpointId {
                got: self.requester_id.len(),
            });
        }
        if self.serialized_addr.is_empty() {
            return Err(ControlFrameError::MissingDirectPathAddress);
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlEnvelope {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.r#gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.r#gen });
        }
        let payloads = [
            self.handshake.is_some(),
            self.request.is_some(),
            self.response.is_some(),
            self.error.is_some(),
        ];
        if payloads.into_iter().filter(|present| *present).count() != 1 {
            return Err(ControlFrameError::MissingControlEnvelope);
        }
        if let Some(handshake) = &self.handshake {
            handshake.validate_frame()?;
        }
        if let Some(request) = &self.request {
            request.validate_frame()?;
        }
        if let Some(response) = &self.response {
            response.validate_frame()?;
        }
        if let Some(error) = &self.error {
            error.validate_frame()?;
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlHandshake {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        let ownership = self
            .ownership
            .as_ref()
            .ok_or(ControlFrameError::MissingControlOwnership)?;
        if ownership.owner_id.trim().is_empty() {
            return Err(ControlFrameError::MissingControlOwnerId);
        }
        validate_public_key_length(ownership.owner_sign_public_key.len())?;
        validate_endpoint_id_length(ownership.node_endpoint_id.len())?;
        if ownership.signature.is_empty() {
            return Err(ControlFrameError::MissingSignature);
        }
        if ownership.signature.len() != 64 {
            return Err(ControlFrameError::InvalidSignatureLength {
                got: ownership.signature.len(),
            });
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.request_id == 0 {
            return Err(ControlFrameError::MissingRequestId);
        }
        let commands = [
            self.get_config.is_some(),
            self.watch_config.is_some(),
            self.apply_config.is_some(),
            self.refresh_inventory.is_some(),
        ];
        if commands.into_iter().filter(|present| *present).count() != 1 {
            return Err(ControlFrameError::MissingControlCommand);
        }
        if let Some(request) = &self.get_config {
            request.validate_frame()?;
        }
        if let Some(request) = &self.watch_config {
            request.validate_frame()?;
        }
        if let Some(request) = &self.apply_config {
            request.validate_frame()?;
        }
        if let Some(request) = &self.refresh_inventory {
            request.validate_frame()?;
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlResponse {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.request_id == 0 {
            return Err(ControlFrameError::MissingRequestId);
        }
        let results = [
            self.get_config.is_some(),
            self.watch_config.is_some(),
            self.apply_config.is_some(),
            self.refresh_inventory.is_some(),
        ];
        if results.into_iter().filter(|present| *present).count() != 1 {
            return Err(ControlFrameError::MissingControlResult);
        }
        if let Some(response) = &self.get_config {
            response.validate_frame()?;
        }
        if let Some(response) = &self.watch_config {
            response.validate_frame()?;
        }
        if let Some(response) = &self.apply_config {
            response.validate_frame()?;
        }
        if let Some(response) = &self.refresh_inventory {
            response.validate_frame()?;
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlError {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if matches!(
            crate::proto::node::OwnerControlErrorCode::try_from(self.code),
            Err(_) | Ok(crate::proto::node::OwnerControlErrorCode::Unspecified)
        ) {
            return Err(ControlFrameError::InvalidOwnerControlErrorCode { got: self.code });
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlGetConfigRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_endpoint_id_length(self.requester_node_id.len())?;
        validate_endpoint_id_length(self.target_node_id.len())?;
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlGetConfigResponse {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        self.snapshot
            .as_ref()
            .ok_or(ControlFrameError::MissingConfig)?
            .validate_frame()
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlWatchConfigRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_endpoint_id_length(self.requester_node_id.len())?;
        validate_endpoint_id_length(self.target_node_id.len())?;
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlWatchConfigResponse {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        let results = [
            self.accepted.is_some(),
            self.snapshot.is_some(),
            self.update.is_some(),
        ];
        if results.into_iter().filter(|present| *present).count() != 1 {
            return Err(ControlFrameError::MissingControlResult);
        }
        if let Some(accepted) = &self.accepted {
            accepted.validate_frame()?;
        }
        if let Some(snapshot) = &self.snapshot {
            snapshot.validate_frame()?;
        }
        if let Some(update) = &self.update {
            update.validate_frame()?;
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlWatchAccepted {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_endpoint_id_length(self.target_node_id.len())?;
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlApplyConfigRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_endpoint_id_length(self.requester_node_id.len())?;
        validate_endpoint_id_length(self.target_node_id.len())?;
        if self.config.is_none() {
            return Err(ControlFrameError::MissingConfig);
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlApplyConfigResponse {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.success || !self.config_hash.is_empty() {
            validate_config_hash_length(self.config_hash.len())?;
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlRefreshInventoryRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_endpoint_id_length(self.requester_node_id.len())?;
        validate_endpoint_id_length(self.target_node_id.len())?;
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlRefreshInventoryResponse {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        self.snapshot
            .as_ref()
            .ok_or(ControlFrameError::MissingConfig)?
            .validate_frame()?;
        if let Some(inventory) = &self.inventory {
            inventory.validate_frame()?;
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlRefreshInventory {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        use crate::proto::node::OwnerControlRefreshInventoryDisposition;

        if !matches!(
            OwnerControlRefreshInventoryDisposition::try_from(self.disposition),
            Ok(OwnerControlRefreshInventoryDisposition::Executed)
                | Ok(OwnerControlRefreshInventoryDisposition::Coalesced)
        ) {
            return Err(ControlFrameError::InvalidInventoryDisposition {
                got: self.disposition,
            });
        }
        let mut previous = None;
        for entry in &self.entries {
            let canonical = entry.canonical_model_ref.trim();
            if canonical.is_empty() {
                return Err(ControlFrameError::MissingInventoryModelRef);
            }
            if previous.is_some_and(|value| value >= canonical) {
                return Err(ControlFrameError::InvalidInventoryOrder);
            }
            previous = Some(canonical);
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlConfigSnapshot {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_endpoint_id_length(self.node_id.len())?;
        validate_config_hash_length(self.config_hash.len())?;
        if self.config.is_none() {
            return Err(ControlFrameError::MissingConfig);
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlConfigUpdate {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_endpoint_id_length(self.node_id.len())?;
        validate_config_hash_length(self.config_hash.len())?;
        if self.config.is_none() {
            return Err(ControlFrameError::MissingConfig);
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::MeshSubprotocolOpen {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.r#gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.r#gen });
        }
        if self.name.trim().is_empty() || self.major == 0 {
            return Err(ControlFrameError::InvalidSubprotocol);
        }
        Ok(())
    }
}

pub(crate) fn validate_peer_announcement(
    pa: &crate::proto::node::PeerAnnouncement,
) -> Result<(), ControlFrameError> {
    if pa.endpoint_id.len() != 32 {
        return Err(ControlFrameError::InvalidEndpointId {
            got: pa.endpoint_id.len(),
        });
    }
    if pa.role == crate::proto::node::NodeRole::Host as i32 && pa.http_port.is_none() {
        return Err(ControlFrameError::MissingHttpPort);
    }
    for subprotocol in &pa.subprotocols {
        if subprotocol.name.trim().is_empty() || subprotocol.major == 0 {
            return Err(ControlFrameError::InvalidSubprotocol);
        }
    }
    Ok(())
}

fn validate_endpoint_id_length(len: usize) -> Result<(), ControlFrameError> {
    if len != 32 {
        return Err(ControlFrameError::InvalidEndpointId { got: len });
    }
    Ok(())
}

fn validate_config_hash_length(len: usize) -> Result<(), ControlFrameError> {
    if len != 32 {
        return Err(ControlFrameError::InvalidConfigHashLength { got: len });
    }
    Ok(())
}

fn validate_public_key_length(len: usize) -> Result<(), ControlFrameError> {
    if len != 32 {
        return Err(ControlFrameError::InvalidPublicKeyLength { got: len });
    }
    Ok(())
}
