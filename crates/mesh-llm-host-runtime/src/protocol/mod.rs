// Protocol infrastructure — extracted from mesh.rs

#[cfg(test)]
use crate::mesh::NodeRole;
use crate::mesh::PeerAnnouncement;

pub(crate) mod config_diagnostic;
pub(crate) mod convert;
use anyhow::Result;
pub(crate) use convert::*;
use iroh::endpoint::Connection;
use iroh::{Endpoint, EndpointAddr, EndpointId};
use prost::Message;

#[cfg(test)]
mod config_tests;
pub const ALPN_CONTROL_V1: &[u8] = b"mesh-llm-control/1";
pub const ALPN_V1: &[u8] = b"mesh-llm/1";
#[cfg(test)]
pub const ALPN: &[u8] = ALPN_V1;
pub(crate) const NODE_PROTOCOL_GENERATION: u32 = 1;
pub(crate) const MAX_CONTROL_FRAME_BYTES: usize = 8 * 1024 * 1024; // 8 MiB

pub(crate) const STREAM_GOSSIP: u8 = 0x01;
pub(crate) const STREAM_TUNNEL: u8 = 0x02;
pub(crate) const STREAM_TUNNEL_MAP: u8 = 0x03;
pub const STREAM_TUNNEL_HTTP: u8 = 0x04;
pub(crate) const STREAM_ROUTE_REQUEST: u8 = 0x05;
pub(crate) const STREAM_PEER_DOWN: u8 = 0x06;
pub(crate) const STREAM_PEER_LEAVING: u8 = 0x07;
pub(crate) const STREAM_PLUGIN_CHANNEL: u8 = 0x08;
pub(crate) const STREAM_PLUGIN_BULK_TRANSFER: u8 = 0x09;
pub(crate) const STREAM_PLUGIN_MESH_STREAM: u8 = 0x0a;
/// Reserved legacy mesh-plane config subscription stream ID.
///
/// Config and inventory control now live exclusively on `mesh-llm-control/1`;
/// keep 0x0b reserved so old wire values are not accidentally reused.
pub(crate) const STREAM_CONFIG_SUBSCRIBE: u8 = 0x0b;
/// Reserved legacy mesh-plane config push stream ID.
///
/// Config and inventory control now live exclusively on `mesh-llm-control/1`;
/// keep 0x0c reserved so old wire values are not accidentally reused.
pub(crate) const STREAM_CONFIG_PUSH: u8 = 0x0c;
pub(crate) const STREAM_SUBPROTOCOL: u8 = 0x0d;
pub(crate) const STREAM_DIRECT_PATH_REQUEST: u8 = 0x0e;
const _: () = {
    let _ = ALPN_CONTROL_V1;
    let _ = STREAM_CONFIG_SUBSCRIBE;
    let _ = STREAM_CONFIG_PUSH;
    let _ = STREAM_SUBPROTOCOL;
    let _ = STREAM_DIRECT_PATH_REQUEST;
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ControlProtocol {
    ProtoV1,
}

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

pub(crate) fn protocol_from_alpn(alpn: &[u8]) -> ControlProtocol {
    let _ = alpn;
    ControlProtocol::ProtoV1
}

pub(crate) fn connection_protocol(conn: &Connection) -> ControlProtocol {
    protocol_from_alpn(conn.alpn())
}

pub(crate) async fn connect_mesh(endpoint: &Endpoint, addr: EndpointAddr) -> Result<Connection> {
    let connecting = endpoint.connect(addr, ALPN_V1).await?;
    Ok(connecting)
}

pub(crate) async fn write_len_prefixed(
    send: &mut iroh::endpoint::SendStream,
    body: &[u8],
) -> Result<()> {
    ensure_control_frame_size(body)?;
    send.write_all(&(body.len() as u32).to_le_bytes()).await?;
    send.write_all(body).await?;
    Ok(())
}

pub(crate) fn ensure_control_frame_size(body: &[u8]) -> Result<(), ControlFrameError> {
    if body.len() > MAX_CONTROL_FRAME_BYTES {
        return Err(ControlFrameError::OversizeFrame { size: body.len() });
    }
    Ok(())
}

pub(crate) async fn read_len_prefixed(recv: &mut iroh::endpoint::RecvStream) -> Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    recv.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_CONTROL_FRAME_BYTES {
        anyhow::bail!("control frame too large: {} bytes", len);
    }
    let mut buf = vec![0u8; len];
    recv.read_exact(&mut buf).await?;
    Ok(buf)
}

pub(crate) async fn write_gossip_payload(
    send: &mut iroh::endpoint::SendStream,
    protocol: ControlProtocol,
    anns: &[PeerAnnouncement],
    sender_id: EndpointId,
) -> Result<()> {
    let _ = protocol;
    let frame = build_gossip_frame(anns, sender_id);
    write_len_prefixed(send, &frame.encode_to_vec()).await?;
    Ok(())
}

pub(crate) fn decode_gossip_payload(
    protocol: ControlProtocol,
    remote: EndpointId,
    buf: &[u8],
) -> Result<Vec<(EndpointAddr, PeerAnnouncement)>> {
    let _ = protocol;
    let frame = crate::proto::node::GossipFrame::decode(buf)
        .map_err(|e| anyhow::anyhow!("gossip decode from {}: {e}", remote.fmt_short()))?;
    frame
        .validate_frame()
        .map_err(|e| anyhow::anyhow!("invalid gossip frame from {}: {e}", remote.fmt_short()))?;
    if frame.sender_id.as_slice() != remote.as_bytes() {
        anyhow::bail!(
            "gossip sender_id mismatch from {}: connection identity does not match frame sender_id",
            remote.fmt_short()
        );
    }
    Ok(frame
        .peers
        .iter()
        .filter_map(proto_ann_to_local)
        .collect::<Vec<_>>())
}

#[cfg(test)]
pub(crate) fn encode_control_frame(stream_type: u8, msg: &impl prost::Message) -> Vec<u8> {
    let proto_bytes = msg.encode_to_vec();
    let len = proto_bytes.len() as u32;
    let mut buf = Vec::with_capacity(1 + 4 + proto_bytes.len());
    buf.push(stream_type);
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&proto_bytes);
    buf
}

#[cfg(test)]
pub(crate) fn decode_control_frame<T: ValidateControlFrame>(
    expected_stream_type: u8,
    data: &[u8],
) -> Result<T, ControlFrameError> {
    const HEADER_LEN: usize = 5;
    if data.len() < HEADER_LEN {
        return Err(ControlFrameError::DecodeError(format!(
            "frame too short: {} bytes (minimum {})",
            data.len(),
            HEADER_LEN
        )));
    }
    let actual_type = data[0];
    if actual_type != expected_stream_type {
        return Err(ControlFrameError::WrongStreamType {
            expected: expected_stream_type,
            got: actual_type,
        });
    }
    let len = u32::from_le_bytes(data[1..5].try_into().unwrap()) as usize;
    if len > MAX_CONTROL_FRAME_BYTES {
        return Err(ControlFrameError::OversizeFrame { size: len });
    }
    let proto_bytes = data.get(5..5 + len).ok_or_else(|| {
        ControlFrameError::DecodeError(format!(
            "frame truncated: header says {} bytes but only {} available",
            len,
            data.len().saturating_sub(5)
        ))
    })?;
    let msg = T::decode(proto_bytes).map_err(|e| ControlFrameError::DecodeError(e.to_string()))?;
    msg.validate_frame()?;
    Ok(msg)
}

#[cfg(test)]
pub(crate) mod tests;
