use crate::mesh::Node;
use crate::network::openai::client_stream::ClientStream;
use anyhow::{Context, Result};
use iroh::EndpointId;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const TUNNELED_HTTP_HEADER_READ_CHUNK_BYTES: usize = 8 * 1024;
const TUNNELED_HTTP_HEADER_READ_TIMEOUT: Duration = Duration::from_secs(5);

#[cfg(test)]
mod tests;

/// Handle an inbound HTTP tunnel bi-stream through direct ingress or the legacy bridge.
pub(super) async fn handle_inbound_http_stream(
    node: Node,
    remote: EndpointId,
    quic_send: iroh::endpoint::SendStream,
    mut quic_recv: iroh::endpoint::RecvStream,
    http_port: u16,
    ingress: Option<super::HttpIngress>,
) -> Result<()> {
    let _inflight = node.begin_inflight_request();

    match node
        .activity_policy_guard
        .check_admission(crate::runtime::IngressType::RemoteQuicHttp)
    {
        crate::runtime::AdmissionResult::Allowed => {}
        crate::runtime::AdmissionResult::Paused { reason, .. } => {
            tracing::debug!(reason, "Inbound HTTP tunnel rejected by activity policy");
            let stream = ClientStream::from_quic_with_prefix(quic_recv, quic_send, Vec::new());
            crate::network::openai::send_503(stream, &format!("inference paused: {reason}"))
                .await
                .context("failed to send paused-inference response")?;
            return Ok(());
        }
    }

    // Only raw mesh ingress that successfully claimed a parent emits the
    // private assertion. Direct API requests have it stripped before they can
    // reach this tunnel, so they retain normal target frontend ownership.
    let prefix = read_tunneled_http_header_prefix(&mut quic_recv).await?;
    let (prefix, _) =
        crate::network::openai::request_parse::ensure_canonical_request_id_in_header_prefix(prefix);
    let caller_metadata =
        remote_tunnel_request_metadata(remote, node.authenticated_peer_path(remote).await);
    let (attribution_request_id, suppression_request_id) = remote_tunnel_request_ids(&prefix);
    let logging_state = crate::logging_runtime_state();
    let _remote_attribution = attribution_request_id.and_then(|request_id| {
        logging_state
            .as_ref()
            .and_then(|state| state.attribute_remote_tunneled_request(request_id, caller_metadata))
    });
    let _remote_suppression = suppression_request_id.and_then(|request_id| {
        logging_state
            .as_ref()
            .and_then(|state| state.suppress_remote_tunneled_request(request_id))
    });
    if let Some(ingress) = ingress {
        let stream = ClientStream::from_quic_with_prefix(quic_recv, quic_send, prefix);
        let targets = ingress.targets.borrow().clone();
        crate::network::openai::ingress::handle_remote_http_stream(
            node,
            stream,
            targets,
            ingress.affinity,
        )
        .await;
        return Ok(());
    }

    // Compatibility for embedders/tests that only configure the legacy port.
    let mut tcp_stream = TcpStream::connect(format!("127.0.0.1:{http_port}")).await?;
    tcp_stream.set_nodelay(true)?;
    tcp_stream.write_all(&prefix).await?;
    let (tcp_read, tcp_write) = tokio::io::split(tcp_stream);
    super::relay_bidirectional(tcp_read, tcp_write, quic_send, quic_recv).await
}

fn remote_tunnel_request_ids(
    prefix: &[u8],
) -> (
    Option<mesh_llm_events::logging::identifiers::RequestId>,
    Option<mesh_llm_events::logging::identifiers::RequestId>,
) {
    (
        crate::network::openai::request_parse::canonical_request_id_from_header_prefix(prefix),
        crate::network::openai::request_parse::raw_lifecycle_owner_from_header_prefix(prefix),
    )
}

fn remote_tunnel_request_metadata(
    remote: EndpointId,
    path: Option<crate::mesh::SelectedPathObservation>,
) -> crate::logging::RequestSummaryMetadata {
    let endpoint_id = hex::encode(remote.as_bytes());
    let (addr, path_type) = match path {
        Some(observation) if observation.path_type == "direct" => (
            observation
                .observed_direct_remote_addr
                .map(|addr| addr.to_string()),
            Some(crate::logging::CallerPathType::RemoteQuicHttp),
        ),
        Some(observation) if observation.path_type == "relay" => {
            (None, Some(crate::logging::CallerPathType::Relay))
        }
        Some(_) | None => (None, None),
    };
    crate::logging::RequestSummaryMetadata::default().with_caller_identity(
        Some(&endpoint_id),
        addr.as_deref(),
        path_type,
    )
}

/// Read at most one bounded HTTP header prefix without changing the bytes that
/// the existing relay will subsequently forward. A read may include a small
/// amount of body data from the same transport chunk; direct ingress consumes
/// it from `ClientStream` before reading the remaining QUIC bytes.
async fn read_tunneled_http_header_prefix<R>(reader: &mut R) -> Result<Vec<u8>>
where
    R: AsyncRead + Unpin,
{
    tokio::time::timeout(TUNNELED_HTTP_HEADER_READ_TIMEOUT, async {
        let mut prefix = Vec::with_capacity(TUNNELED_HTTP_HEADER_READ_CHUNK_BYTES);
        let mut chunk = [0u8; TUNNELED_HTTP_HEADER_READ_CHUNK_BYTES];
        let max_header_bytes = crate::network::openai::request_parse::MAX_HEADER_BYTES;

        while prefix.len() < max_header_bytes {
            if crate::network::openai::request_parse::http_header_terminator(&prefix).is_some() {
                break;
            }
            let read_cap = (max_header_bytes - prefix.len()).min(chunk.len());
            let bytes_read = reader.read(&mut chunk[..read_cap]).await?;
            if bytes_read == 0 {
                break;
            }
            prefix.extend_from_slice(&chunk[..bytes_read]);
        }

        Ok(prefix)
    })
    .await
    .context("timed out reading tunneled HTTP header prefix")?
}
