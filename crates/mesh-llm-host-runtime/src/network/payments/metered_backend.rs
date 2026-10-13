//! Resolve existing local/plugin HTTP backends without another proxy.
use super::request::PaidRequest;
use crate::{mesh::Node, network::openai::client_stream::ClientStream};
use anyhow::{Context, Result, ensure};
use mesh_llm_payments::ledger::metered::MeteredOutcome;
use tokio::io::{AsyncWrite, AsyncWriteExt};

/// Any failure before usage is reported abandons the request: no bill, no
/// retry of the backend request on the seller's initiative.
#[derive(Debug)]
pub(super) struct SubmissionFailure;

impl SubmissionFailure {
    pub(super) fn outcome(self) -> MeteredOutcome {
        MeteredOutcome::Abandoned
    }
}

pub(super) async fn submit_request(
    connection: &mut (impl AsyncWrite + Unpin),
    raw: &[u8],
) -> std::result::Result<(), SubmissionFailure> {
    connection
        .write_all(raw)
        .await
        .map_err(|_| SubmissionFailure)
}

pub(super) enum Backend {
    Local(u16),
    Plugin(String),
}

pub(super) async fn resolve_plugin(node: &Node, model: &str) -> Result<Backend> {
    let manager = node.plugin_manager.lock().await;
    let endpoint = manager
        .as_ref()
        .context("no plugin manager")?
        .inference_endpoint_for_model(model)
        .await?
        .context("no local paid backend")?;
    // Match the existing external-endpoint transport contract (HTTP only).
    let url = url::Url::parse(&endpoint.address)?;
    ensure!(
        url.scheme() == "http" && url.host_str().is_some(),
        "unsupported backend URL"
    );
    Ok(Backend::Plugin(endpoint.address))
}

impl Backend {
    pub(super) async fn connect(
        &self,
        request: &PaidRequest,
        id: &str,
    ) -> std::result::Result<ClientStream, SubmissionFailure> {
        let (address, raw) = self.prepare(request, id).map_err(|_| SubmissionFailure)?;
        let mut connection = ClientStream::connect(address)
            .await
            .map_err(|_| SubmissionFailure)?;
        submit_request(&mut connection, &raw).await?;
        Ok(connection)
    }

    fn prepare(&self, request: &PaidRequest, id: &str) -> Result<(String, Vec<u8>)> {
        Ok(match self {
            Self::Local(port) => (format!("127.0.0.1:{port}"), request.backend_http(id)?),
            Self::Plugin(base) => {
                let url = url::Url::parse(base)?;
                let host = url.host_str().context("missing host")?;
                let port = url.port_or_known_default().context("missing port")?;
                let base_path = url.path().trim_end_matches('/');
                let suffix = request
                    .path
                    .strip_prefix("/v1")
                    .context("invalid inference path")?;
                let path = if base_path.ends_with("/v1") {
                    format!("{base_path}{suffix}")
                } else {
                    format!("{base_path}/v1{suffix}")
                };
                let body = serde_json::to_vec(&request.body)?;
                let mut raw = format!("POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nAccept-Encoding: identity\r\nConnection: close\r\nx-request-id: {id}\r\nContent-Length: {}\r\n\r\n", body.len()).into_bytes();
                raw.extend(body);
                (format!("{host}:{port}"), raw)
            }
        })
    }
}
