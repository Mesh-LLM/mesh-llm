//! The HTTP backend a metered request is dispatched to: the locally served
//! model or a plugin inference endpoint. Both are usage sources only; nothing
//! here prices, gates or bills.

use anyhow::{Context, Result, ensure};
use tokio::io::AsyncWriteExt;

use super::request::PaidRequest;
use crate::{
    inference::election::{InferenceTarget, ModelTargets},
    mesh::Node,
    network::openai::client_stream::ClientStream,
};

pub(super) enum Backend {
    /// A locally served model on this loopback port.
    Local(u16),
    /// A plugin inference endpoint's HTTP base URL.
    Plugin(String),
}

impl Backend {
    /// The backend serving `backend_model`: the local target when there is
    /// one, otherwise the plugin endpoint that lists the public `model`.
    pub(super) async fn resolve(
        node: &Node,
        targets: &ModelTargets,
        backend_model: &str,
        model: &str,
    ) -> Result<Self> {
        let local = targets
            .candidates(backend_model)
            .iter()
            .find_map(|target| match target {
                InferenceTarget::Local(port) => Some(*port),
                _ => None,
            });
        match local {
            Some(port) => Ok(Self::Local(port)),
            None => Self::resolve_plugin(node, model).await,
        }
    }

    async fn resolve_plugin(node: &Node, model: &str) -> Result<Self> {
        let endpoint = node
            .plugin_manager()
            .await
            .context("paid model must be served locally")?
            .inference_endpoint_for_model(model)
            .await?
            .context("paid model must be served locally")?;
        // The same transport contract as external endpoints: plain HTTP.
        let url = url::Url::parse(&endpoint.address)?;
        ensure!(
            url.scheme() == "http" && url.host_str().is_some(),
            "unsupported backend URL"
        );
        Ok(Self::Plugin(endpoint.address))
    }

    pub(super) fn local_port(&self) -> Option<u16> {
        match self {
            Self::Local(port) => Some(*port),
            Self::Plugin(_) => None,
        }
    }

    /// Connect and submit `request`; the returned stream carries the backend's
    /// raw HTTP response.
    pub(super) async fn connect(&self, request: &PaidRequest, id: &str) -> Result<ClientStream> {
        let (address, raw) = self.prepare(request, id)?;
        let mut connection = ClientStream::connect(address).await?;
        connection.write_all(&raw).await?;
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
                let mut raw = format!(
                    "POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nAccept-Encoding: identity\r\nConnection: close\r\nx-request-id: {id}\r\nContent-Length: {}\r\n\r\n",
                    body.len()
                )
                .into_bytes();
                raw.extend(body);
                (format!("{host}:{port}"), raw)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> PaidRequest {
        PaidRequest::parse(
            b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"model\":\"m\",\"messages\":[]}",
        )
        .unwrap()
    }

    #[test]
    fn plugin_backends_get_the_v1_path_once() {
        for (base, expected) in [
            (
                "http://127.0.0.1:8080",
                "POST /v1/chat/completions HTTP/1.1",
            ),
            (
                "http://127.0.0.1:8080/v1",
                "POST /v1/chat/completions HTTP/1.1",
            ),
            (
                "http://127.0.0.1:8080/engine/",
                "POST /engine/v1/chat/completions HTTP/1.1",
            ),
        ] {
            let (address, raw) = Backend::Plugin(base.into())
                .prepare(&request(), "id")
                .unwrap();
            assert_eq!(address, "127.0.0.1:8080");
            let raw = String::from_utf8(raw).unwrap();
            assert!(raw.starts_with(expected), "{base}: {raw}");
            assert!(raw.contains("x-request-id: id\r\n"));
            assert!(raw.contains("Accept-Encoding: identity\r\n"));
        }
    }

    #[test]
    fn local_backends_use_the_loopback_port() {
        let (address, _) = Backend::Local(9337).prepare(&request(), "id").unwrap();
        assert_eq!(address, "127.0.0.1:9337");
        assert_eq!(Backend::Local(9337).local_port(), Some(9337));
        assert_eq!(Backend::Plugin("http://h:1".into()).local_port(), None);
    }
}
