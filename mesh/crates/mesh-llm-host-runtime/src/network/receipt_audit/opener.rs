//! Sends a challenge to the peer that served a turn, over a mesh tunnel to
//! its OpenAI ingress (`POST /v1/receipts/open`, then
//! `POST /v1/receipts/rows` for the attention rows), as a client of that
//! peer would. `x-mesh-target` names the node key that signed the receipt, so
//! the peer opens it itself.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use iroh::EndpointId;
use skippy_commands::receipts::turn_audit::{OpenChallenge, Opened, OpenedRows};
use skippy_receipts::attention::opening::BlockChallenge;
use skippy_receipts::turn;
use skippy_receipts::wire::Challenge;
use tokio::io::AsyncReadExt;

use crate::mesh;
use crate::network::http_framing::chunked::try_decode_chunked_body;

/// The largest opening this node reads.
const MAX_OPENING: u64 = 512 << 20;
/// How long a peer has to answer a challenge.
const OPEN_TIMEOUT: Duration = Duration::from_secs(60);

pub(super) struct TunnelOpener {
    pub(super) node: mesh::Node,
    pub(super) host: EndpointId,
    pub(super) model: String,
    pub(super) node_key: Option<String>,
    pub(super) runtime: tokio::runtime::Handle,
}

impl OpenChallenge for TunnelOpener {
    fn open(&self, challenge: &Challenge) -> Result<Opened> {
        let started = Instant::now();
        let (signature, bytes) = self.post("open", challenge)?;
        let opening = serde_json::from_slice(&bytes).context("decode the opening")?;
        Ok(Opened {
            opening,
            bytes,
            signature,
            seconds: started.elapsed().as_secs_f64(),
        })
    }

    fn rows(&self, challenge: &BlockChallenge) -> Result<OpenedRows> {
        let started = Instant::now();
        let (signature, bytes) = self.post("rows", challenge)?;
        let rows = serde_json::from_slice(&bytes).context("decode the attention rows")?;
        Ok(OpenedRows {
            rows,
            bytes,
            signature,
            seconds: started.elapsed().as_secs_f64(),
        })
    }
}

impl TunnelOpener {
    /// Posts `body`, naming the model, to `/v1/receipts/{action}` on the
    /// peer, and returns its signature header and the answer's body.
    fn post<B: serde::Serialize>(
        &self,
        action: &str,
        body: &B,
    ) -> Result<(Option<String>, Vec<u8>)> {
        let mut body = serde_json::to_value(body)?;
        body["model"] = self.model.clone().into();
        let body = serde_json::to_vec(&body)?;
        // the timer belongs to the runtime, so it is made inside it
        let response = self.runtime.block_on(async {
            tokio::time::timeout(OPEN_TIMEOUT, self.exchange(action, &body)).await
        });
        let response = response
            .with_context(|| format!("the peer did not answer /v1/receipts/{action} in time"))??;
        read_response(&response)
    }

    async fn exchange(&self, action: &str, body: &[u8]) -> Result<Vec<u8>> {
        let (mut send, recv) = self.node.open_http_tunnel(self.host).await?;
        let target = self
            .node_key
            .as_ref()
            .map(|key| format!("x-mesh-target: {key}\r\n"))
            .unwrap_or_default();
        let head = format!(
            "POST /v1/receipts/{action} HTTP/1.1\r\nHost: mesh\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{target}Connection: close\r\n\r\n",
            body.len()
        );
        send.write_all(head.as_bytes()).await?;
        send.write_all(body).await?;
        send.finish()?;
        let mut response = Vec::new();
        recv.take(MAX_OPENING)
            .read_to_end(&mut response)
            .await
            .with_context(|| format!("read the answer to /v1/receipts/{action}"))?;
        Ok(response)
    }
}

/// The opening signature header and the body of a whole HTTP response.
fn read_response(raw: &[u8]) -> Result<(Option<String>, Vec<u8>)> {
    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut response = httparse::Response::new(&mut headers);
    let httparse::Status::Complete(header_end) = response.parse(raw)? else {
        bail!("the peer's answer ended inside its headers");
    };
    let header = |name: &str| {
        response
            .headers
            .iter()
            .find(|h| h.name.eq_ignore_ascii_case(name))
            .and_then(|h| std::str::from_utf8(h.value).ok())
    };
    let chunked = header("transfer-encoding").is_some_and(|v| v.eq_ignore_ascii_case("chunked"));
    let length = header("content-length").and_then(|v| v.trim().parse::<usize>().ok());
    let signature = header(turn::OPENING_SIGNATURE_HEADER).map(str::to_owned);
    let rest = &raw[header_end..];
    let body = match (chunked, length) {
        (true, _) => {
            let limit = usize::try_from(MAX_OPENING).unwrap_or(usize::MAX);
            try_decode_chunked_body(rest, limit)?
                .context("the peer's answer ended inside its body")?
                .1
        }
        (false, Some(length)) => rest
            .get(..length)
            .context("the peer's answer is shorter than it says")?
            .to_vec(),
        (false, None) => rest.to_vec(),
    };
    let status = response.code.unwrap_or(0);
    if status != 200 {
        bail!("{status}: {}", String::from_utf8_lossy(&body));
    }
    Ok((signature, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chunked_answer_is_read_with_its_signature() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nx-receipt-opening-signature: ab\r\n\r\n3\r\n{\"a\r\n4;x=y\r\n\":1}\r\n0\r\n\r\n";
        let (signature, body) = read_response(raw).unwrap();
        assert_eq!(signature.as_deref(), Some("ab"));
        assert_eq!(body, b"{\"a\":1}");
    }

    #[test]
    fn a_sized_answer_is_read_and_an_error_status_fails() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}";
        assert_eq!(read_response(raw).unwrap().1, b"{}");
        let raw = b"HTTP/1.1 404 Not Found\r\nContent-Length: 4\r\n\r\ngone";
        assert!(read_response(raw).is_err());
    }
}
