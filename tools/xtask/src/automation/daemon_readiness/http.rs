use super::{Rejection, correlation::RequestId, ports::Ports, status};
use http_body_util::{BodyExt, Empty};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use std::error::Error as _;
use std::net::Ipv4Addr;
use std::sync::mpsc::{Receiver, SyncSender};
use std::time::Instant;

#[cfg(test)]
#[path = "../../../tests/migration_lifecycle/daemon/http.rs"]
mod tests;

pub(super) const BODY_LIMIT: usize = 1_048_576;
const HEADER_LIMIT: usize = 16_384;

#[derive(Clone, Copy)]
pub(crate) enum Endpoint {
    Status { leader_pid: u32 },
    Models { request_id: RequestId },
}

pub(crate) struct Request {
    pub(super) endpoint: Endpoint,
    pub(super) deadline: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Transfer {
    pub(super) status_code: u16,
    pub(super) body_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransferError {
    Transport,
    Timeout,
    HttpStatus(u16),
    Rejected(Rejection),
}

pub(crate) fn work(
    ports: Ports,
    channels: (
        Receiver<Request>,
        SyncSender<Result<Transfer, TransferError>>,
    ),
    runtime: tokio::runtime::Runtime,
) {
    let (requests, results) = channels;
    while let Ok(request) = requests.recv() {
        let result = runtime.block_on(transfer(ports, request));
        if results.send(result).is_err() {
            break;
        }
    }
}

pub(super) async fn transfer(ports: Ports, request: Request) -> Result<Transfer, TransferError> {
    if Instant::now() >= request.deadline {
        return Err(TransferError::Timeout);
    }
    let result =
        tokio::time::timeout_at(request.deadline.into(), exchange(ports, request.endpoint))
            .await
            .map_err(|_| TransferError::Timeout)?;
    if Instant::now() >= request.deadline {
        return Err(TransferError::Timeout);
    }
    result
}

async fn exchange(ports: Ports, endpoint: Endpoint) -> Result<Transfer, TransferError> {
    let (port, path) = match endpoint {
        Endpoint::Status { .. } => (ports.console, "/api/status"),
        Endpoint::Models { .. } => (ports.api, "/v1/models"),
    };
    let stream = tokio::net::TcpStream::connect((Ipv4Addr::LOCALHOST, port))
        .await
        .map_err(|_| TransferError::Transport)?;
    let (mut sender, connection) = hyper::client::conn::http1::Builder::new()
        .max_buf_size(HEADER_LIMIT)
        .max_headers(HEADER_LIMIT / 4)
        .handshake::<_, Empty<Bytes>>(TokioIo::new(stream))
        .await
        .map_err(classify_error)?;
    let mut request = hyper::Request::builder()
        .uri(path)
        .header("host", format!("127.0.0.1:{port}"))
        .header("connection", "close");
    match endpoint {
        Endpoint::Status { .. } => (),
        Endpoint::Models { request_id } => {
            request = request.header("x-request-id", request_id.header())
        }
    }
    let request = request
        .body(Empty::<Bytes>::new())
        .map_err(|_| TransferError::Transport)?;
    let response = async {
        let response = sender.send_request(request).await.map_err(classify_error)?;
        consume(response, endpoint, ports.api).await
    };
    tokio::pin!(response);
    tokio::select! {
        biased;
        result = &mut response => result,
        result = connection => {
            result.map_err(classify_error)?;
            response.await
        }
    }
}

async fn consume(
    mut response: hyper::Response<hyper::body::Incoming>,
    endpoint: Endpoint,
    api_port: u16,
) -> Result<Transfer, TransferError> {
    let status_code = response.status().as_u16();
    let mut body_bytes = 0;
    let mut status_body = Vec::new();
    while let Some(frame) = response.body_mut().frame().await {
        let frame = frame.map_err(classify_error)?;
        if let Some(data) = frame.data_ref() {
            if data.len() > BODY_LIMIT - body_bytes {
                return Err(TransferError::Rejected(Rejection::ResponseLimit));
            }
            body_bytes += data.len();
            match endpoint {
                Endpoint::Status { .. } => status_body.extend_from_slice(data),
                Endpoint::Models { .. } => (),
            }
        }
    }
    if status_code >= 400 {
        return Err(TransferError::HttpStatus(status_code));
    }
    match endpoint {
        Endpoint::Status { leader_pid } => {
            status::identify(&status_body, (leader_pid, api_port))
                .map_err(TransferError::Rejected)?;
        }
        Endpoint::Models { .. } => (),
    }
    Ok(Transfer {
        status_code,
        body_bytes,
    })
}

fn classify_error(error: hyper::Error) -> TransferError {
    let mut source = error.source();
    while let Some(cause) = source {
        if let Some(error) = cause.downcast_ref::<std::io::Error>()
            && error.kind() == std::io::ErrorKind::InvalidData
            && error.to_string() == "chunk trailers bytes over limit"
        {
            return TransferError::Rejected(Rejection::ResponseLimit);
        }
        source = cause.source();
    }
    if error.is_parse() && error.to_string() == "message head is too large" {
        TransferError::Rejected(Rejection::ResponseLimit)
    } else {
        TransferError::Transport
    }
}
