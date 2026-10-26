//! Loopback HTTP listener with bounded admission and owned upstream cancellation.
use super::{
    forwarding::Forwarder,
    request_projection::{self, BODY_LIMIT, Endpoint},
};
use crate::process::Cancellation;
use http_body_util::{BodyExt, Full};
use hyper::{
    Request, Response, StatusCode,
    body::{Bytes, Incoming},
    service::service_fn,
};
use hyper_util::rt::TokioIo;
use std::{
    convert::Infallible,
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::{net::TcpListener, task::JoinSet};

const CAPTURE_LIMIT: u64 = 64 * 1024 * 1024;
const CONNECTION_LIMIT: usize = 16;
type Reply = Response<Full<Bytes>>;

struct State {
    endpoint: Endpoint,
    forwarder: Forwarder,
    capture: Mutex<File>,
}
struct CancelTransfer(Cancellation);
impl Drop for CancelTransfer {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

pub(super) async fn serve(
    endpoint: Endpoint,
    log: &Path,
    ready: &Path,
    duration: Duration,
    cancellation: Cancellation,
) -> Result<(), String> {
    serve_selected(
        endpoint,
        log,
        ready,
        duration,
        cancellation,
        Forwarder::discover()?,
    )
    .await
}

pub(super) async fn serve_selected(
    endpoint: Endpoint,
    log: &Path,
    ready: &Path,
    duration: Duration,
    cancellation: Cancellation,
    forwarder: Forwarder,
) -> Result<(), String> {
    let state = Arc::new(State {
        endpoint,
        forwarder,
        capture: Mutex::new(
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(log)
                .map_err(|_| "capture log unavailable")?,
        ),
    });
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .map_err(|_| "recording listener bind failed")?;
    let url = format!(
        "http://127.0.0.1:{}/v1",
        listener
            .local_addr()
            .map_err(|_| "listener address unavailable")?
            .port()
    );
    let mut marker = tempfile::NamedTempFile::new_in(ready.parent().unwrap_or(Path::new(".")))
        .map_err(|_| "readiness directory unavailable")?;
    marker
        .write_all(url.as_bytes())
        .map_err(|_| "readiness marker write failed")?;
    marker
        .as_file()
        .sync_all()
        .map_err(|_| "readiness marker sync failed")?;
    marker
        .persist(ready)
        .map_err(|_| "readiness marker publication failed")?;
    println!("{url}");
    let mut connections = JoinSet::new();
    let expires = tokio::time::Instant::now() + duration;
    let result = loop {
        if cancellation.is_cancelled() {
            break Ok(());
        }
        tokio::select! {
            _ = tokio::time::sleep_until(expires) => break Err("recording proxy lifetime expired".into()),
            _ = tokio::time::sleep(Duration::from_millis(10)) => {},
            result = connections.join_next(), if !connections.is_empty() => {
                if result.is_some_and(|result| result.is_err()) { break Err("recording connection task failed".into()); }
            },
            accepted = listener.accept(), if connections.len() < CONNECTION_LIMIT => {
                let (socket, _) = match accepted { Ok(pair) => pair, Err(_) => break Err("recording listener accept failed".into()) };
                let state = state.clone();
                connections.spawn(async move {
                    let service = service_fn(move |request| handle(state.clone(), request));
                    let connection = hyper::server::conn::http1::Builder::new()
                        .max_buf_size(16384).max_headers(64).serve_connection(TokioIo::new(socket), service);
                    // Idle or incomplete clients cannot occupy admission forever.
                    let _ = tokio::time::timeout(Duration::from_secs(390), connection).await;
                });
            }
        }
    };
    drop(listener);
    connections.abort_all();
    while connections.join_next().await.is_some() {}
    // Only remove the readiness publication still owned by this listener.
    if std::fs::read_to_string(ready).ok().as_deref() == Some(&url) {
        std::fs::remove_file(ready).map_err(|_| "readiness marker cleanup failed")?;
    }
    result
}

async fn handle(state: Arc<State>, request: Request<Incoming>) -> Result<Reply, Infallible> {
    let response =
        match tokio::time::timeout(Duration::from_secs(365), exchange(state, request)).await {
            Ok(Ok(response)) => response,
            Ok(Err((status, message))) => failure(status, message),
            Err(_) => failure(
                StatusCode::GATEWAY_TIMEOUT,
                "recording request deadline exceeded",
            ),
        };
    Ok(response)
}

async fn exchange(
    state: Arc<State>,
    request: Request<Incoming>,
) -> Result<Reply, (StatusCode, &'static str)> {
    let (parts, mut incoming) = request.into_parts();
    if parts.method != hyper::Method::GET && parts.method != hyper::Method::POST {
        return Err((
            StatusCode::METHOD_NOT_ALLOWED,
            "recording proxy accepts GET and POST",
        ));
    }
    let path = parts.uri.path_and_query().map_or("/", |path| path.as_str());
    let endpoint = state
        .endpoint
        .target(path)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid recording request path"))?;
    let mut body = Vec::new();
    while let Some(frame) = incoming.frame().await {
        let frame =
            frame.map_err(|_| (StatusCode::BAD_REQUEST, "incomplete recording request body"))?;
        if let Some(bytes) = frame.data_ref() {
            if bytes.len() > BODY_LIMIT.saturating_sub(body.len()) {
                return Err((
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "recording request exceeds 32 MiB",
                ));
            }
            body.extend_from_slice(bytes);
        }
    }
    let text = |key| parts.headers.get(key).and_then(|value| value.to_str().ok());
    let bytes = request_projection::capture(
        parts.method.as_str(),
        path,
        &body,
        text("content-type"),
        text("accept"),
    )
    .map_err(|_| (StatusCode::BAD_REQUEST, "request capture failed"))?;
    append_capture(&state.capture, &bytes)?;
    let token = Cancellation::default();
    let _owner = CancelTransfer(token.clone());
    let forwarder = state.forwarder.clone();
    let forwarded = tokio::task::spawn_blocking(move || {
        forwarder.forward(endpoint, parts.method, parts.headers, body, token)
    })
    .await
    .map_err(|_| (StatusCode::BAD_GATEWAY, "upstream worker failed"))?;
    forwarded.map_err(|_| (StatusCode::BAD_GATEWAY, "upstream transfer failed"))
}

fn append_capture(capture: &Mutex<File>, bytes: &[u8]) -> Result<(), (StatusCode, &'static str)> {
    let mut file = capture
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "capture lock failed"))?;
    let size = file
        .metadata()
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "capture metadata unavailable",
            )
        })?
        .len();
    if bytes.len() as u64 > CAPTURE_LIMIT.saturating_sub(size) {
        return Err((StatusCode::INSUFFICIENT_STORAGE, "capture exceeds 64 MiB"));
    }
    file.write_all(bytes)
        .and_then(|_| file.flush())
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "capture write failed"))
}

fn failure(status: StatusCode, message: &str) -> Reply {
    let mut response = Response::new(Full::new(Bytes::copy_from_slice(message.as_bytes())));
    *response.status_mut() = status;
    response
}
