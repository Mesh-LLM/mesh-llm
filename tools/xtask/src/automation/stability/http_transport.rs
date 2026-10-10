//! HTTP transport uses the existing pure Rust client and drives its connection
//! in the request future. Dropping a cancelled or completed SSE request closes
//! the connection without a detached driver task.
use super::{BODY_LIMIT, Decoder, Failure, Http, Reply, failure, millis};
use http_body_util::{BodyExt, Full};
use hyper::{Method, Request, body::Bytes};
use hyper_util::rt::TokioIo;
use serde_json::Value;
use std::time::Instant;

impl Http {
    pub(super) async fn exchange_http(
        &self,
        method: Method,
        suffix: &str,
        payload: Option<&Value>,
        stream: bool,
        started: Instant,
    ) -> Result<Reply, Failure> {
        let endpoint = format!("{}{suffix}", self.base.as_str().trim_end_matches('/'));
        let uri: hyper::Uri = endpoint
            .parse()
            .map_err(|_| failure("invalid stability HTTP endpoint", None))?;
        let host = self
            .base
            .host_str()
            .ok_or_else(|| failure("missing HTTP host", None))?;
        // URL exposes IPv6 hosts in brackets; TcpStream resolves the bare address.
        let host = host
            .strip_prefix('[')
            .and_then(|h| h.strip_suffix(']'))
            .unwrap_or(host);
        let port = self
            .base
            .port_or_known_default()
            .ok_or_else(|| failure("missing HTTP port", None))?;
        let socket = tokio::net::TcpStream::connect((host, port))
            .await
            .map_err(|_| failure("stability HTTP connection failed", None))?;
        let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(socket))
            .await
            .map_err(|_| failure("stability HTTP handshake failed", None))?;
        let request = http_request(method, &uri, payload, self.token)?;
        let response = async {
            let mut response = sender
                .send_request(request)
                .await
                .map_err(|_| failure("stability HTTP request failed", None))?;
            let status = response.status().as_u16();
            if response.status().is_redirection() {
                return Err(failure("stability endpoint redirected", Some(status)));
            }
            if !response.status().is_success() {
                return Err(failure(&format!("HTTP {status}"), Some(status)));
            }
            let mut size = 0usize;
            let mut body = Vec::new();
            let mut decoder = Decoder::default();
            while let Some(frame) = response.body_mut().frame().await {
                let frame = frame
                    .map_err(|_| failure("stability response body incomplete", Some(status)))?;
                let Some(bytes) = frame.data_ref() else {
                    continue;
                };
                if bytes.len() > BODY_LIMIT.saturating_sub(size) {
                    return Err(failure("stability response exceeds 16 MiB", Some(status)));
                }
                size += bytes.len();
                if stream {
                    decoder
                        .push(bytes, millis(started))
                        .map_err(|detail| failure(&detail, Some(status)))?;
                    if decoder.done {
                        break;
                    }
                } else {
                    body.extend_from_slice(bytes);
                }
            }
            decode_reply(status, body, decoder, stream, started)
        };
        tokio::pin!(response);
        tokio::select! {
            result = &mut response => result,
            result = connection => {
                result.map_err(|_| failure("stability HTTP connection driver failed", None))?;
                response.await
            }
        }
    }
}

fn http_request(
    method: Method,
    uri: &hyper::Uri,
    payload: Option<&Value>,
    token: &str,
) -> Result<Request<Full<Bytes>>, Failure> {
    let mut builder = Request::builder()
        .method(method)
        .uri(uri.path_and_query().map_or("/", |path| path.as_str()))
        .header(
            "host",
            uri.authority()
                .ok_or_else(|| failure("missing HTTP authority", None))?
                .as_str(),
        )
        .header("connection", "close");
    let body = if let Some(payload) = payload {
        let bytes = serde_json::to_vec(payload)
            .map_err(|_| failure("stability request encoding failed", None))?;
        if bytes.len() > 2 * 1024 * 1024 {
            return Err(failure("stability request exceeds 2 MiB", None));
        }
        builder = builder
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {token}"));
        bytes
    } else {
        Vec::new()
    };
    builder
        .body(Full::new(Bytes::from(body)))
        .map_err(|_| failure("invalid stability HTTP request", None))
}

pub(super) fn decode_reply(
    status: u16,
    body: Vec<u8>,
    decoder: Decoder,
    stream: bool,
    started: Instant,
) -> Result<Reply, Failure> {
    if stream {
        let decoder = decoder
            .finish(millis(started))
            .map_err(|detail| failure(&detail, Some(status)))?;
        return Ok(Reply {
            status,
            json: None,
            events: decoder.events,
            first_event_ms: decoder.first_event_ms,
        });
    }
    let json: Value = serde_json::from_slice(&body)
        .map_err(|_| failure("response was not JSON", Some(status)))?;
    if !json.is_object() {
        return Err(failure("response JSON was not an object", Some(status)));
    }
    Ok(Reply {
        status,
        json: Some(json),
        events: Vec::new(),
        first_event_ms: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process::Cancellation;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
        time::Duration,
    };

    fn held_response(bytes: Vec<u8>) -> (url::Url, thread::JoinHandle<bool>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = url::Url::parse(&format!(
            "http://{}/prefix/v1",
            listener.local_addr().unwrap()
        ))
        .unwrap();
        let worker = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(3);
            let mut socket = loop {
                match listener.accept() {
                    Ok((socket, _)) => break socket,
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && Instant::now() < deadline =>
                    {
                        thread::sleep(Duration::from_millis(5))
                    }
                    Err(_) => return false,
                }
            };
            socket.set_nonblocking(false).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            socket
                .set_write_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut request = Vec::new();
            let mut chunk = [0; 1024];
            loop {
                let Ok(count) = socket.read(&mut chunk) else {
                    return false;
                };
                if count == 0 || request.len() > 4096 {
                    return false;
                }
                request.extend_from_slice(&chunk[..count]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            if socket.write_all(&bytes).is_err() {
                return false;
            }
            // No body completion is sent. Success requires the request owner to
            // close the socket on DONE, deadline, or cancellation.
            loop {
                match socket.read(&mut chunk) {
                    Ok(0) => return true,
                    Ok(_) => (),
                    Err(error) => return error.kind() == std::io::ErrorKind::ConnectionReset,
                }
            }
        });
        (base, worker)
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn stability_http_done_finishes_and_closes_a_still_open_chunked_stream() {
        let body =
            "data: {\"choices\":[{\"delta\":{\"content\":\"STREAM_OK\"}}]}\n\ndata: [DONE]\n\n";
        let bytes = format!(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{body}\r\n",
            body.len()
        );
        let (base, worker) = held_response(bytes.into_bytes());
        let http = Http::new(
            base,
            Duration::from_secs(2),
            Cancellation::default(),
            "fixture",
        )
        .unwrap();
        let started = Instant::now();
        let reply = runtime().block_on(http.models_stream_fixture()).unwrap();
        assert_eq!(reply.events.len(), 1);
        assert!(reply.first_event_ms.is_some());
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(
            worker.join().unwrap(),
            "SSE owner did not close its connection"
        );
    }

    impl Http {
        async fn models_stream_fixture(&self) -> Result<Reply, Failure> {
            self.request(Method::GET, "/models", None, true).await
        }
    }

    #[test]
    fn stability_http_deadline_closes_an_incomplete_response() {
        let (base, worker) =
            held_response(b"HTTP/1.1 200 OK\r\nContent-Length: 99\r\n\r\n{".to_vec());
        let http = Http::new(
            base,
            Duration::from_millis(150),
            Cancellation::default(),
            "fixture",
        )
        .unwrap();
        let result = runtime().block_on(http.models());
        let closed = worker.join().unwrap();
        let failure = result.err().unwrap();
        assert!(failure.detail.contains("deadline exceeded"), "{failure:?}");
        assert!(closed, "expired request retained its connection");
    }

    #[test]
    fn stability_http_cancellation_closes_an_incomplete_response() {
        let (base, worker) =
            held_response(b"HTTP/1.1 200 OK\r\nContent-Length: 99\r\n\r\n{".to_vec());
        let cancellation = Cancellation::default();
        let http = Http::new(
            base,
            Duration::from_secs(2),
            cancellation.clone(),
            "fixture",
        )
        .unwrap();
        // Interrupts arrive independently of the HTTP runtime's timer polling.
        let cancel = thread::spawn(move || {
            thread::sleep(Duration::from_millis(150));
            cancellation.cancel();
        });
        let result = runtime().block_on(http.models());
        cancel.join().unwrap();
        let closed = worker.join().unwrap();
        let failure = result.err().unwrap();
        assert!(failure.detail.contains("cancelled"), "{failure:?}");
        assert!(closed, "cancelled request retained its connection");
    }
}
