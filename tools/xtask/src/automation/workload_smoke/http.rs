use crate::command::DynResult;
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;

pub(super) fn post(
    url: &str,
    content_type: &str,
    body: Vec<u8>,
    expected: &str,
) -> DynResult<Vec<u8>> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        tokio::time::timeout(
            std::time::Duration::from_secs(240),
            exchange(url, content_type, body, expected),
        )
        .await?
    })
}

async fn exchange(
    url: &str,
    content_type: &str,
    body: Vec<u8>,
    expected: &str,
) -> DynResult<Vec<u8>> {
    let uri: hyper::Uri = url.parse()?;
    if uri.scheme_str() != Some("http") {
        return Err("workload smoke requires an HTTP endpoint".into());
    }
    let host = uri.host().ok_or("missing HTTP host")?;
    // URI hosts keep IPv6 literal brackets; socket resolution needs the bare address.
    let host = host
        .strip_prefix('[')
        .and_then(|inner| inner.strip_suffix(']'))
        .unwrap_or(host);
    let port = uri.port_u16().unwrap_or(80);
    let stream = tokio::net::TcpStream::connect((host, port)).await?;
    let (mut sender, connection) =
        hyper::client::conn::http1::handshake(TokioIo::new(stream)).await?;
    let request = hyper::Request::builder()
        .method("POST")
        .uri(uri.path_and_query().ok_or("missing HTTP path")?.as_str())
        .header(
            "host",
            uri.authority().ok_or("missing HTTP authority")?.as_str(),
        )
        .header("content-type", content_type)
        .header("connection", "close")
        .body(Full::new(Bytes::from(body)))?;
    let response = async {
        let mut response = sender.send_request(request).await?;
        if !response.status().is_success() {
            return Err("workload HTTP request failed".into());
        }
        let actual = response
            .headers()
            .get("content-type")
            .ok_or("missing response content type")?
            .to_str()?
            .split(';')
            .next()
            .ok_or("empty content type")?
            .trim();
        if actual != expected {
            return Err("unexpected workload response content type".into());
        }
        let mut bytes = Vec::new();
        while let Some(frame) = response.body_mut().frame().await {
            if let Some(data) = frame?.data_ref() {
                if data.len() > (64 * 1024 * 1024_usize).saturating_sub(bytes.len()) {
                    return Err("workload response exceeds 64 MiB".into());
                }
                bytes.extend_from_slice(data);
            }
        }
        Ok(bytes)
    };
    tokio::pin!(response);
    tokio::select! {
        result = &mut response => result,
        result = connection => { result?; response.await }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read as _, Write as _};

    #[test]
    fn bracketed_ipv6_literal_connects_to_bare_address() {
        let Ok(listener) = std::net::TcpListener::bind("[::1]:0") else {
            return;
        };
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request).unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok")
                .unwrap();
        });
        let body = super::post(
            &format!("http://[::1]:{port}/v1/x"),
            "text/plain",
            b"x".to_vec(),
            "text/plain",
        )
        .unwrap();
        server.join().unwrap();
        assert_eq!(body, b"ok");
    }
}
