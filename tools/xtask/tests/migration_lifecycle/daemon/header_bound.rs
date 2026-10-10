use http_body_util::{BodyExt, Empty};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use std::net::Ipv4Addr;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const LIMIT: usize = 16_384;

async fn transfer(wire: Vec<u8>) -> Result<(), hyper::Error> {
    transfer_with(wire, (LIMIT, usize::MAX)).await
}

async fn transfer_with(wire: Vec<u8>, limits: (usize, usize)) -> Result<(), hyper::Error> {
    let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    let server = async {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = [0; 4096];
        let size = stream.read(&mut request).await.unwrap();
        assert!(size > 0);
        for chunk in wire.chunks(limits.1) {
            match stream.write_all(chunk).await {
                Ok(()) => tokio::task::yield_now().await,
                Err(error) => {
                    assert!(matches!(
                        error.kind(),
                        std::io::ErrorKind::BrokenPipe | std::io::ErrorKind::ConnectionReset
                    ));
                    return;
                }
            }
        }
    };
    let client = async {
        let stream = tokio::net::TcpStream::connect(address).await.unwrap();
        let (mut sender, connection) = hyper::client::conn::http1::Builder::new()
            .max_buf_size(limits.0)
            .handshake::<_, Empty<Bytes>>(TokioIo::new(stream))
            .await?;
        let request = hyper::Request::builder()
            .uri("/v1/models")
            .header("host", address.to_string())
            .header("connection", "close")
            .body(Empty::<Bytes>::new())
            .unwrap();
        let response = async {
            let mut response = sender.send_request(request).await?;
            while let Some(frame) = response.body_mut().frame().await {
                frame?;
            }
            Ok::<(), hyper::Error>(())
        };
        let (response, connection) = tokio::join!(response, connection);
        response.and(connection)
    };
    let (_, result) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(server, client)
    })
    .await
    .expect("finite fixture transfer");
    result
}

fn block(prefix: &str, size: usize) -> Vec<u8> {
    let suffix = "\r\n\r\n";
    let mut bytes = prefix.as_bytes().to_vec();
    bytes.resize(size - suffix.len(), b'a');
    bytes.extend_from_slice(suffix.as_bytes());
    assert_eq!(bytes.len(), size);
    bytes
}

#[tokio::test]
async fn d14_accepts_exact_final_head_with_same_packet_body() {
    let mut wire = block("HTTP/1.1 200 OK\r\nContent-Length: 1\r\nX-Pad: ", LIMIT);
    wire.push(b'x');

    let result = transfer(wire).await;

    assert!(result.is_ok(), "{result:?}");
}

#[tokio::test]
async fn d14_rejects_one_byte_over_final_head() {
    let wire = block("HTTP/1.1 200 OK\r\nContent-Length: 0\r\nX-Pad: ", LIMIT + 1);

    let result = transfer(wire).await;

    assert!(result.is_err(), "accepted oversized final head");
}

#[tokio::test]
async fn d14_accepts_independent_exact_informational_and_final_heads() {
    let mut wire = block("HTTP/1.1 103 Early Hints\r\nX-Pad: ", LIMIT);
    wire.extend(block(
        "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nX-Pad: ",
        LIMIT,
    ));

    let result = transfer(wire).await;

    assert!(result.is_ok(), "{result:?}");
}

#[tokio::test]
async fn d14_rejects_one_byte_over_informational_head() {
    let mut wire = block("HTTP/1.1 103 Early Hints\r\nX-Pad: ", LIMIT + 1);
    wire.extend_from_slice(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");

    let result = transfer(wire).await;

    assert!(result.is_err(), "accepted oversized informational head");
}

fn chunked(trailer_size: usize) -> Vec<u8> {
    let mut wire = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1\r\nx\r\n0\r\n".to_vec();
    wire.extend(block("X-Pad: ", trailer_size));
    wire
}

#[tokio::test]
async fn d14_rejects_trailer_at_16384_under_approved_stricter_boundary() {
    let wire = chunked(LIMIT);

    let result = transfer(wire).await;

    assert!(
        result.is_err(),
        "accepted trailer above approved 16383 maximum"
    );
}

#[tokio::test]
async fn d14_rejects_one_byte_over_trailer_block() {
    let wire = chunked(LIMIT + 1);

    let result = transfer(wire).await;

    assert!(result.is_err(), "accepted oversized trailer block");
}

#[tokio::test]
async fn d14_accepts_trailer_one_byte_below_limit() {
    let wire = chunked(LIMIT - 1);

    let result = transfer(wire).await;

    assert!(result.is_ok(), "{result:?}");
}

#[tokio::test]
async fn d14_fragmented_trailer_at_16384_is_rejected() {
    let wire = chunked(LIMIT);

    let result = transfer_with(wire, (LIMIT, 97)).await;

    assert!(
        result.is_err(),
        "accepted trailer above approved 16383 maximum"
    );
}

#[tokio::test]
async fn d14_fragmented_trailer_at_16383_completes() {
    let result = transfer_with(chunked(LIMIT - 1), (LIMIT, 97)).await;

    assert!(result.is_ok(), "{result:?}");
}

#[tokio::test]
async fn d14_fragmented_trailer_at_16385_is_rejected() {
    let result = transfer_with(chunked(LIMIT + 1), (LIMIT, 97)).await;

    assert!(result.is_err(), "accepted oversized fragmented trailer");
}

#[tokio::test]
async fn parser_larger_read_buffer_does_not_raise_trailer_limit() {
    let wire = chunked(LIMIT);

    let result = transfer_with(wire, (LIMIT + 1, usize::MAX)).await;

    assert!(
        result.is_err(),
        "trailer limit followed read-buffer setting"
    );
}

#[tokio::test]
async fn d14_fragmented_exact_final_head_completes() {
    let wire = block("HTTP/1.1 200 OK\r\nContent-Length: 0\r\nX-Pad: ", LIMIT);

    let result = transfer_with(wire, (LIMIT, 97)).await;

    assert!(result.is_ok(), "{result:?}");
}
