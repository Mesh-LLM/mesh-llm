use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn wire(bytes: Vec<u8>, status_endpoint: bool) -> Result<Transfer, TransferError> {
    let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .unwrap();
    let port = listener.local_addr().unwrap().port();
    let bytes = if status_endpoint {
        String::from_utf8(bytes)
            .unwrap()
            .replace("PORT", &format!("{port:05}"))
            .into_bytes()
    } else {
        bytes
    };
    let request = Request {
        endpoint: if status_endpoint {
            Endpoint::Status { leader_pid: 42 }
        } else {
            Endpoint::Models {
                request_id: RequestId::generate().unwrap(),
            }
        },
        deadline: Instant::now() + std::time::Duration::from_secs(5),
    };
    let server = async {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = [0; 4096];
        assert!(stream.read(&mut request).await.unwrap() > 0);
        for chunk in bytes.chunks(97) {
            if let Err(error) = stream.write_all(chunk).await {
                assert!(
                    [
                        std::io::ErrorKind::BrokenPipe,
                        std::io::ErrorKind::ConnectionReset
                    ]
                    .contains(&error.kind())
                );
                return;
            }
            tokio::task::yield_now().await;
        }
    };
    let (_, result) = tokio::time::timeout(std::time::Duration::from_secs(7), async {
        tokio::join!(
            server,
            transfer(
                Ports {
                    api: port,
                    console: port,
                    quic: 1
                },
                request
            )
        )
    })
    .await
    .unwrap();
    result
}

fn block(prefix: &str, size: usize) -> Vec<u8> {
    let mut bytes = prefix.as_bytes().to_vec();
    bytes.resize(size - 4, b'a');
    bytes.extend_from_slice(b"\r\n\r\n");
    bytes
}

#[tokio::test]
async fn d14_adapter_exact_head() {
    let bytes = block("HTTP/1.1 200 OK\r\nContent-Length: 0\r\nX-Pad: ", 16384);
    assert!(wire(bytes, false).await.is_ok());
}
#[tokio::test]
async fn d14_adapter_excess_head() {
    let bytes = block("HTTP/1.1 200 OK\r\nContent-Length: 0\r\nX-Pad: ", 16385);
    assert_eq!(
        wire(bytes, false).await,
        Err(TransferError::Rejected(Rejection::ResponseLimit))
    );
}
#[tokio::test]
async fn d14_adapter_trailer_classification() {
    for size in [16383, 16384, 16385] {
        let mut bytes = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n".to_vec();
        bytes.extend(block("X-Pad: ", size));
        let result = wire(bytes, false).await;
        if size == 16383 {
            assert!(result.is_ok());
        } else {
            assert_eq!(
                result,
                Err(TransferError::Rejected(Rejection::ResponseLimit))
            );
        }
    }
}
#[tokio::test]
async fn d14_status_body_bound() {
    for size in [BODY_LIMIT, BODY_LIMIT + 1] {
        let mut body =
            br#"{"api_port":PORT,"local_instances":[{"pid":42,"is_self":true}]}"#.to_vec();
        body.resize(size - 1, b' ');
        let mut bytes = format!("HTTP/1.1 200 OK\r\nContent-Length: {size}\r\n\r\n").into_bytes();
        bytes.extend(body);
        let result = wire(bytes, true).await;
        if size == BODY_LIMIT {
            assert!(result.is_ok(), "{result:?}");
        } else {
            assert_eq!(
                result,
                Err(TransferError::Rejected(Rejection::ResponseLimit))
            );
        }
    }
}
