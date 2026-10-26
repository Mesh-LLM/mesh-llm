//! Regression coverage for issue #2135: `MeshClient::status()` must report the
//! number of mesh peers whose connection this client actually observed, not a
//! projection of the `connected` lifecycle flag.
//!
//! Every case compares `status().peer_count` with the connections the client
//! really established: zero before any dial, one while a dialed peer's most
//! recent outcome is an established connection, and zero again once that peer
//! is lost or the client leaves the joined state.

use base64::Engine;
use iroh::{Endpoint, EndpointAddr, SecretKey};
use mesh_client::client::builder::{ClientBuilder, InviteToken, MeshClient};
use mesh_client::crypto::keys::OwnerKeypair;
use mesh_client::protocol::{ALPN_V1, STREAM_TUNNEL_HTTP};
use std::str::FromStr;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::mpsc;

const MODELS_BODY: &str = r#"{"data":[{"id":"test-model","metadata":{"context_length":4096}}]}"#;
const MAX_TEST_REQUEST_BYTES: usize = 64 * 1024;

/// Requests to a peer that cannot answer get failed quickly enough to keep the
/// suite fast, but still bounded rather than assumed.
const UNREACHABLE_CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

fn models_response() -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        MODELS_BODY.len(),
        MODELS_BODY,
    )
}

fn invite_token(addr: &EndpointAddr) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(addr).expect("endpoint addr serializes"))
}

/// An invite token for a loopback address that is guaranteed to have no
/// listener: the port came from a socket that was bound and then released.
fn unreachable_invite_token() -> InviteToken {
    let port = {
        let socket = std::net::UdpSocket::bind("127.0.0.1:0").expect("bind ephemeral UDP port");
        socket.local_addr().expect("socket addr").port()
    };
    let addr = EndpointAddr::new(SecretKey::generate().public())
        .with_ip_addr(std::net::SocketAddr::from(([127, 0, 0, 1], port)));
    InviteToken::from_str(&invite_token(&addr)).expect("invite token is non-empty")
}

fn build_client(invite_token: InviteToken) -> MeshClient {
    ClientBuilder::new(OwnerKeypair::generate(), invite_token)
        .build()
        .expect("client builds")
}

/// A stand-in mesh host: it owns a real iroh endpoint on the mesh ALPN, serves
/// the HTTP tunnel stream, and reports the request line it was asked for.
///
/// The receiver is the only synchronisation used — no sleeps.
async fn spawn_mesh_host() -> (Endpoint, InviteToken, mpsc::UnboundedReceiver<String>) {
    let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
        .expect("endpoint builder configures")
        .bind()
        .await
        .expect("endpoint binds");
    let token =
        InviteToken::from_str(&invite_token(&endpoint.addr())).expect("invite token is non-empty");
    let (served_tx, served_rx) = mpsc::unbounded_channel();
    let server = endpoint.clone();
    tokio::spawn(async move {
        while let Some(incoming) = server.accept().await {
            let Ok(connection) = incoming.await else {
                continue;
            };
            let served_tx = served_tx.clone();
            tokio::spawn(async move {
                let Ok((mut send, mut recv)) = connection.accept_bi().await else {
                    return;
                };
                let Ok(request) = recv.read_to_end(MAX_TEST_REQUEST_BYTES).await else {
                    return;
                };
                if request.first() != Some(&STREAM_TUNNEL_HTTP) {
                    return;
                }
                let request_line = String::from_utf8_lossy(&request[1..])
                    .lines()
                    .next()
                    .unwrap_or_default()
                    .to_string();
                if send.write_all(models_response().as_bytes()).await.is_err() {
                    return;
                }
                let _ = send.finish();
                let _ = served_tx.send(request_line);
                connection.closed().await;
            });
        }
    });
    (endpoint, token, served_rx)
}

/// A stand-in OpenAI-compatible HTTP host on the `OpenAiHttp` transport.
async fn spawn_http_host() -> (String, mpsc::UnboundedReceiver<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let base_url = format!("http://{}", listener.local_addr().expect("local addr"));
    let (served_tx, served_rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let mut request = Vec::new();
            let _ = stream.read_to_end(&mut request).await;
            let _ = stream.write_all(models_response().as_bytes()).await;
            let _ = stream.shutdown().await;
            let _ = served_tx.send(());
        }
    });
    (base_url, served_rx)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn status_peer_count_is_zero_before_any_dial() {
    let (_host, token, _served) = spawn_mesh_host().await;
    let mut client = build_client(token);

    client.join().await.expect("join succeeds");
    let status = client.status().await;

    assert!(
        status.connected,
        "join() puts the client in the joined lifecycle state"
    );
    assert_eq!(
        status.peer_count, 0,
        "no dial has completed, so no peer connection has been observed"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn status_peer_count_matches_established_connection() {
    let (_host, token, mut served) = spawn_mesh_host().await;
    let mut client = build_client(token);
    client.join().await.expect("join succeeds");

    let models = client
        .list_models()
        .await
        .expect("local mesh host serves /v1/models");
    let request_line = served.recv().await.expect("host served a request");
    assert_eq!(models.len(), 1, "the served model is reported");
    assert!(
        request_line.starts_with("GET /v1/models"),
        "the request really crossed the tunnel: {request_line}"
    );

    assert_eq!(
        client.status().await.peer_count,
        1,
        "one peer connection was established"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn status_peer_count_excludes_peer_whose_dial_failed() {
    let mut client = ClientBuilder::new(OwnerKeypair::generate(), unreachable_invite_token())
        .with_connect_timeout(UNREACHABLE_CONNECT_TIMEOUT)
        .build()
        .expect("client builds");

    client.join().await.expect("join succeeds");
    client
        .list_models()
        .await
        .expect_err("no listener owns the invited address");

    assert_eq!(
        client.status().await.peer_count,
        0,
        "a failed dial establishes no peer"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn status_peer_count_matches_connection_to_openai_http_host() {
    let (base_url, mut served) = spawn_http_host().await;
    let mut client = ClientBuilder::new(OwnerKeypair::generate(), unreachable_invite_token())
        .with_openai_http_transport(base_url)
        .build()
        .expect("client builds");

    client.join().await.expect("join succeeds");
    assert_eq!(
        client.status().await.peer_count,
        0,
        "join() alone establishes nothing"
    );

    let _ = client.list_models().await;
    served.recv().await.expect("host served a request");

    assert_eq!(
        client.status().await.peer_count,
        1,
        "the host this client connected to is one peer"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn status_peer_count_clears_on_disconnect() {
    let (_host, token, mut served) = spawn_mesh_host().await;
    let mut client = build_client(token);
    client.join().await.expect("join succeeds");
    client
        .list_models()
        .await
        .expect("local mesh host serves /v1/models");
    served.recv().await.expect("host served a request");
    assert_eq!(client.status().await.peer_count, 1);

    client.disconnect().await;
    let status = client.status().await;

    assert!(!status.connected, "disconnect leaves the joined state");
    assert_eq!(
        status.peer_count, 0,
        "a disconnected client holds no peer connections"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn status_peer_count_clears_on_reconnect() {
    let (_host, token, mut served) = spawn_mesh_host().await;
    let mut client = build_client(token);
    client.join().await.expect("join succeeds");
    client
        .list_models()
        .await
        .expect("local mesh host serves /v1/models");
    served.recv().await.expect("host served a request");
    assert_eq!(client.status().await.peer_count, 1);

    client.reconnect().await.expect("reconnect succeeds");

    assert_eq!(
        client.status().await.peer_count,
        0,
        "rejoining has not dialed anything yet"
    );
}
