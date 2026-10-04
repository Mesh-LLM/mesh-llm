//! Receipt completion gates the final cancellation snapshot.
use super::*;
use crate::plugin::{
    config::{PluginHostMode, ResolvedPlugins},
    transport::LocalStream,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::test]
async fn drop_terminal_uses_final_receipt_and_late_emission_state() {
    let (mesh_tx, _mesh_rx) = tokio::sync::mpsc::channel(4);
    let manager = PluginManager::start(
        &ResolvedPlugins {
            externals: Vec::new(),
            inactive: Vec::new(),
        },
        PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        mesh_tx,
    )
    .await
    .unwrap();
    let event = request_event(
        "fixture".into(),
        "chat_completions",
        "POST",
        "/v1/chat/completions",
        b"{}",
        Default::default(),
        false,
    );
    let (mut session, _) = ExchangeSession::begin(&manager, event).await;
    let (stream, mut receiver) = tokio::io::duplex(1024);
    let grant = OpenAiExchangeGrant {
        max_body_bytes: 1024,
        max_queue_bytes: 1024,
        ..Default::default()
    };
    session.emission.lock().unwrap().copies =
        crate::plugin::exchange_streams::ResponseCopies::for_test_stream(
            LocalStream::Memory(stream),
            &grant,
            manager.exchange_grant_revision("observer"),
        );
    let emission = session.emission.clone();
    let observer = EmissionObserver(emission);
    assert!(observer.try_chunk(0, b"abc"));
    let (eof_tx, eof_rx) = tokio::sync::oneshot::channel();
    let (release_tx, release_rx) = tokio::sync::oneshot::channel();
    let peer = tokio::spawn(async move {
        let mut received = Vec::new();
        receiver.read_to_end(&mut received).await.unwrap();
        assert_eq!(received, b"abc");
        eof_tx.send(()).unwrap();
        release_rx.await.unwrap();
        receiver
            .write_all(b"{\"sha256\":\"wrong\",\"byte_count\":3}\n")
            .await
            .unwrap();
        receiver.shutdown().await.unwrap();
    });
    // Transfer exactly as Drop does, with no final commitment available yet.
    session.finished = true;
    let mut owner = session.terminal_owner();
    assert!(owner.terminal_event("client_cancelled")["response_wire_commitment"].is_null());
    drop(session);
    let terminal =
        tokio::spawn(async move { owner.final_terminal_event("client_cancelled").await });
    eof_rx.await.unwrap();
    let mut prefix = commit_wire_bytes(b"abc");
    prefix.incomplete = Some(openai_frontend::wire_bytes::WireBytesIncomplete::Cancelled);
    observer.finish(prefix);
    release_tx.send(()).unwrap();
    let terminal = terminal.await.unwrap();
    assert_eq!(terminal["execution_outcome"], "client_cancelled");
    assert_eq!(terminal["response_wire_commitment"]["byte_count"], 3);
    assert_eq!(
        terminal["response_wire_commitment"]["sha256"],
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(terminal["any_response_bytes_emitted"], true);
    assert_eq!(terminal["evidence_unavailable"], true);
    assert_eq!(terminal["evidence_complete"], false);
    assert_eq!(terminal["observer_response_delivery"]["observer"], false);
    peer.await.unwrap();
    manager.shutdown().await;
}
