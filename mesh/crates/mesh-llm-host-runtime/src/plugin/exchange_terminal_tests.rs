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
    prefix.incomplete = Some(skippy_inference_api::wire_bytes::WireBytesIncomplete::Cancelled);
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

#[tokio::test]
async fn failed_denial_write_terminal_preserves_route_cancellation_without_manual_override() {
    use crate::network::{openai::client_stream::ClientStream, proxy::send_error_observed};
    use tokio::net::{TcpListener, TcpStream};
    for denied in [false, true] {
        let (tx, _rx) = tokio::sync::mpsc::channel(4);
        let manager = PluginManager::start(
            &ResolvedPlugins {
                externals: Vec::new(),
                inactive: Vec::new(),
            },
            PluginHostMode {
                mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
            },
            tx,
        )
        .await
        .unwrap();
        let (mut session, _) = ExchangeSession::begin(
            &manager,
            request_event(
                "failed-denial".into(),
                "chat_completions",
                "POST",
                "/v1/chat/completions",
                b"{}",
                Default::default(),
                false,
            ),
        )
        .await;
        {
            let mut emission = session.emission.lock().unwrap();
            emission.denied = denied;
            emission.required_failure = !denied;
        }
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let _client = TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        let (mut writer, _) = listener.accept().await.unwrap();
        writer.shutdown().await.unwrap();
        let stream = ClientStream::from(writer).with_wire_bytes_observer(session.observer());
        let written = send_error_observed(
            stream,
            if denied { 403 } else { 503 },
            "rejected",
            crate::logging::OpenAiLifecycleAttachment::unowned().route_observer(),
        )
        .await;
        assert!(written.is_err());
        let event = session.final_terminal_event("client_cancelled").await;
        assert_eq!(event["execution_outcome"], "client_cancelled");
        assert_eq!(event["admission_denied"], denied);
        assert_eq!(event["required_admission_failure"], !denied);
        assert_eq!(event["response_wire_commitment"]["byte_count"], 0);
        assert_eq!(
            event["response_wire_commitment"]["sha256"],
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert!(!event["response_wire_commitment"]["incomplete"].is_null());
        assert_eq!(event["any_response_bytes_emitted"], false);
        session.finish("client_cancelled").await;
        manager.shutdown().await;
    }
}

async fn empty_manager() -> PluginManager {
    let (tx, _rx) = tokio::sync::mpsc::channel(4);
    PluginManager::start(
        &ResolvedPlugins {
            externals: Vec::new(),
            inactive: Vec::new(),
        },
        PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        tx,
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn typed_denial_body_drop_reports_cancellation_and_exact_polled_prefix() {
    for denied in [false, true] {
        for poll_prefix in [false, true] {
            assert_typed_error_body_drop(
                denied,
                !denied,
                if denied {
                    "policy_denied"
                } else {
                    "internal_hook_failure"
                },
                if denied { 403 } else { 503 },
                poll_prefix,
            )
            .await;
        }
    }
}

#[tokio::test]
async fn typed_backend_timeout_error_body_drop_reports_cancellation_before_and_after_polling() {
    for poll_prefix in [false, true] {
        assert_typed_error_body_drop(false, false, "timed_out", 504, poll_prefix).await;
    }
}

async fn assert_typed_error_body_drop(
    denied: bool,
    required_failure: bool,
    recorded_outcome: &str,
    status: u16,
    poll_prefix: bool,
) {
    use axum::body::Body;
    use bytes::Bytes;
    use futures_util::StreamExt;
    use http_body_util::BodyExt;
    use skippy_inference_api::wire_bytes::observe_response_body;
    let manager = empty_manager().await;
    let (session, _) = ExchangeSession::begin(
        &manager,
        request_event(
            "typed-denial".into(),
            "chat_completions",
            "POST",
            "/v1/chat/completions",
            b"{}",
            Default::default(),
            false,
        ),
    )
    .await;
    {
        let mut emission = session.emission.lock().unwrap();
        emission.denied = denied;
        emission.required_failure = required_failure;
        emission.execution_outcome = Some(recorded_outcome.into());
    }
    let owner = session.terminal_owner();
    let observer = super::super::exchange_policy::TypedEmission::new(session);
    observer.response_status(status);
    let first = futures_util::stream::once(async {
        Ok::<_, std::io::Error>(Bytes::from_static(b"partial denial"))
    });
    let mut body = observe_response_body(
        Body::from_stream(first.chain(futures_util::stream::pending())),
        observer,
    );
    if poll_prefix {
        assert_eq!(
            body.frame().await.unwrap().unwrap().into_data().unwrap(),
            Bytes::from_static(b"partial denial")
        );
    }
    drop(body);
    let event = owner.terminal_event(recorded_outcome);
    assert_eq!(event["execution_outcome"], "client_cancelled");
    assert_eq!(event["admission_denied"], denied);
    assert_eq!(event["required_admission_failure"], required_failure);
    let expected = commit_wire_bytes(if poll_prefix { b"partial denial" } else { b"" });
    assert_eq!(event["response_wire_commitment"]["sha256"], expected.sha256);
    assert_eq!(
        event["response_wire_commitment"]["byte_count"],
        expected.byte_count
    );
    assert_eq!(event["response_wire_commitment"]["incomplete"], "cancelled");
    assert_eq!(event["evidence_complete"], false);
    tokio::task::yield_now().await;
    manager.shutdown().await;
}

#[tokio::test]
async fn partial_raw_denial_write_keeps_actual_prefix_and_transport_failure() {
    use crate::network::openai::client_stream::ClientStream;
    use tokio::net::{TcpListener, TcpStream};
    for denied in [false, true] {
        let manager = empty_manager().await;
        let (mut session, _) = ExchangeSession::begin(
            &manager,
            request_event(
                "partial-raw-denial".into(),
                "chat_completions",
                "POST",
                "/v1/chat/completions",
                b"{}",
                Default::default(),
                false,
            ),
        )
        .await;
        {
            let mut emission = session.emission.lock().unwrap();
            emission.denied = denied;
            emission.required_failure = !denied;
        }
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        let (writer, _) = listener.accept().await.unwrap();
        let mut stream = ClientStream::from(writer).with_wire_bytes_observer(session.observer());
        let prefix = format!(
            "HTTP/1.1 {} rejected\r\nContent-Length: 20\r\n\r\npartial",
            if denied { 403 } else { 503 }
        );
        stream.write_all(prefix.as_bytes()).await.unwrap();
        let mut received = vec![0u8; prefix.len()];
        client.read_exact(&mut received).await.unwrap();
        assert_eq!(received, prefix.as_bytes());
        stream.shutdown().await.unwrap();
        assert!(stream.write_all(b" remaining denial").await.is_err());
        let event = session.final_terminal_event("policy_denied").await;
        assert_eq!(event["execution_outcome"], "transport_error");
        assert_eq!(event["response_wire_commitment"]["byte_count"], 7);
        assert_eq!(
            event["response_wire_commitment"]["sha256"],
            commit_wire_bytes(b"partial").sha256
        );
        assert_eq!(event["admission_denied"], denied);
        assert_eq!(event["required_admission_failure"], !denied);
        session.finish("policy_denied").await;
        manager.shutdown().await;
    }
}
