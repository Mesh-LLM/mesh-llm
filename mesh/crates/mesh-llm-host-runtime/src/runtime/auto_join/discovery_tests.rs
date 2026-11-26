use super::*;
use base64::Engine;
use futures_util::{SinkExt, StreamExt};
use nostr_sdk::prelude::*;
use tokio_tungstenite::tungstenite::Message;

async fn relay(events: Vec<Event>, eose: bool) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        let Some(Ok(Message::Text(text))) = socket.next().await else {
            panic!("expected subscription request");
        };
        let ClientMessage::Req {
            subscription_id, ..
        } = ClientMessage::from_json(text.as_str()).unwrap()
        else {
            panic!("expected REQ");
        };
        let id = subscription_id.into_owned();
        for event in events {
            socket
                .send(Message::text(
                    RelayMessage::event(id.clone(), event).as_json(),
                ))
                .await
                .unwrap();
        }
        if eose {
            socket
                .send(Message::text(RelayMessage::eose(id).as_json()))
                .await
                .unwrap();
        }
    });
    (url, server)
}

fn listing(name: &str) -> Event {
    let endpoint = iroh::EndpointAddr::new(iroh::SecretKey::from_bytes(&[7; 32]).public());
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&endpoint).unwrap());
    let listing = nostr::MeshListing {
        invite_token: token,
        serving: vec!["test-model".into()],
        wanted: vec![],
        on_disk: vec![],
        total_vram_bytes: 0,
        node_count: 1,
        client_count: 0,
        max_clients: 0,
        name: Some(name.into()),
        region: None,
        mesh_id: Some("test-mesh".into()),
        publisher_serves: Some(true),
    };
    EventBuilder::new(
        Kind::Custom(31990),
        serde_json::to_string(&listing).unwrap(),
    )
    .tag(Tag::parse(["k", "mesh-llm"]).unwrap())
    .finalize(&Keys::generate())
    .unwrap()
}

fn options(relay: String, client: bool) -> RuntimeOptions {
    RuntimeOptions {
        auto: true,
        client,
        nostr_relay: vec![relay],
        ..RuntimeOptions::default()
    }
}

#[tokio::test]
#[serial_test::serial]
async fn offline_discovery_refuses_serving_but_preserves_client_startup() {
    let _catalog = crate::models::remote_catalog::set_catalog_entries_for_test(Vec::new());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}", listener.local_addr().unwrap());
    drop(listener);
    for client in [false, true] {
        let mut options = options(url.clone(), client);
        let mut candidates = Vec::new();
        let result =
            discover_nostr_join_candidates(&mut options, true, &mut candidates, 0.0, None).await;
        if client {
            result.unwrap();
        } else {
            let error = result.unwrap_err();
            assert!(error.to_string().contains("no relay answered"), "{error}");
        }
        assert!(options.nostr_discovery);
        assert!(options.join.is_empty());
        assert!(options.model.is_empty());
        assert!(candidates.is_empty());
    }
}

#[tokio::test]
#[serial_test::serial]
async fn completed_empty_discovery_allows_serving_and_client_startup() {
    let _catalog = crate::models::remote_catalog::set_catalog_entries_for_test(Vec::new());
    for client in [false, true] {
        let (url, server) = relay(vec![], true).await;
        let mut options = options(url, client);
        let mut candidates = Vec::new();
        discover_nostr_join_candidates(&mut options, true, &mut candidates, 0.0, None)
            .await
            .unwrap();
        server.await.unwrap();
        assert!(options.nostr_discovery);
        assert!(candidates.is_empty());
    }
}

#[tokio::test]
#[serial_test::serial]
async fn incomplete_or_truncated_ineligible_results_cannot_start_a_serving_mesh() {
    let _catalog = crate::models::remote_catalog::set_catalog_entries_for_test(Vec::new());
    for (count, eose) in [(1, false), (100, true)] {
        for client in [false, true] {
            let (url, server) = relay((0..count).map(|_| listing("other")).collect(), eose).await;
            let mut options = options(url, client);
            let mut candidates = Vec::new();
            let result =
                discover_nostr_join_candidates(&mut options, true, &mut candidates, 0.0, None)
                    .await;
            if client {
                result.unwrap();
            } else {
                let error = result.unwrap_err();
                assert!(error.to_string().contains("incomplete"), "{error}");
            }
            server.await.unwrap();
            assert!(options.join.is_empty());
            assert!(options.model.is_empty());
            assert!(candidates.is_empty());
        }
    }
}

#[tokio::test]
#[serial_test::serial]
async fn partial_eligible_results_remain_joinable_for_serving_and_clients() {
    let _catalog = crate::models::remote_catalog::set_catalog_entries_for_test(Vec::new());
    for client in [false, true] {
        let (url, server) = relay(vec![listing("mesh-llm")], false).await;
        let mut options = options(url, client);
        let mut candidates = Vec::new();
        discover_nostr_join_candidates(&mut options, true, &mut candidates, 0.0, None)
            .await
            .unwrap();
        server.await.unwrap();
        if client {
            assert_eq!(options.join.len(), 1);
        } else {
            assert_eq!(candidates.len(), 1);
        }
        assert!(options.model.is_empty());
    }
}
