use super::*;
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

/// What a scripted relay does after it receives the REQ.
enum Script {
    /// Send these events, then EOSE, then these live events.
    Answer {
        stored: Vec<Event>,
        live: Vec<Event>,
    },
    /// Send these events, then drop the connection before EOSE.
    DropAfter(Vec<Event>),
    /// Close the subscription without EOSE.
    Closed,
    /// Accept the REQ and never answer.
    Hang,
}

async fn scripted_relay(script: Script) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        let id = loop {
            let Some(Ok(Message::Text(text))) = socket.next().await else {
                return;
            };
            if let Ok(ClientMessage::Req {
                subscription_id, ..
            }) = ClientMessage::from_json(text.as_str())
            {
                break subscription_id.into_owned();
            }
        };
        let send = |message: RelayMessage| Message::text(message.as_json());
        match script {
            Script::Answer { stored, live } => {
                for event in stored {
                    let _ = socket
                        .send(send(RelayMessage::event(id.clone(), event)))
                        .await;
                }
                let _ = socket.send(send(RelayMessage::eose(id.clone()))).await;
                for event in live {
                    let _ = socket
                        .send(send(RelayMessage::event(id.clone(), event)))
                        .await;
                }
            }
            Script::DropAfter(events) => {
                for event in events {
                    let _ = socket
                        .send(send(RelayMessage::event(id.clone(), event)))
                        .await;
                }
                return;
            }
            Script::Closed => {
                let closed = RelayMessage::closed(id, "error: rate-limited");
                let _ = socket.send(send(closed)).await;
            }
            Script::Hang => {}
        }
        // Keep the connection open until the client goes away.
        while socket.next().await.is_some() {}
    });
    format!("ws://{addr}")
}

fn listing(name: &str) -> Event {
    let listing = MeshListing {
        invite_token: super::integration_tests::fake_invite_token(),
        serving: vec![],
        wanted: vec![],
        on_disk: vec![],
        total_vram_bytes: 0,
        node_count: 1,
        client_count: 0,
        max_clients: 0,
        name: Some(name.to_string()),
        region: None,
        mesh_id: Some(format!("id-{name}")),
        publisher_serves: None,
    };
    EventBuilder::new(
        Kind::Custom(MESH_SERVICE_KIND),
        serde_json::to_string(&listing).unwrap(),
    )
    .tag(Tag::parse(["k", "mesh-llm"]).unwrap())
    .finalize(&Keys::generate())
    .unwrap()
}

/// A port nothing listens on: the connection is refused at once.
async fn refused_relay() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    format!("ws://{addr}")
}

const DEADLINE: Duration = Duration::from_secs(2);

#[tokio::test]
async fn relay_urls_are_normalized_before_the_websocket_request() {
    let relay = scripted_relay(Script::Answer {
        stored: vec![],
        live: vec![],
    })
    .await;
    // A raw http::Uri rejects this path; RelayUrl percent-encodes it first.
    let (_, health) = discover_round(&[format!("{relay}/日本")], &MeshFilter::default(), DEADLINE)
        .await
        .unwrap();
    assert!(health.any_query_completed());
    assert_eq!(
        RelayUrl::parse("wss://bücher.example/").unwrap().as_str(),
        "wss://xn--bcher-kva.example/"
    );
}

#[tokio::test]
async fn completed_queries_count_and_refused_relays_are_silent() {
    let relays = vec![
        scripted_relay(Script::Answer {
            stored: vec![listing("a")],
            live: vec![],
        })
        .await,
        scripted_relay(Script::Answer {
            stored: vec![],
            live: vec![],
        })
        .await,
        refused_relay().await,
    ];
    let (meshes, health) = discover_round(&relays, &MeshFilter::default(), DEADLINE)
        .await
        .unwrap();
    assert_eq!(meshes.len(), 1);
    assert_eq!(
        health,
        RelayHealth {
            relays_queried: 3,
            relays_answered: 2,
            queries_completed: 2,
        }
    );
}

#[tokio::test]
async fn a_relay_that_drops_before_eose_keeps_its_events_but_completes_nothing() {
    let relays = vec![scripted_relay(Script::DropAfter(vec![listing("other")])).await];
    let (meshes, health) = discover_round(&relays, &MeshFilter::default(), DEADLINE)
        .await
        .unwrap();
    assert_eq!(meshes.len(), 1);
    assert_eq!(meshes[0].listing.name.as_deref(), Some("other"));
    assert!(health.any_relay_answered());
    assert!(!health.any_query_completed());
}

#[tokio::test]
async fn closed_and_hung_relays_are_silent_and_the_round_ends_on_time() {
    let relays = vec![
        scripted_relay(Script::Closed).await,
        scripted_relay(Script::Hang).await,
    ];
    let (_, health) = discover_round(&relays, &MeshFilter::default(), DEADLINE)
        .await
        .unwrap();
    assert!(!health.any_relay_answered());
}

#[tokio::test]
async fn events_sent_after_eose_are_not_collected() {
    let relays = vec![
        scripted_relay(Script::Answer {
            stored: vec![listing("stored")],
            live: vec![listing("live")],
        })
        .await,
    ];
    let (meshes, health) = discover_round(&relays, &MeshFilter::default(), DEADLINE)
        .await
        .unwrap();
    let names: Vec<_> = meshes
        .iter()
        .filter_map(|m| m.listing.name.as_deref())
        .collect();
    assert_eq!(names, ["stored"]);
    assert_eq!(health.queries_completed, 1);
}

#[tokio::test]
async fn eose_at_the_requested_limit_keeps_listings_but_cannot_establish_absence() {
    for count in [QUERY_EVENT_LIMIT - 1, QUERY_EVENT_LIMIT] {
        let relay = scripted_relay(Script::Answer {
            stored: (0..count).map(|i| listing(&format!("mesh-{i}"))).collect(),
            live: vec![],
        })
        .await;
        let (meshes, health) = discover_round(&[relay], &MeshFilter::default(), DEADLINE)
            .await
            .unwrap();
        assert_eq!(meshes.len(), count);
        assert!(health.any_relay_answered());
        assert_eq!(health.any_query_completed(), count < QUERY_EVENT_LIMIT);
    }
}

#[tokio::test]
async fn a_filtered_out_full_window_does_not_prove_the_target_is_absent() {
    let relay = scripted_relay(Script::Answer {
        stored: (0..QUERY_EVENT_LIMIT)
            .map(|i| listing(&format!("other-{i}")))
            .collect(),
        live: vec![],
    })
    .await;
    let filter = MeshFilter {
        name: Some("target".into()),
        ..Default::default()
    };
    let (meshes, health) = discover_round(&[relay], &filter, DEADLINE).await.unwrap();
    assert!(meshes.is_empty());
    assert!(health.any_relay_answered());
    assert!(!health.any_query_completed());
}
